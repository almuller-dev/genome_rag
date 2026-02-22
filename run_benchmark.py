#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional dependency: qdrant-client
try:
    from qdrant_client import QdrantClient
except Exception:
    QdrantClient = None


# -----------------------------
# Shared helpers
# -----------------------------

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def local_deterministic_embedding(text: str, dim: int = 384) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    out: List[float] = []
    seed = int.from_bytes(h[:8], "big", signed=False)
    x = seed
    for _ in range(dim):
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        v = ((x % 2000000) / 1000000.0) - 1.0
        out.append(float(v))
    return out

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)) + 1e-12
    nb = math.sqrt(sum(y*y for y in b)) + 1e-12
    return dot / (na * nb)


def _openai_client(cfg: Dict[str, Any]):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "OpenAI embeddings requested but openai SDK is not installed. Install with: pip install openai"
        ) from e

    api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI embeddings requested but no API key found. "
            "Set openai_api_key in rag_config.json or OPENAI_API_KEY in env."
        )

    kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = cfg.get("openai_base_url") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    org = cfg.get("openai_organization") or os.getenv("OPENAI_ORG_ID")
    if org:
        kwargs["organization"] = org
    return OpenAI(**kwargs)

def embed_query(text: str, cfg: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
    embed_model = cfg.get("embed_model", "local-deterministic")
    dim = int(cfg.get("embed_dim", 384))

    if embed_model == "local-deterministic":
        return local_deterministic_embedding(text, dim=dim), {"provider": "local", "model": embed_model, "dim": dim}

    if str(embed_model).startswith("text-embedding-"):
        client = _openai_client(cfg)
        req: Dict[str, Any] = {"model": embed_model, "input": [text], "dimensions": dim}
        resp = client.embeddings.create(**req)
        rows = sorted(resp.data, key=lambda r: r.index)
        if not rows:
            raise RuntimeError("OpenAI embeddings returned no vectors for query.")
        vec = list(rows[0].embedding)
        return vec, {"provider": "openai", "model": embed_model, "dim": len(vec)}

    raise RuntimeError(
        f"Unsupported embed_model={embed_model!r}. "
        "Use 'local-deterministic' or an OpenAI model like 'text-embedding-3-large'."
    )

def ensure_qdrant(cfg: Dict[str, Any]) -> "QdrantClient":
    if QdrantClient is None:
        raise RuntimeError("qdrant-client is not installed. Install with: pip install qdrant-client")
    url = cfg.get("qdrant_url", "http://localhost:6333")
    api_key = cfg.get("qdrant_api_key")
    return QdrantClient(url=url, api_key=api_key)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(text or "").lower())


def _ctx_key(ctx: Dict[str, Any]) -> str:
    return f"{ctx.get('doc_id', '')}::{ctx.get('chunk_id', -1)}"


class LexicalIndex:
    """
    Lightweight BM25-style lexical index over chunk payload text.
    Built once per benchmark run for hybrid retrieval mode.
    """

    def __init__(self, rows: List[Dict[str, Any]], *, k1: float = 1.2, b: float = 0.75):
        self.rows = rows
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(rows)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_len: List[int] = []
        for i, r in enumerate(rows):
            toks = _tokenize(r.get("text", ""))
            counts = Counter(toks)
            dlen = int(sum(counts.values()))
            self.doc_len.append(dlen)
            for term, tf in counts.items():
                self.postings.setdefault(term, []).append((i, int(tf)))
        self.avgdl = (sum(self.doc_len) / float(self.n_docs)) if self.n_docs > 0 else 1.0
        if self.avgdl <= 0.0:
            self.avgdl = 1.0
        self.idf: Dict[str, float] = {}
        for term, plist in self.postings.items():
            df = float(len(plist))
            self.idf[term] = math.log(1.0 + ((self.n_docs - df + 0.5) / (df + 0.5)))

    def search(self, query: str, *, top_n: int) -> List[Dict[str, Any]]:
        if self.n_docs == 0:
            return []
        q_terms = sorted(set(_tokenize(query)))
        if not q_terms:
            return []

        scores: Dict[int, float] = {}
        for term in q_terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf.get(term, 0.0)
            for idx, tf in plist:
                dlen = float(self.doc_len[idx]) if idx < len(self.doc_len) else 1.0
                denom = float(tf) + self.k1 * (1.0 - self.b + self.b * (dlen / self.avgdl))
                if denom <= 1e-12:
                    continue
                s = idf * ((float(tf) * (self.k1 + 1.0)) / denom)
                scores[idx] = scores.get(idx, 0.0) + s

        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_n))]
        out: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(ranked, start=1):
            row = dict(self.rows[idx])
            row["score"] = float(score)
            row["lexical_score"] = float(score)
            row["lexical_rank"] = rank
            row["source"] = "lexical"
            out.append(row)
        return out


def iter_collection_payload_rows(
    client: "QdrantClient",
    collection: str,
    *,
    batch_size: int = 512,
) -> Iterable[Dict[str, Any]]:
    """
    Streams payload rows ({doc_id, chunk_id, text}) via Qdrant scroll.
    Handles tuple/object returns across qdrant-client versions.
    """
    offset: Any = None
    while True:
        resp = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=int(batch_size),
            offset=offset,
        )

        points: List[Any] = []
        next_offset: Any = None
        if isinstance(resp, tuple) and len(resp) == 2:
            points, next_offset = resp
        elif hasattr(resp, "points"):
            points = list(getattr(resp, "points", []) or [])
            next_offset = getattr(resp, "next_page_offset", None)
        elif isinstance(resp, list):
            points = resp
            next_offset = None

        for p in points:
            payload = getattr(p, "payload", None) or {}
            chunk_id = payload.get("chunk_id", -1)
            try:
                chunk_id = int(chunk_id)
            except Exception:
                chunk_id = -1
            yield {
                "doc_id": str(payload.get("doc_id", "")),
                "chunk_id": chunk_id,
                "text": str(payload.get("text", "")),
            }

        if next_offset is None or not points:
            break
        offset = next_offset


def build_lexical_index(client: "QdrantClient", collection: str, cfg: Dict[str, Any]) -> LexicalIndex:
    batch = int(cfg.get("hybrid_scroll_batch_size", 512))
    rows = list(iter_collection_payload_rows(client, collection, batch_size=max(64, batch)))
    return LexicalIndex(rows)


def fuse_rrf(
    dense_hits: List[Dict[str, Any]],
    lexical_hits: List[Dict[str, Any]],
    *,
    top_n: int,
    dense_weight: float,
    lexical_weight: float,
    rrf_k: int,
) -> List[Dict[str, Any]]:
    acc: Dict[str, Dict[str, Any]] = {}

    for rank, h in enumerate(dense_hits, start=1):
        key = _ctx_key(h)
        cur = acc.get(key)
        if cur is None:
            cur = dict(h)
            acc[key] = cur
        cur["dense_rank"] = rank
        cur["dense_score"] = float(h.get("score", 0.0))
        cur["source"] = "dense+rrf"
        cur["rrf_score"] = float(cur.get("rrf_score", 0.0)) + (float(dense_weight) / float(int(rrf_k) + rank))

    for rank, h in enumerate(lexical_hits, start=1):
        key = _ctx_key(h)
        cur = acc.get(key)
        if cur is None:
            cur = dict(h)
            acc[key] = cur
        if not cur.get("text"):
            cur["text"] = h.get("text", "")
        cur["lexical_rank"] = rank
        cur["lexical_score"] = float(h.get("score", 0.0))
        cur["source"] = "dense+rrf"
        cur["rrf_score"] = float(cur.get("rrf_score", 0.0)) + (float(lexical_weight) / float(int(rrf_k) + rank))

    out = sorted(acc.values(), key=lambda x: float(x.get("rrf_score", 0.0)), reverse=True)
    return out[: max(1, int(top_n))]


def _lexical_overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q or not t:
        return 0.0
    inter = len(q.intersection(t))
    return float(inter) / (math.sqrt(float(len(q))) * math.sqrt(float(len(t))) + 1e-12)


def rerank_contexts(query: str, candidates: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    top_k = int(cfg.get("top_k", 5))
    rerank_top_n = int(cfg.get("rerank_top_n", max(top_k, 20)))
    pool_n = max(top_k, rerank_top_n)
    pool = [dict(c) for c in candidates[:pool_n]]

    lexical_weight = float(cfg.get("reranker_lexical_weight", 0.8))
    lexical_weight = max(0.0, min(1.0, lexical_weight))
    base_weight = 1.0 - lexical_weight

    for i, c in enumerate(pool, start=1):
        base = float(c.get("rrf_score", c.get("score", 0.0)))
        lex = _lexical_overlap_score(query, str(c.get("text", "")))
        c["pre_rerank_rank"] = i
        c["lexical_overlap"] = lex
        c["rerank_score"] = (lexical_weight * lex) + (base_weight * base)

    pool.sort(key=lambda x: float(x.get("rerank_score", 0.0)), reverse=True)
    return pool[: max(1, top_k)]


# -----------------------------
# Retrieval + prompt
# -----------------------------

def qdrant_search(
    client: "QdrantClient",
    collection: str,
    qvec: List[float],
    *,
    top_k: int,
    ef: Optional[int] = None,
) -> List[Dict[str, Any]]:
    # qdrant-client supports search params; ef is in "search_params"
    search_params = None
    if ef is not None:
        search_params = {"hnsw_ef": int(ef)}

    hits = client.search(
        collection_name=collection,
        query_vector=qvec,
        limit=int(top_k),
        with_payload=True,
        search_params=search_params,
    )
    out: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        out.append({
            "score": float(h.score),
            "text": payload.get("text", ""),
            "doc_id": payload.get("doc_id", ""),
            "chunk_id": payload.get("chunk_id", -1),
        })
    return out

def build_prompt(query: str, contexts: List[Dict[str, Any]], cfg: Dict[str, Any]) -> str:
    # ---- Integration point: your house prompt format ----
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        ctx_lines.append(f"[Context {i}] {c.get('text','')}")
    ctx_block = "\n\n".join(ctx_lines)

    return (
        "You are a helpful support assistant.\n"
        "Answer the user using ONLY the provided context when possible.\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        "ANSWER:\n"
    )


# -----------------------------
# LLM call (stub)
# -----------------------------

def call_llm(prompt: str, cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Calls OpenAI using the same API key config path as embeddings.
    Supports Responses API first, then falls back to Chat Completions.
    """
    model = cfg.get("llm_model", "gpt-5.2")
    temp = float(cfg.get("prompt_temperature", 0.0))
    max_output_tokens = int(cfg.get("llm_max_output_tokens", 512))

    if model == "stub":
        answer = "I found relevant information in the context above. " \
                 "Please follow the steps described there. (stubbed LLM)"
        llm_meta = {
            "provider": "stub",
            "model": model,
            "temperature": temp,
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": 0.0,
        }
        return answer, llm_meta

    client = _openai_client(cfg)

    # Try Responses API first.
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temp,
            max_output_tokens=max_output_tokens,
        )
        answer = (getattr(resp, "output_text", None) or "").strip()
        if not answer:
            # Fallback parse for SDK variants where output_text is unavailable/empty
            chunks: List[str] = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    text = getattr(c, "text", None)
                    if text:
                        chunks.append(str(text))
            answer = "\n".join(chunks).strip()

        usage = getattr(resp, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        return answer, {
            "provider": "openai",
            "api": "responses",
            "model": model,
            "temperature": temp,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": 0.0,
        }
    except Exception:
        # Fall back to chat.completions for compatibility.
        comp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_output_tokens,
        )
        answer = (comp.choices[0].message.content or "").strip()
        usage = comp.usage
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        return answer, {
            "provider": "openai",
            "api": "chat.completions",
            "model": model,
            "temperature": temp,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": 0.0,
        }


# -----------------------------
# Scoring
# -----------------------------

def score_answer(gold: str, pred: str) -> float:
    """
    Simple scoring:
      - 1.0 if pred contains gold (case-insensitive) OR exact match
      - else 0.0
    Replace with your real metric: MRR, rubric grading, LLM judge, etc.
    """
    g = (gold or "").strip().lower()
    p = (pred or "").strip().lower()
    if not g:
        return 0.0
    if p == g:
        return 1.0
    if g in p:
        return 1.0
    return 0.0


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def is_unanswerable_case(case: Dict[str, Any]) -> bool:
    return bool(case.get("expected_unanswerable", False))


def is_abstention(answer: str) -> bool:
    a = _norm(answer)
    if not a:
        return True
    markers = [
        "i don't know",
        "i do not know",
        "not enough information",
        "insufficient information",
        "cannot answer",
        "can't answer",
        "not in the provided context",
        "not provided in the context",
        "context does not contain",
        "unable to determine",
        "no relevant information",
    ]
    return any(m in a for m in markers)


def score_answer_simple(case: Dict[str, Any], gold: str, pred: str) -> float:
    if is_unanswerable_case(case):
        return 1.0 if is_abstention(pred) else 0.0
    return score_answer(gold, pred)


def _extract_first_json_object(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def judge_answer_llm(
    *,
    query: str,
    gold: str,
    answer: str,
    contexts: List[Dict[str, Any]],
    case: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    model = str(cfg.get("judge_model") or cfg.get("llm_model") or "gpt-5.2")
    max_out = int(cfg.get("judge_max_output_tokens", 220))
    temp = float(cfg.get("judge_temperature", 0.0))

    ctx_lines = []
    for i, c in enumerate(contexts[:5], start=1):
        ctx_lines.append(f"[Context {i}] {str(c.get('text', ''))}")
    ctx_block = "\n\n".join(ctx_lines)
    should_abstain = is_unanswerable_case(case)

    prompt = (
        "You are grading a RAG answer. Return ONLY strict JSON with keys:\n"
        "correctness (0..1), groundedness (0..1), abstained (true/false), should_abstain (true/false), notes (short).\n"
        "Rules:\n"
        "- correctness: factual correctness w.r.t. expected answer.\n"
        "- groundedness: how well the answer is supported by provided context only.\n"
        "- If should_abstain is true, correctness should be 1 only if answer abstains.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"EXPECTED_ANSWER:\n{gold}\n\n"
        f"SHOULD_ABSTAIN:\n{str(should_abstain).lower()}\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        f"MODEL_ANSWER:\n{answer}\n"
    )

    client = _openai_client(cfg)
    raw = ""
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temp,
            max_output_tokens=max_out,
        )
        raw = (getattr(resp, "output_text", None) or "").strip()
        if not raw:
            chunks: List[str] = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    t = getattr(c, "text", None)
                    if t:
                        chunks.append(str(t))
            raw = "\n".join(chunks).strip()
    except Exception:
        raw = ""

    obj = _extract_first_json_object(raw) or {}
    abstained = bool(obj.get("abstained", is_abstention(answer)))
    out = {
        "correctness": float(obj.get("correctness", score_answer_simple(case, gold, answer))),
        "groundedness": float(obj.get("groundedness", 0.0)),
        "abstained": abstained,
        "should_abstain": bool(obj.get("should_abstain", should_abstain)),
        "notes": str(obj.get("notes", "")),
        "judge_model": model,
    }
    out["correctness"] = max(0.0, min(1.0, float(out["correctness"])))
    out["groundedness"] = max(0.0, min(1.0, float(out["groundedness"])))
    return out


def _extract_relevant_doc_ids(case: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if case.get("relevant_doc_id") is not None:
        out.append(str(case.get("relevant_doc_id")))
    ids = case.get("relevant_doc_ids")
    if isinstance(ids, list):
        out.extend(str(x) for x in ids if x is not None)
    return [x for x in out if x]


def _extract_relevant_chunk_ids(case: Dict[str, Any]) -> List[int]:
    out: List[int] = []
    if case.get("relevant_chunk_id") is not None:
        try:
            out.append(int(case.get("relevant_chunk_id")))
        except Exception:
            pass
    ids = case.get("relevant_chunk_ids")
    if isinstance(ids, list):
        for x in ids:
            try:
                out.append(int(x))
            except Exception:
                continue
    return out


def _context_match_reason(
    ctx: Dict[str, Any],
    case: Dict[str, Any],
    gold: str,
    *,
    allow_gold_fallback: bool,
) -> Optional[str]:
    ctx_doc = str(ctx.get("doc_id", ""))
    ctx_chunk = ctx.get("chunk_id")
    ctx_text = _norm(ctx.get("text", ""))

    rel_doc_ids = _extract_relevant_doc_ids(case)
    rel_chunk_ids = _extract_relevant_chunk_ids(case)
    rel_text = _norm(case.get("relevant_text", ""))

    # Strongest signal: explicit doc/chunk IDs
    if rel_doc_ids:
        if ctx_doc not in rel_doc_ids:
            return None
        if rel_chunk_ids:
            try:
                if int(ctx_chunk) in rel_chunk_ids:
                    return "doc_id+chunk_id"
                return None
            except Exception:
                return None
        return "doc_id"

    # Next signal: explicit reference text
    if rel_text:
        if rel_text in ctx_text:
            return "relevant_text"
        return None

    # Fallback: answer gold appears in context text
    if allow_gold_fallback:
        g = _norm(gold)
        if g:
            if g in ctx_text:
                return "gold_substring"
    return None


def retrieval_case_metrics(
    contexts: List[Dict[str, Any]],
    case: Dict[str, Any],
    gold: str,
    *,
    allow_gold_fallback: bool,
) -> Tuple[bool, float, Optional[int], str]:
    for rank, ctx in enumerate(contexts, start=1):
        reason = _context_match_reason(ctx, case, gold, allow_gold_fallback=allow_gold_fallback)
        if reason is not None:
            return True, 1.0 / float(rank), rank, reason
    return False, 0.0, None, "none"


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="Path to rag_config.json")
    ap.add_argument("--bench", required=False, type=Path, default=Path("bench/support_tickets.jsonl"),
                    help="JSONL with {'query':..., 'gold':...}")
    ap.add_argument("--collection", required=False, type=str, default=None,
                    help="Override Qdrant collection name. If not set, expects config to provide it.")
    ap.add_argument("--index-ready", required=False, type=Path, default=None,
                    help="Optional path to INDEX_READY.json written by ingest_docs.py to locate collection.")
    ap.add_argument("--debug-cases-out", required=False, type=Path, default=None,
                    help="Optional JSONL path to write per-case debug records (hit rank/matched_by/etc).")
    ap.add_argument("--debug-cases-limit", required=False, type=int, default=0,
                    help="Max number of case debug rows to write (0 = all).")
    args = ap.parse_args(argv)

    cfg = read_json(args.config)
    answer_scoring_mode = str(cfg.get("answer_scoring_mode", "simple")).strip().lower()
    if answer_scoring_mode not in {"simple", "llm_judge"}:
        raise RuntimeError("answer_scoring_mode must be 'simple' or 'llm_judge'")
    allow_gold_fallback = bool(cfg.get("retrieval_label_fallback_to_gold", False))

    # Determine collection name
    collection = args.collection or cfg.get("qdrant_collection_effective")
    if args.index_ready is not None and args.index_ready.exists():
        idx = json.loads(args.index_ready.read_text(encoding="utf-8"))
        collection = idx.get("qdrant_collection") or collection

    if not collection:
        # If you used ingest_docs.py as provided, it prints the collection; in practice
        # you'd stash it to a known place or put it in cfg before running benchmark.
        raise RuntimeError(
            "No Qdrant collection provided. Pass --collection or --index-ready, "
            "or set qdrant_collection_effective in rag_config.json."
        )

    client = ensure_qdrant(cfg)

    top_k = int(cfg.get("top_k", 5))
    qdrant_ef = cfg.get("qdrant_ef")
    qdrant_ef = int(qdrant_ef) if qdrant_ef is not None else None
    retrieval_mode = str(cfg.get("retrieval_mode", "dense")).strip().lower()
    if retrieval_mode not in {"dense", "hybrid_rrf"}:
        raise RuntimeError("retrieval_mode must be one of: dense, hybrid_rrf")
    use_reranker = bool(cfg.get("use_reranker", False))
    rerank_top_n = int(cfg.get("rerank_top_n", max(top_k, 20)))
    candidate_k = int(cfg.get("retrieval_candidate_k", max(top_k, rerank_top_n, top_k * 4)))
    candidate_k = max(top_k, candidate_k)

    hybrid_rrf_k = int(cfg.get("hybrid_rrf_k", 60))
    hybrid_dense_weight = float(cfg.get("hybrid_dense_weight", 1.0))
    hybrid_lexical_weight = float(cfg.get("hybrid_lexical_weight", 1.0))
    hybrid_lexical_top_k = int(cfg.get("hybrid_lexical_top_k", candidate_k))

    lexical_index: Optional[LexicalIndex] = None
    lexical_index_docs = 0
    if retrieval_mode == "hybrid_rrf":
        lexical_index = build_lexical_index(client, collection, cfg)
        lexical_index_docs = int(lexical_index.n_docs)

    bench_path = args.bench
    cases = list(iter_jsonl(bench_path))
    if not cases:
        raise RuntimeError(f"No benchmark cases found in {bench_path}")

    t0 = time.perf_counter()
    total_score = 0.0
    total_latency_ms = 0.0
    total_cost = 0.0
    total_correctness = 0.0
    total_groundedness = 0.0
    n_unanswerable = 0
    n_unanswerable_correct = 0
    retrieval_hits = 0.0
    retrieval_rr_sum = 0.0
    retrieval_eval_cases = 0
    debug_rows: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        query = str(case.get("query", "")).strip()
        gold = str(case.get("gold", "")).strip()
        if not query:
            continue
        expected_unanswerable = is_unanswerable_case(case)
        if expected_unanswerable:
            n_unanswerable += 1

        # Embed query
        qvec, embed_meta = embed_query(query, cfg)

        # Retrieve
        r0 = time.perf_counter()
        dense_candidates = qdrant_search(client, collection, qvec, top_k=candidate_k, ef=qdrant_ef)
        for rank, c in enumerate(dense_candidates, start=1):
            c["dense_rank"] = rank
            c["dense_score"] = float(c.get("score", 0.0))
            c["source"] = "dense"

        if retrieval_mode == "hybrid_rrf" and lexical_index is not None:
            lexical_candidates = lexical_index.search(query, top_n=hybrid_lexical_top_k)
            candidates = fuse_rrf(
                dense_candidates,
                lexical_candidates,
                top_n=candidate_k,
                dense_weight=hybrid_dense_weight,
                lexical_weight=hybrid_lexical_weight,
                rrf_k=hybrid_rrf_k,
            )
        else:
            candidates = dense_candidates

        if use_reranker:
            contexts = rerank_contexts(query, candidates, cfg)
        else:
            contexts = candidates[:top_k]
        r_ms = (time.perf_counter() - r0) * 1000.0

        retrieval_hit: Optional[bool] = None
        retrieval_rr: Optional[float] = None
        retrieval_hit_rank: Optional[int] = None
        retrieval_matched_by: str = "no_label"

        # Retrieval metrics (Recall@K + MRR@K)
        has_retrieval_label = bool(
            _extract_relevant_doc_ids(case)
            or _extract_relevant_chunk_ids(case)
            or _norm(case.get("relevant_text", ""))
            or (allow_gold_fallback and (not expected_unanswerable) and _norm(gold))
        )
        if has_retrieval_label:
            retrieval_eval_cases += 1
            hit, rr, hit_rank, matched_by = retrieval_case_metrics(
                contexts,
                case,
                gold,
                allow_gold_fallback=allow_gold_fallback,
            )
            retrieval_hit = hit
            retrieval_rr = rr
            retrieval_hit_rank = hit_rank
            retrieval_matched_by = matched_by
            retrieval_hits += 1.0 if hit else 0.0
            retrieval_rr_sum += rr

        # Prompt + LLM
        ptxt = build_prompt(query, contexts, cfg)

        l0 = time.perf_counter()
        answer, llm_meta = call_llm(ptxt, cfg)
        l_ms = (time.perf_counter() - l0) * 1000.0

        # Score (simple or stricter LLM judge)
        judge: Optional[Dict[str, Any]] = None
        if answer_scoring_mode == "llm_judge":
            judge = judge_answer_llm(
                query=query,
                gold=gold,
                answer=answer,
                contexts=contexts,
                case=case,
                cfg=cfg,
            )
            correctness = float(judge.get("correctness", 0.0))
            groundedness = float(judge.get("groundedness", 0.0))
            # Weighted score favors correctness but includes grounding signal.
            s = (0.7 * correctness) + (0.3 * groundedness)
        else:
            correctness = score_answer_simple(case, gold, answer)
            groundedness = 0.0
            s = correctness

        total_score += s
        total_correctness += correctness
        total_groundedness += groundedness
        if expected_unanswerable and correctness >= 0.5:
            n_unanswerable_correct += 1

        total_latency_ms += (r_ms + l_ms)
        total_cost += float(llm_meta.get("cost_usd", 0.0))

        if args.debug_cases_out is not None:
            if args.debug_cases_limit <= 0 or len(debug_rows) < args.debug_cases_limit:
                top_contexts = []
                for rank, c in enumerate(contexts, start=1):
                    if rank > min(5, len(contexts)):
                        break
                    top_contexts.append(
                        {
                            "rank": rank,
                            "score": float(c.get("score", 0.0)),
                            "source": c.get("source"),
                            "rrf_score": c.get("rrf_score"),
                            "rerank_score": c.get("rerank_score"),
                            "doc_id": c.get("doc_id"),
                            "chunk_id": c.get("chunk_id"),
                            "text_snippet": str(c.get("text", "")).replace("\n", " ")[:240],
                        }
                    )
                debug_rows.append(
                    {
                        "case_index": i,
                        "query": query,
                        "gold": gold,
                        "answer": answer,
                        "answer_score": float(s),
                        "answer_correctness": float(correctness),
                        "answer_groundedness": float(groundedness),
                        "expected_unanswerable": expected_unanswerable,
                        "abstained": is_abstention(answer),
                        "judge": judge,
                        "retrieval_labeled": has_retrieval_label,
                        "retrieval_hit": retrieval_hit,
                        "retrieval_rr": retrieval_rr,
                        "hit_rank": retrieval_hit_rank,
                        "matched_by": retrieval_matched_by,
                        "retrieval_ms": round(r_ms, 3),
                        "llm_ms": round(l_ms, 3),
                        "top_contexts": top_contexts,
                    }
                )

    n = max(1, len(cases))
    overall_score = total_score / n
    avg_latency_ms = total_latency_ms / n
    cost_usd = total_cost
    avg_correctness = total_correctness / n
    avg_groundedness = total_groundedness / n if answer_scoring_mode == "llm_judge" else None
    unanswerable_accuracy = (n_unanswerable_correct / n_unanswerable) if n_unanswerable > 0 else None
    recall_at_k = retrieval_hits / max(1, retrieval_eval_cases)
    mrr_at_k = retrieval_rr_sum / max(1, retrieval_eval_cases)

    elapsed_s = time.perf_counter() - t0

    if args.debug_cases_out is not None:
        args.debug_cases_out.parent.mkdir(parents=True, exist_ok=True)
        with args.debug_cases_out.open("w", encoding="utf-8") as f:
            for row in debug_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Final line MUST be JSON (your evaluator parses last stdout line)
    print(json.dumps({
        "overall_score": round(overall_score, 6),
        "latency_ms": round(avg_latency_ms, 3),
        "cost_usd": round(cost_usd, 6),
        "n_cases": n,
        "n_retrieval_eval_cases": retrieval_eval_cases,
        "elapsed_s": round(elapsed_s, 3),
        "collection": collection,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "retrieval_candidate_k": candidate_k,
        "use_reranker": use_reranker,
        "rerank_top_n": rerank_top_n,
        "qdrant_ef": qdrant_ef,
        "hybrid_rrf_k": (hybrid_rrf_k if retrieval_mode == "hybrid_rrf" else None),
        "hybrid_dense_weight": (hybrid_dense_weight if retrieval_mode == "hybrid_rrf" else None),
        "hybrid_lexical_weight": (hybrid_lexical_weight if retrieval_mode == "hybrid_rrf" else None),
        "hybrid_lexical_top_k": (hybrid_lexical_top_k if retrieval_mode == "hybrid_rrf" else None),
        "hybrid_lexical_index_docs": lexical_index_docs,
        "recall_at_k": round(recall_at_k, 6),
        "mrr_at_k": round(mrr_at_k, 6),
        "answer_scoring_mode": answer_scoring_mode,
        "avg_correctness": round(avg_correctness, 6),
        "avg_groundedness": (round(float(avg_groundedness), 6) if avg_groundedness is not None else None),
        "n_unanswerable_cases": n_unanswerable,
        "unanswerable_accuracy": (round(float(unanswerable_accuracy), 6) if unanswerable_accuracy is not None else None),
        "retrieval_label_fallback_to_gold": allow_gold_fallback,
        "debug_cases_written": len(debug_rows),
        "llm_model": cfg.get("llm_model"),
        "embed_model": cfg.get("embed_model"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
