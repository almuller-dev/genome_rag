#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Optional dependency: qdrant-client
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    QdrantClient = None
    qmodels = None


# -----------------------------
# Config + utilities
# -----------------------------

CHUNKER_VERSION = "chunker_v1"
INGEST_CODE_VERSION = "ingest_v1"

def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def now_ms() -> int:
    return int(time.time() * 1000)


def corpus_fingerprint(docs_dir: Path) -> str:
    """
    Robust content fingerprint: sha256 of sorted (path + sha256(content)).
    """
    h = hashlib.sha256()
    files = sorted([p for p in docs_dir.rglob("*") if p.is_file()])
    for p in files:
        rel = p.relative_to(docs_dir).as_posix()
        data = p.read_bytes()
        fh = hashlib.sha256(data).hexdigest()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(fh.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


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


# -----------------------------
# Chunking
# -----------------------------

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Simple character-based chunker (fast + deterministic).
    If you prefer token-based, swap this for a tokenizer-based splitter.
    """
    text = normalize_text(text)
    if not text:
        return []

    chunk_size = max(1, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, int(chunk_size * 0.2))

    chunks: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
    return chunks


# -----------------------------
# Embeddings (pluggable)
# -----------------------------

def local_deterministic_embedding(text: str, dim: int = 384) -> List[float]:
    """
    Deterministic, API-free embedding for pipeline testing.
    Not "semantic" like real embeddings, but stable and useful for wiring.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # expand bytes into dim floats in [-1, 1]
    out: List[float] = []
    seed = int.from_bytes(h[:8], "big", signed=False)
    x = seed
    for _ in range(dim):
        # xorshift-ish
        x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 7) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
        # map to [-1, 1]
        v = ((x % 2000000) / 1000000.0) - 1.0
        out.append(float(v))
    return out

def embed_texts(texts: List[str], cfg: Dict[str, Any]) -> Tuple[List[List[float]], Dict[str, Any]]:
    """
    Returns (vectors, embed_meta).
    Replace this with real embeddings (OpenAI, local model, etc.).
    """
    embed_model = cfg.get("embed_model", "local-deterministic")
    dim = int(cfg.get("embed_dim", 384))

    if embed_model == "local-deterministic":
        vectors = [local_deterministic_embedding(t, dim=dim) for t in texts]
        meta = {"provider": "local", "model": embed_model, "dim": dim}
        return vectors, meta

    # OpenAI embeddings path
    if str(embed_model).startswith("text-embedding-"):
        client = _openai_client(cfg)
        batch_size = max(1, int(cfg.get("embed_batch_size", 64)))
        dimensions = int(cfg.get("embed_dim", 384))

        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            req: Dict[str, Any] = {"model": embed_model, "input": batch}
            req["dimensions"] = dimensions
            resp = client.embeddings.create(**req)
            rows = sorted(resp.data, key=lambda r: r.index)
            vectors.extend([list(r.embedding) for r in rows])

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding response size mismatch: got {len(vectors)} vectors for {len(texts)} texts."
            )
        out_dim = len(vectors[0]) if vectors else int(dim or 0)
        meta = {
            "provider": "openai",
            "model": embed_model,
            "dim": out_dim,
            "batch_size": batch_size,
        }
        return vectors, meta

    raise RuntimeError(
        f"Unsupported embed_model={embed_model!r}. "
        "Use 'local-deterministic' or an OpenAI model like 'text-embedding-3-large'."
    )


# -----------------------------
# Qdrant ingestion
# -----------------------------

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: int
    text: str
    source_path: str

def iter_docs(docs_dir: Path) -> Iterable[Tuple[str, str]]:
    """
    Yields (doc_id, text). Supports .txt and .md by default.
    Add PDF/HTML loaders if needed.
    """
    for p in sorted(docs_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".txt", ".md"):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        doc_id = p.relative_to(docs_dir).as_posix()
        yield doc_id, text

def build_chunks(docs_dir: Path, cfg: Dict[str, Any]) -> List[DocChunk]:
    cs = int(cfg["chunk_size"])
    ov = int(cfg["chunk_overlap"])

    out: List[DocChunk] = []
    for doc_id, text in iter_docs(docs_dir):
        chs = chunk_text(text, chunk_size=cs, chunk_overlap=ov)
        for i, ch in enumerate(chs):
            out.append(DocChunk(doc_id=doc_id, chunk_id=i, text=ch, source_path=doc_id))
    return out

def ensure_qdrant(cfg: Dict[str, Any]) -> "QdrantClient":
    if QdrantClient is None or qmodels is None:
        raise RuntimeError(
            "qdrant-client is not installed. Install it with: pip install qdrant-client"
        )
    url = cfg.get("qdrant_url", "http://localhost:6333")
    api_key = cfg.get("qdrant_api_key")
    return QdrantClient(url=url, api_key=api_key)

def ensure_collection(
    client: "QdrantClient",
    collection: str,
    vector_dim: int,
    cfg: Dict[str, Any],
) -> None:
    hnsw_m = int(cfg.get("qdrant_hnsw_m", 16))
    ef_construct = int(cfg.get("qdrant_ef_construct", 200))

    # Create or recreate collection
    recreate = bool(cfg.get("qdrant_recreate_collection", True))

    if recreate:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(
                size=vector_dim,
                distance=qmodels.Distance.COSINE,
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=hnsw_m,
                ef_construct=ef_construct,
            ),
        )
    else:
        # Try create if doesn't exist; ignore if exists
        try:
            client.create_collection(
                collection_name=collection,
                vectors_config=qmodels.VectorParams(size=vector_dim, distance=qmodels.Distance.COSINE),
                hnsw_config=qmodels.HnswConfigDiff(m=hnsw_m, ef_construct=ef_construct),
            )
        except Exception:
            pass

def upsert_chunks(
    client: "QdrantClient",
    collection: str,
    chunks: List[DocChunk],
    vectors: List[List[float]],
) -> None:
    assert len(chunks) == len(vectors)
    points = []
    for idx, (ch, vec) in enumerate(zip(chunks, vectors)):
        # Qdrant point IDs must be uint or UUID; use deterministic UUID per chunk.
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{ch.doc_id}:{ch.chunk_id}"))
        payload = {
            "doc_id": ch.doc_id,
            "chunk_id": ch.chunk_id,
            "text": ch.text,
            "source_path": ch.source_path,
        }
        points.append(qmodels.PointStruct(id=point_id, vector=vec, payload=payload))

    # batch upserts
    batch = 128
    for i in range(0, len(points), batch):
        client.upsert(collection_name=collection, points=points[i : i + batch])


# -----------------------------
# Main
# -----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path, help="Path to rag_config.json")
    ap.add_argument("--docs-dir", required=True, type=Path, help="Directory with .txt/.md docs")
    ap.add_argument("--artifact-dir", required=False, type=Path, default=None, help="Artifact reuse dir for index signature")
    ap.add_argument("--collection", required=False, type=str, default=None, help="Override Qdrant collection name")
    args = ap.parse_args(argv)

    cfg = read_json(args.config)
    docs_fp = corpus_fingerprint(args.docs_dir)

    # Build index signature (only params that require rebuild)
    index_sig = {
        "chunker_version": CHUNKER_VERSION,
        "ingest_code_version": INGEST_CODE_VERSION,
        "docs_fingerprint": docs_fp,
        "chunk_size": cfg["chunk_size"],
        "chunk_overlap": cfg["chunk_overlap"],
        "embed_model": cfg.get("embed_model", "local-deterministic"),
        "embed_dim": int(cfg.get("embed_dim", 384)),
        "qdrant_hnsw_m": int(cfg.get("qdrant_hnsw_m", 16)),
        "qdrant_ef_construct": int(cfg.get("qdrant_ef_construct", 200)),
    }
    index_key = stable_hash(index_sig)

    artifact_dir = args.artifact_dir
    if artifact_dir is not None:
        artifact_dir = artifact_dir / index_key
        artifact_dir.mkdir(parents=True, exist_ok=True)

        marker = artifact_dir / "INDEX_READY.json"
        if marker.exists():
            # Artifact exists; validate marker before reuse.
            try:
                ready = json.loads(marker.read_text(encoding="utf-8"))
            except Exception:
                ready = {}
            ready_sig = ready.get("index_sig")
            if isinstance(ready_sig, dict) and ready_sig == index_sig:
                print(json.dumps({"status": "cached", "index_key": index_key}))
                return 0

    # Qdrant collection name: include index_key so parallel runs don’t collide
    base_collection = args.collection or cfg.get("qdrant_collection", "rag_support")
    collection = f"{base_collection}_{index_key[:12]}"

    t0 = time.perf_counter()

    chunks = build_chunks(args.docs_dir, cfg)
    if not chunks:
        print(json.dumps({"status": "no_docs"}))
        return 2

    texts = [c.text for c in chunks]
    vectors, embed_meta = embed_texts(texts, cfg)
    dim = len(vectors[0])

    client = ensure_qdrant(cfg)
    ensure_collection(client, collection, dim, cfg)
    upsert_chunks(client, collection, chunks, vectors)

    elapsed = time.perf_counter() - t0

    # Write artifact marker
    if artifact_dir is not None:
        write_json(artifact_dir / "INDEX_READY.json", {
            "status": "ok",
            "index_key": index_key,
            "index_sig": index_sig,
            "qdrant_collection": collection,
            "embed_meta": embed_meta,
            "n_chunks": len(chunks),
            "built_at_ms": now_ms(),
            "elapsed_s": elapsed,
        })

    print(json.dumps({
        "status": "ok",
        "index_key": index_key,
        "qdrant_collection": collection,
        "n_chunks": len(chunks),
        "elapsed_s": elapsed,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
