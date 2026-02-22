import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from evo_engine_v1 import FitnessResult, NEG_INF, RepoStagedEvaluator, Stage

CHUNKER_VERSION = "chunker_v1"
INGEST_CODE_VERSION = "ingest_v1"


def stable_key(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


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


def index_signature(genome: Dict[str, Any], docs_fp: str) -> Dict[str, Any]:
    return {
        "chunker_version": CHUNKER_VERSION,
        "ingest_code_version": INGEST_CODE_VERSION,
        "docs_fingerprint": docs_fp,
        "chunk_size": genome["chunk_size"],
        "chunk_overlap": genome["chunk_overlap"],
        "embed_model": genome.get("embed_model", "local-deterministic"),
        "embed_dim": int(genome.get("embed_dim", 384)),
        "qdrant_hnsw_m": genome["qdrant_hnsw_m"],
        "qdrant_ef_construct": genome["qdrant_ef_construct"],
    }


def marker_matches(marker_path: Path, expected_sig: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not marker_path.exists():
        return False, None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    if not isinstance(marker, dict):
        return False, None
    sig = marker.get("index_sig")
    collection = marker.get("qdrant_collection")
    if not isinstance(sig, dict) or sig != expected_sig:
        return False, marker
    if not isinstance(collection, str) or not collection:
        return False, marker
    return True, marker


def canonicalize(g: Dict[str, Any]) -> Dict[str, Any]:
    g = dict(g)

    # clamp + normalize
    g["chunk_size"] = int(max(128, min(1024, g["chunk_size"])))
    g["chunk_overlap"] = int(max(0, min(256, g["chunk_overlap"])))
    if g["chunk_overlap"] >= g["chunk_size"]:
        g["chunk_overlap"] = max(0, int(g["chunk_size"] * 0.2))

    g["qdrant_hnsw_m"] = int(max(8, min(64, g["qdrant_hnsw_m"])))
    g["qdrant_ef_construct"] = int(max(64, min(512, g["qdrant_ef_construct"])))

    g["top_k"] = int(max(2, min(20, g["top_k"])))
    g["qdrant_ef"] = int(max(32, min(512, g["qdrant_ef"])))

    g["use_reranker"] = bool(g["use_reranker"])
    g["rerank_top_n"] = int(max(5, min(50, g.get("rerank_top_n", 20))))
    if not g["use_reranker"]:
        g["rerank_top_n"] = 0

    t = float(g["prompt_temperature"])
    g["prompt_temperature"] = round(max(0.0, min(0.7, t)), 2)

    return g


def apply_rag_config(ws: Path, genome: Dict[str, Any]) -> None:
    (ws / "rag_config.json").write_text(json.dumps(genome, indent=2), encoding="utf-8")


class RAGEvaluator(RepoStagedEvaluator):
    """
    Stages:
      - ingest (optional if validated index marker exists)
      - benchmark (always)

    Assumes benchmark prints last line JSON: {"overall_score":0.87,"latency_ms":450,"cost_usd":0.02}
    """

    def __init__(
        self,
        base_repo: Path,
        apply_genome_fn,
        stages,
        *,
        docs_dir_rel: str = "docs",
        bench_rel: str = "bench/support_tickets.jsonl",
        persistent_artifact_root: Path = Path(".rag_artifacts/index"),
        collection_prefix: str = "rag_support",
    ):
        super().__init__(base_repo=base_repo, apply_genome_fn=apply_genome_fn, stages=stages)
        self.docs_dir_rel = docs_dir_rel
        self.bench_rel = bench_rel
        self.persistent_artifact_root = Path(persistent_artifact_root).resolve()
        self.collection_prefix = collection_prefix

    def __call__(self, genome: Any) -> FitnessResult:
        g = canonicalize(genome)

        with self._workspace() as ws:
            apply_rag_config(ws, g)

            docs_dir = ws / self.docs_dir_rel
            if not docs_dir.exists():
                return FitnessResult(
                    NEG_INF,
                    {
                        "status": "missing_docs_dir",
                        "docs_dir": str(docs_dir),
                    },
                )

            self.persistent_artifact_root.mkdir(parents=True, exist_ok=True)
            docs_fp = corpus_fingerprint(docs_dir)
            sig = index_signature(g, docs_fp)
            sig_key = stable_key(sig)
            sig_dir = self.persistent_artifact_root / sig_key
            marker_path = sig_dir / "INDEX_READY.json"

            details: Dict[str, Any] = {
                "stages": [],
                "index_sig": sig,
                "index_key": sig_key,
                "artifact_root": str(self.persistent_artifact_root),
            }

            marker_ok, marker = marker_matches(marker_path, sig)

            # 1) ingest only if validated marker is absent/mismatched
            if not marker_ok:
                sig_dir.mkdir(parents=True, exist_ok=True)
                ingest_cmd = [
                    "python3",
                    "ingest_docs.py",
                    "--config",
                    "rag_config.json",
                    "--docs-dir",
                    self.docs_dir_rel,
                    "--artifact-dir",
                    str(self.persistent_artifact_root),
                    "--collection",
                    self.collection_prefix,
                ]
                ok, elapsed, out, err = self._run_stage(ws, Stage("ingest", ingest_cmd, 600))
                details["stages"].append({"name": "ingest", "ok": ok, "elapsed_s": elapsed, "stdout": out, "stderr": err})
                if not ok:
                    return FitnessResult(NEG_INF, {**details, "status": "failed_ingest"})

                marker_ok, marker = marker_matches(marker_path, sig)
                if not marker_ok:
                    return FitnessResult(
                        NEG_INF,
                        {
                            **details,
                            "status": "failed_ingest_marker_validation",
                            "marker_path": str(marker_path),
                        },
                    )
            else:
                details["stages"].append({"name": "ingest", "ok": True, "elapsed_s": 0.0, "cached": True})

            # 2) benchmark always
            benchmark_cmd = [
                "python3",
                "run_benchmark.py",
                "--config",
                "rag_config.json",
                "--bench",
                self.bench_rel,
                "--index-ready",
                str(marker_path),
            ]
            ok, elapsed, out, err = self._run_stage(ws, Stage("benchmark", benchmark_cmd, 900))
            details["stages"].append({"name": "benchmark", "ok": ok, "elapsed_s": elapsed, "stdout": out, "stderr": err})
            if not ok:
                return FitnessResult(NEG_INF, {**details, "status": "failed_benchmark"})

            try:
                last = out.strip().splitlines()[-1]
                metrics = json.loads(last)
            except Exception as e:
                return FitnessResult(NEG_INF, {**details, "status": "parse_error", "error": repr(e)})

            score = float(metrics.get("overall_score", 0.0))
            latency_ms = float(metrics.get("latency_ms", 1000.0))
            cost_usd = float(metrics.get("cost_usd", 0.0))

            # fitness = accuracy - latency penalty - cost penalty
            fitness = score - (latency_ms * 0.0001) - (cost_usd * 1.0)

            return FitnessResult(
                fitness=fitness,
                details={
                    **details,
                    "status": "ok",
                    "overall_score": score,
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "qdrant_collection": marker.get("qdrant_collection") if isinstance(marker, dict) else None,
                    "genome": g,
                },
            )

    # --- helpers: keep these inside to avoid copy/paste ---
    from contextlib import contextmanager

    @contextmanager
    def _workspace(self):
        import tempfile

        with tempfile.TemporaryDirectory(prefix="rag_ws_") as td:
            ws = Path(td) / "repo"
            self._copy_repo(ws)
            yield ws

    def _run_stage(self, ws: Path, st: Stage):
        from evo_engine_v1 import _run_cmd

        return _run_cmd(st.cmd, cwd=ws, timeout_s=st.timeout_s)
