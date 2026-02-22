import math
from typing import Any, Dict

NUMERIC_KEYS = {
    "chunk_size", "chunk_overlap", "qdrant_hnsw_m", "qdrant_ef_construct",
    "top_k", "qdrant_ef", "rerank_top_n", "prompt_temperature",
}
CATEGORICAL_KEYS = {"llm_model", "embed_model", "use_reranker"}

WEIGHTS = {
    # index-time (more impactful)
    "chunk_size": 1.2,
    "chunk_overlap": 1.1,
    "embed_model": 1.0,
    "qdrant_hnsw_m": 1.0,
    "qdrant_ef_construct": 0.9,

    # query-time
    "top_k": 0.9,
    "qdrant_ef": 0.8,
    "use_reranker": 0.8,
    "rerank_top_n": 0.6,
    "llm_model": 0.8,
    "prompt_temperature": 0.5,
}

def _norm(key: str, x: float, space: Dict[str, Any]) -> float:
    lo, hi = space[key]
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        return 0.0
    x = max(lo, min(hi, float(x)))
    return (x - lo) / (hi - lo)

def rag_distance(a: Dict[str, Any], b: Dict[str, Any], space: Dict[str, Any]) -> float:
    acc = 0.0

    for k in NUMERIC_KEYS:
        if k not in a or k not in b or k not in space:
            continue
        w = WEIGHTS.get(k, 1.0)
        d = _norm(k, a[k], space) - _norm(k, b[k], space)
        acc += (w * d) * (w * d)

    for k in CATEGORICAL_KEYS:
        if k not in a or k not in b:
            continue
        w = WEIGHTS.get(k, 1.0)
        diff = 0.0 if a[k] == b[k] else 1.0
        acc += (w * diff) * (w * diff)

    return math.sqrt(acc)
