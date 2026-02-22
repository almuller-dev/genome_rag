RAG_SPACE = {
  # index-time (expensive)
  "chunk_size": (128, 1024),
  "chunk_overlap": (0, 256),
  "embed_model": ["text-embedding-3-large"],
  "qdrant_hnsw_m": (8, 64),
  "qdrant_ef_construct": (64, 512),

  # query-time (cheap)
  "top_k": (2, 20),
  "qdrant_ef": (32, 512),
  "use_reranker": [False, True],
  "rerank_top_n": (5, 50),  # only relevant if use_reranker True
  "llm_model": ["gpt-5.2"],
  "prompt_temperature": (0.0, 0.7),
}
