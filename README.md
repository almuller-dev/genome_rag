# genome_rag

[![RAG Retrieval Regression](https://github.com/almuller-dev/genome_rag/actions/workflows/rag-regression.yml/badge.svg)](https://github.com/almuller-dev/genome_rag/actions/workflows/rag-regression.yml)
[![RAG Retrieval Regression (Full)](https://github.com/almuller-dev/genome_rag/actions/workflows/rag-regression-full.yml/badge.svg)](https://github.com/almuller-dev/genome_rag/actions/workflows/rag-regression-full.yml)
[![Bench Smoke Thresholds](https://github.com/almuller-dev/genome_rag/actions/workflows/bench-smoke.yml/badge.svg)](https://github.com/almuller-dev/genome_rag/actions/workflows/bench-smoke.yml)
[![Bench Label Validation](https://github.com/almuller-dev/genome_rag/actions/workflows/bench-labels.yml/badge.svg)](https://github.com/almuller-dev/genome_rag/actions/workflows/bench-labels.yml)

RAG tuning and evaluation workspace with:
- Qdrant indexing (`ingest_docs.py`)
- Benchmarking and judged scoring (`run_benchmark.py`)
- Evolutionary search scaffolding (`evo_engine_v1.py`)
- CI regression gates in GitHub Actions

## CI Workflows

1. `RAG Retrieval Regression`
- File: `.github/workflows/rag-regression.yml`
- Bench: `bench/arxiv_llm_rag_eval_120_strict.jsonl`
- Trigger: manual + weekly schedule

2. `RAG Retrieval Regression (Full)`
- File: `.github/workflows/rag-regression-full.yml`
- Bench: `bench/arxiv_llm_rag_eval_400_strict.jsonl`
- Trigger: manual + weekly schedule

Both run:
1. Start Qdrant (`qdrant/qdrant:v1.7.0`)
2. Build CI config from `rag_config_prod.json`
3. Ingest corpus
4. Run benchmark
5. Enforce regression thresholds
6. Upload artifacts

## Required GitHub Secret

Set repository secret:
- `OPENAI_API_KEY`

Path:
- GitHub repo -> `Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`

Important:
- The key is not stored in repository files.
- Workflows use `${{ secrets.OPENAI_API_KEY }}` at runtime.

## Regression Thresholds

Default gate thresholds:
- `overall_score >= 0.93`
- `recall_at_k >= 0.98`
- `mrr_at_k >= 0.95`
- `latency_ms <= 1500`

Gate logic:
- Script: `tools/check_regression.py`
- CI wrapper: `tools/check_retrieval_regression_ci.sh`

## Runbooks

### Run in GitHub (recommended)

1. Open repo `Actions` tab.
2. Select `RAG Retrieval Regression` (or `RAG Retrieval Regression (Full)`).
3. Click `Run workflow`, select `main`, run.
4. On completion, open artifact `rag-regression-results` (or full variant) for:
- `metrics.json`
- `gate.json`
- benchmark stdout/stderr captures

### Run locally

Example (requires Qdrant and OpenAI key in env):

```bash
tools/check_retrieval_regression_ci.sh \
  --config rag_config_prod.json \
  --bench bench/arxiv_llm_rag_eval_120_strict.jsonl \
  --index-ready .rag_artifacts/arxiv_llm_20/5a2006cc806e15e86dd1e8eba78fa41983167fd4b8f1096c41ea2aef50e20d71/INDEX_READY.json
```

## Primary Files

- `ingest_docs.py`: corpus chunking, embeddings, Qdrant ingestion
- `run_benchmark.py`: retrieval + generation + scoring metrics
- `rag_config_prod.json`: current best production-leaning config
- `tools/ab_retrieval.py`: A/B compare retrieval configs
- `tools/check_regression.py`: metric gate checker
- `tools/check_retrieval_regression_ci.sh`: benchmark + gate wrapper

## Troubleshooting

1. Workflow not visible
- Ensure the workflow file is pushed to `main`, then refresh Actions.

2. Qdrant startup issues in Actions
- Workflows start Qdrant via `docker run` and wait on `/readyz`/`/healthz`.

3. Benchmark failure with little detail
- `tools/check_retrieval_regression_ci.sh` prints stderr/stdout tails on failure.

4. API issues
- Verify `OPENAI_API_KEY` secret exists and has quota/permissions.
