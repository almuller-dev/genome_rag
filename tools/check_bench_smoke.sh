#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="${SMOKE_WORK_DIR:-$ROOT_DIR/.smoke_bench}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
MIN_RECALL_AT_K="${MIN_RECALL_AT_K:-1.0}"
MIN_MRR_AT_K="${MIN_MRR_AT_K:-0.8}"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR/docs" "$WORK_DIR/bench"

cat > "$WORK_DIR/docs/account_help.txt" <<'TXT'
Password reset instructions:
Go to Settings > Security > Reset Password and confirm via email.

Billing address update:
Go to Billing > Address and save your changes.
TXT

cat > "$WORK_DIR/docs/subscription_help.txt" <<'TXT'
Cancellation instructions:
Open Plans > Manage Plan > Cancel Subscription.

Invoice download:
Go to Billing > Invoices and click Download PDF.
TXT

cat > "$WORK_DIR/bench/smoke.jsonl" <<'JSONL'
{"query":"How do I reset my password?","gold":"Reset Password","relevant_doc_id":"account_help.txt","relevant_text":"Reset Password"}
{"query":"Where do I download an invoice PDF?","gold":"Download PDF","relevant_doc_id":"subscription_help.txt","relevant_text":"Download PDF"}
JSONL

cat > "$WORK_DIR/rag_config.smoke.json" <<JSON
{
  "embed_model": "local-deterministic",
  "embed_dim": 384,
  "chunk_size": 300,
  "chunk_overlap": 40,
  "qdrant_url": "$QDRANT_URL",
  "qdrant_collection": "smoke_rag",
  "qdrant_recreate_collection": true,
  "qdrant_hnsw_m": 16,
  "qdrant_ef_construct": 100,
  "top_k": 3,
  "qdrant_ef": 64,
  "llm_model": "stub",
  "prompt_temperature": 0.0
}
JSON

# Wait for Qdrant readiness
for _ in {1..40}; do
  if curl -fsS "$QDRANT_URL/readyz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "$QDRANT_URL/readyz" >/dev/null 2>&1; then
  echo "ERROR: Qdrant not ready at $QDRANT_URL"
  exit 1
fi

python3 "$ROOT_DIR/ingest_docs.py" \
  --config "$WORK_DIR/rag_config.smoke.json" \
  --docs-dir "$WORK_DIR/docs" \
  --artifact-dir "$WORK_DIR/artifacts" \
  --collection "smoke_rag"

IDX_READY="$(find "$WORK_DIR/artifacts" -maxdepth 2 -type f -name INDEX_READY.json | head -n 1)"
if [[ -z "$IDX_READY" || ! -f "$IDX_READY" ]]; then
  echo "ERROR: INDEX_READY.json not found under $WORK_DIR/artifacts"
  exit 1
fi

OUT_JSON="$(python3 "$ROOT_DIR/run_benchmark.py" \
  --config "$WORK_DIR/rag_config.smoke.json" \
  --bench "$WORK_DIR/bench/smoke.jsonl" \
  --index-ready "$IDX_READY" | tail -n 1)"

python3 - <<PY
import json, sys
metrics = json.loads('''$OUT_JSON''')
recall = float(metrics.get('recall_at_k', 0.0))
mrr = float(metrics.get('mrr_at_k', 0.0))
min_recall = float('$MIN_RECALL_AT_K')
min_mrr = float('$MIN_MRR_AT_K')
print(json.dumps(metrics, indent=2))
if recall < min_recall:
    raise SystemExit(f"recall_at_k {recall:.6f} below threshold {min_recall:.6f}")
if mrr < min_mrr:
    raise SystemExit(f"mrr_at_k {mrr:.6f} below threshold {min_mrr:.6f}")
PY
