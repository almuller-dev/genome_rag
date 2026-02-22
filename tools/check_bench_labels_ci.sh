#!/usr/bin/env bash
set -euo pipefail

BENCH_PATH="${BENCH_PATH:-bench/support_tickets.jsonl}"
if [[ ! -f "$BENCH_PATH" ]]; then
  echo "ERROR: benchmark file not found: $BENCH_PATH"
  exit 1
fi

python3 tools/validate_bench_labels.py \
  --bench "$BENCH_PATH" \
  --min-label-coverage 1.0 \
  --reject-placeholders
