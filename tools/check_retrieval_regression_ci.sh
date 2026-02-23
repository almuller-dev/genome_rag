#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

CONFIG="${CONFIG:-$ROOT_DIR/rag_config_prod.json}"
BENCH="${BENCH:-$ROOT_DIR/bench/arxiv_llm_rag_eval_120_strict.jsonl}"
INDEX_READY="${INDEX_READY:-}"
COLLECTION="${COLLECTION:-}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/slice_eval/ci_gate}"

MIN_OVERALL_SCORE="${MIN_OVERALL_SCORE:-0.93}"
MIN_RECALL_AT_K="${MIN_RECALL_AT_K:-0.98}"
MIN_MRR_AT_K="${MIN_MRR_AT_K:-0.95}"
MAX_LATENCY_MS="${MAX_LATENCY_MS:-1500}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --bench)
      BENCH="$2"
      shift 2
      ;;
    --index-ready)
      INDEX_READY="$2"
      shift 2
      ;;
    --collection)
      COLLECTION="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --min-overall-score)
      MIN_OVERALL_SCORE="$2"
      shift 2
      ;;
    --min-recall-at-k)
      MIN_RECALL_AT_K="$2"
      shift 2
      ;;
    --min-mrr-at-k)
      MIN_MRR_AT_K="$2"
      shift 2
      ;;
    --max-latency-ms)
      MAX_LATENCY_MS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 2
      ;;
  esac
done

if [[ -z "$INDEX_READY" && -z "$COLLECTION" ]]; then
  echo "ERROR: Pass --index-ready or --collection (or set INDEX_READY/COLLECTION env)."
  exit 2
fi

mkdir -p "$OUT_DIR"

BENCH_STDOUT="$OUT_DIR/benchmark.stdout.txt"
BENCH_STDERR="$OUT_DIR/benchmark.stderr.txt"
METRICS_JSON="$OUT_DIR/metrics.json"
GATE_JSON="$OUT_DIR/gate.json"

run_cmd=(python3 "$ROOT_DIR/run_benchmark.py" --config "$CONFIG" --bench "$BENCH")
if [[ -n "$INDEX_READY" ]]; then
  run_cmd+=(--index-ready "$INDEX_READY")
fi
if [[ -n "$COLLECTION" ]]; then
  run_cmd+=(--collection "$COLLECTION")
fi

echo "Running benchmark:"
printf '  %q ' "${run_cmd[@]}"
echo

if ! "${run_cmd[@]}" >"$BENCH_STDOUT" 2>"$BENCH_STDERR"; then
  rc=$?
  echo "Benchmark command failed with exit code $rc"
  echo "--- benchmark.stderr (tail -200) ---"
  tail -n 200 "$BENCH_STDERR" || true
  echo "--- benchmark.stdout (tail -200) ---"
  tail -n 200 "$BENCH_STDOUT" || true
  exit "$rc"
fi

if ! python3 - "$BENCH_STDOUT" "$METRICS_JSON" <<'PY'
import json
import sys
from pathlib import Path

stdout_path = Path(sys.argv[1])
metrics_path = Path(sys.argv[2])
lines = stdout_path.read_text(encoding="utf-8").splitlines()
obj = None
for line in reversed(lines):
    line = line.strip()
    if not line:
        continue
    try:
        maybe = json.loads(line)
    except Exception:
        continue
    if isinstance(maybe, dict):
        obj = maybe
        break
if obj is None:
    raise SystemExit(f"No JSON metrics line found in {stdout_path}")
metrics_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(obj, indent=2, sort_keys=True))
PY
then
  echo "Failed to parse benchmark metrics JSON from stdout."
  echo "--- benchmark.stderr (tail -200) ---"
  tail -n 200 "$BENCH_STDERR" || true
  echo "--- benchmark.stdout (tail -200) ---"
  tail -n 200 "$BENCH_STDOUT" || true
  exit 1
fi

python3 "$ROOT_DIR/tools/check_regression.py" \
  --metrics-json "$METRICS_JSON" \
  --min-overall-score "$MIN_OVERALL_SCORE" \
  --min-recall-at-k "$MIN_RECALL_AT_K" \
  --min-mrr-at-k "$MIN_MRR_AT_K" \
  --max-latency-ms "$MAX_LATENCY_MS" \
  --out "$GATE_JSON"

echo "Gate check passed."
echo "Metrics: $METRICS_JSON"
echo "Gate:    $GATE_JSON"
echo "Stdout:  $BENCH_STDOUT"
echo "Stderr:  $BENCH_STDERR"
