#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_metrics_shape(obj: Dict[str, Any]) -> bool:
    return "overall_score" in obj and "latency_ms" in obj


def extract_metrics(obj: Dict[str, Any], arm: str) -> Dict[str, Any]:
    if is_metrics_shape(obj):
        return obj
    if "a" in obj and "b" in obj:
        arm_obj = obj.get(arm, {})
        metrics = arm_obj.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    raise RuntimeError("Unsupported input JSON shape. Expect run_benchmark metrics or ab_retrieval summary.")


def as_float(metrics: Dict[str, Any], key: str) -> float:
    if key not in metrics:
        raise RuntimeError(f"Missing metric key: {key}")
    return float(metrics[key])


def evaluate(
    metrics: Dict[str, Any],
    *,
    min_overall_score: float,
    min_recall_at_k: float,
    min_mrr_at_k: float,
    max_latency_ms: float,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    overall = as_float(metrics, "overall_score")
    recall = as_float(metrics, "recall_at_k")
    mrr = as_float(metrics, "mrr_at_k")
    latency = as_float(metrics, "latency_ms")

    failures: List[str] = []
    if overall < min_overall_score:
        failures.append(f"overall_score {overall:.6f} < {min_overall_score:.6f}")
    if recall < min_recall_at_k:
        failures.append(f"recall_at_k {recall:.6f} < {min_recall_at_k:.6f}")
    if mrr < min_mrr_at_k:
        failures.append(f"mrr_at_k {mrr:.6f} < {min_mrr_at_k:.6f}")
    if latency > max_latency_ms:
        failures.append(f"latency_ms {latency:.3f} > {max_latency_ms:.3f}")

    summary = {
        "overall_score": overall,
        "recall_at_k": recall,
        "mrr_at_k": mrr,
        "latency_ms": latency,
        "thresholds": {
            "min_overall_score": min_overall_score,
            "min_recall_at_k": min_recall_at_k,
            "min_mrr_at_k": min_mrr_at_k,
            "max_latency_ms": max_latency_ms,
        },
    }
    return len(failures) == 0, failures, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-json", required=True, type=Path, help="Path to metrics JSON from run_benchmark or ab_retrieval output")
    ap.add_argument("--arm", required=False, choices=["a", "b"], default="b", help="Arm to evaluate if using ab_retrieval summary")
    ap.add_argument("--min-overall-score", required=False, type=float, default=0.93)
    ap.add_argument("--min-recall-at-k", required=False, type=float, default=0.98)
    ap.add_argument("--min-mrr-at-k", required=False, type=float, default=0.95)
    ap.add_argument("--max-latency-ms", required=False, type=float, default=1500.0)
    ap.add_argument("--out", required=False, type=Path, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    raw = read_json(args.metrics_json)
    metrics = extract_metrics(raw, args.arm)
    ok, failures, summary = evaluate(
        metrics,
        min_overall_score=float(args.min_overall_score),
        min_recall_at_k=float(args.min_recall_at_k),
        min_mrr_at_k=float(args.min_mrr_at_k),
        max_latency_ms=float(args.max_latency_ms),
    )

    result = {
        "ok": ok,
        "arm": args.arm,
        "metrics_source": str(args.metrics_json),
        "summary": summary,
        "failures": failures,
    }

    blob = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob + "\n", encoding="utf-8")
    print(blob)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
