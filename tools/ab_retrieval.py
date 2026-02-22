#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def parse_json_object(s: str) -> Dict[str, Any]:
    for line in reversed(s.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError("No JSON object found in command output.")


def _tail(s: str, max_chars: int = 20000) -> str:
    return s if len(s) <= max_chars else s[-max_chars:]


def run_benchmark_once(
    *,
    repo_root: Path,
    config_path: Path,
    bench: Path,
    index_ready: Optional[Path],
    collection: Optional[str],
    debug_out: Optional[Path],
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "run_benchmark.py",
        "--config",
        str(config_path),
        "--bench",
        str(bench),
    ]
    if index_ready is not None:
        cmd += ["--index-ready", str(index_ready)]
    if collection:
        cmd += ["--collection", collection]
    if debug_out is not None:
        cmd += ["--debug-cases-out", str(debug_out)]

    p = subprocess.run(
        cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out = _tail(p.stdout)
    err = _tail(p.stderr)
    if p.returncode != 0:
        return {"ok": False, "returncode": p.returncode, "stdout_tail": out, "stderr_tail": err, "metrics": None}
    try:
        metrics = parse_json_object(out)
    except Exception as e:
        return {
            "ok": False,
            "returncode": 0,
            "stdout_tail": out,
            "stderr_tail": err,
            "metrics": None,
            "error": f"parse_failed: {e}",
        }
    return {"ok": True, "returncode": 0, "stdout_tail": out, "stderr_tail": err, "metrics": metrics}


def build_config(base: Path, override_json: Optional[str]) -> Path:
    if not override_json:
        return base
    cfg = read_json(base)
    override_obj = json.loads(override_json)
    if not isinstance(override_obj, dict):
        raise RuntimeError("--override-* must parse to a JSON object")
    cfg.update(override_obj)
    tf = tempfile.NamedTemporaryFile(prefix="ab_cfg_", suffix=".json", delete=False)
    tmp_path = Path(tf.name)
    tf.close()
    write_json(tmp_path, cfg)
    return tmp_path


def metric_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(b) - float(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-a", required=True, type=Path, help="Baseline config JSON")
    ap.add_argument("--config-b", required=True, type=Path, help="Variant config JSON")
    ap.add_argument("--override-a", required=False, type=str, default=None, help="Inline JSON object to merge into config A")
    ap.add_argument("--override-b", required=False, type=str, default=None, help="Inline JSON object to merge into config B")
    ap.add_argument("--bench", required=True, type=Path, help="Benchmark JSONL path")
    ap.add_argument("--index-ready", required=False, type=Path, default=None, help="INDEX_READY.json")
    ap.add_argument("--collection", required=False, type=str, default=None, help="Optional collection override")
    ap.add_argument("--label-a", required=False, type=str, default="A")
    ap.add_argument("--label-b", required=False, type=str, default="B")
    ap.add_argument("--debug-dir", required=False, type=Path, default=None, help="Optional directory for per-case debug JSONL")
    ap.add_argument("--out", required=False, type=Path, default=None, help="Optional output JSON summary")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg_a = build_config(args.config_a, args.override_a)
    cfg_b = build_config(args.config_b, args.override_b)

    debug_a = None
    debug_b = None
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        debug_a = args.debug_dir / f"{args.label_a}.jsonl"
        debug_b = args.debug_dir / f"{args.label_b}.jsonl"

    try:
        res_a = run_benchmark_once(
            repo_root=repo_root,
            config_path=cfg_a,
            bench=args.bench,
            index_ready=args.index_ready,
            collection=args.collection,
            debug_out=debug_a,
        )
        res_b = run_benchmark_once(
            repo_root=repo_root,
            config_path=cfg_b,
            bench=args.bench,
            index_ready=args.index_ready,
            collection=args.collection,
            debug_out=debug_b,
        )
    finally:
        # Clean up temp config files created by overrides.
        if args.override_a:
            try:
                os.unlink(cfg_a)
            except Exception:
                pass
        if args.override_b:
            try:
                os.unlink(cfg_b)
            except Exception:
                pass

    m_a = res_a.get("metrics") or {}
    m_b = res_b.get("metrics") or {}
    summary = {
        "ok": bool(res_a.get("ok")) and bool(res_b.get("ok")),
        "labels": {"a": args.label_a, "b": args.label_b},
        "a": res_a,
        "b": res_b,
        "delta_b_minus_a": {
            "overall_score": metric_delta(m_a.get("overall_score"), m_b.get("overall_score")),
            "recall_at_k": metric_delta(m_a.get("recall_at_k"), m_b.get("recall_at_k")),
            "mrr_at_k": metric_delta(m_a.get("mrr_at_k"), m_b.get("mrr_at_k")),
            "latency_ms": metric_delta(m_a.get("latency_ms"), m_b.get("latency_ms")),
        },
    }

    blob = json.dumps(summary, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob + "\n", encoding="utf-8")
    print(blob)
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
