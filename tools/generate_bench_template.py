#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("bench/support_tickets_template_120.jsonl"))
    ap.add_argument("--n", type=int, default=120, help="Number of template cases to generate")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for i in range(1, args.n + 1):
            row = {
                "case_id": f"CASE_{i:04d}",
                "query": f"TODO: write user query #{i}",
                "gold": f"TODO: write expected answer fragment #{i}",
                "relevant_doc_id": "TODO:doc_path.ext",
                "relevant_chunk_ids": [0],
                "relevant_text": "TODO: exact supporting text snippet",
                "notes": "Replace TODO fields with real labels before evaluation",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(args.out), "n_cases": args.n}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
