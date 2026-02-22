#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            yield lineno, json.loads(raw)


def has_retrieval_label(case: Dict[str, Any]) -> bool:
    if case.get("relevant_doc_id"):
        return True
    if isinstance(case.get("relevant_doc_ids"), list) and len(case["relevant_doc_ids"]) > 0:
        return True
    if case.get("relevant_chunk_id") is not None:
        return True
    if isinstance(case.get("relevant_chunk_ids"), list) and len(case["relevant_chunk_ids"]) > 0:
        return True
    if str(case.get("relevant_text", "")).strip():
        return True
    return False


def is_unanswerable(case: Dict[str, Any]) -> bool:
    return bool(case.get("expected_unanswerable", False))


def has_placeholder_value(case: Dict[str, Any]) -> bool:
    keys = [
        "query",
        "gold",
        "relevant_doc_id",
        "relevant_text",
        "notes",
    ]
    for k in keys:
        v = str(case.get(k, ""))
        if "todo" in v.lower():
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True, help="Path to benchmark JSONL")
    ap.add_argument(
        "--min-label-coverage",
        type=float,
        default=1.0,
        help="Fail if labeled-case coverage is below this ratio (default: 1.0)",
    )
    ap.add_argument(
        "--reject-placeholders",
        action="store_true",
        help="Fail if any case contains placeholder values like TODO",
    )
    args = ap.parse_args()

    total = 0
    with_labels = 0
    unanswerable = 0
    issues: List[str] = []

    for lineno, case in iter_jsonl(args.bench):
        total += 1
        q = str(case.get("query", "")).strip()
        g = str(case.get("gold", "")).strip()
        labeled = has_retrieval_label(case)
        ua = is_unanswerable(case)

        if not q:
            issues.append(f"line {lineno}: missing/empty query")
        if not g:
            issues.append(f"line {lineno}: missing/empty gold")
        if ua:
            unanswerable += 1
            if g.strip().lower() not in {"__no_answer__", "no_answer", "unanswerable"}:
                issues.append(
                    f"line {lineno}: expected_unanswerable=true but gold is not one of "
                    "__NO_ANSWER__/NO_ANSWER/UNANSWERABLE"
                )
        elif not labeled:
            issues.append(
                f"line {lineno}: missing retrieval label "
                "(relevant_doc_id(s)/relevant_chunk_id(s)/relevant_text)"
            )
        else:
            with_labels += 1
        if args.reject_placeholders and has_placeholder_value(case):
            issues.append(f"line {lineno}: placeholder value detected (contains TODO)")

    answerable = total - unanswerable
    if total == 0:
        coverage = 0.0
    elif answerable <= 0:
        # All cases are explicitly unanswerable: retrieval-label coverage is not applicable.
        coverage = 1.0
    else:
        coverage = with_labels / answerable
    print(json.dumps({
        "bench": str(args.bench),
        "n_cases": total,
        "n_unanswerable_cases": unanswerable,
        "n_cases_with_retrieval_labels": with_labels,
        "label_coverage": coverage,
        "min_label_coverage": args.min_label_coverage,
        "n_issues": len(issues),
    }, indent=2))

    if coverage < float(args.min_label_coverage):
        issues.append(
            f"label coverage {coverage:.6f} is below required {float(args.min_label_coverage):.6f}"
        )

    if issues:
        print("\nIssues:")
        for s in issues[:100]:
            print(f"- {s}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
