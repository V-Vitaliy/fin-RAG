#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


QUESTION_TYPE_TARGETS = {
    "metrics-generated": 6,
    "domain-relevant": 8,
    "novel-generated": 6,
}

DOC_TYPE_TARGETS = {
    "10k": 12,
    "10q": 3,
    "8k": 3,
    "Earnings": 2,
}

RARE_DOC_TYPE_BONUS = {
    "8k": 20,
    "Earnings": 20,
    "10q": 10,
}


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")

    return data


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def with_split(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for row in rows:
        item = copy.deepcopy(row)
        item["split"] = split
        result.append(item)

    return result


def build_document_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}

    for row in rows:
        doc_name = row["doc_name"]

        if doc_name not in grouped:
            grouped[doc_name] = {
                "doc_name": doc_name,
                "filename": f"{doc_name}.pdf",
                "company": row.get("company"),
                "doc_type": row.get("doc_type"),
                "doc_period": row.get("doc_period"),
                "doc_link": row.get("doc_link"),
                "splits": set(),
                "question_ids": [],
                "question_count": 0,
            }

        grouped[doc_name]["splits"].add(row.get("split"))
        grouped[doc_name]["question_ids"].append(row.get("financebench_id"))
        grouped[doc_name]["question_count"] += 1

    manifest = []

    for row in sorted(grouped.values(), key=lambda item: item["doc_name"]):
        item = dict(row)
        item["splits"] = sorted(split for split in row["splits"] if split)
        manifest.append(item)

    return manifest


def select_train_subset(train_rows: list[dict[str, Any]], *, n: int = 20) -> list[dict[str, Any]]:
    """
    Deterministic coverage-oriented subset selection.

    The goal is not to optimize model score. The goal is to create a small subset
    that exercises:
      - all question types;
      - 10-K, 10-Q, 8-K and Earnings documents;
      - multiple companies;
      - extraction, numerical reasoning and financial reasoning cases.

    The algorithm is greedy and operates on the original train rows only.
    """

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    question_type_counts: Counter[str] = Counter()
    doc_type_counts: Counter[str] = Counter()
    covered_companies: set[str] = set()
    covered_docs: set[str] = set()

    for _ in range(n):
        best_row: dict[str, Any] | None = None
        best_key: tuple[float, int] | None = None

        for index, row in enumerate(train_rows):
            case_id = row["financebench_id"]

            if case_id in selected_ids:
                continue

            question_type = row["question_type"]
            doc_type = row["doc_type"]
            company = row["company"]
            doc_name = row["doc_name"]

            score = 0.0

            if question_type_counts[question_type] < QUESTION_TYPE_TARGETS[question_type]:
                score += 100
                score += 10 * (QUESTION_TYPE_TARGETS[question_type] - question_type_counts[question_type])
            else:
                score -= 50

            if doc_type_counts[doc_type] < DOC_TYPE_TARGETS[doc_type]:
                score += 80
                score += 8 * (DOC_TYPE_TARGETS[doc_type] - doc_type_counts[doc_type])
            else:
                score -= 30

            score += RARE_DOC_TYPE_BONUS.get(doc_type, 0)

            if company not in covered_companies:
                score += 25

            if doc_name in covered_docs:
                score += 8
            else:
                score += 3

            # Tiny stable source-order preference.
            score += max(0.0, 5.0 - index / 20.0)

            key = (score, -index)

            if best_key is None or key > best_key:
                best_key = key
                best_row = row

        if best_row is None:
            break

        selected.append(best_row)
        selected_ids.add(best_row["financebench_id"])
        question_type_counts[best_row["question_type"]] += 1
        doc_type_counts[best_row["doc_type"]] += 1
        covered_companies.add(best_row["company"])
        covered_docs.add(best_row["doc_name"])

    # Return rows in original train order to keep the dataset easy to inspect.
    selected_id_set = {row["financebench_id"] for row in selected}
    return [copy.deepcopy(row) for row in train_rows if row["financebench_id"] in selected_id_set]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "questions": len(rows),
        "unique_documents": len({row["doc_name"] for row in rows}),
        "unique_companies": len({row["company"] for row in rows}),
        "question_type_counts": dict(Counter(row["question_type"] for row in rows)),
        "doc_type_counts": dict(Counter(row["doc_type"] for row in rows)),
        "company_counts": dict(Counter(row["company"] for row in rows)),
        "documents": sorted(Counter(row["doc_name"] for row in rows).items()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=20)
    args = parser.parse_args()

    train = load_json(args.train)
    val = load_json(args.val)

    train_with_split = with_split(train, "train")
    val_with_split = with_split(val, "val")
    all_questions = train_with_split + val_with_split

    subset = with_split(
        select_train_subset(train, n=args.subset_size),
        "train",
    )

    all_manifest = build_document_manifest(all_questions)
    subset_manifest = build_document_manifest(subset)

    summary = {
        "train": summarize(train_with_split),
        "val": summarize(val_with_split),
        "all": summarize(all_questions),
        "train_subset": summarize(subset),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dump_json(args.out_dir / "all_questions.json", all_questions)
    dump_json(args.out_dir / "train_subset_20.json", subset)
    dump_jsonl(args.out_dir / "document_manifest_all.jsonl", all_manifest)
    dump_jsonl(args.out_dir / "document_manifest_train_subset_20.jsonl", subset_manifest)
    dump_json(args.out_dir / "eval_data_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
