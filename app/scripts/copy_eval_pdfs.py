from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestDocument:
    doc_name: str
    expected_filename: str


def normalize_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(".pdf", "")
        .replace("-", "_")
        .replace(" ", "_")
    )


def load_manifest(path: Path) -> list[ManifestDocument]:
    documents: list[ManifestDocument] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw = line.strip()
            if not raw:
                continue

            item = json.loads(raw)

            doc_name = str(item.get("doc_name") or "").strip()
            if not doc_name:
                raise ValueError(f"Missing doc_name in manifest line {line_number}")

            expected_filename = str(
                item.get("expected_filename") or f"{doc_name}.pdf"
            ).strip()

            documents.append(
                ManifestDocument(
                    doc_name=doc_name,
                    expected_filename=expected_filename,
                )
            )

    unique: dict[str, ManifestDocument] = {}
    for document in documents:
        unique[document.doc_name] = document

    return list(unique.values())


def index_pdfs(source_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}

    for path in source_dir.rglob("*.pdf"):
        if not path.is_file():
            continue

        keys = {
            normalize_name(path.name),
            normalize_name(path.stem),
        }

        for key in keys:
            index.setdefault(key, []).append(path)

    return index


def find_pdf(
    document: ManifestDocument,
    pdf_index: dict[str, list[Path]],
) -> tuple[Path | None, list[Path]]:
    candidate_keys = [
        normalize_name(document.expected_filename),
        normalize_name(document.doc_name),
    ]

    matches: list[Path] = []
    seen: set[Path] = set()

    for key in candidate_keys:
        for path in pdf_index.get(key, []):
            if path not in seen:
                matches.append(path)
                seen.add(path)

    if len(matches) == 1:
        return matches[0], matches

    return None, matches


def copy_documents(
    *,
    manifest_path: Path,
    source_dir: Path,
    target_dir: Path,
    dry_run: bool,
    overwrite: bool,
) -> int:
    documents = load_manifest(manifest_path)
    pdf_index = index_pdfs(source_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_existing = 0
    missing: list[str] = []
    ambiguous: list[tuple[str, list[Path]]] = []

    print(f"Manifest: {manifest_path}")
    print(f"Source dir: {source_dir}")
    print(f"Target dir: {target_dir}")
    print(f"Documents required: {len(documents)}")
    print()

    for document in documents:
        source_pdf, matches = find_pdf(document, pdf_index)

        if source_pdf is None:
            if matches:
                ambiguous.append((document.doc_name, matches))
            else:
                missing.append(document.doc_name)
            continue

        target_pdf = target_dir / document.expected_filename

        if target_pdf.exists() and not overwrite:
            skipped_existing += 1
            print(f"SKIP existing: {target_pdf.name}")
            continue

        print(f"COPY: {source_pdf} -> {target_pdf}")

        if not dry_run:
            shutil.copy2(source_pdf, target_pdf)

        copied += 1

    print()
    print("Summary")
    print(f"  copied: {copied}")
    print(f"  skipped_existing: {skipped_existing}")
    print(f"  missing: {len(missing)}")
    print(f"  ambiguous: {len(ambiguous)}")

    if missing:
        print()
        print("Missing documents:")
        for doc_name in missing:
            print(f"  - {doc_name}.pdf")

    if ambiguous:
        print()
        print("Ambiguous documents:")
        for doc_name, matches in ambiguous:
            print(f"  - {doc_name}")
            for match in matches:
                print(f"      {match}")

    if missing or ambiguous:
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy evaluation PDFs from a source folder into import/pdfs."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/data/document_manifest_train_subset_20.jsonl"),
        help="Path to JSONL document manifest.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Folder where all available PDFs are stored.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("import/pdfs"),
        help="Target folder used by Docker bind mount.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Actually copy files. Without this flag, script runs in dry-run mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files already present in target folder.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    return copy_documents(
        manifest_path=args.manifest,    
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        dry_run=not args.copy,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())