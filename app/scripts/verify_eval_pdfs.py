from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_required_filenames(manifest_path: Path) -> list[str]:
    filenames: list[str] = []

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            raw = line.strip()
            if not raw:
                continue

            item = json.loads(raw)

            doc_name = str(item.get("doc_name") or "").strip()
            expected_filename = str(
                item.get("expected_filename") or f"{doc_name}.pdf"
            ).strip()

            if not doc_name:
                raise ValueError(f"Missing doc_name in line {line_number}")

            filenames.append(expected_filename)

    return sorted(set(filenames))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that all PDFs required by an evaluation manifest exist."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    required = load_required_filenames(args.manifest)

    rows: list[dict] = []
    missing: list[str] = []

    for filename in required:
        path = args.pdf_dir / filename
        exists = path.exists() and path.is_file()
        size_bytes = path.stat().st_size if exists else None

        row = {
            "filename": filename,
            "path": str(path),
            "exists": exists,
            "size_bytes": size_bytes,
        }
        rows.append(row)

        if not exists:
            missing.append(filename)

    report = {
        "manifest": str(args.manifest),
        "pdf_dir": str(args.pdf_dir),
        "required_count": len(required),
        "present_count": len(required) - len(missing),
        "missing_count": len(missing),
        "missing": missing,
        "files": rows,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())