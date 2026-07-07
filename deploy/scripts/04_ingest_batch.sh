#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
load_env

MANIFEST="${MANIFEST:-/data/import/document_manifest.jsonl}"
PDF_DIR="${PDF_DIR:-/data/import/pdfs}"
RUN_NAME="${RUN_NAME:-prod_batch}"
RUN_ID="${RUN_ID:-}"
DOCUMENT_LIMIT="${DOCUMENT_LIMIT:-0}"
STOP_ON_ERROR="${STOP_ON_ERROR:-false}"
SKIP_EXISTING_READY="${SKIP_EXISTING_READY:-true}"

TMP_MANIFEST="/data/batches/_tmp_manifest_${RUN_NAME}_limit_${DOCUMENT_LIMIT}.jsonl"

if [[ "$DOCUMENT_LIMIT" != "0" ]]; then
  compose run --rm \
    -e MANIFEST="$MANIFEST" \
    -e TMP_MANIFEST="$TMP_MANIFEST" \
    -e DOCUMENT_LIMIT="$DOCUMENT_LIMIT" \
    api python - <<'PY'
import os
from pathlib import Path
src = Path(os.environ["MANIFEST"])
dst = Path(os.environ["TMP_MANIFEST"])
limit = int(os.environ["DOCUMENT_LIMIT"])
lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
if limit > 0:
    lines = lines[:limit]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {len(lines)} manifest lines to {dst}")
PY
  MANIFEST="$TMP_MANIFEST"
fi

ARGS=(
  python -m app.scripts.ingest_global_batch
  --manifest "$MANIFEST"
  --pdf-dir "$PDF_DIR"
  --workspace-id "${RAG_GLOBAL_WORKSPACE_ID}"
  --workspace-name "${WORKSPACE_NAME:-Global FinanceBench Workspace}"
  --run-name "$RUN_NAME"
  --eval-root /data/evals
  --batch-root /data/batches
)

if [[ -n "$RUN_ID" ]]; then
  ARGS+=(--run-id "$RUN_ID")
fi

if [[ "$STOP_ON_ERROR" == "true" ]]; then
  ARGS+=(--stop-on-error)
fi

if [[ "$SKIP_EXISTING_READY" == "true" ]]; then
  ARGS+=(--skip-existing-ready)
else
  ARGS+=(--no-skip-existing-ready)
fi

compose run --rm api "${ARGS[@]}"
