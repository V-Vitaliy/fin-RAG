#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
load_env

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
BACKUP_DIR="${BACKUP_ROOT:-/srv/finrag/backups}/$STAMP"
mkdir -p "$BACKUP_DIR"

# 1) Postgres logical dump
compose exec -T postgres pg_dump -U "${POSTGRES_USER:-rag_user}" "${POSTGRES_DB:-financial_rag_db}" > "$BACKUP_DIR/postgres.dump"

# 2) DuckDB file copy. Stop API if you need a strictly cold copy.
DUCKDB_HOST_PATH="${FINRAG_DATA_ROOT:-/srv/finrag/data}/duckdb/finance.duckdb"
if [[ -f "$DUCKDB_HOST_PATH" ]]; then
  cp "$DUCKDB_HOST_PATH" "$BACKUP_DIR/finance.duckdb"
fi

# 3) Qdrant collection snapshot through localhost admin port.
SNAPSHOT_META="$BACKUP_DIR/qdrant_snapshot.json"
SNAPSHOT_NAME=""
if curl -fsS "http://127.0.0.1:${QDRANT_PORT:-6333}/readyz" >/dev/null; then
  curl -fsS -X POST "http://127.0.0.1:${QDRANT_PORT:-6333}/collections/${RAG_COLLECTION_NAME}/snapshots" > "$SNAPSHOT_META"
  SNAPSHOT_NAME="$(python3 - <<PY
import json
from pathlib import Path
meta = json.loads(Path('$SNAPSHOT_META').read_text())
print((meta.get('result') or {}).get('name') or '')
PY
)"
  if [[ -n "$SNAPSHOT_NAME" ]]; then
    curl -fsS \
      "http://127.0.0.1:${QDRANT_PORT:-6333}/collections/${RAG_COLLECTION_NAME}/snapshots/${SNAPSHOT_NAME}" \
      -o "$BACKUP_DIR/qdrant_${RAG_COLLECTION_NAME}_${SNAPSHOT_NAME}"
  fi
fi

# 4) Batch/eval artifacts
if [[ -d "${FINRAG_DATA_ROOT:-/srv/finrag/data}/batches" ]]; then
  tar -C "${FINRAG_DATA_ROOT:-/srv/finrag/data}" -czf "$BACKUP_DIR/batches.tar.gz" batches
fi
if [[ -d "${FINRAG_DATA_ROOT:-/srv/finrag/data}/evals" ]]; then
  tar -C "${FINRAG_DATA_ROOT:-/srv/finrag/data}" -czf "$BACKUP_DIR/evals.tar.gz" evals
fi

sha256sum "$BACKUP_DIR"/* > "$BACKUP_DIR/SHA256SUMS" || true

if [[ -n "${BACKUP_S3_URI:-}" ]]; then
  aws s3 cp "$BACKUP_DIR" "$BACKUP_S3_URI/$STAMP" --recursive
fi

echo "Backup written to $BACKUP_DIR"
