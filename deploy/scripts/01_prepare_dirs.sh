#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
load_env

sudo mkdir -p \
  "${FINRAG_DATA_ROOT:-/srv/finrag/data}/postgres" \
  "${FINRAG_DATA_ROOT:-/srv/finrag/data}/qdrant" \
  "${FINRAG_DATA_ROOT:-/srv/finrag/data}/duckdb" \
  "${FINRAG_DATA_ROOT:-/srv/finrag/data}/evals" \
  "${FINRAG_DATA_ROOT:-/srv/finrag/data}/batches" \
  "${FINRAG_IMPORT_ROOT:-/srv/finrag/import}/pdfs" \
  "${FINRAG_MODELS_ROOT:-/srv/finrag/models}/huggingface" \
  "${FINRAG_MODELS_ROOT:-/srv/finrag/models}/sentence-transformers" \
  "${BACKUP_ROOT:-/srv/finrag/backups}"

sudo chown -R "$USER":"$USER" \
  "${FINRAG_DATA_ROOT:-/srv/finrag/data}" \
  "${FINRAG_IMPORT_ROOT:-/srv/finrag/import}" \
  "${FINRAG_MODELS_ROOT:-/srv/finrag/models}" \
  "${BACKUP_ROOT:-/srv/finrag/backups}"

echo "Prepared persistent directories."
