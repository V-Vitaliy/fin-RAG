#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
load_env

cd "$REPO_ROOT"

if [[ "${GIT_PULL:-false}" == "true" ]]; then
  git pull --ff-only
fi

echo "Deploying commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

"$SCRIPT_DIR/01_prepare_dirs.sh"

compose build api ui
compose up -d postgres qdrant
compose run --rm api alembic upgrade head
compose up -d api ui
compose ps
