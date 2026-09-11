#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"
load_env

RUN_ID="${RUN_ID:?Set RUN_ID, e.g. RUN_ID=prod_batch_20260707_120000}"
DATASET="${DATASET:-/workspace/evaluation/data/train_subset_20.json}"
DOCUMENT_MAP="${DOCUMENT_MAP:-/data/batches/$RUN_ID/document_id_map.json}"
QUESTION_LIMIT="${QUESTION_LIMIT:-0}"
NO_JUDGE="${NO_JUDGE:-true}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
STOP_ON_ERROR="${STOP_ON_ERROR:-false}"

TMP_DATASET="/data/evals/$RUN_ID/tmp/eval_dataset_limit_${QUESTION_LIMIT}.json"

if [[ "$QUESTION_LIMIT" != "0" ]]; then
  compose run --rm \
    -e DATASET="$DATASET" \
    -e TMP_DATASET="$TMP_DATASET" \
    -e QUESTION_LIMIT="$QUESTION_LIMIT" \
    api python - <<'PY'
import json, os
from pathlib import Path
src = Path(os.environ["DATASET"])
dst = Path(os.environ["TMP_DATASET"])
limit = int(os.environ["QUESTION_LIMIT"])
data = json.loads(src.read_text(encoding="utf-8"))
if limit > 0:
    data = data[:limit]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(data)} questions to {dst}")
PY
  DATASET="$TMP_DATASET"
fi

ARGS=(
  python -m app.scripts.run_eval
  --dataset "$DATASET"
  --document-map "$DOCUMENT_MAP"
  --workspace-id "${RAG_GLOBAL_WORKSPACE_ID}"
  --run-id "$RUN_ID"
  --eval-root /data/evals
  --judge-model "$JUDGE_MODEL"
)

if [[ "$NO_JUDGE" == "true" ]]; then
  ARGS+=(--no-judge)
fi

if [[ "$STOP_ON_ERROR" == "true" ]]; then
  ARGS+=(--stop-on-error)
fi

compose run --rm api "${ARGS[@]}"
