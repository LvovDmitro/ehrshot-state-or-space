#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUTPUT_DIR="${REPRO_OUTPUT_DIR:-state_or_space_reproducibility_package}"
TASK_NAME="${REPRO_TASK_NAME:-state_or_space_prepare_reproducibility_package}"
SOURCE_ROOT="${REPRO_SOURCE_ROOT:-}"
SKIP_UPLOAD="${SKIP_REPRO_UPLOAD:-0}"
ENABLE_CLEARML="${REPRO_ENABLE_CLEARML:-1}"
if [[ -n "$SOURCE_ROOT" && "$SKIP_UPLOAD" == "1" ]]; then
  ENABLE_CLEARML=0
fi

if [[ "$ENABLE_CLEARML" == "1" ]]; then
  : "${CLEARML_PROJECT:?Set CLEARML_PROJECT}"
  : "${CLEARML_OUTPUT_URI:?Set CLEARML_OUTPUT_URI}"
fi
if [[ -z "$SOURCE_ROOT" || "$SKIP_UPLOAD" != "1" ]]; then
  : "${EHRSHOT_S3_BASE:?Set EHRSHOT_S3_BASE}"
fi

cd "$PROJECT_ROOT"
mkdir -p logs reproducibility

ARGS=(
  --repo-root "$PROJECT_ROOT"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "${EHRSHOT_S3_BASE:-}" ]]; then
  ARGS+=(--storage-base-s3-prefix "$EHRSHOT_S3_BASE")
fi
if [[ "$SKIP_UPLOAD" != "1" ]]; then
  ARGS+=(--output-s3-prefix "$EHRSHOT_S3_BASE/state_or_space_reproducibility_package")
fi
if [[ "$ENABLE_CLEARML" == "1" ]]; then
  ARGS+=(
    --clearml-project "$CLEARML_PROJECT"
    --clearml-output-uri "$CLEARML_OUTPUT_URI"
    --clearml-task-name "$TASK_NAME"
    --enable-clearml
    --clearml-upload-artifacts
  )
fi

if [[ -n "$SOURCE_ROOT" ]]; then
  ARGS+=(--source-root "$SOURCE_ROOT")
fi
if [[ -f reproducibility/clearml_tasks.csv ]]; then
  ARGS+=(--clearml-tasks-file reproducibility/clearml_tasks.csv)
fi
if [[ "$SKIP_UPLOAD" == "1" ]]; then
  ARGS+=(--skip-upload)
fi

export CLEARML_TASK_NO_REUSE="${CLEARML_TASK_NO_REUSE:-1}"
export PYTHONUNBUFFERED=1

python final_exps/06_prepare_final_reproducibility_package.py \
  "${ARGS[@]}" \
  2>&1 | tee "logs/reproducibility_package_$(date +%Y%m%d_%H%M%S).log"
