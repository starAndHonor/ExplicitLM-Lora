#!/usr/bin/env bash
# 方案二：P1-Router -> P3-FusionInference
#
# 用法：
#   bash scripts/run_scheme2_p1_p3_infer.sh \
#     --question "..." \
#     --option-a "..." \
#     --option-b "..." \
#     --option-c "..." \
#     --option-d "..."
#
#   使用指定知识源先重建 Phase1 store，再做推理：
#   REBUILD_KNOWLEDGE=data/my_knowledge.jsonl \
#   bash scripts/run_scheme2_p1_p3_infer.sh \
#     --question "..." \
#     --option-a "..." \
#     --option-b "..." \
#     --option-c "..." \
#     --option-d "..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

DEVICE="${DEVICE:-cpu}"
PHASE1_CKPT="${PHASE1_CKPT:-${PROJECT_ROOT}/checkpoints/phase1_best}"
PHASE3_CKPT="${PHASE3_CKPT:-${PROJECT_ROOT}/checkpoints/phase3_best}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
REBUILD_KNOWLEDGE="${REBUILD_KNOWLEDGE:-}"
REBUILD_OUTPUT_DIR="${REBUILD_OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/phase1_rebuilt_for_scheme2}"
REBUILD_DEVICE="${REBUILD_DEVICE:-${DEVICE}}"
REBUILD_CHUNK_SIZE="${REBUILD_CHUNK_SIZE:-}"
REBUILD_TOKENIZE_BATCH_SIZE="${REBUILD_TOKENIZE_BATCH_SIZE:-}"
REBUILD_LIMIT="${REBUILD_LIMIT:--1}"

if [ -n "${REBUILD_KNOWLEDGE}" ]; then
    echo "[Scheme2] Rebuilding Phase1 store from knowledge source: ${REBUILD_KNOWLEDGE}"
    REBUILD_CMD=(
        conda run --no-capture-output -n ExplicitLLM
        python "${SCRIPT_DIR}/rebuild_phase1_store.py"
        --config "${CONFIG}"
        --input "${REBUILD_KNOWLEDGE}"
        --phase1-ckpt "${PHASE1_CKPT}"
        --output-dir "${REBUILD_OUTPUT_DIR}"
        --device "${REBUILD_DEVICE}"
        --limit "${REBUILD_LIMIT}"
    )
    if [ -n "${REBUILD_CHUNK_SIZE}" ]; then
        REBUILD_CMD+=(--chunk-size "${REBUILD_CHUNK_SIZE}")
    fi
    if [ -n "${REBUILD_TOKENIZE_BATCH_SIZE}" ]; then
        REBUILD_CMD+=(--tokenize-batch-size "${REBUILD_TOKENIZE_BATCH_SIZE}")
    fi
    "${REBUILD_CMD[@]}"
    PHASE1_CKPT="${REBUILD_OUTPUT_DIR}"
    echo "[Scheme2] Using rebuilt Phase1 checkpoint: ${PHASE1_CKPT}"
fi

conda run --no-capture-output -n ExplicitLLM \
    python "${SCRIPT_DIR}/run_phase1_phase3_infer.py" \
    --config "${CONFIG}" \
    --phase1-ckpt "${PHASE1_CKPT}" \
    --phase3-ckpt "${PHASE3_CKPT}" \
    --device "${DEVICE}" \
    "$@"
