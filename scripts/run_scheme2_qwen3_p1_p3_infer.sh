#!/usr/bin/env bash
# 方案二（qwen3）：Phase1-Router -> Phase3-FusionInference
#
# 用法：
#   MODEL_PATH=/home/undergraduate/zcy/Explicit-Lora/Qwen3-0.6B \
#   DEVICE=cuda:0 \
#   bash scripts/run_scheme2_qwen3_p1_p3_infer.sh \
#     --question "..." \
#     --option-a "..." \
#     --option-b "..." \
#     --option-c "..." \
#     --option-d "..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/Qwen3-0.6B}"
export DEVICE="${DEVICE:-cpu}"
export CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
export PHASE1_CKPT="${PHASE1_CKPT:-${PROJECT_ROOT}/checkpoints/phase1_best}"
export PHASE3_CKPT="${PHASE3_CKPT:-${PROJECT_ROOT}/checkpoints/p3_from_p2_qwen3_10ep/phase3_best}"

bash "${SCRIPT_DIR}/run_scheme2_p1_p3_infer.sh" "$@"
