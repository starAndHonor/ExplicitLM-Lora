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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

DEVICE="${DEVICE:-cpu}"
PHASE1_CKPT="${PHASE1_CKPT:-${PROJECT_ROOT}/checkpoints/phase1_best}"
PHASE3_CKPT="${PHASE3_CKPT:-${PROJECT_ROOT}/checkpoints/phase3_best}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"

conda run --no-capture-output -n ExplicitLLM \
    python "${SCRIPT_DIR}/run_phase1_phase3_infer.py" \
    --config "${CONFIG}" \
    --phase1-ckpt "${PHASE1_CKPT}" \
    --phase3-ckpt "${PHASE3_CKPT}" \
    --device "${DEVICE}" \
    "$@"

