#!/usr/bin/env bash
# 运行 E3 的多 token 预算版本，默认 k=32/64/128/256
#
# 用法：
#   ENC_MODE=qwen3 \
#   PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   bash scripts/run_e3_multik.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-6,7}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
K_LIST="${K_LIST:-32 64 128 256}"
PHASE2_WEIGHTS="${PHASE2_WEIGHTS:-checkpoints/phase2_best}"
PHASE3_WEIGHTS="${PHASE3_WEIGHTS:-checkpoints/phase3_best}"
DRY_RUN="${DRY_RUN:-0}"

for K in ${K_LIST}; do
    echo "[E3-MULTIK] running k=${K}"
    CMD=(
      conda run --no-capture-output -n ExplicitLLM
      python "${PROJECT_ROOT}/experiments/e3/run_e3.py"
      --config "${CONFIG}"
      --phase2-weights "${PHASE2_WEIGHTS}"
      --phase3-weights "${PHASE3_WEIGHTS}"
      --k "${K}"
      --device "${DEVICE}"
      --max-samples "${MAX_SAMPLES}"
      --output "${PROJECT_ROOT}/results/e3/e3_fair_compare_k${K}_$(basename "${PHASE2_WEIGHTS%/}")__$(basename "${PHASE3_WEIGHTS%/}").json"
    )
    if [ "${ENC_MODE}" = "qwen3" ]; then
        CMD+=(--override "model.knowledge_encoder_mode=qwen3")
    fi
    printf '[E3-MULTIK] Command:'
    for arg in "${CMD[@]}"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
    if [ "${DRY_RUN}" = "1" ]; then
        continue
    fi
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
done
