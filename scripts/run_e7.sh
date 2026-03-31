#!/usr/bin/env bash
# 运行 E7：Qwen3-0.6B / Qwen3-0.6B+RAG / 三种方案，在 MedQA / ARC / MMLU 上统一对比
#
# 用法：
#   PHASE1_WEIGHTS=checkpoints/phase1_best \
#   SCHEME1_WEIGHTS=checkpoints/scheme1_final/phase3_best \
#   SCHEME2_WEIGHTS=checkpoints/phase3_best \
#   SCHEME3_WEIGHTS=checkpoints/scheme3_final/phase3_best \
#   bash scripts/run_e7.sh
#
#   GPU_IDS=2 \
#   MAX_SAMPLES=50 \
#   bash scripts/run_e7.sh
#
# 说明：
#   - 默认 `ENC_MODE=qwen3`，即默认使用 Qwen 嵌入
#   - 如果想显式切回可训练编码器，可传 `ENC_MODE=trainable`

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
GPU_IDS="${GPU_IDS:-6}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DRY_RUN="${DRY_RUN:-0}"

PHASE1_WEIGHTS="${PHASE1_WEIGHTS:-checkpoints/phase1_best}"
SCHEME1_WEIGHTS="${SCHEME1_WEIGHTS:-checkpoints/scheme1_final/phase3_best}"
SCHEME2_WEIGHTS="${SCHEME2_WEIGHTS:-checkpoints/phase3_best}"
SCHEME3_WEIGHTS="${SCHEME3_WEIGHTS:-checkpoints/scheme3_final/phase3_best}"

exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [ -z "${OUTPUT:-}" ]; then
    OUTPUT="${PROJECT_ROOT}/results/e7/e7_$(exp_ckpt_tag "${SCHEME1_WEIGHTS}")__$(exp_ckpt_tag "${SCHEME2_WEIGHTS}")__$(exp_ckpt_tag "${SCHEME3_WEIGHTS}").json"
fi

declare -a CMD=(
    conda run --no-capture-output -n ExplicitLLM
    python "${PROJECT_ROOT}/experiments/e7/run_e7.py"
    --config "${CONFIG}"
    --phase1-weights "${PHASE1_WEIGHTS}"
    --scheme1-weights "${SCHEME1_WEIGHTS}"
    --scheme2-weights "${SCHEME2_WEIGHTS}"
    --scheme3-weights "${SCHEME3_WEIGHTS}"
    --device "${DEVICE}"
    --max-samples "${MAX_SAMPLES}"
    --output "${OUTPUT}"
)

if [ "${ENC_MODE}" = "qwen3" ]; then
    CMD+=(--override "model.knowledge_encoder_mode=qwen3")
fi

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "[E7] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[E7] config=${CONFIG}"
echo "[E7] device=${DEVICE}"
echo "[E7] enc_mode=${ENC_MODE}"
echo "[E7] phase1_weights=${PHASE1_WEIGHTS}"
echo "[E7] scheme1_weights=${SCHEME1_WEIGHTS}"
echo "[E7] scheme2_weights=${SCHEME2_WEIGHTS}"
echo "[E7] scheme3_weights=${SCHEME3_WEIGHTS}"
echo "[E7] output=${OUTPUT}"
exp_print_cmd "${CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

"${CMD[@]}"
