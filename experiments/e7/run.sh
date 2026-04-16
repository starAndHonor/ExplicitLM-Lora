#!/usr/bin/env bash
# 运行 E7（Dense 版）：B0 / Dense training-free / Dense RAG 在 MedQA / ARC / MMLU 上统一对比
#
# 用法：
#   TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash experiments/e7/run.sh
#
#   K_SIZE=128 TRAINING_FREE_WEIGHTS=... bash experiments/e7/run.sh
#
# 说明：
#   - K_SIZE 控制 overlay 索引的 fusion_length，默认 64
#   - 索引路径由 K_SIZE 自动推导，也可手动覆盖 DENSE_INDEX_* 变量

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/_experiment_common.sh"

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
GPU_IDS="${GPU_IDS:-6}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DRY_RUN="${DRY_RUN:-0}"
K_SIZE="${K_SIZE:-64}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"

DENSE_INDEX_MEDQA="${DENSE_INDEX_MEDQA:-${CHECKPOINT_DIR}/dense_fineweb_medqa_overlay_k${K_SIZE}_flat_r0_qwen3.pt}"
DENSE_INDEX_ARC="${DENSE_INDEX_ARC:-${CHECKPOINT_DIR}/dense_fineweb_arc_overlay_k${K_SIZE}_flat_r0_qwen3.pt}"
DENSE_INDEX_MMLU="${DENSE_INDEX_MMLU:-${CHECKPOINT_DIR}/dense_fineweb_mmlu_overlay_k${K_SIZE}_flat_r0_qwen3.pt}"
TRAINING_FREE_WEIGHTS="${TRAINING_FREE_WEIGHTS:-checkpoints/p3_from_p2_qwen3_10ep/phase3_best}"
QUERY_MODE="${QUERY_MODE:-question_only}"

exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [ -z "${OUTPUT:-}" ]; then
    OUTPUT="${PROJECT_ROOT}/results/e7/e7_dense_k${K_SIZE}_r0_$(exp_ckpt_tag "${TRAINING_FREE_WEIGHTS}").json"
fi

declare -a CMD=(
    conda run --no-capture-output -n ExplicitLLM
    python "${PROJECT_ROOT}/experiments/e7/run_e7.py"
    --config "${CONFIG}"
    --dense-index-medqa "${DENSE_INDEX_MEDQA}"
    --dense-index-arc "${DENSE_INDEX_ARC}"
    --dense-index-mmlu "${DENSE_INDEX_MMLU}"
    --training-free-weights "${TRAINING_FREE_WEIGHTS}"
    --device "${DEVICE}"
    --max-samples "${MAX_SAMPLES}"
    --output "${OUTPUT}"
    --query-mode "${QUERY_MODE}"
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
echo "[E7] k_size=${K_SIZE}"
echo "[E7] dense_index_medqa=${DENSE_INDEX_MEDQA}"
echo "[E7] dense_index_arc=${DENSE_INDEX_ARC}"
echo "[E7] dense_index_mmlu=${DENSE_INDEX_MMLU}"
echo "[E7] training_free_weights=${TRAINING_FREE_WEIGHTS}"
echo "[E7] query_mode=${QUERY_MODE}"
echo "[E7] output=${OUTPUT}"
exp_print_cmd "${CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

"${CMD[@]}"
