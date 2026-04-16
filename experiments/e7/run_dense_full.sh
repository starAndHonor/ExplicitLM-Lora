#!/usr/bin/env bash
# 一键自动化 Dense E7（单视图，fusion_length 由 K_SIZE 指定）：
#   1. 构建 1M FineWeb base（可跳过：BUILD_BASE=0）
#   2. 生成 MedQA / ARC / MMLU 三份 overlay index（可跳过：BUILD_OVERLAYS=0）
#   3. 评测检索正确率（可跳过：RUN_RETRIEVAL_EVAL=0）
#   4. 运行 Dense E7 评测（可跳过：SKIP_E7=1）
#
# 用法示例：
#   BUILD_BASE=0 BUILD_OVERLAYS=0 TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash experiments/e7/run_dense_full.sh
#
#   K_SIZE=128 BUILD_BASE=0 \
#     bash experiments/e7/run_dense_full.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/_experiment_common.sh"

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
GPU_IDS="${GPU_IDS:-6}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
QUERY_MODE="${QUERY_MODE:-question_only}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DRY_RUN="${DRY_RUN:-0}"
K_SIZE="${K_SIZE:-64}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-10000}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PARQUET_DIR="${PARQUET_DIR:-${DATA_ROOT}/compressed/v2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
TRAINING_FREE_WEIGHTS="${TRAINING_FREE_WEIGHTS:-${PROJECT_ROOT}/checkpoints/p3_from_p2_qwen3_10ep/phase3_best}"

BASE_INDEX="${BASE_INDEX:-${CHECKPOINT_DIR}/dense_fineweb_1m_flat_r0_qwen3_fv.pt}"

# 输入使用对应 k-size 的预处理 knowledge 文件（k64 无后缀，其余带 _k{K_SIZE}）
if [ "${K_SIZE}" = "64" ]; then
    MEDQA_INPUT="${MEDQA_INPUT:-${DATA_ROOT}/medqa_knowledge.jsonl}"
    ARC_INPUT="${ARC_INPUT:-${DATA_ROOT}/arc_knowledge.jsonl}"
    MMLU_INPUT="${MMLU_INPUT:-${DATA_ROOT}/mmlu_knowledge.jsonl}"
else
    MEDQA_INPUT="${MEDQA_INPUT:-${DATA_ROOT}/medqa_knowledge_k${K_SIZE}.jsonl}"
    ARC_INPUT="${ARC_INPUT:-${DATA_ROOT}/arc_knowledge_k${K_SIZE}.jsonl}"
    MMLU_INPUT="${MMLU_INPUT:-${DATA_ROOT}/mmlu_knowledge_k${K_SIZE}.jsonl}"
fi

# 输出文件名带 k${K_SIZE}
MEDQA_INDEX="${MEDQA_INDEX:-${CHECKPOINT_DIR}/dense_fineweb_medqa_overlay_k${K_SIZE}_flat_r0_qwen3.pt}"
ARC_INDEX="${ARC_INDEX:-${CHECKPOINT_DIR}/dense_fineweb_arc_overlay_k${K_SIZE}_flat_r0_qwen3.pt}"
MMLU_INDEX="${MMLU_INDEX:-${CHECKPOINT_DIR}/dense_fineweb_mmlu_overlay_k${K_SIZE}_flat_r0_qwen3.pt}"

BUILD_BASE="${BUILD_BASE:-1}"
BUILD_OVERLAYS="${BUILD_OVERLAYS:-1}"
SKIP_E7="${SKIP_E7:-0}"
RUN_RETRIEVAL_EVAL="${RUN_RETRIEVAL_EVAL:-1}"

declare -a EXTRA_ARGS=("$@")

exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

run_cmd() {
    exp_print_cmd "$@"
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    "$@"
}

echo "[E7DenseFull] config=${CONFIG}"
echo "[E7DenseFull] device=${DEVICE}"
echo "[E7DenseFull] enc_mode=${ENC_MODE}"
echo "[E7DenseFull] k_size=${K_SIZE}"
echo "[E7DenseFull] base_index=${BASE_INDEX}"
echo "[E7DenseFull] medqa_index=${MEDQA_INDEX}"
echo "[E7DenseFull] arc_index=${ARC_INDEX}"
echo "[E7DenseFull] mmlu_index=${MMLU_INDEX}"
echo "[E7DenseFull] training_free_weights=${TRAINING_FREE_WEIGHTS}"
echo "[E7DenseFull] query_mode=${QUERY_MODE}"
echo "[E7DenseFull] max_samples=${MAX_SAMPLES}"
echo "[E7DenseFull] build_base=${BUILD_BASE}"
echo "[E7DenseFull] build_overlays=${BUILD_OVERLAYS}"
echo "[E7DenseFull] run_retrieval_eval=${RUN_RETRIEVAL_EVAL}"
echo "[E7DenseFull] skip_e7=${SKIP_E7}"

if [ "${BUILD_BASE}" = "1" ]; then
    run_cmd \
        conda run --no-capture-output -n ExplicitLLM \
        python "${PROJECT_ROOT}/scripts/build_dense_index_from_fineweb.py" \
        --config "${CONFIG}" \
        --parquet-dir "${PARQUET_DIR}" \
        --output "${BASE_INDEX}" \
        --sample-size 1048576 \
        --seed 0 \
        --device "${DEVICE}" \
        --batch-size "${BATCH_SIZE}" \
        --tokenize-batch-size "${TOKENIZE_BATCH_SIZE}" \
        --index-type flat \
        --override "model.fusion_length=${K_SIZE}" \
        "${EXTRA_ARGS[@]}"
fi

if [ "${BUILD_OVERLAYS}" = "1" ]; then
    run_cmd \
        conda run --no-capture-output -n ExplicitLLM \
        python "${PROJECT_ROOT}/scripts/overlay_dense_index.py" \
        --config "${CONFIG}" \
        --index "${BASE_INDEX}" \
        --input "${MEDQA_INPUT}" \
        --output "${MEDQA_INDEX}" \
        --device "${DEVICE}" \
        --batch-size "${BATCH_SIZE}" \
        --tokenize-batch-size "${TOKENIZE_BATCH_SIZE}" \
        --seed 42 \
        --override "model.fusion_length=${K_SIZE}" \
        "${EXTRA_ARGS[@]}"

    run_cmd \
        conda run --no-capture-output -n ExplicitLLM \
        python "${PROJECT_ROOT}/scripts/overlay_dense_index.py" \
        --config "${CONFIG}" \
        --index "${BASE_INDEX}" \
        --input "${ARC_INPUT}" \
        --output "${ARC_INDEX}" \
        --device "${DEVICE}" \
        --batch-size "${BATCH_SIZE}" \
        --tokenize-batch-size "${TOKENIZE_BATCH_SIZE}" \
        --seed 42 \
        --override "model.fusion_length=${K_SIZE}" \
        "${EXTRA_ARGS[@]}"

    run_cmd \
        conda run --no-capture-output -n ExplicitLLM \
        python "${PROJECT_ROOT}/scripts/overlay_dense_index.py" \
        --config "${CONFIG}" \
        --index "${BASE_INDEX}" \
        --input "${MMLU_INPUT}" \
        --output "${MMLU_INDEX}" \
        --device "${DEVICE}" \
        --batch-size "${BATCH_SIZE}" \
        --tokenize-batch-size "${TOKENIZE_BATCH_SIZE}" \
        --seed 42 \
        --override "model.fusion_length=${K_SIZE}" \
        "${EXTRA_ARGS[@]}"
fi

if [ "${RUN_RETRIEVAL_EVAL}" = "1" ]; then
    run_cmd \
        conda run --no-capture-output -n ExplicitLLM \
        python "${PROJECT_ROOT}/experiments/e7/eval_dense_retrieval.py" \
        --config "${CONFIG}" \
        --dense-index-medqa "${MEDQA_INDEX}" \
        --dense-index-arc "${ARC_INDEX}" \
        --dense-index-mmlu "${MMLU_INDEX}" \
        --device "${DEVICE}" \
        --top-k 16 \
        --query-mode "${QUERY_MODE}" \
        --limit "${MAX_SAMPLES}" \
        --output "${PROJECT_ROOT}/results/e7/e7_dense_retrieval_precheck_k${K_SIZE}_r0.json" \
        "${EXTRA_ARGS[@]}"
fi

if [ "${SKIP_E7}" = "1" ]; then
    exit 0
fi

run_cmd \
    env \
    CONFIG="${CONFIG}" \
    GPU_IDS="${GPU_IDS}" \
    DEVICE="${DEVICE}" \
    ENC_MODE="${ENC_MODE}" \
    QUERY_MODE="${QUERY_MODE}" \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    DENSE_INDEX_MEDQA="${MEDQA_INDEX}" \
    DENSE_INDEX_ARC="${ARC_INDEX}" \
    DENSE_INDEX_MMLU="${MMLU_INDEX}" \
    TRAINING_FREE_WEIGHTS="${TRAINING_FREE_WEIGHTS}" \
    OUTPUT="${PROJECT_ROOT}/results/e7/e7_dense_k${K_SIZE}_r0_$(basename "${TRAINING_FREE_WEIGHTS}").json" \
    bash "${PROJECT_ROOT}/experiments/e7/run.sh" \
    "${EXTRA_ARGS[@]}"
