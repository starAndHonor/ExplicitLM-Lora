#!/usr/bin/env bash
# 一键自动化 Dense E7：
#   1. 构建 1M FineWeb base
#   2. 生成 MedQA / ARC / MMLU 三份 overlay index
#   3. 先评测检索正确率
#   4. 再运行 Dense E7 评测
#
# 默认同时跑两套 anchor 版本：
#   - original_text
#   - k256（将 knowledge_ids 解码成检索文本）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/_experiment_common.sh"

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
GPU_IDS="${GPU_IDS:-2}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
QUERY_MODE="${QUERY_MODE:-question_only}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DRY_RUN="${DRY_RUN:-0}"
ANCHOR_VARIANTS="${ANCHOR_VARIANTS:-original_text,k256}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data}"
PARQUET_DIR="${PARQUET_DIR:-${DATA_ROOT}/compressed/v2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
TRAINING_FREE_WEIGHTS="${TRAINING_FREE_WEIGHTS:-${PROJECT_ROOT}/checkpoints/p3_from_p2_qwen3_10ep/phase3_best}"

BASE_INDEX="${BASE_INDEX:-${CHECKPOINT_DIR}/dense_fineweb_1m_flat_r24_qwen3.pt}"

BUILD_BASE="${BUILD_BASE:-1}"
BUILD_OVERLAYS="${BUILD_OVERLAYS:-1}"
SKIP_E7="${SKIP_E7:-0}"
RUN_RETRIEVAL_EVAL="${RUN_RETRIEVAL_EVAL:-1}"

MEDQA_ANCHOR="${MEDQA_ANCHOR:-${DATA_ROOT}/medqa_knowledge_original_text.jsonl}"
MEDQA_FUSION="${MEDQA_FUSION:-${DATA_ROOT}/medqa_knowledge.jsonl}"
ARC_ANCHOR="${ARC_ANCHOR:-${DATA_ROOT}/arc_knowledge_original_text.jsonl}"
ARC_FUSION="${ARC_FUSION:-${DATA_ROOT}/arc_knowledge.jsonl}"
MMLU_ANCHOR="${MMLU_ANCHOR:-${DATA_ROOT}/mmlu_knowledge_original_text.jsonl}"
MMLU_FUSION="${MMLU_FUSION:-${DATA_ROOT}/mmlu_knowledge.jsonl}"

MEDQA_ANCHOR_K256="${MEDQA_ANCHOR_K256:-${DATA_ROOT}/medqa_knowledge_k256.jsonl}"
ARC_ANCHOR_K256="${ARC_ANCHOR_K256:-${DATA_ROOT}/arc_knowledge_k256.jsonl}"
MMLU_ANCHOR_K256="${MMLU_ANCHOR_K256:-${DATA_ROOT}/mmlu_knowledge_k256.jsonl}"

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
echo "[E7DenseFull] data_root=${DATA_ROOT}"
echo "[E7DenseFull] parquet_dir=${PARQUET_DIR}"
echo "[E7DenseFull] base_index=${BASE_INDEX}"
echo "[E7DenseFull] training_free_weights=${TRAINING_FREE_WEIGHTS}"
echo "[E7DenseFull] query_mode=${QUERY_MODE}"
echo "[E7DenseFull] max_samples=${MAX_SAMPLES}"
echo "[E7DenseFull] run_retrieval_eval=${RUN_RETRIEVAL_EVAL}"
echo "[E7DenseFull] anchor_variants=${ANCHOR_VARIANTS}"

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
        --batch-size 256 \
        --index-type flat
fi

variant_medqa_anchor() {
    case "$1" in
        original_text) printf '%s' "${MEDQA_ANCHOR}" ;;
        k256) printf '%s' "${MEDQA_ANCHOR_K256}" ;;
        *) echo "[E7DenseFull] unknown variant: $1" >&2; exit 1 ;;
    esac
}

variant_arc_anchor() {
    case "$1" in
        original_text) printf '%s' "${ARC_ANCHOR}" ;;
        k256) printf '%s' "${ARC_ANCHOR_K256}" ;;
        *) echo "[E7DenseFull] unknown variant: $1" >&2; exit 1 ;;
    esac
}

variant_mmlu_anchor() {
    case "$1" in
        original_text) printf '%s' "${MMLU_ANCHOR}" ;;
        k256) printf '%s' "${MMLU_ANCHOR_K256}" ;;
        *) echo "[E7DenseFull] unknown variant: $1" >&2; exit 1 ;;
    esac
}

variant_medqa_index() {
    printf '%s' "${CHECKPOINT_DIR}/dense_fineweb_medqa_overlay_$1_flat_r24_qwen3.pt"
}

variant_arc_index() {
    printf '%s' "${CHECKPOINT_DIR}/dense_fineweb_arc_overlay_$1_flat_r24_qwen3.pt"
}

variant_mmlu_index() {
    printf '%s' "${CHECKPOINT_DIR}/dense_fineweb_mmlu_overlay_$1_flat_r24_qwen3.pt"
}

IFS=',' read -r -a VARIANT_LIST <<< "${ANCHOR_VARIANTS}"
for variant in "${VARIANT_LIST[@]}"; do
    variant="${variant// /}"
    [ -n "${variant}" ] || continue

    MEDQA_INDEX="$(variant_medqa_index "${variant}")"
    ARC_INDEX="$(variant_arc_index "${variant}")"
    MMLU_INDEX="$(variant_mmlu_index "${variant}")"
    MEDQA_ANCHOR_CUR="$(variant_medqa_anchor "${variant}")"
    ARC_ANCHOR_CUR="$(variant_arc_anchor "${variant}")"
    MMLU_ANCHOR_CUR="$(variant_mmlu_anchor "${variant}")"

    echo "[E7DenseFull] ===== variant=${variant} ====="
    echo "[E7DenseFull] medqa_index=${MEDQA_INDEX}"
    echo "[E7DenseFull] arc_index=${ARC_INDEX}"
    echo "[E7DenseFull] mmlu_index=${MMLU_INDEX}"

    if [ "${BUILD_OVERLAYS}" = "1" ]; then
        run_cmd \
            conda run --no-capture-output -n ExplicitLLM \
            python "${PROJECT_ROOT}/scripts/overlay_dense_index.py" \
            --config "${CONFIG}" \
            --index "${BASE_INDEX}" \
            --anchor-input "${MEDQA_ANCHOR_CUR}" \
            --fusion-input "${MEDQA_FUSION}" \
            --output "${MEDQA_INDEX}" \
            --device "${DEVICE}" \
            --batch-size 256 \
            --seed 42

        run_cmd \
            conda run --no-capture-output -n ExplicitLLM \
            python "${PROJECT_ROOT}/scripts/overlay_dense_index.py" \
            --config "${CONFIG}" \
            --index "${BASE_INDEX}" \
            --anchor-input "${ARC_ANCHOR_CUR}" \
            --fusion-input "${ARC_FUSION}" \
            --output "${ARC_INDEX}" \
            --device "${DEVICE}" \
            --batch-size 256 \
            --seed 42

        run_cmd \
            conda run --no-capture-output -n ExplicitLLM \
            python "${PROJECT_ROOT}/scripts/overlay_dense_index.py" \
            --config "${CONFIG}" \
            --index "${BASE_INDEX}" \
            --anchor-input "${MMLU_ANCHOR_CUR}" \
            --fusion-input "${MMLU_FUSION}" \
            --output "${MMLU_INDEX}" \
            --device "${DEVICE}" \
            --batch-size 256 \
            --seed 42
    fi

    if [ "${SKIP_E7}" = "1" ]; then
        continue
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
            --output "${PROJECT_ROOT}/results/e7/e7_dense_retrieval_precheck_${variant}.json"
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
        OUTPUT="${PROJECT_ROOT}/results/e7/e7_dense_${variant}_$(basename "${TRAINING_FREE_WEIGHTS}").json" \
        bash "${PROJECT_ROOT}/experiments/e7/run.sh"
done
