#!/usr/bin/env bash
# 运行 E8 Editable Memory Benchmark
#
# 用法：
#   FULL_INDEX=checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   bash scripts/run_e8.sh e8a
#
#   FULL_INDEX=checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   GPU_IDS=2 \
#   N_EDITS=100 \
#   bash scripts/run_e8.sh e8b
#
#   BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r24_qwen3.pt \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   GPU_IDS=2 \
#   MEMORY_SETTING=overlay_1m \
#   ANCHOR_VARIANT=original_text \
#   STEPS=1,2,3,10,11,12,100,101,102 \
#   LOCALITY_SAMPLES=200 \
#   bash scripts/run_e8.sh e8c
#
#   BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r24_qwen3.pt \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   GPU_IDS=2 \
#   MEMORY_SETTING=overlay_1m \
#   ANCHOR_VARIANT=k256 \
#   N_EDITS=100 \
#   LOCALITY_SAMPLES=200 \
#   bash scripts/run_e8.sh e8d_a
#
#   BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r24_qwen3.pt \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   GPU_IDS=2 \
#   MEMORY_SETTING=overlay_1m \
#   ANCHOR_VARIANT=k256 \
#   N_EDITS=100 \
#   UPDATE_BATCH_SIZE=10 \
#   LOCALITY_SAMPLES=200 \
#   bash scripts/run_e8.sh e8d_b

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_e8.sh <e8a|e8b|e8c|e8d_a|e8d_b> [extra args ...]"
    exit 1
fi

EXPERIMENT="$1"
shift

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
GPU_IDS="${GPU_IDS:-6}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
QUERY_MODE="${QUERY_MODE:-question_only}"
N_EDITS="${N_EDITS:-100}"
SEED="${SEED:-0}"
STEPS="${STEPS:-1,2,3,10,11,12,100,101,102}"
LOCALITY_SAMPLES="${LOCALITY_SAMPLES:-200}"
UPDATE_BATCH_SIZE="${UPDATE_BATCH_SIZE:-10}"
MEMORY_SETTING="${MEMORY_SETTING:-controlled}"
ANCHOR_VARIANT="${ANCHOR_VARIANT:-original_text}"
OVERLAY_SEED="${OVERLAY_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"

FULL_INDEX="${FULL_INDEX:-}"
BASE_INDEX="${BASE_INDEX:-}"
PHASE3_WEIGHTS="${PHASE3_WEIGHTS:-checkpoints/p3_from_p2_qwen3_10ep/phase3_best}"

case "${MEMORY_SETTING}" in
    controlled)
        if [ -z "${FULL_INDEX}" ]; then
            echo "[E8] ERROR: FULL_INDEX 必须设置（MEMORY_SETTING=controlled）"
            exit 1
        fi
        ;;
    overlay_1m)
        if [ -z "${BASE_INDEX}" ]; then
            echo "[E8] ERROR: BASE_INDEX 必须设置（MEMORY_SETTING=overlay_1m）"
            exit 1
        fi
        ;;
    *)
        echo "[E8] ERROR: unsupported MEMORY_SETTING='${MEMORY_SETTING}', expected controlled|overlay_1m"
        exit 1
        ;;
esac

if [ -z "${OUTPUT:-}" ]; then
    RESULT_TAG="${MEMORY_SETTING}"
    if [ "${MEMORY_SETTING}" = "overlay_1m" ]; then
        RESULT_TAG="${RESULT_TAG}_${ANCHOR_VARIANT}"
    fi
    case "${EXPERIMENT}" in
        e8a)
            OUTPUT="${PROJECT_ROOT}/results/e8/e8a_upsert_${RESULT_TAG}_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e8b)
            OUTPUT="${PROJECT_ROOT}/results/e8/e8b_delete_rollback_${RESULT_TAG}_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e8c)
            OUTPUT="${PROJECT_ROOT}/results/e8/e8c_sequential_${RESULT_TAG}_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e8d_a)
            OUTPUT="${PROJECT_ROOT}/results/e8/e8d_a_batch_ingest_${RESULT_TAG}_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e8d_b)
            OUTPUT="${PROJECT_ROOT}/results/e8/e8d_b_incremental_${RESULT_TAG}_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
    esac
fi

case "${EXPERIMENT}" in
    e8a|e8b|e8c|e8d_a|e8d_b)
        ;;
    *)
        echo "[E8] ERROR: unsupported experiment '${EXPERIMENT}', expected e8a|e8b|e8c|e8d_a|e8d_b"
        exit 1
        ;;
esac

exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

declare -a CMD=(
    conda run --no-capture-output -n ExplicitLLM
    python "${PROJECT_ROOT}/experiments/e8/run_e8.py"
    --config "${CONFIG}"
    --experiment "${EXPERIMENT}"
    --memory-setting "${MEMORY_SETTING}"
    --phase3-weights "${PHASE3_WEIGHTS}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --query-mode "${QUERY_MODE}"
    --output "${OUTPUT}"
)

if [ -n "${FULL_INDEX}" ]; then
    CMD+=(--full-index "${FULL_INDEX}")
fi
if [ -n "${BASE_INDEX}" ]; then
    CMD+=(--base-index "${BASE_INDEX}")
fi
CMD+=(--anchor-variant "${ANCHOR_VARIANT}" --overlay-seed "${OVERLAY_SEED}")

case "${EXPERIMENT}" in
    e8a|e8b)
        CMD+=(--n-edits "${N_EDITS}")
        ;;
    e8c)
        CMD+=(--steps "${STEPS}" --locality-samples "${LOCALITY_SAMPLES}")
        ;;
    e8d_a)
        CMD+=(--n-edits "${N_EDITS}" --locality-samples "${LOCALITY_SAMPLES}")
        ;;
    e8d_b)
        CMD+=(--n-edits "${N_EDITS}" --locality-samples "${LOCALITY_SAMPLES}" --update-batch-size "${UPDATE_BATCH_SIZE}")
        ;;
esac

if [ "${ENC_MODE}" = "qwen3" ]; then
    CMD+=(--override "model.knowledge_encoder_mode=qwen3")
fi

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

echo "[E8] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[E8] config=${CONFIG}"
echo "[E8] experiment=${EXPERIMENT}"
echo "[E8] device=${DEVICE}"
echo "[E8] enc_mode=${ENC_MODE}"
echo "[E8] memory_setting=${MEMORY_SETTING}"
echo "[E8] full_index=${FULL_INDEX:-<auto>}"
echo "[E8] base_index=${BASE_INDEX:-<none>}"
echo "[E8] anchor_variant=${ANCHOR_VARIANT}"
echo "[E8] overlay_seed=${OVERLAY_SEED}"
echo "[E8] phase3_weights=${PHASE3_WEIGHTS}"
echo "[E8] query_mode=${QUERY_MODE}"
if [ "${EXPERIMENT}" = "e8a" ] || [ "${EXPERIMENT}" = "e8b" ] || [ "${EXPERIMENT}" = "e8d_a" ] || [ "${EXPERIMENT}" = "e8d_b" ]; then
    echo "[E8] n_edits=${N_EDITS}"
fi
if [ "${EXPERIMENT}" = "e8c" ]; then
    echo "[E8] steps=${STEPS}"
    echo "[E8] locality_samples=${LOCALITY_SAMPLES}"
fi
if [ "${EXPERIMENT}" = "e8d_a" ] || [ "${EXPERIMENT}" = "e8d_b" ]; then
    echo "[E8] locality_samples=${LOCALITY_SAMPLES}"
fi
if [ "${EXPERIMENT}" = "e8d_b" ]; then
    echo "[E8] update_batch_size=${UPDATE_BATCH_SIZE}"
fi
echo "[E8] output=${OUTPUT}"
exp_print_cmd "${CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

"${CMD[@]}"
