#!/usr/bin/env bash
# 一键运行 E8 全套 Editable Memory Benchmark
#
# 默认顺序：
#   e8a -> e8b -> e8c -> e8d_a -> e8d_b
#
# 示例：
#   BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r0_qwen3.pt \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   GPU_IDS=2 \
#   MEMORY_SETTING=overlay_1m \
#   ANCHOR_VARIANT=k256 \
#   bash scripts/run_e8_full.sh
#
#   DRY_RUN=1 \
#   BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r0_qwen3.pt \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   MEMORY_SETTING=overlay_1m \
#   ANCHOR_VARIANT=k256 \
#   bash scripts/run_e8_full.sh --override model.retrieval_encoder_depth=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

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
EXPERIMENTS="${EXPERIMENTS:-e8a,e8b,e8c,e8d_a,e8d_b}"

declare -a EXTRA_ARGS=("$@")

exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

IFS=',' read -r -a EXP_LIST <<< "${EXPERIMENTS}"

echo "[E8Full] config=${CONFIG}"
echo "[E8Full] device=${DEVICE}"
echo "[E8Full] enc_mode=${ENC_MODE}"
echo "[E8Full] memory_setting=${MEMORY_SETTING}"
echo "[E8Full] full_index=${FULL_INDEX:-<none>}"
echo "[E8Full] base_index=${BASE_INDEX:-<none>}"
echo "[E8Full] anchor_variant=${ANCHOR_VARIANT}"
echo "[E8Full] overlay_seed=${OVERLAY_SEED}"
echo "[E8Full] phase3_weights=${PHASE3_WEIGHTS}"
echo "[E8Full] query_mode=${QUERY_MODE}"
echo "[E8Full] n_edits=${N_EDITS}"
echo "[E8Full] steps=${STEPS}"
echo "[E8Full] locality_samples=${LOCALITY_SAMPLES}"
echo "[E8Full] update_batch_size=${UPDATE_BATCH_SIZE}"
echo "[E8Full] experiments=${EXPERIMENTS}"

run_one() {
    local exp_name="$1"
    local -a cmd=(
        env
        CONFIG="${CONFIG}"
        GPU_IDS="${GPU_IDS}"
        DEVICE="${DEVICE}"
        ENC_MODE="${ENC_MODE}"
        QUERY_MODE="${QUERY_MODE}"
        N_EDITS="${N_EDITS}"
        SEED="${SEED}"
        STEPS="${STEPS}"
        LOCALITY_SAMPLES="${LOCALITY_SAMPLES}"
        UPDATE_BATCH_SIZE="${UPDATE_BATCH_SIZE}"
        MEMORY_SETTING="${MEMORY_SETTING}"
        ANCHOR_VARIANT="${ANCHOR_VARIANT}"
        OVERLAY_SEED="${OVERLAY_SEED}"
        DRY_RUN="${DRY_RUN}"
        PHASE3_WEIGHTS="${PHASE3_WEIGHTS}"
    )
    if [ -n "${FULL_INDEX}" ]; then
        cmd+=(FULL_INDEX="${FULL_INDEX}")
    fi
    if [ -n "${BASE_INDEX}" ]; then
        cmd+=(BASE_INDEX="${BASE_INDEX}")
    fi
    cmd+=(bash "${PROJECT_ROOT}/scripts/run_e8.sh" "${exp_name}")
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi
    exp_print_cmd "${cmd[@]}"
    if [ "${DRY_RUN}" = "1" ]; then
        return 0
    fi
    "${cmd[@]}"
}

for exp_name in "${EXP_LIST[@]}"; do
    exp_name="${exp_name// /}"
    [ -n "${exp_name}" ] || continue
    echo "[E8Full] ===== experiment=${exp_name} ====="
    case "${exp_name}" in
        e8a|e8b|e8c|e8d_a|e8d_b)
            run_one "${exp_name}"
            ;;
        *)
            echo "[E8Full] ERROR: unsupported experiment '${exp_name}', expected e8a|e8b|e8c|e8d_a|e8d_b"
            exit 1
            ;;
    esac
done
