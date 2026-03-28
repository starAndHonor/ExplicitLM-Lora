#!/usr/bin/env bash
# 单实验运行脚本
#
# 用法：
#   bash scripts/run_experiment.sh e1
#   ENC_MODE=qwen3 FUSION_CKPT=checkpoints/p2_qwen3_10ep/phase2_best \
#     bash scripts/run_experiment.sh e2
#   ENC_MODE=qwen3 PHASE1_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE2_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment.sh e3
#   DRY_RUN=1 bash scripts/run_experiment.sh e6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_experiment.sh <e1|e2|e3|e4|e5|e6> [extra args ...]"
    exit 1
fi

EXPERIMENT="$1"
shift

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-6,7}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-trainable}"
OUTPUT="${OUTPUT:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
N_WARMUP="${N_WARMUP:-10}"
N_MEASURE="${N_MEASURE:-200}"
E3_RESULT="${E3_RESULT:-}"
E5_RESULT="${E5_RESULT:-}"
BUILD_ONLY="${BUILD_ONLY:-0}"
BUILD_MISSING="${BUILD_MISSING:-0}"
REBUILD="${REBUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"

FUSION_CKPT="${FUSION_CKPT:-checkpoints/phase2_best}"
PHASE1_WEIGHTS="${PHASE1_WEIGHTS:-checkpoints/phase2_best}"
PHASE2_WEIGHTS="${PHASE2_WEIGHTS:-checkpoints/phase3_best}"

exp_load_env

case "${EXPERIMENT}" in
    e6)
        export CUDA_VISIBLE_DEVICES="$(exp_first_gpu "${GPU_IDS}")"
        if [[ "${DEVICE}" == cuda:* ]]; then
            DEVICE="cuda:0"
        fi
        ;;
    *)
        export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
        ;;
esac

declare -a CMD
CMD=(conda run --no-capture-output -n ExplicitLLM python)

declare -a OVERRIDE_ARGS=()
if [ "${ENC_MODE}" = "qwen3" ]; then
    OVERRIDE_ARGS+=(--override "model.knowledge_encoder_mode=qwen3")
fi

case "${EXPERIMENT}" in
    e1)
        CMD+=("${PROJECT_ROOT}/experiments/e1/run_e1.py" --config "${CONFIG}" --fusion-ckpt "${FUSION_CKPT}" --max-samples "${MAX_SAMPLES}")
        if [ -n "${OUTPUT}" ]; then
            CMD+=(--output "${OUTPUT}")
        fi
        ;;
    e2)
        CMD+=("${PROJECT_ROOT}/experiments/e2/run_e2.py" --config "${CONFIG}" --fusion-ckpt "${FUSION_CKPT}" --device "${DEVICE}" --max-samples "${MAX_SAMPLES}")
        if [ -n "${OUTPUT}" ]; then
            CMD+=(--output "${OUTPUT}")
        fi
        ;;
    e3)
        CMD+=("${PROJECT_ROOT}/experiments/e3/run_e3.py" --config "${CONFIG}" --phase1-weights "${PHASE1_WEIGHTS}" --phase2-weights "${PHASE2_WEIGHTS}" --device "${DEVICE}" --max-samples "${MAX_SAMPLES}")
        if [ -n "${OUTPUT}" ]; then
            CMD+=(--output "${OUTPUT}")
        fi
        ;;
    e4)
        CMD+=("${PROJECT_ROOT}/experiments/e4/run_e4.py" --config "${CONFIG}" --phase1-weights "${PHASE1_WEIGHTS}" --phase2-weights "${PHASE2_WEIGHTS}" --device "${DEVICE}" --max-samples "${MAX_SAMPLES}")
        if [ -n "${OUTPUT}" ]; then
            CMD+=(--output "${OUTPUT}")
        fi
        ;;
    e5)
        CMD+=("${PROJECT_ROOT}/experiments/e5/run_e5.py" --config "${CONFIG}")
        if [ "${BUILD_ONLY}" = "1" ]; then
            CMD+=(--build-only --max-samples "${MAX_SAMPLES}")
            if [ "${REBUILD}" = "1" ]; then
                CMD+=(--rebuild)
            fi
        else
            CMD+=(--phase1-weights "${PHASE1_WEIGHTS}" --phase2-weights "${PHASE2_WEIGHTS}" --device "${DEVICE}" --max-samples "${MAX_SAMPLES}")
            if [ "${BUILD_MISSING}" = "1" ]; then
                CMD+=(--build-missing)
            fi
            if [ -n "${OUTPUT}" ]; then
                CMD+=(--output "${OUTPUT}")
            fi
        fi
        ;;
    e6)
        CMD+=("${PROJECT_ROOT}/experiments/e6/run_e6.py" --config "${CONFIG}" --phase2-weights "${PHASE2_WEIGHTS}" --device "${DEVICE}" --n-warmup "${N_WARMUP}" --n-measure "${N_MEASURE}")
        if [ -n "${OUTPUT}" ]; then
            CMD+=(--output "${OUTPUT}")
        fi
        if [ -n "${E3_RESULT}" ]; then
            CMD+=(--e3-result "${E3_RESULT}")
        fi
        if [ -n "${E5_RESULT}" ]; then
            CMD+=(--e5-result "${E5_RESULT}")
        fi
        ;;
    *)
        echo "[Experiment] Unknown experiment: ${EXPERIMENT}"
        exit 1
        ;;
esac

CMD+=("${OVERRIDE_ARGS[@]}")
CMD+=("$@")

echo "[Experiment] name=${EXPERIMENT}"
echo "[Experiment] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Experiment] config=${CONFIG}"
echo "[Experiment] device=${DEVICE}"
echo "[Experiment] enc_mode=${ENC_MODE}"
if [ -n "${OUTPUT}" ]; then
    echo "[Experiment] output=${OUTPUT}"
fi
exp_print_cmd "${CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

"${CMD[@]}"
