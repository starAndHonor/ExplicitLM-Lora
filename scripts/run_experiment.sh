#!/usr/bin/env bash
# 单实验运行脚本
#
# 用法：
#   ENC_MODE=qwen3 PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment.sh e1
#   ENC_MODE=qwen3 PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment.sh e2
#   ENC_MODE=qwen3 PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment.sh e3_multik
#   ENC_MODE=qwen3 PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment.sh e3
#   DRY_RUN=1 bash scripts/run_experiment.sh e6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_experiment.sh <e1|e2|e3|e3_multik|e4|e5|e6> [extra args ...]"
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
E1_PHASE2_OUTPUT="${E1_PHASE2_OUTPUT:-}"
E1_PHASE3_OUTPUT="${E1_PHASE3_OUTPUT:-}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
N_WARMUP="${N_WARMUP:-10}"
N_MEASURE="${N_MEASURE:-200}"
E3_RESULT="${E3_RESULT:-}"
E5_RESULT="${E5_RESULT:-}"
BUILD_ONLY="${BUILD_ONLY:-0}"
BUILD_MISSING="${BUILD_MISSING:-0}"
REBUILD="${REBUILD:-0}"
DRY_RUN="${DRY_RUN:-0}"

PHASE1_WEIGHTS="${PHASE1_WEIGHTS:-checkpoints/phase1_best}"
PHASE2_WEIGHTS="${PHASE2_WEIGHTS:-checkpoints/phase2_best}"
PHASE3_WEIGHTS="${PHASE3_WEIGHTS:-checkpoints/phase3_best}"
E1_WEIGHTS="${E1_WEIGHTS:-${PHASE2_WEIGHTS}}"

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

if [ "${EXPERIMENT}" = "e3_multik" ] || [ "${EXPERIMENT}" = "e3k" ]; then
    echo "[Experiment] name=${EXPERIMENT}"
    echo "[Experiment] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[Experiment] config=${CONFIG}"
    echo "[Experiment] device=${DEVICE}"
    echo "[Experiment] enc_mode=${ENC_MODE}"
    CONFIG="${CONFIG}" \
    NUM_GPUS="${NUM_GPUS}" \
    GPU_IDS="${GPU_IDS}" \
    DEVICE="${DEVICE}" \
    ENC_MODE="${ENC_MODE}" \
    MAX_SAMPLES="${MAX_SAMPLES}" \
    PHASE2_WEIGHTS="${PHASE2_WEIGHTS}" \
    PHASE3_WEIGHTS="${PHASE3_WEIGHTS}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${SCRIPT_DIR}/run_e3_multik.sh"
    exit 0
fi

if [ "${EXPERIMENT}" = "e1" ]; then
    declare -a E1_OVERRIDE_ARGS=()
    if [ "${ENC_MODE}" = "qwen3" ]; then
        E1_OVERRIDE_ARGS+=(--override "model.knowledge_encoder_mode=qwen3")
    fi

    E1_PHASE2_OUTPUT="${E1_PHASE2_OUTPUT:-${OUTPUT}}"
    E1_PHASE3_OUTPUT="${E1_PHASE3_OUTPUT:-}"

    if [ -z "${E1_PHASE2_OUTPUT}" ]; then
        E1_PHASE2_OUTPUT="results/e1/e1_sanity_check_$(exp_ckpt_tag "${PHASE2_WEIGHTS}").json"
    fi
    if [ -z "${E1_PHASE3_OUTPUT}" ]; then
        E1_PHASE3_OUTPUT="results/e1/e1_sanity_check_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
    fi

    echo "[Experiment] name=${EXPERIMENT}"
    echo "[Experiment] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[Experiment] config=${CONFIG}"
    echo "[Experiment] device=${DEVICE}"
    echo "[Experiment] enc_mode=${ENC_MODE}"
    echo "[Experiment] phase2_weights=${PHASE2_WEIGHTS}"
    echo "[Experiment] phase3_weights=${PHASE3_WEIGHTS}"

    for e1_weight in "${PHASE2_WEIGHTS}" "${PHASE3_WEIGHTS}"; do
        if [ "${e1_weight}" = "${PHASE2_WEIGHTS}" ]; then
            current_output="${E1_PHASE2_OUTPUT}"
        else
            current_output="${E1_PHASE3_OUTPUT}"
        fi
        declare -a E1_CMD=(
            conda run --no-capture-output -n ExplicitLLM
            python "${PROJECT_ROOT}/experiments/e1/run_e1.py"
            --config "${CONFIG}"
            --weights "${e1_weight}"
            --max-samples "${MAX_SAMPLES}"
            --output "${current_output}"
        )
        E1_CMD+=("${E1_OVERRIDE_ARGS[@]}")
        E1_CMD+=("$@")
        echo "[Experiment] e1_weights=${e1_weight}"
        echo "[Experiment] output=${current_output}"
        exp_print_cmd "${E1_CMD[@]}"
        if [ "${DRY_RUN}" = "1" ]; then
            continue
        fi
        "${E1_CMD[@]}"
    done
    exit 0
fi

declare -a CMD
CMD=(conda run --no-capture-output -n ExplicitLLM python)

declare -a OVERRIDE_ARGS=()
if [ "${ENC_MODE}" = "qwen3" ]; then
    OVERRIDE_ARGS+=(--override "model.knowledge_encoder_mode=qwen3")
fi

case "${EXPERIMENT}" in
    e2)
        CMD+=("${PROJECT_ROOT}/experiments/e2/run_e2.py" --config "${CONFIG}" --phase2-weights "${PHASE2_WEIGHTS}" --phase3-weights "${PHASE3_WEIGHTS}" --device "${DEVICE}" --max-samples "${MAX_SAMPLES}")
        if [ -n "${OUTPUT}" ]; then
            CMD+=(--output "${OUTPUT}")
        fi
        ;;
    e3)
        CMD+=("${PROJECT_ROOT}/experiments/e3/run_e3.py" --config "${CONFIG}" --phase2-weights "${PHASE2_WEIGHTS}" --phase3-weights "${PHASE3_WEIGHTS}" --device "${DEVICE}" --max-samples "${MAX_SAMPLES}")
        if [ -n "${OUTPUT}" ]; then
            CMD+=(--output "${OUTPUT}")
        fi
        ;;
    e4)
        CMD+=("${PROJECT_ROOT}/experiments/e4/run_e4.py" --config "${CONFIG}" --phase2-weights "${PHASE2_WEIGHTS}" --phase3-weights "${PHASE3_WEIGHTS}" --device "${DEVICE}" --max-samples "${MAX_SAMPLES}")
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
            CMD+=(--phase2-weights "${PHASE2_WEIGHTS}" --phase3-weights "${PHASE3_WEIGHTS}" --device "${DEVICE}" --max-samples "${MAX_SAMPLES}")
            if [ "${BUILD_MISSING}" = "1" ]; then
                CMD+=(--build-missing)
            fi
            if [ -n "${OUTPUT}" ]; then
                CMD+=(--output "${OUTPUT}")
            fi
        fi
        ;;
    e6)
        CMD+=("${PROJECT_ROOT}/experiments/e6/run_e6.py" --config "${CONFIG}" --phase3-weights "${PHASE3_WEIGHTS}" --device "${DEVICE}" --n-warmup "${N_WARMUP}" --n-measure "${N_MEASURE}")
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
echo "[Experiment] phase1_weights=${PHASE1_WEIGHTS}"
echo "[Experiment] phase2_weights=${PHASE2_WEIGHTS}"
echo "[Experiment] phase3_weights=${PHASE3_WEIGHTS}"
if [ -n "${OUTPUT}" ]; then
    echo "[Experiment] output=${OUTPUT}"
fi
exp_print_cmd "${CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

"${CMD[@]}"
