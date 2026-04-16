#!/usr/bin/env bash
# 通用评测脚本：按 checkpoint / 模式自动生成结果文件名
#
# 用法：
#   ENC_MODE=qwen3 PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment_auto.sh e2
#   ENC_MODE=qwen3 PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment_auto.sh e3
#   ENC_MODE=qwen3 PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment_auto.sh e3_multik
#   DRY_RUN=1 bash scripts/run_experiment_auto.sh e6
#   DENSE_INDEX_MEDQA=checkpoints/dense_fineweb_medqa_overlay_k64_flat_r0_qwen3.pt \
#     DENSE_INDEX_ARC=checkpoints/dense_fineweb_arc_overlay_k64_flat_r0_qwen3.pt \
#     DENSE_INDEX_MMLU=checkpoints/dense_fineweb_mmlu_overlay_k64_flat_r0_qwen3.pt \
#     TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment_auto.sh e7
#   TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment_auto.sh e7_full

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_experiment_auto.sh <e1|e2|e3|e3_multik|e4|e5|e6|e7|e7_full> [extra args ...]"
    exit 1
fi

EXPERIMENT="$1"
shift

PHASE1_WEIGHTS="${PHASE1_WEIGHTS:-checkpoints/phase1_best}"
PHASE2_WEIGHTS="${PHASE2_WEIGHTS:-checkpoints/phase2_best}"
PHASE3_WEIGHTS="${PHASE3_WEIGHTS:-checkpoints/phase3_best}"
TRAINING_FREE_WEIGHTS="${TRAINING_FREE_WEIGHTS:-${PHASE3_WEIGHTS}}"
E1_WEIGHTS="${E1_WEIGHTS:-${PHASE2_WEIGHTS}}"
E1_PHASE2_OUTPUT="${E1_PHASE2_OUTPUT:-}"
E1_PHASE3_OUTPUT="${E1_PHASE3_OUTPUT:-}"
OUTPUT="${OUTPUT:-}"
E3_RESULT="${E3_RESULT:-}"
E5_RESULT="${E5_RESULT:-}"
E5_RESULT_AUTO="${E5_RESULT_AUTO:-1}"

if [ -z "${OUTPUT}" ]; then
    case "${EXPERIMENT}" in
        e1)
            E1_PHASE2_OUTPUT="${E1_PHASE2_OUTPUT:-results/e1/e1_sanity_check_$(exp_ckpt_tag "${PHASE2_WEIGHTS}").json}"
            E1_PHASE3_OUTPUT="${E1_PHASE3_OUTPUT:-results/e1/e1_sanity_check_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json}"
            OUTPUT=""
            ;;
        e2)
            OUTPUT="results/e2/e2_cross_domain_$(exp_ckpt_tag "${PHASE2_WEIGHTS}")__$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e3)
            OUTPUT="results/e3/e3_fair_compare_$(exp_ckpt_tag "${PHASE2_WEIGHTS}")__$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e3_multik|e3k)
            OUTPUT=""
            ;;
        e4)
            OUTPUT="results/e4/e4_sft_ablation_$(exp_ckpt_tag "${PHASE2_WEIGHTS}")__$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e5)
            OUTPUT="results/e5/e5_knowledge_analysis_$(exp_ckpt_tag "${PHASE2_WEIGHTS}")__$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e6)
            OUTPUT="results/e6/e6_inference_efficiency_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
            ;;
        e7)
            OUTPUT="results/e7/e7_dense_$(exp_ckpt_tag "${TRAINING_FREE_WEIGHTS}").json"
            ;;
        e7_full)
            OUTPUT=""
            ;;
        *)
            echo "[ExperimentAuto] Unknown experiment: ${EXPERIMENT}"
            exit 1
            ;;
    esac
fi

if [ "${EXPERIMENT}" = "e6" ] && [ -z "${E5_RESULT}" ] && [ "${E5_RESULT_AUTO}" = "1" ]; then
    E5_RESULT="results/e5/e5_knowledge_analysis_$(exp_ckpt_tag "${PHASE2_WEIGHTS}")__$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
fi

if [ "${EXPERIMENT}" = "e6" ] && [ -z "${E3_RESULT}" ]; then
    E3_RESULT="results/e3/e3_fair_compare_$(exp_ckpt_tag "${PHASE2_WEIGHTS}")__$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
fi

echo "[ExperimentAuto] experiment=${EXPERIMENT}"
if [ "${EXPERIMENT}" = "e1" ]; then
    echo "[ExperimentAuto] e1_phase2_output=${E1_PHASE2_OUTPUT}"
    echo "[ExperimentAuto] e1_phase3_output=${E1_PHASE3_OUTPUT}"
elif [ -n "${OUTPUT}" ]; then
    echo "[ExperimentAuto] output=${OUTPUT}"
fi
if [ -n "${E3_RESULT}" ] && [ "${EXPERIMENT}" = "e6" ]; then
    echo "[ExperimentAuto] e3_result=${E3_RESULT}"
fi
if [ -n "${E5_RESULT}" ] && [ "${EXPERIMENT}" = "e6" ]; then
    echo "[ExperimentAuto] e5_result=${E5_RESULT}"
fi

OUTPUT="${OUTPUT}" \
E1_WEIGHTS="${E1_WEIGHTS}" \
E1_PHASE2_OUTPUT="${E1_PHASE2_OUTPUT}" \
E1_PHASE3_OUTPUT="${E1_PHASE3_OUTPUT}" \
E3_RESULT="${E3_RESULT}" \
E5_RESULT="${E5_RESULT}" \
PHASE1_WEIGHTS="${PHASE1_WEIGHTS}" \
PHASE2_WEIGHTS="${PHASE2_WEIGHTS}" \
PHASE3_WEIGHTS="${PHASE3_WEIGHTS}" \
TRAINING_FREE_WEIGHTS="${TRAINING_FREE_WEIGHTS}" \
bash "${SCRIPT_DIR}/run_experiment.sh" "${EXPERIMENT}" "$@"
