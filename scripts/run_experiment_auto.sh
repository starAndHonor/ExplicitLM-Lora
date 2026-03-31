#!/usr/bin/env bash
# 通用评测脚本：按 checkpoint / 模式自动生成结果文件名
#
# 用法：
#   ENC_MODE=qwen3 FUSION_CKPT=checkpoints/p2_qwen3_10ep/phase2_best \
#     bash scripts/run_experiment_auto.sh e2
#   ENC_MODE=qwen3 PHASE1_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE2_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment_auto.sh e3
#   ENC_MODE=qwen3 PHASE1_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#     PHASE2_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#     bash scripts/run_experiment_auto.sh e3_multik
#   DRY_RUN=1 bash scripts/run_experiment_auto.sh e6

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_experiment_auto.sh <e1|e2|e3|e3_multik|e4|e5|e6> [extra args ...]"
    exit 1
fi

EXPERIMENT="$1"
shift

FUSION_CKPT="${FUSION_CKPT:-checkpoints/phase2_best}"
PHASE1_WEIGHTS="${PHASE1_WEIGHTS:-checkpoints/phase2_best}"
PHASE2_WEIGHTS="${PHASE2_WEIGHTS:-checkpoints/phase3_best}"
E2_PHASE3_CKPT="${E2_PHASE3_CKPT:-${PHASE2_WEIGHTS}}"
OUTPUT="${OUTPUT:-}"
E3_RESULT="${E3_RESULT:-}"
E5_RESULT="${E5_RESULT:-}"
E5_RESULT_AUTO="${E5_RESULT_AUTO:-1}"

if [ -z "${OUTPUT}" ]; then
    case "${EXPERIMENT}" in
        e1)
            OUTPUT="results/e1/e1_sanity_check_$(exp_ckpt_tag "${FUSION_CKPT}").json"
            ;;
        e2)
            OUTPUT="results/e2/e2_cross_domain_$(exp_ckpt_tag "${FUSION_CKPT}")__$(exp_ckpt_tag "${E2_PHASE3_CKPT}").json"
            ;;
        e3)
            OUTPUT="results/e3/e3_fair_compare_$(exp_ckpt_tag "${PHASE1_WEIGHTS}")__$(exp_ckpt_tag "${PHASE2_WEIGHTS}").json"
            ;;
        e3_multik|e3k)
            OUTPUT=""
            ;;
        e4)
            OUTPUT="results/e4/e4_sft_ablation_$(exp_ckpt_tag "${PHASE1_WEIGHTS}")__$(exp_ckpt_tag "${PHASE2_WEIGHTS}").json"
            ;;
        e5)
            OUTPUT="results/e5/e5_knowledge_analysis_$(exp_ckpt_tag "${PHASE1_WEIGHTS}")__$(exp_ckpt_tag "${PHASE2_WEIGHTS}").json"
            ;;
        e6)
            OUTPUT="results/e6/e6_inference_efficiency_$(exp_ckpt_tag "${PHASE2_WEIGHTS}").json"
            ;;
        *)
            echo "[ExperimentAuto] Unknown experiment: ${EXPERIMENT}"
            exit 1
            ;;
    esac
fi

if [ "${EXPERIMENT}" = "e6" ] && [ -z "${E5_RESULT}" ] && [ "${E5_RESULT_AUTO}" = "1" ]; then
    E5_RESULT="results/e5/e5_knowledge_analysis_$(exp_ckpt_tag "${PHASE1_WEIGHTS}")__$(exp_ckpt_tag "${PHASE2_WEIGHTS}").json"
fi

if [ "${EXPERIMENT}" = "e6" ] && [ -z "${E3_RESULT}" ]; then
    E3_RESULT="results/e3/e3_fair_compare_$(exp_ckpt_tag "${PHASE1_WEIGHTS}")__$(exp_ckpt_tag "${PHASE2_WEIGHTS}").json"
fi

echo "[ExperimentAuto] experiment=${EXPERIMENT}"
if [ -n "${OUTPUT}" ]; then
    echo "[ExperimentAuto] output=${OUTPUT}"
fi
if [ -n "${E3_RESULT}" ] && [ "${EXPERIMENT}" = "e6" ]; then
    echo "[ExperimentAuto] e3_result=${E3_RESULT}"
fi
if [ -n "${E5_RESULT}" ] && [ "${EXPERIMENT}" = "e6" ]; then
    echo "[ExperimentAuto] e5_result=${E5_RESULT}"
fi

OUTPUT="${OUTPUT}" \
E3_RESULT="${E3_RESULT}" \
E5_RESULT="${E5_RESULT}" \
FUSION_CKPT="${FUSION_CKPT}" \
E2_PHASE3_CKPT="${E2_PHASE3_CKPT}" \
PHASE1_WEIGHTS="${PHASE1_WEIGHTS}" \
PHASE2_WEIGHTS="${PHASE2_WEIGHTS}" \
bash "${SCRIPT_DIR}/run_experiment.sh" "${EXPERIMENT}" "$@"
