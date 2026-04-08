#!/usr/bin/env bash
# 一键实验总控脚本
#
# 默认顺序：E1 -> E2 -> E3 -> E3 multi-k -> E4 -> E5 -> E6 -> E7
#
# 用法：
#   ENC_MODE=qwen3 \
#   PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   bash scripts/run_experiment_suite.sh
#
#   只跑部分实验：
#   EXPERIMENTS="e1 e2 e6" bash scripts/run_experiment_suite.sh
#   EXPERIMENTS="e1 e2 e3_multik e5" bash scripts/run_experiment_suite.sh
#
#   只看命令不执行：
#   DRY_RUN=1 bash scripts/run_experiment_suite.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

EXPERIMENTS="${EXPERIMENTS:-e1 e2 e3 e3_multik e4 e5 e6 e7}"
PHASE1_WEIGHTS="${PHASE1_WEIGHTS:-checkpoints/phase1_best}"
PHASE2_WEIGHTS="${PHASE2_WEIGHTS:-checkpoints/phase2_best}"
PHASE3_WEIGHTS="${PHASE3_WEIGHTS:-checkpoints/phase3_best}"
E1_WEIGHTS="${E1_WEIGHTS:-${PHASE2_WEIGHTS}}"
DRY_RUN="${DRY_RUN:-0}"

echo "[ExperimentSuite] experiments=${EXPERIMENTS}"
echo "[ExperimentSuite] phase1_weights=${PHASE1_WEIGHTS}"
echo "[ExperimentSuite] phase2_weights=${PHASE2_WEIGHTS}"
echo "[ExperimentSuite] phase3_weights=${PHASE3_WEIGHTS}"

for exp_name in ${EXPERIMENTS}; do
    echo ""
    echo "[ExperimentSuite] >>> Running ${exp_name}"
    E1_WEIGHTS="${E1_WEIGHTS}" \
    PHASE1_WEIGHTS="${PHASE1_WEIGHTS}" \
    PHASE2_WEIGHTS="${PHASE2_WEIGHTS}" \
    PHASE3_WEIGHTS="${PHASE3_WEIGHTS}" \
    DRY_RUN="${DRY_RUN}" \
    bash "${SCRIPT_DIR}/run_experiment_auto.sh" "${exp_name}" "$@"
done
