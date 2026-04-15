#!/usr/bin/env bash
# 全实验自动运行脚本：E1 → E2 → E3 → E3_multik → E4 → E5 → E6 → E7 → E8
#
# 用法（最简）：
#   bash scripts/run_all_experiments.sh
#
# 只跑部分实验：
#   EXPERIMENTS="e1 e2 e3" bash scripts/run_all_experiments.sh
#
# 只看命令不执行：
#   DRY_RUN=1 bash scripts/run_all_experiments.sh
#
# E8 子实验选择（默认全跑）：
#   E8_EXPERIMENTS="e8a e8b" bash scripts/run_all_experiments.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

# ── 全局参数 ─────────────────────────────────────────────────────────────────
EXPERIMENTS="${EXPERIMENTS:-e1 e2 e3 e3_multik e4 e5 e6 e7 e8}"
E8_EXPERIMENTS="${E8_EXPERIMENTS:-e8a e8b e8c e8d_a e8d_b}"
GPU_IDS="${GPU_IDS:-6}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
DRY_RUN="${DRY_RUN:-0}"
QUERY_MODE="${QUERY_MODE:-question_only}"

# ── Checkpoint 路径 ──────────────────────────────────────────────────────────
PHASE2_WEIGHTS="${PHASE2_WEIGHTS:-checkpoints/p2_qwen3_10ep/phase2_best}"
PHASE3_WEIGHTS="${PHASE3_WEIGHTS:-checkpoints/p3_from_p2_qwen3_10ep/phase3_best}"
PHASE1_WEIGHTS="${PHASE1_WEIGHTS:-checkpoints/phase1_best}"

# ── E7 Dense Index（单视图，depth=0，fusion_length=64）──────────────────────
# 优先用新单视图 index（_fv 后缀），不存在时 fallback 旧版
_pick_index() {
    local new_path="$1"
    local old_path="$2"
    if [ -f "${PROJECT_ROOT}/${new_path}" ]; then
        echo "${new_path}"
    else
        echo "${old_path}"
    fi
}

DENSE_INDEX_MEDQA="${DENSE_INDEX_MEDQA:-$(_pick_index \
    "checkpoints/dense_fineweb_medqa_overlay_original_text_flat_fv_qwen3.pt" \
    "checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt")}"
DENSE_INDEX_ARC="${DENSE_INDEX_ARC:-$(_pick_index \
    "checkpoints/dense_fineweb_arc_overlay_original_text_flat_fv_qwen3.pt" \
    "checkpoints/dense_fineweb_arc_overlay_original_text_flat_r24_qwen3.pt")}"
DENSE_INDEX_MMLU="${DENSE_INDEX_MMLU:-$(_pick_index \
    "checkpoints/dense_fineweb_mmlu_overlay_original_text_flat_fv_qwen3.pt" \
    "checkpoints/dense_fineweb_mmlu_overlay_original_text_flat_r24_qwen3.pt")}"
TRAINING_FREE_WEIGHTS="${TRAINING_FREE_WEIGHTS:-${PHASE3_WEIGHTS}}"

# ── E8 Index（单视图）────────────────────────────────────────────────────────
E8_FULL_INDEX="${E8_FULL_INDEX:-$(_pick_index \
    "checkpoints/dense_fineweb_medqa_overlay_original_text_flat_fv_qwen3.pt" \
    "checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt")}"
E8_BASE_INDEX="${E8_BASE_INDEX:-$(_pick_index \
    "checkpoints/dense_fineweb_1m_flat_r0_qwen3_fv.pt" \
    "checkpoints/dense_fineweb_1m_flat_r0_qwen3.pt")}"
E8_MEMORY_SETTING="${E8_MEMORY_SETTING:-controlled}"

# ── 打印配置 ─────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════"
echo "  全实验运行：${EXPERIMENTS}"
echo "══════════════════════════════════════════════════════"
echo "  GPU_IDS         = ${GPU_IDS}"
echo "  ENC_MODE        = ${ENC_MODE}"
echo "  PHASE2_WEIGHTS  = ${PHASE2_WEIGHTS}"
echo "  PHASE3_WEIGHTS  = ${PHASE3_WEIGHTS}"
echo "  E7 MedQA index  = ${DENSE_INDEX_MEDQA}"
echo "  E7 ARC index    = ${DENSE_INDEX_ARC}"
echo "  E7 MMLU index   = ${DENSE_INDEX_MMLU}"
echo "  E8 full index   = ${E8_FULL_INDEX}"
echo "  E8 experiments  = ${E8_EXPERIMENTS}"
echo "══════════════════════════════════════════════════════"
echo ""

exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# ── 辅助：运行单个实验 ────────────────────────────────────────────────────────
run_exp() {
    local name="$1"
    echo ""
    echo "────────────────────────────────────────────────────"
    echo "  >>> ${name}"
    echo "────────────────────────────────────────────────────"

    case "${name}" in
        e1|e2|e3|e3_multik|e4|e5|e6)
            ENC_MODE="${ENC_MODE}" \
            GPU_IDS="${GPU_IDS}" \
            DEVICE="${DEVICE}" \
            PHASE1_WEIGHTS="${PHASE1_WEIGHTS}" \
            PHASE2_WEIGHTS="${PHASE2_WEIGHTS}" \
            PHASE3_WEIGHTS="${PHASE3_WEIGHTS}" \
            MAX_SAMPLES="${MAX_SAMPLES}" \
            DRY_RUN="${DRY_RUN}" \
            bash "${SCRIPT_DIR}/run_experiment_auto.sh" "${name}"
            ;;

        e7)
            ENC_MODE="${ENC_MODE}" \
            GPU_IDS="${GPU_IDS}" \
            DEVICE="${DEVICE}" \
            DENSE_INDEX_MEDQA="${DENSE_INDEX_MEDQA}" \
            DENSE_INDEX_ARC="${DENSE_INDEX_ARC}" \
            DENSE_INDEX_MMLU="${DENSE_INDEX_MMLU}" \
            TRAINING_FREE_WEIGHTS="${TRAINING_FREE_WEIGHTS}" \
            QUERY_MODE="${QUERY_MODE}" \
            MAX_SAMPLES="${MAX_SAMPLES}" \
            DRY_RUN="${DRY_RUN}" \
            bash "${SCRIPT_DIR}/run_experiment_auto.sh" e7
            ;;

        e8)
            for e8_sub in ${E8_EXPERIMENTS}; do
                echo ""
                echo "  [E8] >>> ${e8_sub}"
                case "${E8_MEMORY_SETTING}" in
                    controlled)
                        FULL_INDEX="${E8_FULL_INDEX}" \
                        PHASE3_WEIGHTS="${PHASE3_WEIGHTS}" \
                        GPU_IDS="${GPU_IDS}" \
                        DEVICE="${DEVICE}" \
                        MEMORY_SETTING="controlled" \
                        QUERY_MODE="${QUERY_MODE}" \
                        DRY_RUN="${DRY_RUN}" \
                        bash "${SCRIPT_DIR}/run_e8.sh" "${e8_sub}"
                        ;;
                    overlay_1m)
                        BASE_INDEX="${E8_BASE_INDEX}" \
                        PHASE3_WEIGHTS="${PHASE3_WEIGHTS}" \
                        GPU_IDS="${GPU_IDS}" \
                        DEVICE="${DEVICE}" \
                        MEMORY_SETTING="overlay_1m" \
                        QUERY_MODE="${QUERY_MODE}" \
                        DRY_RUN="${DRY_RUN}" \
                        bash "${SCRIPT_DIR}/run_e8.sh" "${e8_sub}"
                        ;;
                esac
            done
            ;;

        *)
            echo "[run_all] 未知实验：${name}，跳过"
            ;;
    esac
}

# ── 顺序执行 ─────────────────────────────────────────────────────────────────
for exp_name in ${EXPERIMENTS}; do
    run_exp "${exp_name}"
done

echo ""
echo "══════════════════════════════════════════════════════"
echo "  所有实验完成"
echo "══════════════════════════════════════════════════════"
