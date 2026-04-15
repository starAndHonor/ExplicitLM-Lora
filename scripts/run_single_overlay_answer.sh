#!/usr/bin/env bash
# 单样本知识构建 + Overlay + 检索 + 融合回答 完整流程
#
# 流程：
#   Step 1: 从数据集取原始（未压缩）知识文本
#   Step 2: 使用 LLMLingua 压缩为 fusion token IDs（检索视图 + 注入视图）
#   Step 3: 将新知识条目 overlay 到 dense index（替换一个随机位置）
#   Step 4: 用 dense retriever 检索最相关知识
#   Step 5: 用 injection 模型（Phase3 checkpoint）融合知识回答问题
#
# 必填变量：
#   BASE_INDEX        - 基础 dense index 文件路径（.pt）
#   PHASE3_CKPT       - Phase3 checkpoint 目录（含 injection_modules.pt）
#   QUESTION          - 问题文本
#   OPTION_A/B/C/D    - 四个候选答案
#
# 可选变量（知识来源，三选一）：
#   SOURCE_TEXT       - 直接提供原始知识文本（优先级最高）
#   SOURCE_QUESTION   - 用另一个问题的 key 来查数据集原始知识
#   SOURCE_KEY        - 直接指定 key（默认用 QUESTION[:200].strip()）
#
# 可选变量（运行控制）：
#   DATASET           - 数据集名称：medqa|arc|mmlu（默认 medqa）
#   COMPRESSION_BACKEND - llmlingua|mock_tokenize（默认 llmlingua，无模型时用 mock_tokenize）
#   COMPRESSION_RATE  - LLMLingua 压缩率（默认 0.25）
#   QUERY_MODE        - question_only|question_choices（默认 question_only）
#   ANCHOR_SOURCE     - original_text|compressed_decode（默认 original_text）
#   TOP_K             - 检索 top-k（默认 3）
#   GPU_IDS           - CUDA_VISIBLE_DEVICES（默认 6）
#   DEVICE            - cuda:0|cpu（默认 cuda:0）
#   ENC_MODE          - qwen3|trainable（默认 qwen3）
#   SAVE_OVERLAY      - 是否保存 overlay 后的 index（默认 0）
#   ARTIFACTS_DIR     - 中间产物保存目录（默认 results/single_overlay）
#   OUTPUT_JSON       - 是否以 JSON 输出（默认 0，输出人类可读格式）
#   SEED              - 随机种子，控制 overlay 替换位置（默认 42）
#
# 用法示例：
#
#   # 用 MedQA 数据集中该问题对应的原始知识，使用 mock_tokenize（无需 LLMLingua）：
#   BASE_INDEX=checkpoints/dense_medqa_flat.pt \
#   PHASE3_CKPT=checkpoints/phase3_best \
#   COMPRESSION_BACKEND=mock_tokenize \
#   QUESTION="A 65-year-old man presents with chest pain." \
#   OPTION_A="Myocardial infarction" \
#   OPTION_B="Pulmonary embolism" \
#   OPTION_C="Aortic dissection" \
#   OPTION_D="Pneumothorax" \
#   bash scripts/run_single_overlay_answer.sh
#
#   # 直接提供原始知识文本（不查数据集），使用 LLMLingua：
#   BASE_INDEX=checkpoints/dense_medqa_flat.pt \
#   PHASE3_CKPT=checkpoints/phase3_best \
#   SOURCE_TEXT="Chest pain in elderly patients can indicate myocardial infarction..." \
#   QUESTION="A 65-year-old man presents with chest pain." \
#   OPTION_A="MI" OPTION_B="PE" OPTION_C="AD" OPTION_D="PTX" \
#   bash scripts/run_single_overlay_answer.sh
#
#   # 使用 E8 预建的 FineWeb 1M base index + overlay_1m 设置：
#   BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r24_qwen3.pt \
#   PHASE3_CKPT=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   COMPRESSION_BACKEND=mock_tokenize \
#   GPU_IDS=6 \
#   QUESTION="Which of the following is the most common cause of community-acquired pneumonia?" \
#   OPTION_A="Streptococcus pneumoniae" \
#   OPTION_B="Staphylococcus aureus" \
#   OPTION_C="Haemophilus influenzae" \
#   OPTION_D="Mycoplasma pneumoniae" \
#   bash scripts/run_single_overlay_answer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

# ── 必填参数校验 ──────────────────────────────────────────
BASE_INDEX="${BASE_INDEX:-}"
PHASE3_CKPT="${PHASE3_CKPT:-}"
QUESTION="${QUESTION:-}"
OPTION_A="${OPTION_A:-}"
OPTION_B="${OPTION_B:-}"
OPTION_C="${OPTION_C:-}"
OPTION_D="${OPTION_D:-}"

if [ -z "${BASE_INDEX}" ]; then
    echo "[SingleOverlay] ERROR: BASE_INDEX 必须设置（dense index 文件路径）"
    echo "[SingleOverlay] 示例: BASE_INDEX=checkpoints/dense_medqa_flat.pt"
    exit 1
fi
if [ -z "${PHASE3_CKPT}" ]; then
    echo "[SingleOverlay] ERROR: PHASE3_CKPT 必须设置（Phase3 checkpoint 目录）"
    echo "[SingleOverlay] 示例: PHASE3_CKPT=checkpoints/phase3_best"
    exit 1
fi
if [ -z "${QUESTION}" ]; then
    echo "[SingleOverlay] ERROR: QUESTION 必须设置"
    exit 1
fi
if [ -z "${OPTION_A}" ] || [ -z "${OPTION_B}" ] || [ -z "${OPTION_C}" ] || [ -z "${OPTION_D}" ]; then
    echo "[SingleOverlay] ERROR: OPTION_A/B/C/D 均必须设置"
    exit 1
fi

# ── 可选参数默认值 ────────────────────────────────────────
SOURCE_TEXT="${SOURCE_TEXT:-}"
SOURCE_QUESTION="${SOURCE_QUESTION:-}"
SOURCE_KEY="${SOURCE_KEY:-}"
DATASET="${DATASET:-medqa}"
COMPRESSION_BACKEND="${COMPRESSION_BACKEND:-llmlingua}"
COMPRESSION_RATE="${COMPRESSION_RATE:-0.25}"
QUERY_MODE="${QUERY_MODE:-question_only}"
ANCHOR_SOURCE="${ANCHOR_SOURCE:-original_text}"
TOP_K="${TOP_K:-3}"
GPU_IDS="${GPU_IDS:-6}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
SAVE_OVERLAY="${SAVE_OVERLAY:-0}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${PROJECT_ROOT}/results/single_overlay}"
OUTPUT_JSON="${OUTPUT_JSON:-0}"
SEED="${SEED:-42}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
DRY_RUN="${DRY_RUN:-0}"

# ── 加载 .env 环境变量（MODEL_PATH 等）─────────────────────
exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# ── 修复 libstdc++ 兼容性（llmlingua → nltk → sqlite3 需要 CXXABI_1.3.15）──
# 系统 /lib/x86_64-linux-gnu/libstdc++.so.6 版本过旧，强制预加载 conda 环境版本
_CONDA_LIBSTDCXX="/home/undergraduate/.conda/envs/ExplicitLLM/lib/libstdc++.so.6"
if [ -f "${_CONDA_LIBSTDCXX}" ]; then
    export LD_PRELOAD="${_CONDA_LIBSTDCXX}"
fi

# ── 打印运行参数 ──────────────────────────────────────────
echo "======================================================"
echo "[SingleOverlay] 单样本知识构建 → Overlay → 检索 → 融合回答"
echo "======================================================"
echo "[SingleOverlay] CUDA_VISIBLE_DEVICES=${GPU_IDS}"
echo "[SingleOverlay] device=${DEVICE}"
echo "[SingleOverlay] config=${CONFIG}"
echo "[SingleOverlay] base_index=${BASE_INDEX}"
echo "[SingleOverlay] phase3_ckpt=${PHASE3_CKPT}"
echo "[SingleOverlay] dataset=${DATASET}"
echo "[SingleOverlay] compression_backend=${COMPRESSION_BACKEND}"
echo "[SingleOverlay] compression_rate=${COMPRESSION_RATE}"
echo "[SingleOverlay] query_mode=${QUERY_MODE}"
echo "[SingleOverlay] anchor_source=${ANCHOR_SOURCE}"
echo "[SingleOverlay] top_k=${TOP_K}"
echo "[SingleOverlay] enc_mode=${ENC_MODE}"
echo "[SingleOverlay] seed=${SEED}"
echo "[SingleOverlay] artifacts_dir=${ARTIFACTS_DIR}"
if [ -n "${SOURCE_TEXT}" ]; then
    echo "[SingleOverlay] knowledge_source=direct_text"
elif [ -n "${SOURCE_QUESTION}" ]; then
    echo "[SingleOverlay] knowledge_source=dataset_via_question: ${SOURCE_QUESTION:0:80}..."
elif [ -n "${SOURCE_KEY}" ]; then
    echo "[SingleOverlay] knowledge_source=dataset_via_key: ${SOURCE_KEY:0:80}..."
else
    echo "[SingleOverlay] knowledge_source=dataset_via_question_prefix (default)"
fi
echo "[SingleOverlay] question: ${QUESTION:0:100}..."
echo "[SingleOverlay] options: A=${OPTION_A} | B=${OPTION_B} | C=${OPTION_C} | D=${OPTION_D}"
echo "------------------------------------------------------"

# ── 构建命令 ──────────────────────────────────────────────
declare -a CMD=(
    conda run --no-capture-output -n ExplicitLLM
    python "${PROJECT_ROOT}/scripts/run_single_knowledge_overlay_answer.py"
    --config "${CONFIG}"
    --dataset "${DATASET}"
    --base-index "${BASE_INDEX}"
    --phase3-ckpt "${PHASE3_CKPT}"
    --question "${QUESTION}"
    --option-a "${OPTION_A}"
    --option-b "${OPTION_B}"
    --option-c "${OPTION_C}"
    --option-d "${OPTION_D}"
    --compression-backend "${COMPRESSION_BACKEND}"
    --compression-rate "${COMPRESSION_RATE}"
    --query-mode "${QUERY_MODE}"
    --anchor-source "${ANCHOR_SOURCE}"
    --top-k "${TOP_K}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --artifacts-dir "${ARTIFACTS_DIR}"
)

# 知识来源：SOURCE_TEXT > SOURCE_QUESTION > SOURCE_KEY > 默认（用 QUESTION 前缀）
if [ -n "${SOURCE_TEXT}" ]; then
    CMD+=(--source-original-text "${SOURCE_TEXT}")
elif [ -n "${SOURCE_QUESTION}" ]; then
    CMD+=(--source-question "${SOURCE_QUESTION}")
elif [ -n "${SOURCE_KEY}" ]; then
    CMD+=(--source-key "${SOURCE_KEY}")
fi

# 是否保存 overlay 后的 index
if [ "${SAVE_OVERLAY}" = "1" ]; then
    CMD+=(--save-overlay-index)
fi

# 输出格式
if [ "${OUTPUT_JSON}" = "1" ]; then
    CMD+=(--json)
fi

# encoder 模式覆盖
if [ "${ENC_MODE}" = "qwen3" ]; then
    CMD+=(--override "model.knowledge_encoder_mode=qwen3")
fi

exp_print_cmd "${CMD[@]}"
echo "------------------------------------------------------"

if [ "${DRY_RUN}" = "1" ]; then
    echo "[SingleOverlay] DRY_RUN=1，跳过执行"
    exit 0
fi

"${CMD[@]}"
