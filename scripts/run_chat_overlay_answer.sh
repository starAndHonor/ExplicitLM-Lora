#!/usr/bin/env bash
# 两轮对话式知识更新 + 融合生成
#
# 必填：
#   BASE_INDEX       dense index 文件路径
#   PHASE3_CKPT      Phase3 checkpoint 目录
#
# 可选：
#   COMPRESSION_BACKEND  llmlingua|mock_tokenize（默认 llmlingua）
#   COMPRESSION_RATE     压缩率（默认 0.25）
#   MAX_NEW_TOKENS       最大生成 token 数（默认 256）
#   TEMPERATURE          采样温度（默认 1.0，greedy）
#   TOP_P                top-p 采样（默认 0.9）
#   GPU_IDS              CUDA_VISIBLE_DEVICES（默认 6）
#   DEVICE               cuda:0|cpu（默认 cuda:0）
#   ENC_MODE             qwen3|trainable（默认 qwen3）
#
# 用法：
#   BASE_INDEX=checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
#   PHASE3_CKPT=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   bash scripts/run_chat_overlay_answer.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
source "${SCRIPT_DIR}/_experiment_common.sh"

BASE_INDEX="${BASE_INDEX:-}"
PHASE3_CKPT="${PHASE3_CKPT:-}"

# 默认使用 FineWeb 1M r0 base index（需已用单视图重建）
BASE_INDEX="${BASE_INDEX:-checkpoints/dense_fineweb_1m_flat_r0_qwen3_fv.pt}"

if [ -z "${BASE_INDEX}" ]; then
    echo "[Chat] ERROR: BASE_INDEX 必须设置"
    exit 1
fi
if [ -z "${PHASE3_CKPT}" ]; then
    echo "[Chat] ERROR: PHASE3_CKPT 必须设置"
    exit 1
fi

COMPRESSION_BACKEND="${COMPRESSION_BACKEND:-llmlingua}"
COMPRESSION_RATE="${COMPRESSION_RATE:-0.25}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
GPU_IDS="${GPU_IDS:-6}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
SEED="${SEED:-42}"

exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# 修复 libstdc++ 兼容性（llmlingua → nltk → sqlite3）
_CONDA_LIBSTDCXX="/home/undergraduate/.conda/envs/ExplicitLLM/lib/libstdc++.so.6"
if [ -f "${_CONDA_LIBSTDCXX}" ]; then
    export LD_PRELOAD="${_CONDA_LIBSTDCXX}"
fi

echo "[Chat] CUDA_VISIBLE_DEVICES=${GPU_IDS}"
echo "[Chat] device=${DEVICE}"
echo "[Chat] base_index=${BASE_INDEX}"
echo "[Chat] phase3_ckpt=${PHASE3_CKPT}"
echo "[Chat] compression=${COMPRESSION_BACKEND} (rate=${COMPRESSION_RATE})"
echo "[Chat] generation: max_new_tokens=${MAX_NEW_TOKENS}, temperature=${TEMPERATURE}, top_p=${TOP_P}"
echo "------------------------------------------------------"

declare -a CMD=(
    conda run --no-capture-output -n ExplicitLLM
    python "${PROJECT_ROOT}/scripts/chat_overlay_answer.py"
    --config "${CONFIG}"
    --base-index "${BASE_INDEX}"
    --phase3-ckpt "${PHASE3_CKPT}"
    --compression-backend "${COMPRESSION_BACKEND}"
    --compression-rate "${COMPRESSION_RATE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --seed "${SEED}"
    --device "${DEVICE}"
)

if [ "${ENC_MODE}" = "qwen3" ]; then
    CMD+=(--override "model.knowledge_encoder_mode=qwen3")
fi

exp_print_cmd "${CMD[@]}"
echo "------------------------------------------------------"

"${CMD[@]}"
