#!/usr/bin/env bash
# E9: Sequential Write + Closed-book Probe Benchmark
#
# 用法：
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   bash scripts/run_e9.sh
#
#   # mock_tokenize（跳过 LLMLingua，用于快速调试）
#   COMPRESSION_BACKEND=mock_tokenize \
#   PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
#   bash scripts/run_e9.sh
#
#   # 指定压缩率
#   COMPRESSION_RATE=0.5 PHASE3_WEIGHTS=... bash scripts/run_e9.sh
#
#   # 指定 k-size
#   K_SIZE=128 PHASE3_WEIGHTS=... bash scripts/run_e9.sh
#
# 说明：
#   - 从 MedQA / ARC / MMLU 各随机抽取 N_WRITES=100 条
#   - 用 original_text 在线 LLMLingua 压缩后，逐条写入 base dense index
#   - 写入完成后对 N_PROBES=10 条做闭卷探测，评测写入前后的检索命中率和 QA 准确率
#   - BASE_INDEX 使用 FineWeb 1M base（不含任务知识）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_experiment_common.sh"

CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
GPU_IDS="${GPU_IDS:-6}"
DEVICE="${DEVICE:-cuda:0}"
ENC_MODE="${ENC_MODE:-qwen3}"
K_SIZE="${K_SIZE:-64}"
N_WRITES="${N_WRITES:-100}"
N_PROBES="${N_PROBES:-10}"
SEED="${SEED:-0}"
QUERY_MODE="${QUERY_MODE:-question_only}"
COMPRESSION_BACKEND="${COMPRESSION_BACKEND:-llmlingua}"
COMPRESSION_RATE="${COMPRESSION_RATE:-0.25}"
DRY_RUN="${DRY_RUN:-0}"

BASE_INDEX="${BASE_INDEX:-checkpoints/dense_fineweb_1m_flat_r0_qwen3_fv.pt}"
PHASE3_WEIGHTS="${PHASE3_WEIGHTS:-checkpoints/p3_from_p2_qwen3_10ep/phase3_best}"

if [ -z "${OUTPUT:-}" ]; then
    OUTPUT="${PROJECT_ROOT}/results/e9/e9_seq_write_probe_k${K_SIZE}_$(exp_ckpt_tag "${PHASE3_WEIGHTS}").json"
fi

exp_load_env
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# conda run 不完全等价于 conda activate，某些情况下会回退到系统 libstdc++。
# 显式将 conda 环境自己的 lib 目录置于首位，确保 libicui18n.so 等找到兼容版本。
CONDA_LIB="$(conda run -n ExplicitLLM python -c 'import sys,os; print(os.path.join(sys.prefix,"lib"))' 2>/dev/null || true)"
if [ -n "${CONDA_LIB}" ]; then
    export LD_LIBRARY_PATH="${CONDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

declare -a CMD=(
    conda run --no-capture-output -n ExplicitLLM
    python "${PROJECT_ROOT}/experiments/e9/main.py"
    --config "${CONFIG}"
    --base-index "${BASE_INDEX}"
    --phase3-weights "${PHASE3_WEIGHTS}"
    --device "${DEVICE}"
    --n-writes "${N_WRITES}"
    --n-probes "${N_PROBES}"
    --seed "${SEED}"
    --query-mode "${QUERY_MODE}"
    --compression-backend "${COMPRESSION_BACKEND}"
    --compression-rate "${COMPRESSION_RATE}"
    --output "${OUTPUT}"
    --override "model.fusion_length=${K_SIZE}"
)

if [ "${ENC_MODE}" = "qwen3" ]; then
    CMD+=(--override "model.knowledge_encoder_mode=qwen3")
fi

echo "[E9] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[E9] config=${CONFIG}"
echo "[E9] base_index=${BASE_INDEX}"
echo "[E9] phase3_weights=${PHASE3_WEIGHTS}"
echo "[E9] k_size=${K_SIZE}"
echo "[E9] n_writes=${N_WRITES}"
echo "[E9] n_probes=${N_PROBES}"
echo "[E9] seed=${SEED}"
echo "[E9] query_mode=${QUERY_MODE}"
echo "[E9] compression_backend=${COMPRESSION_BACKEND}"
echo "[E9] compression_rate=${COMPRESSION_RATE}"
echo "[E9] output=${OUTPUT}"
exp_print_cmd "${CMD[@]}"

if [ "${DRY_RUN}" = "1" ]; then
    exit 0
fi

"${CMD[@]}"
