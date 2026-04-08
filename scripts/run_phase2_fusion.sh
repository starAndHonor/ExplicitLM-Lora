#!/usr/bin/env bash
# Phase 2 Fusion 预训练一键启动脚本
#
# 用法：
#   默认双卡（GPU 6,7）：bash scripts/run_phase2_fusion.sh
#   指定 GPU：       NUM_GPUS=1 GPU_IDS=3 bash scripts/run_phase2_fusion.sh
#   覆盖配置：        bash scripts/run_phase2_fusion.sh --override train.phase2_max_epochs=1
#
# 环境变量：
#   NUM_GPUS        使用 GPU 数量（默认 2）
#   GPU_IDS         CUDA_VISIBLE_DEVICES（默认 6,7）
#   CONFIG          配置文件路径（默认 config/default.yaml）
#   ENC_MODE        知识编码模式（默认 qwen3，可选 trainable）
#   EPOCHS          Phase 2 训练轮数（默认 10）
#   TAG             额外命名标签（可选，如 norm）
#   FROM_PHASE1     可选：Phase 1 checkpoint 目录（启用 frozen router 检索知识）
#   KNOWLEDGE_SOURCE 知识来源（默认 oracle，可选 phase1_router）
#   MAIN_PROCESS_PORT accelerate 主进程端口（默认 29501）

set -euo pipefail

# ── GPU 配置 ──
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-6,7}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

# ── 路径 ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
CONFIG="${CONFIG:-${PROJECT_ROOT}/config/default.yaml}"
ENV_FILE="${PROJECT_ROOT}/.env"
ENC_MODE="${ENC_MODE:-qwen3}"
EPOCHS="${EPOCHS:-10}"
TAG="${TAG:-}"
FROM_PHASE1="${FROM_PHASE1:-}"
KNOWLEDGE_SOURCE="${KNOWLEDGE_SOURCE:-oracle}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29501}"

CKP_NAME="p2_${ENC_MODE}_${EPOCHS}ep"
if [ -n "${TAG}" ]; then
    CKP_NAME="${CKP_NAME}_${TAG}"
fi
CHECKPOINT_PATH="checkpoints/${CKP_NAME}"

# ── 项目环境变量（如 SwanLab API Key） ──
if [ -f "${ENV_FILE}" ]; then
    # 仅加载 .env 中的简单 KEY=VALUE 行，避免影响脚本严格模式
    set -a
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
    set +a
    echo "[Phase2Fusion] 已加载项目环境变量: ${ENV_FILE}"
else
    echo "[Phase2Fusion] 未找到项目 .env，继续使用当前 shell 环境变量"
fi

echo "[Phase2Fusion] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, num_processes=${NUM_GPUS}"
echo "[Phase2Fusion] Config: ${CONFIG}"
echo "[Phase2Fusion] Project: ${PROJECT_ROOT}"
echo "[Phase2Fusion] Encoder mode: ${ENC_MODE}"
echo "[Phase2Fusion] Epochs: ${EPOCHS}"
echo "[Phase2Fusion] Checkpoint name: ${CKP_NAME}"
echo "[Phase2Fusion] Knowledge source: ${KNOWLEDGE_SOURCE}"
echo "[Phase2Fusion] Main process port: ${MAIN_PROCESS_PORT}"
if [ -n "${FROM_PHASE1}" ]; then
    echo "[Phase2Fusion] Phase 1 Checkpoint: ${FROM_PHASE1}"
fi

# ── 检查预压缩数据目录 ──
PARQUET_DIR="${PROJECT_ROOT}/data/compressed/v2"
if [ ! -d "${PARQUET_DIR}" ]; then
    echo "[ERROR] 预压缩 FineWeb-Edu 数据目录不存在: ${PARQUET_DIR}"
    echo "[INFO]  Phase 2 复用 Phase 1 数据目录（data/compressed/v2/）"
    echo "[INFO]  请先准备预压缩数据，或通过 --override data.phase1_parquet_dir=<path> 指定"
    exit 1
fi

PARQUET_COUNT=$(find "${PARQUET_DIR}" -name "*.parquet" | wc -l)
if [ "${PARQUET_COUNT}" -eq 0 ]; then
    echo "[ERROR] ${PARQUET_DIR} 中没有找到 .parquet 文件"
    exit 1
fi
echo "[Phase2Fusion] 找到 ${PARQUET_COUNT} 个 Parquet 文件于 ${PARQUET_DIR}"

# ── 检查检查点目录 ──
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
mkdir -p "${CHECKPOINT_DIR}"
echo "[Phase2Fusion] Checkpoint 目录: ${CHECKPOINT_DIR}"

# ── 检查依赖 ──
conda run --no-capture-output -n ExplicitLLM python -c "import accelerate; import swanlab; import pandas" 2>/dev/null || {
    echo "[INFO] 安装缺失依赖..."
    conda run --no-capture-output -n ExplicitLLM pip install accelerate swanlab pandas -q
}

# ── Accelerate 启动 ──
# 注意：用户参数 "$@" 必须放在子命令 train --phase 2 之前，
# 因为 main.py 的 --override 使用 nargs="*"，会贪婪消耗后续参数
# 注意：不能直接用 `conda run ... accelerate launch`
# 否则可能命中 ~/.local/bin/accelerate，落回系统 Python。
# 这里强制使用 ExplicitLLM 环境中的 `python -m accelerate.commands.launch`。
echo "[Phase2Fusion] 启动训练..."
CMD=(
    conda run --no-capture-output -n ExplicitLLM python -m accelerate.commands.launch
    --num_processes "${NUM_GPUS}"
    --mixed_precision bf16
    --main_process_port "${MAIN_PROCESS_PORT}"
    "${PROJECT_ROOT}/main.py"
    --config "${CONFIG}"
    --device cuda
    --override model.knowledge_encoder_mode="${ENC_MODE}"
    --override train.phase2_max_epochs="${EPOCHS}"
    --override paths.checkpoint_dir="${CHECKPOINT_PATH}"
)
if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi
CMD+=(train --phase 2 --knowledge-source "${KNOWLEDGE_SOURCE}")
if [ -n "${FROM_PHASE1}" ]; then
    CMD+=(--from-phase1 "${FROM_PHASE1}")
fi
"${CMD[@]}"
