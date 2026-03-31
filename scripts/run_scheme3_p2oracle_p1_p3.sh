#!/usr/bin/env bash
# 方案三：P2-OracleFusion -> P1-Router -> P3-SFT
#
# 用法：
#   bash scripts/run_scheme3_p2oracle_p1_p3.sh
#
# 环境变量：
#   PHASE1_CKPT    Phase 1 checkpoint 目录（默认 checkpoints/phase1_best）
#   ENC_MODE       知识编码模式（默认 trainable）
#   EPOCHS         Phase 2 训练轮数（默认 10）
#   TAG            Phase 2 标签（默认 oracle）
#   SWANLAB_PROJECT_P2  Phase 2 SwanLab 项目名（默认 explicit-lora-p2）
#   SWANLAB_PROJECT_P3  Phase 3 SwanLab 项目名（默认 explicit-lora-p3）
#   NUM_GPUS       GPU 数量
#   GPU_IDS        CUDA_VISIBLE_DEVICES

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

PHASE1_CKPT="${PHASE1_CKPT:-${PROJECT_ROOT}/checkpoints/phase1_best}"
ENC_MODE="${ENC_MODE:-trainable}"
EPOCHS="${EPOCHS:-10}"
TAG="${TAG:-oracle}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_IDS="${GPU_IDS:-6,7}"
SWANLAB_PROJECT_P2="${SWANLAB_PROJECT_P2:-explicit-lora-p2}"
SWANLAB_PROJECT_P3="${SWANLAB_PROJECT_P3:-explicit-lora-p3}"

P2_RUN_NAME="scheme3_oracle_${ENC_MODE}_${EPOCHS}ep"
P3_RUN_NAME="scheme3_from_oracle_${ENC_MODE}"

echo "[Scheme3] Step 1/3: Phase 1 Router"
NUM_GPUS="${NUM_GPUS}" GPU_IDS="${GPU_IDS}" \
    bash "${SCRIPT_DIR}/run_phase1_router.sh"

echo "[Scheme3] Step 2/3: Phase 2 with oracle knowledge"
NUM_GPUS="${NUM_GPUS}" GPU_IDS="${GPU_IDS}" \
KNOWLEDGE_SOURCE=oracle \
ENC_MODE="${ENC_MODE}" \
EPOCHS="${EPOCHS}" \
TAG="${TAG}" \
SWANLAB_PROJECT="${SWANLAB_PROJECT_P2}" \
SWANLAB_RUN_NAME="${P2_RUN_NAME}" \
    bash "${SCRIPT_DIR}/run_phase2_fusion.sh"

P2_DIR="checkpoints/p2_${ENC_MODE}_${EPOCHS}ep"
if [ -n "${TAG}" ]; then
    P2_DIR="${P2_DIR}_${TAG}"
fi

echo "[Scheme3] Step 3/3: Phase 3 SFT with Phase 1 retrieval"
NUM_GPUS="${NUM_GPUS}" GPU_IDS="${GPU_IDS}" \
KNOWLEDGE_SOURCE=phase1_router \
FROM_PHASE1="${PHASE1_CKPT}" \
FROM_PHASE2="${P2_DIR}/phase2_best" \
FROM_TAG="$(basename "${P2_DIR}")" \
ENC_MODE="${ENC_MODE}" \
SWANLAB_PROJECT="${SWANLAB_PROJECT_P3}" \
SWANLAB_RUN_NAME="${P3_RUN_NAME}" \
    bash "${SCRIPT_DIR}/run_phase3_sft.sh" --override data.num_workers=0
