# Explicit-Lora

当前主线已经切到：

- `dense retrieval`
- `qwen3 / trainable` 两种知识编码模式
- `Phase2 -> Phase3` 注入训练
- `E1 ~ E7` 自动实验与结果汇总

旧的 `scheme1 / scheme3` shell 入口已经移除，当前建议直接使用：

- 单阶段训练脚本
- `scheme2` 推理脚本
- `run_experiment_auto.sh`
- `run_experiment_suite.sh`
- `experiments/e7/` 下的正式 E7 入口

## 环境约定

常用环境变量：

- `MODEL_PATH=/path/to/Qwen3-0.6B`
- `GPU_IDS=2` 或 `GPU_IDS=2,3`
- `ENC_MODE=qwen3` 或 `ENC_MODE=trainable`

常用 Conda 环境：

- `ExplicitLLM`

## 训练

### 1. 训练 Phase2 Fusion

`qwen3` 模式：

```bash
cd /path/to/Explicit-Lora

MODEL_PATH=/path/to/Qwen3-0.6B \
ENC_MODE=qwen3 \
NUM_GPUS=1 GPU_IDS=2 \
bash scripts/run_phase2_fusion.sh
```

`trainable` 模式：

```bash
cd /path/to/Explicit-Lora

ENC_MODE=trainable \
NUM_GPUS=1 GPU_IDS=2 \
bash scripts/run_phase2_fusion.sh
```

### 2. 训练 Phase3 SFT

从 `Phase2` 接着训练：

`qwen3` 模式：

```bash
cd /path/to/Explicit-Lora

MODEL_PATH=/path/to/Qwen3-0.6B \
ENC_MODE=qwen3 \
FROM_PHASE2=checkpoints/p2_qwen3_10ep/phase2_best \
NUM_GPUS=1 GPU_IDS=2 \
bash scripts/run_phase3_sft.sh --override data.num_workers=0
```

`trainable` 模式：

```bash
cd /path/to/Explicit-Lora

ENC_MODE=trainable \
FROM_PHASE2=checkpoints/phase2_best \
NUM_GPUS=1 GPU_IDS=2 \
bash scripts/run_phase3_sft.sh --override data.num_workers=0
```

### 3. Dense 检索版 Phase3 训练

如果要接已经构建好的 dense index：

```bash
cd /path/to/Explicit-Lora

MODEL_PATH=/path/to/Qwen3-0.6B \
NUM_GPUS=1 GPU_IDS=2 \
ENC_MODE=qwen3 \
DATA_ROOT=/path/to/Explicit-Lora/data \
FROM_PHASE2=/path/to/Explicit-Lora/checkpoints/p2_qwen3_10ep/phase2_best \
FROM_DENSE_INDEX=/path/to/Explicit-Lora/checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
KNOWLEDGE_SOURCE=dense_retriever \
bash scripts/run_phase3_sft.sh
```

## 推理

### 1. Scheme2 推理

普通版本：

```bash
cd /path/to/Explicit-Lora

bash scripts/run_scheme2_p1_p3_infer.sh \
  --question "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. Which of the following is the best treatment for this patient?" \
  --option-a "Ampicillin" \
  --option-b "Ceftriaxone" \
  --option-c "Doxycycline" \
  --option-d "Nitrofurantoin"
```

`qwen3` 版本：

```bash
cd /path/to/Explicit-Lora

MODEL_PATH=/path/to/Qwen3-0.6B \
CUDA_VISIBLE_DEVICES=2 \
DEVICE=cuda:0 \
bash scripts/run_scheme2_qwen3_p1_p3_infer.sh \
  --question "A 65-year-old man is brought to the emergency department 30 minutes after the onset of acute chest pain." \
  --option-a "Aspirin only" \
  --option-b "Clopidogrel" \
  --option-c "Warfarin" \
  --option-d "Amiodarone"
```

### 2. Dense 检索 + Phase3 推理

```bash
cd /path/to/Explicit-Lora

CUDA_VISIBLE_DEVICES=2 \
MODEL_PATH=/path/to/Qwen3-0.6B \
conda run --no-capture-output -n ExplicitLLM \
python scripts/run_dense_phase3_infer.py \
  --config config/default.yaml \
  --dense-index checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
  --phase3-ckpt checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --device cuda:0 \
  --query-mode question_only \
  --question "A 65-year-old man is brought to the emergency department 30 minutes after the onset of acute chest pain." \
  --option-a "Aspirin only" \
  --option-b "Clopidogrel" \
  --option-c "Warfarin" \
  --option-d "Amiodarone" \
  --json
```

## 实验评测

### 单独运行某个实验

统一入口：

```bash
cd /path/to/Explicit-Lora

bash scripts/run_experiment.sh <e1|e2|e3|e3_multik|e4|e5|e6|e7|e7_full>
```

自动命名输出：

```bash
cd /path/to/Explicit-Lora

bash scripts/run_experiment_auto.sh <e1|e2|e3|e3_multik|e4|e5|e6|e7|e7_full>
```

一键顺序运行：

```bash
cd /path/to/Explicit-Lora

bash scripts/run_experiment_suite.sh
```

默认顺序：

- `e1`
- `e2`
- `e3`
- `e3_multik`
- `e4`
- `e5`
- `e6`
- `e7`

### E1

```bash
cd /path/to/Explicit-Lora

ENC_MODE=qwen3 \
PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e1
```

### E2

```bash
cd /path/to/Explicit-Lora

ENC_MODE=qwen3 \
PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e2
```

### E3

```bash
cd /path/to/Explicit-Lora

ENC_MODE=qwen3 \
PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e3
```

### E3 multi-k

```bash
cd /path/to/Explicit-Lora

ENC_MODE=qwen3 \
PHASE1_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE2_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e3_multik
```

### E4

```bash
cd /path/to/Explicit-Lora

ENC_MODE=qwen3 \
PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e4
```

### E5

```bash
cd /path/to/Explicit-Lora

ENC_MODE=qwen3 \
PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e5
```

### E6

```bash
cd /path/to/Explicit-Lora

ENC_MODE=qwen3 \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e6
```

### E7：只跑最终 dense E7

需要先准备三份 overlay index：

- `DENSE_INDEX_MEDQA`
- `DENSE_INDEX_ARC`
- `DENSE_INDEX_MMLU`

运行：

```bash
cd /path/to/Explicit-Lora

DENSE_INDEX_MEDQA=checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
DENSE_INDEX_ARC=checkpoints/dense_fineweb_arc_overlay_original_text_flat_r24_qwen3.pt \
DENSE_INDEX_MMLU=checkpoints/dense_fineweb_mmlu_overlay_original_text_flat_r24_qwen3.pt \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e7
```

### E7：从建库到评测全自动

这一版会自动做：

1. 构建 `1M FineWeb base`
2. 构建 `MedQA / ARC / MMLU` overlay
3. 跑 retrieval precheck
4. 跑最终 dense E7

默认会同时跑两套检索文本版本：

- `original_text`
- `k256`

运行：

```bash
cd /path/to/Explicit-Lora

GPU_IDS=2 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e7_full
```

如果只想看命令：

```bash
cd /path/to/Explicit-Lora

DRY_RUN=1 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e7_full
```

正式入口也可以直接用：

- [run.sh](experiments/e7/run.sh)
- [run_dense_full.sh](experiments/e7/run_dense_full.sh)

## 结果汇总

收集所有实验结果：

```bash
cd /path/to/Explicit-Lora

python scripts/collect_results.py
```

输出文件：

- [results_summary.md](results/results_summary.md)

当前 `collect_results.py` 已支持：

- `E1 ~ E6`
- `E7` 最终结果
- `E7 retrieval precheck`

## 当前主线建议

如果你只想跑“当前推荐配置”，最短路径是：

1. 训练 `Phase2`
2. 训练或加载 `Phase3`
3. 构建 dense index / overlay
4. 跑 `e7` 或 `e7_full`

推荐命令组合：

```bash
cd /path/to/Explicit-Lora

MODEL_PATH=/path/to/Qwen3-0.6B \
ENC_MODE=qwen3 \
NUM_GPUS=1 GPU_IDS=2 \
bash scripts/run_phase2_fusion.sh
```

```bash
cd /path/to/Explicit-Lora

MODEL_PATH=/path/to/Qwen3-0.6B \
NUM_GPUS=1 GPU_IDS=2 \
ENC_MODE=qwen3 \
FROM_PHASE2=checkpoints/p2_qwen3_10ep/phase2_best \
bash scripts/run_phase3_sft.sh
```

```bash
cd /path/to/Explicit-Lora

GPU_IDS=2 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e7_full
```
