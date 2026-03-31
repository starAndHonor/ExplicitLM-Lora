# Explicit-Lora

本仓库当前常用流程可以分成两部分：

- 训练
- 实验评测（E1 / E2 / E3 / E4 / E5 / E6）

## 训练

### Phase 2 Fusion

默认 `trainable` 模式：

```bash
ENC_MODE=trainable EPOCHS=10 TAG=norm \
NUM_GPUS=2 GPU_IDS=2,3 \
bash scripts/run_phase2_fusion.sh
```

输出目录示例：

```text
checkpoints/p2_trainable_10ep_norm
```

`qwen3` 模式：

```bash
ENC_MODE=qwen3 EPOCHS=10 \
NUM_GPUS=2 GPU_IDS=2,3 \
bash scripts/run_phase2_fusion.sh
```

输出目录示例：

```text
checkpoints/p2_qwen3_10ep
```

### Phase 3 SFT

从 `trainable` 的 Phase 2 接着训练：

```bash
ENC_MODE=trainable \
FROM_PHASE2=checkpoints/p2_trainable_10ep_norm/phase2_best \
FROM_TAG=p2_trainable_10ep_norm \
NUM_GPUS=2 GPU_IDS=2,3 \
bash scripts/run_phase3_sft.sh --override data.num_workers=0
```

输出目录示例：

```text
checkpoints/p3_from_p2_trainable_10ep_norm
```

从 `qwen3` 的 Phase 2 接着训练：

```bash
ENC_MODE=qwen3 \
FROM_PHASE2=checkpoints/p2_qwen3_10ep/phase2_best \
FROM_TAG=p2_qwen3_10ep \
NUM_GPUS=2 GPU_IDS=2,3 \
bash scripts/run_phase3_sft.sh --override data.num_workers=0
```

输出目录示例：

```text
checkpoints/p3_from_p2_qwen3_10ep
```

## 实验评测

### E1 Sanity Check

`E1` 按 Reference 语义实现：

- `correct knowledge`
- `counterfactual knowledge`
- `no knowledge`（all-pad knowledge）

跑 `qwen3` 的 `Phase 2 best`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e1/run_e1.py \
  --config config/default.yaml \
  --fusion-ckpt checkpoints/p2_qwen3_10ep/phase2_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --output results/e1/e1_sanity_check_p2_qwen3_10ep_phase2_best.json
```

跑 `qwen3` 的 `Phase 3 best`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e1/run_e1.py \
  --config config/default.yaml \
  --fusion-ckpt checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --output results/e1/e1_sanity_check_p3_from_p2_qwen3_10ep_phase3_best.json
```

### E2 Cross-Domain

当前 `E2` 会同时评测：

- `Phase 2 checkpoint`
- `Phase 3 checkpoint`

并输出：

- `Phase 2` 表
- `Phase 3` 表
- `Phase3 vs Phase2` 对比

跑 `qwen3` 的 `Phase 2 / Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e2/run_e2.py \
  --config config/default.yaml \
  --phase2-ckpt checkpoints/p2_qwen3_10ep/phase2_best \
  --phase3-ckpt checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0 \
  --output results/e2/e2_cross_domain_p2_qwen3_10ep_phase2_best__p3_from_p2_qwen3_10ep_phase3_best.json
```

跑当前默认 `trainable` 的 `Phase 2 / Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e2/run_e2.py \
  --config config/default.yaml \
  --phase2-ckpt checkpoints/phase2_best \
  --phase3-ckpt checkpoints/phase3_best \
  --device cuda:0 \
  --output results/e2/e2_cross_domain_phase2_best__phase3_best.json
```

### E3 Fair Compare

默认 `E3` 仍然是 `k=64`。

跑 `qwen3` 的 `Phase 2/Phase 3`（默认 `k=64`）：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e3/run_e3.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase2-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0
```

跑当前默认 `trainable` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e3/run_e3.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase2_best \
  --phase2-weights checkpoints/phase3_best \
  --device cuda:0
```

跑单个自定义 token 预算（例如 `k=128`）：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e3/run_e3.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase2-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --k 128 \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0
```

一口气跑 `k=32/64/128/256`：

```bash
ENC_MODE=qwen3 \
GPU_IDS=2,3 \
PHASE1_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE2_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_e3_multik.sh
```

也可以走统一实验脚本：

```bash
ENC_MODE=qwen3 \
PHASE1_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE2_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash scripts/run_experiment_auto.sh e3_multik
```

### E4 SFT Ablation

跑 `qwen3` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e4/run_e4.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase2-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0
```

跑当前默认 `trainable` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e4/run_e4.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase2_best \
  --phase2-weights checkpoints/phase3_best \
  --device cuda:0
```

### E5 Knowledge Analysis

跑 `qwen3` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase2-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0 \
  --output results/e5/e5_knowledge_analysis_p2p3_qwen3.json
```

跑当前默认 `trainable` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase2_best \
  --phase2-weights checkpoints/phase3_best \
  --device cuda:0
```

如果只想构建 E5 需要的知识映射：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --build-only
```

### E6 Inference Efficiency

跑 `qwen3` 的 `Phase 3 best`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e6/run_e6.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --e5-result results/e5/e5_knowledge_analysis_p2p3_qwen3.json \
  --device cuda:0
```

跑当前默认 `trainable` 的 `Phase 3 best`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e6/run_e6.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase3_best \
  --device cuda:0
```

## 说明

- `qwen3` 模式复用 Qwen3 encoder 与原始 `norm`，因此评测 `qwen3` checkpoint 时要加：
  - `--override model.knowledge_encoder_mode=qwen3`
- `trainable` 模式是当前默认配置。
- `scripts/run_experiment_auto.sh` 会根据 checkpoint 自动生成结果文件名。
- `scripts/run_experiment_suite.sh` 会自动串联实验依赖。
  - 例如跑 `E3 -> E5 -> E6` 时，会把前面生成的 `E3/E5` 结果路径自动传给 `E6`。
  - 推荐示例：

```bash
ENC_MODE=qwen3 \
FUSION_CKPT=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE1_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE2_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
EXPERIMENTS="e3 e5 e6" \
bash scripts/run_experiment_suite.sh
```

- `result.md` 汇总了当前主要 E1 / E2 / E3 结果。
