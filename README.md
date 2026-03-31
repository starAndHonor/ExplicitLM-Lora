# Explicit-Lora

本仓库当前常用流程可以分成两部分：

- 训练
- 实验评测（E1 / E2 / E3 / E4 / E5 / E6）

## 训练

### 三种方案

#### 方案一：`P1-Router -> P2-RouterFusion -> P3-SFT`

含义：

- `Phase1` 先训练 router
- `Phase2` 使用 `Phase1` 检索出来的知识训练 fusion
- `Phase3` 继续使用 `Phase1` 检索知识做 SFT

可直接复制执行：

```bash
cd /tmp/Explicit-Lora-model-dev
bash scripts/run_scheme1_p1_p2_p3.sh
```

最终模型示例：

```text
checkpoints/p3_from_p2_trainable_10ep_router/phase3_best
```

#### 方案二：`P1-Router -> P3-FusionInference`

含义：

- `Phase1` 只负责检索
- `Phase3` 使用已有 `phase3_best` 做融合推理
- 不训练，不做 SFT

当前建议：

- 使用 `Phase1` 权重：`/home/undergraduate/zcy/Explicit-Lora/checkpoints/phase1_best`
- 使用 `Phase3` 权重：`/home/undergraduate/zcy/Explicit-Lora/checkpoints/phase3_best`

可直接复制执行：

```bash
cd /tmp/Explicit-Lora-model-dev
bash scripts/run_scheme2_p1_p3_infer.sh \
  --question "A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. Which of the following is the best treatment for this patient?" \
  --option-a "Ampicillin" \
  --option-b "Ceftriaxone" \
  --option-c "Doxycycline" \
  --option-d "Nitrofurantoin"
```

#### 方案三：`P2-OracleFusion -> P1-Router -> P3-SFT`

含义：

- `Phase2` 先用正确知识（oracle knowledge）训练 fusion
- `Phase3` 再切换到 `Phase1` 检索知识做 SFT
- 最终模型学到的是“正确融合能力 + 检索噪声适应能力”

可直接复制执行：

```bash
cd /tmp/Explicit-Lora-model-dev
bash scripts/run_scheme3_p2oracle_p1_p3.sh
```

最终模型示例：

```text
checkpoints/p3_from_p2_trainable_10ep_oracle/phase3_best
```

### 基础单阶段命令

#### 单独训练 `Phase2 Fusion`

`trainable` 模式：

```bash
ENC_MODE=trainable EPOCHS=10 TAG=norm \
NUM_GPUS=2 GPU_IDS=2,3 \
bash scripts/run_phase2_fusion.sh
```

`qwen3` 模式：

```bash
ENC_MODE=qwen3 EPOCHS=10 \
NUM_GPUS=2 GPU_IDS=2,3 \
bash scripts/run_phase2_fusion.sh
```

#### 单独训练 `Phase3 SFT`

从 `trainable` 的 `Phase2` 接着训练：

```bash
ENC_MODE=trainable \
FROM_PHASE2=checkpoints/p2_trainable_10ep_norm/phase2_best \
FROM_TAG=p2_trainable_10ep_norm \
NUM_GPUS=2 GPU_IDS=2,3 \
bash scripts/run_phase3_sft.sh --override data.num_workers=0
```

从 `qwen3` 的 `Phase2` 接着训练：

```bash
ENC_MODE=qwen3 \
FROM_PHASE2=checkpoints/p2_qwen3_10ep/phase2_best \
FROM_TAG=p2_qwen3_10ep \
NUM_GPUS=2 GPU_IDS=2,3 \
bash scripts/run_phase3_sft.sh --override data.num_workers=0
```

## 实验评测

### E1 Sanity Check

`E1` 按 Reference 语义实现：

- `correct knowledge`
- `counterfactual knowledge`
- `no knowledge`（all-pad knowledge）

自动实验链里，`e1` 会默认连续评测：

- `PHASE2_WEIGHTS`
- `PHASE3_WEIGHTS`

并生成两份 JSON，供 `results_summary.md` 汇总成对比表。

跑 `qwen3` 的 `Phase 2 best`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e1/run_e1.py \
  --config config/default.yaml \
  --weights checkpoints/p2_qwen3_10ep/phase2_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --output results/e1/e1_sanity_check_p2_qwen3_10ep_phase2_best.json
```

跑 `qwen3` 的 `Phase 3 best`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e1/run_e1.py \
  --config config/default.yaml \
  --weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
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
  --phase2-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase3-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0 \
  --output results/e2/e2_cross_domain_p2_qwen3_10ep_phase2_best__p3_from_p2_qwen3_10ep_phase3_best.json
```

跑当前默认 `trainable` 的 `Phase 2 / Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e2/run_e2.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
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
  --phase2-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase3-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0
```

跑当前默认 `trainable` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e3/run_e3.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
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
  --phase2-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase3-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0
```

跑当前默认 `trainable` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e4/run_e4.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
  --device cuda:0
```

### E5 Knowledge Analysis

跑 `qwen3` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase3-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0 \
  --output results/e5/e5_knowledge_analysis_p2p3_qwen3.json
```

跑当前默认 `trainable` 的 `Phase 2/Phase 3`：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
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
  --phase3-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --e5-result results/e5/e5_knowledge_analysis_p2p3_qwen3.json \
  --device cuda:0
```

跑当前默认 `trainable` 的 `Phase 3 best`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e6/run_e6.py \
  --config config/default.yaml \
  --phase3-weights checkpoints/phase3_best \
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
PHASE2_WEIGHTS=checkpoints/p2_qwen3_10ep/phase2_best \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
EXPERIMENTS="e3 e5 e6" \
bash scripts/run_experiment_suite.sh
```

- `result.md` 汇总了当前主要 E1 / E2 / E3 结果。
