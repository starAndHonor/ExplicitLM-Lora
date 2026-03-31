# E7 Multi-Benchmark Compare

E7 用来在三个 benchmark 上统一比较 5 组方法：

- `B0_qwen3_base`
  `Qwen3-0.6B` 直接做多选题打分
- `B1_qwen3_rag`
  `Qwen3-0.6B + RAG`，使用现有 `*_knowledge.jsonl` 压缩知识映射
- `S1_p1_p2_p3`
  方案一：`Phase1 -> Phase2 -> Phase3`
- `S2_p1_p3_infer`
  方案二：`Phase1 -> Phase3` 冻结推理
- `S3_p2oracle_p1_p3`
  方案三：`Phase2(oracle) -> Phase1 -> Phase3`

评测数据集：

- `MedQA`
- `ARC-Challenge`
- `MMLU`

说明：

- `S1/S2/S3` 都使用同一个 `Phase1` router 做在线检索。
- 检索 query 使用和下游打分一致的完整 prompt：
  `Question + A/B/C/D + Answer:`
- `B1_qwen3_rag` 目前使用仓库里已有的压缩知识映射，作为 `Qwen3-0.6B with RAG` 基线。

## 运行

推荐直接用脚本：

```bash
PHASE1_WEIGHTS=checkpoints/phase1_best \
SCHEME1_WEIGHTS=checkpoints/scheme1_final/phase3_best \
SCHEME2_WEIGHTS=checkpoints/phase3_best \
SCHEME3_WEIGHTS=checkpoints/scheme3_final/phase3_best \
bash scripts/run_e7.sh
```

默认情况下，`scripts/run_e7.sh` 会使用：

- `ENC_MODE=qwen3`

也就是默认使用 `Qwen` 嵌入，而不是可训练知识编码器。

如果你想显式切回可训练模式：

```bash
ENC_MODE=trainable \
PHASE1_WEIGHTS=checkpoints/phase1_best \
SCHEME1_WEIGHTS=checkpoints/scheme1_final/phase3_best \
SCHEME2_WEIGHTS=checkpoints/phase3_best \
SCHEME3_WEIGHTS=checkpoints/scheme3_final/phase3_best \
bash scripts/run_e7.sh
```

等价的 Python 命令：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e7/run_e7.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase1_best \
  --scheme1-weights checkpoints/scheme1_final/phase3_best \
  --scheme2-weights checkpoints/phase3_best \
  --scheme3-weights checkpoints/scheme3_final/phase3_best \
  --device cuda:0
```

如果只想先做小样本 smoke test：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e7/run_e7.py \
  --config config/default.yaml \
  --phase1-weights checkpoints/phase1_best \
  --scheme1-weights checkpoints/scheme1_final/phase3_best \
  --scheme2-weights checkpoints/phase3_best \
  --scheme3-weights checkpoints/scheme3_final/phase3_best \
  --device cuda:0 \
  --max-samples 20
```

如果你直接走 Python 入口、而不是脚本，并且想评测 `qwen3` 模式 checkpoint，需要补充：

```bash
--override model.knowledge_encoder_mode=qwen3
```

结果默认写到：

- [results/e7/e7_benchmark_compare.json](/home/undergraduate/zcy/Explicit-Lora/results/e7/e7_benchmark_compare.json)
