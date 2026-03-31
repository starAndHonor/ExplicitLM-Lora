# E5 Knowledge Analysis

E5 包含两部分：

- `E5-A`：知识 token 预算分析
  - `k=32/64/128/256`
  - 方法：`Baseline / RAG / Fusion`
  - 数据集：`MedQA / ARC / MMLU`
  - 权重：`Phase 2 / Phase 3`
- `E5-B`：知识相关性分析
  - 条件：`Oracle / Shuffled / Empty`
  - 固定 `k=64`
  - 数据集：`MedQA / ARC / MMLU`
  - 权重：`Phase 2 / Phase 3`

## 运行 E5

默认复用已有知识映射：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/phase2_best \
  --phase3-weights checkpoints/phase3_best \
  --device cuda:0
```

如果评测 `qwen3` checkpoint，需要补充：

```bash
--override model.knowledge_encoder_mode=qwen3
```

例如：

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --phase2-weights checkpoints/p2_qwen3_10ep/phase2_best \
  --phase3-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0
```

## 构建知识映射

只构建 E5 需要的知识映射：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --build-only
```

强制重建：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e5/run_e5.py \
  --config config/default.yaml \
  --build-only \
  --rebuild
```

说明：

- `k=32/64/128` 使用 LLMLingua 压缩率 `0.125/0.25/0.5`
- `k=256` 直接 tokenize `question + correct_answer`
- 结果默认保存到 [results/e5/e5_knowledge_analysis.json](/home/undergraduate/zcy/Explicit-Lora/results/e5/e5_knowledge_analysis.json)
