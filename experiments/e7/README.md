# E7 Dense Benchmark

当前 `E7` 使用 dense retrieval，在三个 benchmark 上统一比较 3 组方法：

- `B0_qwen3_base`
  `Qwen3-0.6B` 直接做多选题打分
- `TF_dense_p3_infer`
  dense 检索 + `Phase3` 注入推理
- `RAG_dense`
  dense 检索 + 直接拼接知识文本的 RAG 基线

评测数据集：

- `MedQA`
- `ARC-Challenge`
- `MMLU`

## 运行

如果三份 overlay index 已经准备好，直接运行：

```bash
DENSE_INDEX_MEDQA=checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
DENSE_INDEX_ARC=checkpoints/dense_fineweb_arc_overlay_original_text_flat_r24_qwen3.pt \
DENSE_INDEX_MMLU=checkpoints/dense_fineweb_mmlu_overlay_original_text_flat_r24_qwen3.pt \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash experiments/e7/run.sh
```

如果希望自动完成：

1. 构建 `1M FineWeb base`
2. 构建三个数据集的 overlay index
3. 先做 retrieval precheck
4. 再跑最终 E7

可以直接运行：

```bash
GPU_IDS=2 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash experiments/e7/run_dense_full.sh
```

默认会同时跑两套 anchor 版本：

- `original_text`
- `k256`

如果只想跑其中一个版本，例如 `k256`：

```bash
GPU_IDS=2 \
ANCHOR_VARIANTS=k256 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash experiments/e7/run_dense_full.sh
```

等价的 Python 主入口：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
  python experiments/e7/run_e7.py \
  --config config/default.yaml \
  --dense-index-medqa checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
  --dense-index-arc checkpoints/dense_fineweb_arc_overlay_original_text_flat_r24_qwen3.pt \
  --dense-index-mmlu checkpoints/dense_fineweb_mmlu_overlay_original_text_flat_r24_qwen3.pt \
  --training-free-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --device cuda:0 \
  --query-mode question_only \
  --override model.knowledge_encoder_mode=qwen3
```

结果默认写到 `results/e7/`，例如：

- `results/e7/e7_dense_original_text_phase3_best.json`
- `results/e7/e7_dense_k256_phase3_best.json`
- `results/e7/e7_dense_retrieval_precheck_original_text.json`
- `results/e7/e7_dense_retrieval_precheck_k256.json`
