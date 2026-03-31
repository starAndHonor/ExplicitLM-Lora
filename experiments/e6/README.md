# E6 Inference Efficiency

E6 用来回答一个实际部署问题：

- `E3` 已经证明 `Fusion` 在同等压缩知识下优于 `RAG`
- 但 `RAG-original` 在绝对准确率上仍可能高于 `Fusion Phase 2`
- 所以需要从推理效率角度说明，为什么实际部署时不直接用 `RAG-original`

## 评测内容

单 GPU、单样本推理基准，方法包括：

- `Baseline`
- `RAG-compressed@64`
- `Fusion-Phase2@64`
- `RAG-original@~256`

输出指标：

- `latency_ms`
- `throughput`
- `peak_memory_mb`
- `avg_input_len`
- `context_tokens`

如果存在 `E3` / `E5` 结果，还会自动生成综合六维对比表。

## 运行

默认跑 `N=200`、`warmup=10`：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e6/run_e6.py \
  --config config/default.yaml \
  --phase3-weights checkpoints/phase3_best \
  --device cuda:0
```

如果跑 `qwen3` 版本：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e6/run_e6.py \
  --config config/default.yaml \
  --phase3-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --device cuda:0
```

如果想显式指定 E5 结果文件：

```bash
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n ExplicitLLM \
python experiments/e6/run_e6.py \
  --config config/default.yaml \
  --phase3-weights checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --override model.knowledge_encoder_mode=qwen3 \
  --e5-result results/e5/e5_knowledge_analysis_p2p3_qwen3.json \
  --device cuda:0
```

结果默认保存到 [results/e6/e6_inference_efficiency.json](/home/undergraduate/zcy/Explicit-Lora/results/e6/e6_inference_efficiency.json)。
