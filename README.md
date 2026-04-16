# Explicit-Lora

为大语言模型构建内置持续学习模块，使模型具备显式的动态知识更新能力。

核心思路：将知识存储在外部向量索引中，通过 dense retrieval 检索，再通过 forward hook 将知识注入冻结的 LLM，规避参数重训的算力成本与遗忘风险。

**基座模型**：Qwen3-0.6B（冻结，不更新参数）

---

## 架构

### 端到端数据流

```
用户问题
  ↓ tokenize (fusion_length)
DenseRetriever.retrieve_from_texts()
  ↓ KnowledgeEncoder.encode_mean()   ← r0：纯词嵌入均值，无 Transformer 层
FAISS Flat 精确搜索 top-k
  ↓ fusion_ids [1, fusion_length]
ModifiedQwen forward hook（层 6 / 12 / 18 / 24）
  ↓ AttentionInjection cross-attention + zero-init 残差
最终 logits → 答案
```

### 核心模块

| 模块 | 类 | 职责 |
|------|-----|------|
| `retrieval/dense_index.py` | `DenseKnowledgeIndex` | Dense 检索索引（FAISS Flat/HNSW 后端） |
| `training/dense_retriever.py` | `DenseRetriever` | 编码 query → 检索 → 返回 fusion_ids |
| `models/qwen_wrapper.py` | `KnowledgeEncoder` | Qwen3 前 N 层 + 词嵌入均值编码知识 |
| `models/modified_model.py` | `ModifiedQwen` | 冻结 Qwen3 + hook 注入 |
| `models/injection_modules.py` | `AttentionInjection` | Cross-Attention + Null KV + zero-init 残差 |

### 知识编码器

- **`encoder_depth=0`（r0，当前主线）**：只取 embedding 层均值，无 Transformer 层
- **`encoder_depth=N`（rN）**：前 N 层 Qwen3 双向注意力，质量更高但速度慢
- 当前默认：`knowledge_encoder_mode=qwen3`，`retrieval_encoder_depth=0`

### 三层训练管线

| 阶段 | 脚本 | 说明 |
|------|------|------|
| Phase 2 | `scripts/run_phase2_fusion.sh` | 训练 KnowledgeEncoder + AttentionInjection（FineWeb-Edu 上预训练） |
| Phase 3 | `scripts/run_phase3_sft.sh` | 在 MedQA 上 SFT，接 dense index 作为知识源 |

---

## 数据

### 知识数据文件（`data/` 目录）

所有知识文件均为 JSONL 格式，每行：

```json
{"key": "<问题文本前200字>", "knowledge_ids": [token_id, ...], "text": "原文"}
```

`knowledge_ids` 为 LLMLingua 压缩后的 token ID 列表，长度等于对应 k-size。

| 文件 | 数据集 | fusion_length | 说明 |
|------|--------|---------------|------|
| `medqa_knowledge.jsonl` | MedQA | 64 | k64 默认（无后缀） |
| `medqa_knowledge_k32.jsonl` | MedQA | 32 | |
| `medqa_knowledge_k128.jsonl` | MedQA | 128 | |
| `medqa_knowledge_k256.jsonl` | MedQA | 256 | |
| `arc_knowledge.jsonl` | ARC | 64 | |
| `arc_knowledge_k32/128/256.jsonl` | ARC | 32/128/256 | |
| `mmlu_knowledge.jsonl` | MMLU | 64 | |
| `mmlu_knowledge_k32/128/256.jsonl` | MMLU | 32/128/256 | |
| `medqa_knowledge_train.jsonl` | MedQA | — | 训练集分割 |
| `medqa_knowledge_validation.jsonl` | MedQA | — | 验证集分割 |
| `medqa_knowledge_counterfactual.jsonl` | MedQA | — | 反事实知识（E1 实验用） |

> 代码中统一按 `fusion_length` 自动选路：`fusion_length==64` → `{ds}_knowledge.jsonl`，否则 → `{ds}_knowledge_k{N}.jsonl`

### Dense 索引文件（`checkpoints/` 目录）

命名规则：`dense_fineweb_{dataset}_overlay_k{K_SIZE}_flat_r{depth}_{enc_mode}.pt`

| 文件 | 说明 |
|------|------|
| `dense_fineweb_1m_flat_r0_qwen3_fv.pt` | FineWeb 1M base 索引 |
| `dense_fineweb_medqa_overlay_k64_flat_r0_qwen3.pt` | MedQA overlay（k64） |
| `dense_fineweb_arc_overlay_k64_flat_r0_qwen3.pt` | ARC overlay（k64） |
| `dense_fineweb_mmlu_overlay_k64_flat_r0_qwen3.pt` | MMLU overlay（k64） |

每个 `.pt` 索引包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `embeddings` | `[N, hidden_dim]` | 每条知识的向量表示（float32） |
| `fusion_ids` | `[N, fusion_length]` | LLMLingua 压缩后的 token IDs（注入用） |
| `keys` | `List[str]` | 知识条目唯一标识 |
| `texts` | `List[str]` | 解码后可读文本（调试用） |

### Overlay 机制

```
FineWeb 1M base index（1,048,576 条）
        ↓  overlay_dense_index.py（seed=42 随机替换）
Task overlay index（总大小不变）
= N 条任务知识 + (1M - N) 条 FineWeb 背景
```

构建 overlay 时：
- 使用 `--input` 传入 k-specific knowledge 文件（单视图模式）
- 知识文件中的 `knowledge_ids` 直接用作 `fusion_ids`，同时解码为文本用于 embedding 编码

---

## 环境

```bash
# Conda 环境
conda activate ExplicitLLM   # Python 3.11

# GPU 约束：只使用 GPU 6
export CUDA_VISIBLE_DEVICES=6

# 推荐执行方式
conda run --no-capture-output -n ExplicitLLM python xxx
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n ExplicitLLM python xxx
```

关键环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_PATH` | `config/default.yaml` 中的 `paths.model_dir` | Qwen3-0.6B 路径 |
| `GPU_IDS` | `6` | 物理 GPU 编号 |
| `DEVICE` | `cuda:0` | PyTorch 设备（配合 `CUDA_VISIBLE_DEVICES` 使用） |
| `ENC_MODE` | `qwen3` | 编码模式：`qwen3` 或 `trainable` |
| `K_SIZE` | `64` | fusion_length，控制知识压缩长度 |

---

## 训练

### Phase 2 Fusion

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
ENC_MODE=qwen3 \
NUM_GPUS=1 GPU_IDS=6 \
bash scripts/run_phase2_fusion.sh
```

### Phase 3 SFT

```bash
MODEL_PATH=/path/to/Qwen3-0.6B \
ENC_MODE=qwen3 \
FROM_PHASE2=checkpoints/p2_qwen3_10ep/phase2_best \
NUM_GPUS=1 GPU_IDS=6 \
bash scripts/run_phase3_sft.sh
```

---

## Dense 索引构建

### 1. 构建 FineWeb 1M base 索引

```bash
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n ExplicitLLM \
python scripts/build_dense_index_from_fineweb.py \
  --config config/default.yaml \
  --parquet-dir data/compressed/v2 \
  --output checkpoints/dense_fineweb_1m_flat_r0_qwen3_fv.pt \
  --sample-size 1048576 \
  --seed 0 --device cuda:0 \
  --batch-size 256 --index-type flat \
  --override model.fusion_length=64
```

### 2. 构建任务 overlay 索引

```bash
# MedQA k64
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n ExplicitLLM \
python scripts/overlay_dense_index.py \
  --config config/default.yaml \
  --index checkpoints/dense_fineweb_1m_flat_r0_qwen3_fv.pt \
  --input data/medqa_knowledge.jsonl \
  --output checkpoints/dense_fineweb_medqa_overlay_k64_flat_r0_qwen3.pt \
  --device cuda:0 --batch-size 256 --seed 42 \
  --override model.fusion_length=64
```

其他数据集（arc / mmlu）和 k-size（32 / 128 / 256）类似，修改 `--input`、`--output` 和 `fusion_length` 即可。

### 3. 一键全流程（E7-full）

```bash
# 构建 base + 三份 overlay + retrieval precheck + E7 评测
GPU_IDS=6 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash experiments/e7/run_dense_full.sh

# 跳过 base 和 overlay 构建，只跑 precheck + E7
BUILD_BASE=0 BUILD_OVERLAYS=0 \
GPU_IDS=6 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash experiments/e7/run_dense_full.sh

# 指定 k-size
K_SIZE=128 BUILD_BASE=0 BUILD_OVERLAYS=0 \
GPU_IDS=6 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
bash experiments/e7/run_dense_full.sh
```

---

## 实验评测

### 实验矩阵

| 实验 | 内容 |
|------|------|
| E1 | 反事实知识注入——测试模型是否真的使用注入知识 |
| E2 | MedQA / ARC / MMLU 跨域 QA 评测 |
| E3 | RAG 基线对比（BM25 / E5 / dense） |
| E3-multik | 多 k-size 消融（32 / 64 / 128 / 256） |
| E4 | 知识注入深度消融（注入层数） |
| E5 | 检索器质量评测（E5 vs dense） |
| E6 | 推理效率评测（延迟 / 吞吐） |
| E7 | Dense retrieval 全面评测（单视图，r0，qwen3） |
| E8 | Editable Memory Benchmark（upsert / delete / rollback） |

### 快速运行单个实验

```bash
bash scripts/run_experiment_auto.sh <e1|e2|e3|e3_multik|e4|e5|e6|e7|e7_full>
```

### E7：只跑评测（索引已就绪）

```bash
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=6 \
bash experiments/e7/run.sh
```

使用非默认 k-size：

```bash
K_SIZE=128 \
TRAINING_FREE_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=6 \
bash experiments/e7/run.sh
```

### E7 Retrieval Precheck

检验 overlay 索引的检索正确率（key 匹配）：

```bash
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n ExplicitLLM \
python experiments/e7/eval_dense_retrieval.py \
  --config config/default.yaml \
  --dense-index-medqa checkpoints/dense_fineweb_medqa_overlay_k64_flat_r0_qwen3.pt \
  --dense-index-arc   checkpoints/dense_fineweb_arc_overlay_k64_flat_r0_qwen3.pt \
  --dense-index-mmlu  checkpoints/dense_fineweb_mmlu_overlay_k64_flat_r0_qwen3.pt \
  --device cuda:0 --top-k 16 --query-mode question_only \
  --output results/e7/e7_dense_retrieval_precheck_k64_r0.json \
  --override model.fusion_length=64
```

### E8 Editable Memory Benchmark

```bash
# e8a: 单条 upsert
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=6 \
bash scripts/run_e8.sh e8a

# e8b: delete + rollback
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=6 N_EDITS=100 \
bash scripts/run_e8.sh e8b

# e8c: 顺序编辑（在线 upsert/delete/rollback 序列）
MEMORY_SETTING=overlay_1m \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
STEPS=1,2,3,10,11,12,100,101,102 LOCALITY_SAMPLES=200 \
bash scripts/run_e8.sh e8c

# e8d_a: 批量 ingest
K_SIZE=128 MEMORY_SETTING=overlay_1m \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
N_EDITS=100 LOCALITY_SAMPLES=200 \
bash scripts/run_e8.sh e8d_a
```

E8 的 `MEMORY_SETTING` 选项：

| 值 | 说明 |
|----|------|
| `controlled` | 使用预构建的 task overlay 索引（默认） |
| `overlay_1m` | 运行时从 FineWeb base 动态构建 MedQA overlay |

### 汇总所有实验结果

```bash
conda run --no-capture-output -n ExplicitLLM python scripts/collect_results.py
# 输出：results/results_summary.md
```

---

## 推理

### Dense 检索 + Phase3 单样本推理

```bash
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n ExplicitLLM \
python scripts/run_dense_phase3_infer.py \
  --config config/default.yaml \
  --dense-index checkpoints/dense_fineweb_medqa_overlay_k64_flat_r0_qwen3.pt \
  --phase3-ckpt checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --device cuda:0 --query-mode question_only \
  --question "A 65-year-old man is brought to the emergency department 30 minutes after the onset of acute chest pain." \
  --option-a "Aspirin only" \
  --option-b "Clopidogrel" \
  --option-c "Warfarin" \
  --option-d "Amiodarone" \
  --json
```

### 单条知识 overlay 推理（在线压缩）

```bash
CUDA_VISIBLE_DEVICES=6 conda run --no-capture-output -n ExplicitLLM \
python scripts/run_single_knowledge_overlay_answer.py \
  --config config/default.yaml \
  --base-index checkpoints/dense_fineweb_medqa_overlay_k64_flat_r0_qwen3.pt \
  --phase3-ckpt checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
  --dataset medqa \
  --question "..." \
  --option-a "..." --option-b "..." --option-c "..." --option-d "..." \
  --json
```

---

## 配置

主配置文件：`config/default.yaml`

关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.fusion_length` | `64` | 知识压缩后 token 数（K_SIZE） |
| `model.retrieval_encoder_depth` | `0` | 检索编码器深度（r0=纯词嵌入） |
| `model.knowledge_encoder_mode` | `qwen3` | 编码模式 |
| `model.hidden_dim` | `1024` | 知识向量维度 |
| `train.bf16` | `true` | 混合精度训练 |

CLI override 方式：`--override model.fusion_length=128`

---

## 代码质量

```bash
# 格式化
conda run --no-capture-output -n ExplicitLLM ruff format <file>

# Lint
conda run --no-capture-output -n ExplicitLLM ruff check <file> --fix

# 单元测试
conda run --no-capture-output -n ExplicitLLM pytest tests/unit/ --cov=models --cov-report=term-missing

# 集成测试
conda run --no-capture-output -n ExplicitLLM pytest tests/integration/
```
