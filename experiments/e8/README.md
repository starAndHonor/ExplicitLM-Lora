# E8 Editable Memory Benchmark

`E8` 用来验证当前 `dense retrieval + external memory` 主线是否真的具备：

- 写入 `UPSERT`
- 删除 `DELETE`
- 回滚 `ROLLBACK`
- 连续编辑 `SEQUENTIAL EDITS`

对应 claim：

- `C4 — 记忆可编辑`

## 核心原则

`E8` 只测试 **外部记忆库编辑能力**，不测试参数编辑能力。

因此实验中固定两样东西：

- 固定 `Phase3` 权重
- 固定 query / evaluation protocol

只允许变化：

- dense memory store
- dense index 内容

换句话说，`E8` 的目标不是“重新训练后能不能学会新知识”，而是：

> 在不改模型参数的前提下，仅通过编辑 memory store，系统能否正确写入、删除、恢复知识。

## Memory Settings

`E8` 现在支持两种 memory setting：

- `controlled`
  - 直接使用一个已经准备好的 task full index
  - 适合做干净、低噪声的编辑因果分析
- `overlay_1m`
  - 从 `FineWeb 1M base index` 出发，临时 overlay 一份 `MedQA` full index
  - 适合做更贴近真实部署的 realistic setting

其中 `overlay_1m` 会显式记录：

- `base_index_path`
- `anchor_variant`
- `overlay_seed`
- `overlay_deleted`

因此后续可以稳定复现实验，而不是只把结果解释成“某个手工准备好的 full index”。

## 检索与编辑对象

当前主线里每条知识有两种视图：

- 检索视图：
  - `anchor text` / `knowledge text`
  - 用于编码成 dense embedding
- 注入视图：
  - `knowledge_ids` / `fusion_ids`
  - 用于下游 `Phase3` 注入

因此 `E8` 的编辑单位定义为：

- 一个 knowledge entry
  - `key`
  - `anchor text`
  - `fusion ids`

## E8a: Knowledge Write / UPSERT

### 目标

验证知识从“不在库中”到“写入库后”是否能立刻生效。

### 数据构造

以 `MedQA` 为第一版实验对象：

1. 从完整 `MedQA` 知识库中抽取 `N` 条条目作为 `edit set`
2. 从 base memory 中移除这些条目，得到 `base_missing index`
3. 在 `base_missing index` 上评测对应问题
4. 把这些知识逐条或逐批 `upsert` 回去
5. 再次评测同一批问题

### 指标

- `pre_write_acc`
- `post_write_acc`
- `write_success_rate`
  - 写入后从错误变为正确的比例
- `write_latency_ms`
  - 单条写入耗时
- `retrieval_top1_before`
- `retrieval_top1_after`

### Baseline

- `Dense RAG`
  - 同样更新知识库，但走 prompt 拼接
- `Base model`
  - 不接知识库

## E8b: Knowledge Delete / Rollback

### 目标

验证删除知识后系统是否回退，回滚后是否恢复。

### 流程

对每个目标知识条目 `K`：

1. `full index`
   - 包含 `K`
2. `delete(K)`
   - 得到 `deleted index`
3. `rollback(K)`
   - 得到 `restored index`

然后对对应问题 `Q` 分别评测三次。

### 指标

- `delete_success_rate`
  - 删除后不再命中 / 不再正确回答的比例
- `rollback_fidelity`
  - 回滚后恢复到删除前表现的比例
- `retrieval_drop_after_delete`
- `qa_drop_after_delete`
- `qa_recovery_after_rollback`

### 解释

这里要同时看两层：

- 检索层
  - 被删知识是否真的不再被检到
- 任务层
  - 最终答案是否失去该知识支持

原因是删除知识后，模型可能依靠参数先验仍答对。

## E8c: Sequential Edits

### 目标

验证连续编辑后的稳定性与可扩展性。

### 流程

编辑步数：

- `1, 2, 3`
- `10, 11, 12`
- `100, 101, 102`

推荐按三元组取样，因为当前在线操作流是：

- step `3n+1`: `upsert`
- step `3n+2`: `delete`
- step `3n+3`: `rollback`

这样每一组 step 都能显式覆盖三种状态，而不是只观察到 present / rollback。

在每个 edit budget 下执行：

- 同一个 index 上顺序执行一系列 `upsert` / `delete` / `rollback`
- 然后在固定评测集上测试

建议拆成两类评测题：

- `edited queries`
  - 对应刚编辑过的知识
- `locality queries`
  - 与编辑无关的旧知识问题

### 指标

- `edit_success_rate`
- `delete_success_rate`
- `rollback_fidelity`
- `locality_retention`
- `retrieval_latency_ms`
- `index_size`
- `active_entries`
- `tombstone_ratio`
- `rebuild_count`

当前实现使用真实在线操作流：

- `upsert -> delete -> rollback`

并且这些操作会持续累积在同一个 `DenseKnowledgeIndex` 上，而不是每个 step 从同一个初始状态重新构造。

## 一键运行全套 E8

可以直接使用：

```bash
BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r0_qwen3.pt \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=2 \
MEMORY_SETTING=overlay_1m \
ANCHOR_VARIANT=k256 \
bash scripts/run_e8_full.sh \
  --override model.retrieval_encoder_depth=0 \
  --override model.knowledge_encoder_mode=qwen3
```

默认顺序：

- `e8a`
- `e8b`
- `e8c`
- `e8d_a`
- `e8d_b`

如果只想跑其中一部分，可以通过环境变量 `EXPERIMENTS` 指定，例如：

```bash
EXPERIMENTS=e8a,e8b,e8d_a \
BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r0_qwen3.pt \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=2 \
MEMORY_SETTING=overlay_1m \
ANCHOR_VARIANT=k256 \
bash scripts/run_e8_full.sh \
  --override model.retrieval_encoder_depth=0 \
  --override model.knowledge_encoder_mode=qwen3
```

## E8d-A: 批量灌入后旧知识保持

### 目标

验证一次性批量灌入一批新知识后：

- 新知识是否立刻生效
- 旧知识是否保持稳定

### 流程

1. 从 full index 中拿掉一批 `ingest set`
2. 在 base-missing 状态下评测：
   - `ingest set`
   - `old/locality set`
3. 将 `ingest set` 一次性批量写回
4. 再次评测新知识恢复和旧知识保持

### 指标

- `ingest_qa_acc_before / after`
- `ingest_retrieval_top1_before / after`
- `old_qa_retention`
- `old_retrieval_retention`
- `bulk_ingest_latency_ms`

## E8d-B: 增量 Add/Delete 后性能

### 目标

验证小批次持续热更新时：

- 新增知识是否逐步生效
- 删除知识后性能是否按预期下降
- 不相关旧知识是否保持稳定

### 流程

1. 从 full index 中先移除一批 `add set`
2. 以小批次持续 add 回去
3. 再对另一批 `delete set` 以小批次持续 delete
4. 在每个阶段都记录：
   - 新增集合表现
   - 删除集合表现
   - locality 集合保持率
   - index 大小和 tombstone

### 指标

- `add_qa_acc_before / after`
- `delete_qa_acc_before / after`
- `old_qa_retention_after_updates`
- `mean_add_latency_ms`
- `mean_delete_latency_ms`
- `tombstone_ratio_final`

## 推荐结果文件

统一放到：

- `results/e8/`

建议输出：

- `results/e8/e8a_upsert_<tag>.json`
- `results/e8/e8b_delete_rollback_<tag>.json`
- `results/e8/e8c_sequential_<tag>.json`

## 建议的代码结构

建议最小实现为：

```text
experiments/e8/
  README.md
  __init__.py
  common.py
  run_e8.py
  upsert.py
  delete_rollback.py
  sequential_edits.py
```

职责建议：

- `common.py`
  - 统一加载模型、dense index、评测数据
  - 公共指标计算
- `upsert.py`
  - E8a 主逻辑
- `delete_rollback.py`
  - E8b 主逻辑
- `sequential_edits.py`
  - E8c 主逻辑
- `run_e8.py`
  - CLI 总入口

## JSON 输出建议

### E8a

```json
{
  "experiment": "e8a",
  "dataset": "medqa",
  "n_edits": 100,
  "phase3_weights": "...",
  "base_index": "...",
  "updated_index": "...",
  "metrics": {
    "pre_write_acc": 0.12,
    "post_write_acc": 0.81,
    "write_success_rate": 0.76,
    "mean_write_latency_ms": 8.5,
    "retrieval_top1_before": 0.05,
    "retrieval_top1_after": 0.92
  }
}
```

### E8b

```json
{
  "experiment": "e8b",
  "dataset": "medqa",
  "n_cases": 100,
  "metrics": {
    "delete_success_rate": 0.84,
    "rollback_fidelity": 0.97,
    "retrieval_drop_after_delete": 0.88,
    "qa_drop_after_delete": 0.63,
    "qa_recovery_after_rollback": 0.95
  }
}
```

### E8c

```json
{
  "experiment": "e8c",
  "dataset": "medqa",
  "steps": [1, 10, 100, 1000],
  "series": {
    "edit_success_rate": [1.0, 0.98, 0.95, 0.88],
    "locality_retention": [1.0, 0.99, 0.98, 0.94],
    "retrieval_latency_ms": [3.1, 3.2, 3.6, 5.4],
    "index_size": [1000001, 1000010, 1000100, 1001000]
  }
}
```

## 第一版落地顺序

建议先做最小可跑版本：

## 运行方式

### Controlled setting

```bash
FULL_INDEX=checkpoints/dense_fineweb_medqa_overlay_original_text_flat_r24_qwen3.pt \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=2 \
bash scripts/run_e8.sh e8a
```

### FineWeb 1M + overlay realistic setting

```bash
BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r24_qwen3.pt \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=2 \
MEMORY_SETTING=overlay_1m \
ANCHOR_VARIANT=original_text \
bash scripts/run_e8.sh e8a
```

也可以把 `ANCHOR_VARIANT` 切到：

- `original_text`
- `k256`

### E8d-A

```bash
BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r24_qwen3.pt \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=2 \
MEMORY_SETTING=overlay_1m \
ANCHOR_VARIANT=k256 \
N_EDITS=100 \
LOCALITY_SAMPLES=200 \
bash scripts/run_e8.sh e8d_a
```

### E8d-B

```bash
BASE_INDEX=checkpoints/dense_fineweb_1m_flat_r24_qwen3.pt \
PHASE3_WEIGHTS=checkpoints/p3_from_p2_qwen3_10ep/phase3_best \
GPU_IDS=2 \
MEMORY_SETTING=overlay_1m \
ANCHOR_VARIANT=k256 \
N_EDITS=100 \
UPDATE_BATCH_SIZE=10 \
LOCALITY_SAMPLES=200 \
bash scripts/run_e8.sh e8d_b
```

1. `E8a-min`
   - 数据集：`MedQA`
   - `N=100`
   - 先支持 `UPSERT`
2. `E8b-min`
   - 在同一批 `100` 条上做 `DELETE + ROLLBACK`
3. `E8c-min`
   - 先支持 `1 / 10 / 100`
   - 暂不做 `1000`

当前实现状态：

- `E8a-min`：已实现
- `E8b-min`：已实现
- `E8c-min`：已实现

## 当前已知依赖

实现 `E8` 会直接依赖：

- `retrieval/dense_index.py`
- `training/dense_retriever.py`
- `scripts/overlay_dense_index.py`
- `experiments/e7/comparison.py`
  - 可复用部分推理与评测逻辑

## 设计结论

`E8` 最重要的不是证明“模型学会了可编辑知识”，而是证明：

> 在固定参数下，外部 dense memory 可以被低成本、可逆、可持续地编辑，并被下游问答能力立即利用。
