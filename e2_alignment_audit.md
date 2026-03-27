# E2 对拍记录与模型差异说明

更新时间：2026-03-27

## 1. 已对拍过的部分

### 1.1 评测 prompt 与打分方式

- 当前版 E2 的多选题 prompt 构造与 Reference 一致。
- 当前版 E2 的打分方式与 Reference 一致：
  - 对 `" A" / " B" / " C" / " D"` 四个 continuation 分别做 forward
  - 取 continuation token 的 log-softmax
  - 累加 log-likelihood
  - 选择分数最高的选项

对应实现：
- 当前版：[experiments/e2/scoring.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2/scoring.py)
- Reference：[Reference/Explicit-Lora-fusion/evaluation/compare_eval.py](/home/undergraduate/zcy/Explicit-Lora/Reference/Explicit-Lora-fusion/evaluation/compare_eval.py)
- Reference：[Reference/Explicit-Lora-fusion/evaluation/counterfactual_eval.py](/home/undergraduate/zcy/Explicit-Lora/Reference/Explicit-Lora-fusion/evaluation/counterfactual_eval.py)

已有测试：
- [tests/unit/test_experiments_e2.py](/home/undergraduate/zcy/Explicit-Lora/tests/unit/test_experiments_e2.py)

结论：
- `scoring` 不是当前结果偏低的主要原因。

### 1.2 数据字段归一化与样本过滤

- MedQA：当前版 `sent1/ending0-3/label` 映射到统一 row 结构，对齐 Reference 语义。
- ARC：当前版会在加载阶段过滤非 4 选项样本，Reference 多在评测阶段跳过，语义接近，但 `skipped` 统计可能不同。
- MMLU：当前版对 4 选项样本的字段映射与 Reference 对齐。

对应实现：
- 当前版：[experiments/e2/medqa_eval.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2/medqa_eval.py)
- 当前版：[experiments/e2/arc_eval.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2/arc_eval.py)
- 当前版：[experiments/e2/mmlu_eval.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2/mmlu_eval.py)

结论：
- 数据字段映射基本不是当前主要差异来源。
- ARC 的 `skipped` 统计口径可能略有不同。

### 1.3 knowledge map 载入

- 已对拍当前版 `load_knowledge_map()` 与 `e2-ref` / Reference 的知识加载逻辑。
- 三份知识文件：
  - `data/medqa_knowledge.jsonl`
  - `data/arc_knowledge.jsonl`
  - `data/mmlu_knowledge.jsonl`
- 对拍结果：
  - key 数一致
  - key 集合一致
  - 每个 key 对应的 `knowledge_ids` 一致

对应实现：
- 当前版：[experiments/e2/common.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2/common.py)
- e2-ref：[experiments/e2-ref/medqa_knowledge_builder.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2-ref/medqa_knowledge_builder.py)
- Reference：[Reference/Explicit-Lora-fusion/evaluation/medqa_knowledge_builder.py](/home/undergraduate/zcy/Explicit-Lora/Reference/Explicit-Lora-fusion/evaluation/medqa_knowledge_builder.py)

结论：
- `knowledge_map` 本身是一致的。

### 1.4 knowledge_tensor

- 已用代码实际对拍当前版 `prepare_knowledge_tensor()` 与 Reference 语义下的 `torch.tensor([knowledge_ids])`。
- 抽样覆盖 MedQA / ARC / MMLU 三个数据集的前 5 条和后 5 条。
- 结果：
  - `shape` 一致：`(1, 64)`
  - `dtype` 一致：`torch.int64`
  - 数值一致：`same=True`
  - 三个数据集都得到 `sampled_all_equal=True`

对应实现：
- 当前版：[experiments/e2/common.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2/common.py)
- Reference：[Reference/Explicit-Lora-fusion/evaluation/explicit_lm.py](/home/undergraduate/zcy/Explicit-Lora/Reference/Explicit-Lora-fusion/evaluation/explicit_lm.py)

结论：
- `knowledge_tensor` 不是差异来源。

### 1.5 knowledge_mask

- 已用代码对拍“送进模型前”的 `knowledge_mask`。
- 比较方式：
  - 当前版：`(prepare_knowledge_tensor(ids, K, pad) != pad).long()`
  - Reference 语义：`(torch.tensor([ids]) != pad).long()`
- 结果：
  - MedQA / ARC / MMLU 抽样全相同
  - `Fusion+empty` 的空知识也相同
  - `empty_mask_equal=True`
  - `empty_valid=0`

结论：
- `knowledge_mask` 不是差异来源。

### 1.6 Fusion+empty 的输入知识

- 当前版与 Reference 版在 `Fusion+empty` 下都使用全 `pad_token_id` 的知识张量。
- 所以：
  - `knowledge_tensor` 相同
  - `knowledge_mask` 相同

结论：
- 如果 `Fusion+empty` 结果不同，原因不在空知识输入本身，而在后续编码与融合实现。

### 1.7 knowledge_encoder 输出

- 已用同一条真实 `knowledge_ids` 对拍：
  - 当前版 `KnowledgeEncoder.forward()`
  - Reference `QwenWrapper.encode_knowledge()`
- 实测结果：
  - `shape` 相同：`(1, 64, 1024)`
  - `dtype` 相同：`torch.bfloat16`
  - 但数值明显不同
  - `max_abs = 50.125`
  - `mean_abs = 3.007232904434204`
  - `allclose(atol=1e-4) = False`
  - `allclose(atol=1e-2) = False`

对应实现：
- 当前版：[models/qwen_wrapper.py](/home/undergraduate/zcy/Explicit-Lora/models/qwen_wrapper.py)
- Reference：[Reference/Explicit-Lora-fusion/models/qwen_wrapper.py](/home/undergraduate/zcy/Explicit-Lora/Reference/Explicit-Lora-fusion/models/qwen_wrapper.py)

结论：
- 真正开始分叉的是 `knowledge_encoder` 本身。

### 1.8 单题注入链路检查

- 已写最小调试脚本：
  - [experiments/e2/debug_single_case.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2/debug_single_case.py)
- 该脚本可直接打印：
  - `knowledge_ids`
  - `knowledge_mask`
  - `Fusion+knowledge` 与 `Fusion+empty` 下的注入层触发情况
  - `baseline / fusion+knowledge / fusion+empty` 四选项分数

已经确认：
- 注入层确实按配置触发
- 当前默认层为 `[6, 12, 18, 24]`
- `Fusion+knowledge` 和 `Fusion+empty` 都会走注入结构
- `Fusion+empty` 使用全 pad 知识，属于预期消融语义

### 1.9 模型权重载入

- 当前版 E2 现在使用当前仓库 `models/` 下的结构，并载入完整 checkpoint 结构，而不再只加载 `injection_modules.pt`。
- 现在会依次加载：
  - `injection_modules.pt`
  - `encoder_layers.pt`
  - `encoder_norm.pt`
- 当前实现会打印：
  - `[E2Load] loaded ...`

对应实现：
- [experiments/e2/common.py](/home/undergraduate/zcy/Explicit-Lora/experiments/e2/common.py)

结论：
- 当前 E2 评测的是“当前完整模型”的能力，不再是早期 Reference-compatible 的“只恢复注入模块”模式。

## 2. 当前 checkpoint 结构检查

已检查：
- [checkpoints/phase2_best](/home/undergraduate/zcy/Explicit-Lora/checkpoints/phase2_best)
- [checkpoints/phase3_best](/home/undergraduate/zcy/Explicit-Lora/checkpoints/phase3_best)

两者结论一致：
- 都有：
  - `injection_modules.pt`
  - `encoder_layers.pt`
  - `encoder_norm.pt`
- `injection_modules.pt` 都表现为 4 个模块前缀：
  - `0`
  - `1`
  - `2`
  - `3`
- 没有 `knowledge_adapter` 权重

结论：
- `phase2_best` 和 `phase3_best` 都是当前主线的：
  - 4 层注入 `[6,12,18,24]`
  - 无 adapter

## 3. 当前模型结构与 Reference 的差异

### 3.1 KnowledgeEncoder 实现不同

当前版：
- 使用独立的 `KnowledgeEncoder` 类
- 共享 `embed_tokens`
- 共享前 `encoder_depth` 层
- 独立深拷贝一份 `norm`
- 显式接收 `attention_mask`
- 构造双向 attention bias

Reference：
- 没有独立 `KnowledgeEncoder` 类
- 使用 `QwenWrapper.encode_knowledge()`
- 直接走前 `encoder_depth` 层
- 最后直接使用 base model 自带的 `norm`
- 不走当前版这套显式 mask/bias 路径

这是当前 `knowledge_encoder` 输出不一致的直接原因。

### 3.2 模型组装方式不同

当前版：
- 外部显式组装 `base_model + knowledge_encoder + injection_modules + ModifiedQwen`

Reference：
- 更偏一体化封装
- 通过 `QwenWrapper` / `create_model()` 风格创建

### 3.3 注入模块组织方式不同

当前版：
- `injection_modules` 用 `nn.ModuleList`

Reference：
- 原始实现更偏 `ModuleDict` / old factory 风格

这会影响 `state_dict` key 形式和旧 checkpoint 兼容逻辑。

### 3.4 AttentionInjection 内部实现不同

当前版：
- 自定义 `W_q / W_k / W_v / out_proj`
- 使用 `scaled_dot_product_attention`
- mask 约定：`1=有效, 0=padding`

Reference：
- 更偏 `nn.MultiheadAttention`
- mask 约定和传法与当前版不同

高层语义类似，但内部实现并非同一个模块。

### 3.5 权重保存/恢复格式不同

当前主线 checkpoint：
- `injection_modules.pt`
- `encoder_layers.pt`
- `encoder_norm.pt`

Reference 历史实现：
- 更偏 `injection_modules.safetensors`
- 可选 `knowledge_adapter`
- 配套 `injection_config.json`

### 3.6 前向接口不同

当前版：
- `ModifiedQwen.forward(...)`
- 更偏训练主线接口
- 支持 `knowledge_ids`
- 支持更完整的 `attention_mask` / `labels` 等参数

Reference：
- 更偏旧评测接口
- 常见调用为 `model(input_ids, knowledge_ids)`

## 4. 目前最稳的结论

以下部分已经基本排除为结果差异来源：
- 多选题 prompt 构造
- 评分算法
- `knowledge_map`
- `knowledge_tensor`
- `knowledge_mask`
- `Fusion+empty` 的空知识输入

目前最可疑、且已实测确实不同的部分是：
- `knowledge_encoder` 的实现
- 当前主线模型结构与 Reference 结构的差异
- checkpoint 所属代次与阶段语义差异
- 当前 E2 现在评测的是“完整本地模型”，不再是旧的 Reference-compatible 子图

## 5. 阶段语义提醒

当前仓库中：
- `phase1_best` 是 Router，不是旧文档里的 Fusion Phase 1
- `phase2_best` 最接近旧文档里的 Phase 1 Fusion
- `phase3_best` 最接近旧文档里的 Phase 2 SFT

可参考：
- [phase_mapping_reference.txt](/home/undergraduate/zcy/Explicit-Lora/phase_mapping_reference.txt)

## 6. 当前判断

如果当前 E2 分数明显低于历史 plan：
- 不能再把原因归到知识载入或 scoring
- 更应该优先检查：
  - `knowledge_encoder` 差异
  - 是否在用正确阶段的 checkpoint
  - 当前训练轮数是否足够
  - 当前完整模型与历史 Reference 体系是否本来就不是同一条结构线
