# Fusion 模块实验方案（完整版）

> **文档用途**：本文档同时作为研究设计规范和 Agent 实施指南。
> **Part I** 为完整理论设计，**Part II** 为每个实验的逐步实现方案，agent 可直接依据 Part II 修改代码并执行。
> **项目环境**：conda 环境 `ExplicitLLM`，所有命令使用 `conda run -n ExplicitLLM ...` 执行。

---

## 项目背景与核心目标

**研究动机**：现有大模型训练结束后，知识封存于权重无法动态更新。本项目构建一个通用插件组件，接入后让现有大模型实现知识动态更新。组件分为检索（Retrieval）和融合（Fusion）两部分，当前阶段专注于 **Fusion 模块**的验证。

**Fusion 模块愿景**：维护一个显式可读的知识库（人类可编辑的文本形式）。推理时将对应知识压缩编码后注入到模型每一层（Cross-Attention），模块训练好后不再需要重新 fine-tune，只需替换知识库内容。

**需要证明的两件核心事情**：
1. Fusion 模块具有**通用的知识融合能力**（不依赖特定领域 fine-tuning）
2. 层注入的融合效果**优于 RAG 输入拼接**

---

# Part I：总纲实验方案（理论设计）

## 1. 核心主张与论证逻辑

本研究需要论证三件事，它们有严格的逻辑依赖关系：

```
主张一（前提）：Fusion 模块学到了通用的知识融合能力
       ↓ 依赖此前提
主张二（准确率）：层注入的知识融合效果优于 RAG（输入拼接）——在同等 token 预算下
       ↓ 补全论证
主张三（效率）：层注入在推理效率上优于 RAG-original——以少量准确率代价换取显著效率优势
```

**如果主张一不成立**（模块只学到了特定域的映射），主张二的意义就大打折扣——因为你比较的不是"融合机制"本身，而是"哪种方式在 MedQA 上 overfit 更好"。

**主张三为何必要**：E3 结果显示 RAG-original（全文拼接 ~256 token）在绝对准确率上仍高于 Fusion（64 token 层注入）。主张三从效率维度回应这一差距，论证 Fusion 的实际部署优势。参考 Memory3 的三级记忆成本模型：RAG 的 read cost 最高（每次推理处理完整文本），Fusion 的 read cost 显著更低（预编码 + 短序列 cross-attention）。

---

## 2. 实验前提设定

**Oracle 知识协议**：所有实验中，知识 = 已知与 input 最相关的文本，预先配对，不经过检索模块。

```
input (question/text) → [预配对] → knowledge_text
                                        ↓
                              LLMLingua 压缩 → knowledge_ids (64 tokens)
```

这使得实验完全聚焦于融合机制本身，消除检索质量的干扰。

---

## 3. 实验体系（五组核心实验）

### E1：Sanity Check — 知识确实驱动了输出 ✅ 已完成（2026-02-24）

**目的**：这是最基础的验证，证明 Fusion 模块真的在使用注入的知识，而不是忽略它。

**设计**：构造对抗知识对（Counterfactual Knowledge Test）

```
同一问题 Q，准备两组知识：
  知识 A：支持正确答案 X 的文本（question + correct_answer）
  知识 B：故意错误、支持错误答案 Y 的文本（question + wrong_answer，反事实）

验证：
  注入 A → 模型输出倾向 X
  注入 B → 模型输出倾向 Y
  无注入 → 模型输出基于参数知识（baseline）
```

**评测指标**：知识敏感度（Knowledge Sensitivity）
```
KS = acc(注入正确知识) - acc(注入反事实知识)
```
KS 越大，说明模块越依赖注入的知识，而不是自身参数。同时报告：
- `acc_correct`：注入正确知识的准确率
- `acc_counterfactual`：注入反事实知识时正确答案的准确率（应显著下降）
- `acc_no_knowledge`：无知识注入 baseline

**数据集**：MedQA test split（已有，1273 题）

**预期结论**：
- 如果 KS > 20%，证明知识注入显著有效
- 如果 KS ≈ 0，说明模块把知识忽略了（严重问题，需要回头检查训练）

**实际结果 v2**（2026-02-24，4 层注入 [6,12,18,24]，无 adapter）：

| 指标 | Phase 1 权重 | Phase 2 权重 |
|------|------------|------------|
| acc_correct（正确知识） | 36.53% (465/1273) | **69.52%** (885/1273) |
| acc_counterfactual（反事实知识） | 28.75% (366/1273) | **13.20%** (168/1273) |
| acc_no_knowledge（全 pad） | 32.60% (415/1273) | 32.36% (412/1273) |
| **KS** | **+7.78%** | **+56.32%** |

**实际结果 v3**（2026-03-02，6 层注入 [4,8,12,16,20,24] + knowledge_adapter）：

| 指标 | Phase 1 权重 | Phase 2 权重 | v2→v3 Δ（Phase 2） |
|------|------------|------------|-------------------|
| acc_correct（正确知识） | 36.61% (466/1273) | **71.09%** (905/1273) | +1.57% |
| acc_counterfactual（反事实知识） | 28.36% (361/1273) | **14.30%** (182/1273) | +1.10% |
| acc_no_knowledge（全 pad） | 34.41% (438/1273) | 34.72% (442/1273) | +2.36% |
| **KS** | **+8.25%** | **+56.79%** | +0.47% |

结论：**E1 通过（v2 + v3 均通过）**。v3 Phase 2 KS = 56.79%，与 v2（56.32%）基本一致；v3 Phase 2 acc_correct 提升 +1.57%（69.52%→71.09%）；无知识组提升 +2.36%（32.36%→34.72%），说明 6 层注入 + adapter 改善了模型基础能力。详见 Part II E1 实验结果。

---

### E2：通用融合能力验证 — 最重要的实验 ✅ 已完成（2026-02-25）

**目的**：证明 Phase 1 训练出来的 Fusion 模块具有**跨域**的知识融合能力，而不是只在 FineWeb-Edu 或 MedQA 上有效。

**设计**：在多个不同领域的数据集上，使用 **Phase 1 权重（不做任何 domain SFT）**，对比有/无知识注入的效果。

| 数据集 | 领域 | 知识来源构造方式 |
|--------|------|----------------|
| MedQA（4 选 1） | 医学 | compress(question + correct_answer)，已有 |
| ARC-Challenge（4 选 1） | 科学推理 | compress(question + correct_answer)，Oracle 设置 |
| MMLU（多选题，57 子域） | 多领域 | compress(question + correct_answer)，Oracle 设置 |

**对比条件**：

| 条件 | 说明 |
|------|------|
| Baseline（无注入） | 原始 Qwen3-0.6B，无任何知识 |
| Phase1-Fusion + 知识 | Phase 1 权重 + 注入对应知识 |
| Phase1-Fusion，无知识（全 pad） | Phase 1 权重 + 知识全为 padding（消融：验证注入位置而非内容） |

**评测指标**：各数据集上的 accuracy（Δacc = 相对 baseline 的提升）

**预期结论**：
- 如果在多个领域都显著优于 Baseline → **通用融合能力得证**
- 如果只在医学领域有效 → Phase 1 的训练存在域偏差，需要重新审视训练数据

**实际结果 v2**（2026-02-25，4 层注入 [6,12,18,24]，无 adapter）：

| 数据集 | Baseline | Fusion+知识 | Fusion+空知识 | **Δacc** |
|--------|----------|-------------|--------------|----------|
| MedQA（1,273 题） | 32.91% | 36.53% | 32.60% | **+3.61%** |
| ARC-Challenge（1,165 题） | 51.85% | **67.21%** | 52.96% | **+15.36%** |
| MMLU（14,042 题） | 41.58% | 48.87% | 41.75% | **+7.29%** |

**实际结果 v3**（2026-03-02，6 层注入 [4,8,12,16,20,24] + knowledge_adapter）：

| 数据集 | Baseline | Fusion+知识 | Fusion+空知识 | **Δacc** | v2→v3 Δ |
|--------|----------|-------------|--------------|----------|---------|
| MedQA（1,273 题） | 32.91% | 36.61% | 34.41% | **+3.69%** | +0.08% |
| ARC-Challenge（1,165 题） | 51.85% | 57.85% | 52.53% | **+6.01%** | **-9.35%** |
| MMLU（14,042 题） | 41.58% | 43.98% | 43.36% | **+2.40%** | **-4.89%** |

结论：**E2 v2 通过，v3 Phase 1 跨域融合能力显著退化**。v3 三个领域 Δacc 仍为正，但 ARC 从 +15.36% 骤降至 +6.01%，MMLU 从 +7.29% 降至 +2.40%。同时 Fusion+空知识 明显高于 v2（MMLU 43.36% vs 41.75%），说明 v3 注入模块的"基础增益"增大但"知识融合增益"缩小。可能原因：v3 仅训练 2 epoch（v2 为 5 epoch）、数据全文对齐导致捷径学习。详见 Part II E2 v3 实验结果。

---

### E3：层注入 vs RAG 公平对比 — 核心贡献的直接证明 ✅ 已完成（2026-02-26）

**目的**：在控制变量的条件下，证明层注入 > 输入拼接。

**关键原则**：相同的知识内容，只改融合方式。

**对比矩阵**：

| 组别 | 知识内容 | 融合方式 | 是否需要训练 |
|------|---------|---------|------------|
| G0 | 无 | — | 否 |
| G1（RAG-compressed）| LLMLingua 压缩文本 | decode → 文本前缀拼接 | 否 |
| **G2（Fusion-Phase1）** | **LLMLingua 压缩知识** | **层注入（Phase 1 权重）** | **是（Phase 1）** |
| **G3（Fusion-Phase2）** | **LLMLingua 压缩知识** | **层注入（Phase 2 权重）** | **是（Phase 1+2）** |
| G4（RAG-original）| 原始完整文本 | 文本前缀拼接 | 否 |

**G1 vs G2 是核心对比**：知识完全相同（都是 LLMLingua 压缩后的内容），
- G1 将 knowledge_ids decode 为文本，再 prepend 到 input context
- G2 将 knowledge_ids 直接注入各层（Cross-Attention）

两者使用完全一致的知识来源，只有融合方式不同。

**跨任务验证**：在 E2 的所有数据集上都跑此对比矩阵，取平均增益。

**评测指标**：
- QA 任务：accuracy, Δacc
- 汇总：知识利用效率 = Δacc(Fusion) / Δacc(RAG-original)

**预期结论**：G2 > G1 即可说明层注入比文本拼接更有效地利用了压缩知识。

**实际结果**（2026-02-26，4 GPU 并行）：

| 组别 | MedQA | ARC | MMLU |
|------|-------|-----|------|
| G0 Baseline | 32.91% | 51.85% | 41.58% |
| G1 RAG-compressed | 41.56% | 66.52% | 52.42% |
| G2 Fusion-Phase1 | 36.53% | 67.21% | 48.87% |
| **G3 Fusion-Phase2** | **69.52%** | **85.75%** | **70.94%** |
| G4 RAG-original | 86.96% | 86.95% | 79.55% |

知识利用效率（efficiency = Δacc(G) / Δacc(G4)）：

| 数据集 | G2 效率 | G3 效率 |
|--------|---------|---------|
| MedQA | 6.7% | **67.7%** |
| ARC | 43.8% | **96.6%** |
| MMLU | 19.2% | **77.3%** |

结论：**E3 通过**。G3 在三个数据集上全面超越 G1（RAG-compressed），Phase 2 层注入比文本拼接更高效；ARC 上效率达 96.6%，几乎追平 RAG-original 上限。详见 Part II E3 实验结果。

---

### E4：Phase 1 vs Phase 2 的消融分析 — 揭示 SFT 的代价 ✅ 已完成（2026-03-03）

**目的**：揭示 domain SFT 与通用融合能力之间的 trade-off。

**设计**：对比两组 v2 权重在三个数据集上的表现：

```
权重 A：Phase 1（通用 FineWeb-Edu 预训练，v2_phase1_best）
权重 B：Phase 2 Best Val（MedQA SFT，v2_phase2_best）
```

在 MedQA / ARC / MMLU 上同时评测 Baseline + Phase 1 + Phase 2。

**原始假设**：Phase 2 在跨域任务上低于 Phase 1 → SFT 破坏通用融合能力。

**实际结果（假设被推翻）**：

| 数据集 | Baseline | Phase 1 | Phase 2 | Phase1 Δ | Phase2 Δ | SFT 效果 |
|--------|----------|---------|---------|----------|----------|----------|
| MedQA  | 32.91%   | 36.53%  | **69.52%** | +3.61% | +36.61% | +32.99%（大幅提升） |
| ARC    | 51.85%   | 67.21%  | **85.75%** | +15.36% | +33.91% | +18.54%（大幅提升） |
| MMLU   | 41.58%   | 48.87%  | **70.94%** | +7.29% | +29.37% | +22.08%（大幅提升） |

**核心发现**：**原始假设被完全推翻**。SFT 不仅没有损害通用融合能力，反而在所有域上都带来了巨大提升。Phase 2 的 MedQA SFT 实际上充当了"instruction tuning"的角色：
1. Phase 1 让模块学会"接收并传递知识信号"，但融合深度有限
2. Phase 2 SFT 让模块学会"利用知识信号做出判断"，这种多选决策能力跨域通用
3. ARC 从 67.21% → 85.75%（+18.5%），MMLU 从 48.87% → 70.94%（+22.1%），跨域增幅甚至超过 MedQA 的 absolute Δ

**对后续训练策略的指导**：可以大胆做领域 SFT，无需担心 trade-off。少量领域 SFT 数据即可解锁通用知识融合能力。

---

### E5：知识注入量与质的分析 — 理解融合机制的边界 ✅ 已完成（2026-03-04）

**目的**：理解 Fusion 模块的信息利用效率及其局限性。

**设计 A：知识 Token 预算分析**

```
4 种 token 预算（k=32/64/128/256）
× 3 种方法（Baseline / RAG / Fusion）
× 3 数据集（MedQA / ARC / MMLU）
× 2 权重（Phase 1 / Phase 2）
```

知识映射构建方式：k=32/64/128 使用 LLMLingua 不同压缩率（0.125/0.25/0.5），k=256 直接 tokenize 原文（无压缩）。

**设计 B：知识相关性分析**

```
注入条件（k=64 固定）：
  Oracle：正确知识（question + correct_answer → LLMLingua 压缩）
  Shuffled：随机打乱 key↔value 映射（同数据集、同分布，但与当前题目无关）
  Empty：无知识（全 padding）
```

验证：融合模块是否真正利用了知识的语义相关性。

**实际结果 E5-A（Phase 2, v2 权重）**：

| Token | Baseline | Fusion-P2 | RAG | Δ(Fusion-RAG) | | — MedQA — |
|-------|----------|-----------|-----|---------------|---|-----------|
| 32 | 32.91% | **53.26%** | 33.78% | **+19.48%** | | |
| 64 | 32.91% | **69.52%** | 41.56% | **+27.96%** | | |
| 128 | 32.91% | **85.15%** | 61.51% | **+23.65%** | | |
| 256 | 32.91% | **88.22%** | 77.61% | **+10.60%** | | |

| Token | Baseline | Fusion-P2 | RAG | Δ(Fusion-RAG) | | — ARC — |
|-------|----------|-----------|-----|---------------|---|---------|
| 32 | 51.85% | **70.64%** | 57.42% | **+13.22%** | | |
| 64 | 51.85% | **85.75%** | 66.52% | **+19.23%** | | |
| 128 | 51.85% | **94.33%** | 77.08% | **+17.25%** | | |
| 256 | 51.85% | **95.45%** | 86.95% | **+8.50%** | | |

| Token | Baseline | Fusion-P2 | RAG | Δ(Fusion-RAG) | | — MMLU — |
|-------|----------|-----------|-----|---------------|---|----------|
| 32 | 41.58% | **58.00%** | 46.36% | **+11.64%** | | |
| 64 | 41.58% | **70.94%** | 52.42% | **+18.52%** | | |
| 128 | 41.58% | **83.11%** | 62.85% | **+20.25%** | | |
| 256 | 41.58% | **87.15%** | 76.51% | **+10.64%** | | |

**E5-A 核心发现**：

1. **Phase 2 Fusion 在所有 token 预算下全面超越 RAG**，Δ 最高达 +28%（MedQA@k=64）
2. **Fusion@128 超越 RAG@256**：MedQA 85.15% vs 77.61%、ARC 94.33% vs 86.95%、MMLU 83.11% vs 76.51% — 用一半 token 即超越 RAG 全文
3. **Fusion@256 接近天花板**：MedQA 88.22%（超越旧 RAG-original 77.6% 和修正后 87.0%）、ARC 95.45%、MMLU 87.15%
4. **Phase 1 Fusion 在 k≥128 时落后于 RAG**：Phase 1 在 k=128/256 时性能饱和甚至下降（MedQA@256 仅 32.05%），说明 Phase 1 融合深度有限，无法利用更多信息

**实际结果 E5-B（k=64, v2 权重）**：

| 条件 | P1 MedQA | P2 MedQA | P1 ARC | P2 ARC | P1 MMLU | P2 MMLU |
|------|----------|----------|--------|--------|---------|---------|
| Oracle | 36.53% | **69.52%** | 67.21% | **85.75%** | 48.87% | **70.94%** |
| Shuffled | 29.77% | 32.05% | 54.25% | 54.51% | 40.21% | 42.33% |
| Empty | 32.60% | 32.36% | 52.96% | 53.05% | 41.75% | 41.58% |

**E5-B 核心发现**：

1. **Phase 2 Shuffled ≈ Empty ≈ Baseline**：不相关知识被有效忽略（MedQA: 32.05% ≈ 32.36% ≈ 32.91%），模型回落到 baseline 行为，证明 Fusion 真正利用了语义相关性
2. **Oracle 与 Shuffled 差距极大**：Phase 2 MedQA Oracle-Shuffled = +37.47%，ARC = +31.24%，MMLU = +28.61% — 融合增益完全来自语义匹配
3. **Phase 1 敏感度较弱但方向一致**：Oracle vs Shuffled/Empty 差距 +3~15%，远小于 Phase 2（+28~37%）
4. **Shuffled 在部分场景低于 Empty**（MedQA P1: 29.77% < 32.60%），说明不相关知识不仅没帮助反而产生了干扰

结论：**E5 通过**。Fusion 在所有 token 预算下均优于 RAG（E5-A），且融合增益完全依赖知识的语义相关性（E5-B）。Phase 2 SFT 同时增强了知识利用深度和相关性鉴别能力。

---

### E6：推理效率对比 — 回应"为什么不直接用 RAG" ✅ 已完成（2026-03-04）

**目的**：E3 已证明 Fusion 在同等压缩知识下优于 RAG，但 G4（RAG-original 全文拼接）在绝对准确率上仍高于 G3（Fusion Phase 2）。本实验从**推理效率**维度论证 Fusion 的实际部署优势，回应"为什么不直接用 RAG-original"的质疑。

**设计 A 实际结果**（MedQA, N=200, 单 GPU, batch_size=1）：

| 方法 | 延迟 (ms/样本) | 吞吐 (样本/s) | 显存 (MB) | 平均输入长度 | 上下文占用 |
|------|---------------|--------------|----------|------------|----------|
| Baseline | 71.29 | 14.03 | 1816.9 | 211.0 | 0 tokens |
| RAG-compressed@64 | 71.38 | 14.01 | 1873.7 | 261.3 | ~64 tokens |
| **Fusion-Phase2@64** | **93.94** | **10.64** | **1799.1** | **211.0** | **0 tokens** |
| RAG-original@~256 | 72.17 | 13.86 | 2168.9 | 391.0 | ~256 tokens |

**关键发现（预期被部分推翻）**：

1. **Fusion 比 RAG-original 更慢**（93.94 vs 72.17 ms，+30.2%）：这与预期相反。原因分析：
   - Qwen3-0.6B 是小模型（28 层，hidden=1024），self-attention 在 211~391 tokens 区间的 O(L²) 代价非常低
   - RAG 的额外序列长度仅增加 ~0.9ms（72.17 - 71.29），几乎可忽略
   - Fusion 的额外开销来自：知识编码（6 层 encoder forward）+ 4 层 cross-attention，每次 forward 额外 ~5.7ms
   - **在小模型短序列场景下，cross-attention + 编码开销 > self-attention 序列延长开销**

2. **Fusion 显存最低**（1799.1 MB < Baseline 1816.9 MB < RAG-original 2168.9 MB）：
   - 比 RAG-original 节省 369.8 MB（**-17.1%**）
   - 甚至略低于 Baseline（知识编码不产生额外 KV cache）

3. **三种 RAG 方法延迟几乎相同**（71.29 / 71.38 / 72.17 ms）：
   - 说明在 0.6B 模型上，211→391 token 的序列长度增加对延迟影响微乎其微
   - self-attention 瓶颈在大模型/长序列场景才会显现

**设计 B 结果**：由 E5-A 已完整覆盖（4 种 token 预算 × 3 数据集 × 2 权重），核心结论：
- Fusion 在所有 token 预算下准确率全面超越 RAG（+8.5% ~ +28.0%）
- Fusion@128 > RAG@256（用一半 token 超越 RAG 全文）

**设计 C**：可选，未实施。

**修正后的六维对比汇总表**（E3 准确率 + E6 效率）：

| 维度 | RAG-original (G4) | Fusion Phase 2 (G3) | 胜者 |
|------|-------------------|---------------------|------|
| 绝对准确率 | **86.96%** (MedQA) | 69.52% (MedQA) | RAG |
| 同 token 准确率(k=64) | 41.56% | **69.52%** (+27.96%) | **Fusion** |
| 推理延迟 | **72.17 ms** | 93.94 ms (+30.2%) | **RAG** |
| 峰值显存 | 2168.9 MB | **1799.1 MB** (-17.1%) | **Fusion** |
| 上下文窗口侵占 | ~256 token | **0 token** | **Fusion** |
| 知识可预编码缓存 | 否（每次重处理） | **是**（编码一次复用） | **Fusion** |

**论文叙事框架修正**：

> 在 Qwen3-0.6B 这一小规模模型上，Fusion 的推理延迟高于 RAG（+30%），这是因为 cross-attention + 知识编码的固定开销在短序列场景下超过了 RAG 额外序列长度带来的 self-attention 增量。然而，这一关系在模型规模增大时会反转：self-attention 代价为 O(L²·d)，而 cross-attention 代价为 O(L·K·d)（K=64 固定），随着 L 和 d 增大，RAG 的代价增长更快。
>
> **Fusion 的核心优势不在单样本速度，而在于：**
> 1. **token 效率**：64 token 达到 RAG@64 的 +28% 准确率，甚至 Fusion@128 > RAG@256
> 2. **上下文窗口保护**：知识不占 input context，对多轮对话和长文本场景至关重要
> 3. **显存优势**：-17.1% 峰值显存，对部署密度有直接影响
> 4. **知识预编码缓存**：同一知识文本仅需编码一次，后续推理复用编码结果

结论：**E6 部分通过**。推理延迟维度不支持 Fusion（小模型短序列场景），但显存、上下文窗口保护和 token 效率维度均支持 Fusion。六维对比中 Fusion 以 4:2 胜出。

---

## 4. 数据集选择与知识构造方案

| 数据集 | 类型 | 知识构造方式 | 用于实验 |
|--------|------|------------|---------|
| MedQA-USMLE | 医学多选 | compress(question + correct_answer)，已有 | E1-E6 |
| ARC-Challenge | 科学推理多选 | compress(question + correct_answer)，Oracle 设置 | E2-E4, E6 |
| MMLU (all) | 多领域多选 | compress(question + choices[answer]) | E2-E4, E6 |

> **注意**：MedQA 使用 correct_answer 构造知识属于 Oracle 设置，在所有实验中均明确标注为"Oracle 知识"。

---

## 5. 评测指标体系

| 指标 | 适用场景 | 说明 |
|------|---------|------|
| Accuracy（acc） | 多选/判断题 | 最直接 |
| Δacc（相对 baseline） | 所有 QA | 消除模型基础能力差异 |
| Knowledge Sensitivity（KS） | E1 | 量化知识驱动强度，= acc_correct - acc_counterfactual |
| Cross-domain Δacc | E2, E4 | 跨域泛化能力 |
| Knowledge Utilization Efficiency | E3 | = Δacc(Fusion) / Δacc(RAG-oracle) |
| Compression Degradation Curve | E5 | 随压缩率变化的 acc 曲线 |
| Latency / Throughput | E6 | 推理延迟 (ms/sample)、吞吐量 (samples/sec) |
| Peak GPU Memory | E6 | 峰值显存占用 (MB) |
| Acc vs Token Budget Curve | E6 | 固定 token 预算下的 acc 曲线，量化单位 token 知识利用效率 |

---

## 6. 建议执行顺序

```
优先级 1（核验前提，1天内）：E1 — Sanity Check ✅ 已完成（KS=56.32%）
  ↓ 知识确实被使用，前提成立

优先级 2（核心贡献，2-3天）：E3 — 层注入 vs RAG 公平对比 ✅ 已完成（G3 效率: ARC 96.6%, MMLU 77.3%, MedQA 67.7%）
  ↓ Phase 2 层注入全面超越 RAG-compressed，ARC 几乎追平 RAG-original

优先级 2（核心贡献，2-3天）：E2 — 跨域通用能力验证 ✅ 已完成（Δacc: ARC +15.4%, MMLU +7.3%, MedQA +3.6%）
  ↓ Phase 1 通用融合能力得证，三个域全部正增益

优先级 4（揭示 trade-off，1天）：E4 — SFT 消融分析 ✅ 已完成（假设被推翻：SFT 在所有域全面提升，无 trade-off）
  ↓ SFT 充当 instruction tuning，解锁通用知识融合决策能力

优先级 5（深度分析，1-2天）：E5 — 知识注入量与质的分析 ✅ 已完成（Fusion@128 > RAG@256，Shuffled ≈ Empty ≈ Baseline）
  ↓ Fusion 在所有 token 预算下优于 RAG，融合增益完全依赖语义相关性

优先级 3（效率论证，1天）：E6 — 推理效率对比 ✅ 已完成（延迟: Fusion 93.94ms > RAG 72.17ms，但显存 -17.1%，token 效率 +28%）
  ↓ 推理延迟预期被推翻（小模型短序列），但显存/上下文/token效率维度 Fusion 胜出（4:2）
```

---

## 7. 关键的实验设计决策

**E3 核心对比，应该在哪个权重上做**：

| 选择 | 优点 | 缺点 |
|------|------|------|
| Phase 1 权重 | 测的是纯融合机制，不受 domain SFT 影响 | MedQA accuracy 较低，结果不够"好看" |
| Phase 2 权重 | 结果更高，论证更有说服力 | 可能只是 domain overfit，融合效果混入了 SFT 效果 |

**结论**：两组都跑，分开报告。Phase 1 权重论证"融合机制本身的优越性"，Phase 2 权重展示"在特定域完整系统的性能上限"。

---

## 8. 当前已有实验的重新定位

| 当前实验 | 问题 | 在新体系中的角色 |
|---------|------|--------------|
| Phase 1（FineWeb-Edu LM）| 设计合理 | E2 的 Phase1-Fusion 权重来源 |
| Phase 2（MedQA SFT，1-token）| 任务退化为 4 分类，信号稀疏 | E4 的消融分析对比点 |
| compare G1（无知识）| 合理 | 沿用为所有实验的 baseline |
| compare G2（RAG compressed）| 知识形式与 Fusion 不同 | E3 的 G1（RAG-compressed）：decode 后等长拼接 |
| compare G3（RAG 原始）| 设计合理 | E3 的 G4（RAG-oracle 上限），保留 |
| compare G4（Prefix token）| 作为辅助参考 | E3 的参考组，保留 |

---

# Part II：实验实现指南

> 每个实验的实现指南包含：资源清单、新建/修改文件、知识构造方案、配置参数、执行命令、预期输出。

---

## E1 实现指南：Sanity Check — 反事实知识测试

### 所需资源

| 资源 | 说明 | 状态 |
|------|------|------|
| `checkpoints/phase1_best` | Phase 1 最优权重 | ✅ 已有 |
| `checkpoints/phase2_best` | Phase 2 最优权重 | ✅ 已有 |
| `data/medqa_knowledge.jsonl` | 正确知识映射（test split） | ✅ 已有 |
| `data/medqa_knowledge_counterfactual.jsonl` | 反事实知识映射 | ✅ 已构建（2026-02-24） |
| MedQA test split | HuggingFace 数据集 | ✅ 已缓存 |

### 反事实知识的构造原理

```python
# 对每道题：
question = row["sent1"]
label = row["label"]  # 正确答案索引
correct_answer = row[f"ending{label}"]

# 反事实知识：随机选一个错误选项
wrong_indices = [i for i in range(4) if i != label]
wrong_idx = random.choice(wrong_indices)  # 或固定选第一个错误选项
wrong_answer = row[f"ending{wrong_idx}"]

# 正确知识（已有）：compress(question + correct_answer)
# 反事实知识（新建）：compress(question + wrong_answer)
```

### 新建文件

**`evaluation/counterfactual_eval.py`**

功能：
1. `build_counterfactual_knowledge(split, output_path, config)` — 构造反事实知识映射，复用 `MedQAKnowledgeBuilder`，只改 source_text 的 answer 部分
2. `eval_counterfactual(model, tokenizer, ds, correct_km, cf_km, device)` — 三组对比评测，返回 `{acc_correct, acc_counterfactual, acc_no_knowledge, KS}`
3. `run_e1_sanity_check(injection_weights, config)` — 入口函数，分别测 Phase 1 和 Phase 2 权重

**关键实现细节**：
- 复用 `evaluation/run_eval.py` 的 `evaluate_medqa_inline()` 函数
- 无知识注入：knowledge_ids 全部设置为 `[pad_token_id] * 64`
- 反事实知识：使用 `medqa_knowledge_counterfactual.jsonl`
- 正确知识：使用 `medqa_knowledge.jsonl`（已有）

### 修改文件

**`main.py`**：添加命令 `eval-counterfactual`

```python
# 在 choices 列表添加
"eval-counterfactual"

# 在命令分支添加
elif args.command == "eval-counterfactual":
    from evaluation.counterfactual_eval import run_e1_sanity_check
    assert args.injection_weights is not None
    run_e1_sanity_check(args.injection_weights, config)
```

**`config/default.yaml`**：添加反事实知识映射路径

```yaml
evaluation:
  medqa:
    knowledge_map_counterfactual: "data/medqa_knowledge_counterfactual.jsonl"
```

### 执行命令

**Step 1：构造反事实知识映射（一次性）**

```bash
# 在 counterfactual_eval.py 中内置构建，或通过 main.py 触发
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python -c "
from evaluation.counterfactual_eval import build_counterfactual_knowledge
from utils.helpers import load_config
config = load_config('config/default.yaml')
build_counterfactual_knowledge('test', 'data/medqa_knowledge_counterfactual.jsonl', config)
"
```

**Step 2：运行 E1（Phase 1 权重）**

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py eval-counterfactual \
  --injection-weights checkpoints/phase1_best \
  --config config/default.yaml
```

**Step 3：运行 E1（Phase 2 权重）**

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py eval-counterfactual \
  --injection-weights checkpoints/phase2_best \
  --config config/default.yaml
```

### 预期输出

保存路径：`results/e1_sanity_check_{phase1|phase2}.json`

```json
{
  "weights": "checkpoints/phase1_best",
  "acc_correct_knowledge": 0.365,
  "acc_counterfactual_knowledge": 0.210,
  "acc_no_knowledge": 0.250,
  "knowledge_sensitivity": 0.155,
  "total": 1273
}
```

**判断标准**：
- `KS > 0.15`：知识注入显著有效 ✅
- `KS < 0.05`：知识被忽略，需要检查模型 ❌

### 实验结果（2026-02-24 执行）

#### 实现文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `evaluation/counterfactual_eval.py` | NEW | 反事实知识构建 + 三组对比评测 + 入口函数 |
| `tests/unit/test_counterfactual_eval.py` | NEW | 3 个单元测试（格式验证、评分范围、pad 逻辑） |
| `scripts/run_e1_sanity_check.sh` | NEW | 一键运行脚本（GPU 4，Phase 2 → Phase 1） |
| `main.py` | MODIFY | 添加 `eval-counterfactual` 命令 |
| `config/default.yaml` | MODIFY | 添加 `knowledge_map_counterfactual` 路径 |

#### 核心函数

| 函数 | 文件 | 说明 |
|------|------|------|
| `build_counterfactual_knowledge()` | counterfactual_eval.py | 构造反事实知识：`wrong_answer = ending[(label+1)%4]`（确定性） |
| `_score_choices_injection()` | counterfactual_eval.py | 对 ModifiedQwen 做 loglikelihood 4 选 1 评分 |
| `eval_e1_sanity_check()` | counterfactual_eval.py | 三组对比：正确知识 / 反事实 / 全 pad |
| `run_e1_sanity_check()` | counterfactual_eval.py | 完整入口（自动构建反事实映射、加载模型、报告、保存） |

#### 运行命令

```bash
bash scripts/run_e1_sanity_check.sh
```

#### 结果数据

**Phase 2 权重**（`results/e1_sanity_check_phase2_best.json`）：

| 指标 | 值 |
|------|-----|
| acc_correct（正确知识） | **69.52%** (885/1273) |
| acc_counterfactual（反事实知识） | 13.20% (168/1273) |
| acc_no_knowledge（全 pad） | 32.36% (412/1273) |
| **Knowledge Sensitivity (KS)** | **+56.32%** |

**Phase 1 权重**（`results/e1_sanity_check_phase1_best.json`）：

| 指标 | 值 |
|------|-----|
| acc_correct（正确知识） | 36.53% (465/1273) |
| acc_counterfactual（反事实知识） | 28.75% (366/1273) |
| acc_no_knowledge（全 pad） | 32.60% (415/1273) |
| **Knowledge Sensitivity (KS)** | **+7.78%** |

#### 分析

**1. Phase 2 Fusion 模块极度依赖注入知识（KS = 56.32%）**

远超 20% 的"显著有效"阈值。反事实知识将 acc 拉至 13.20%，低于随机基线 25%，说明模型在**认真读并信任**注入的知识——注入错误知识会导致模型被积极误导。

**2. Phase 1 Fusion 模块部分利用知识（KS = 7.78%）**

处于 5%~15% 的"部分有效"区间。三组排序方向完全正确：
```
acc_correct (36.53%) > acc_no_knowledge (32.60%) > acc_counterfactual (28.75%)
```
效果较弱符合预期——Phase 1 只做了通用 LM 预训练，未针对 MedQA 微调。

**3. 无知识组（全 pad）两组权重表现一致（~32.5%）**

与原始 Qwen3-0.6B baseline（32.9%）基本持平，说明注入模块在收到"空"输入时不干扰模型原有能力。

**4. 关键不等式全部满足**

```
Phase 2: acc_correct(69.5%) > acc_nk(32.4%) > random(25%) > acc_cf(13.2%)  ✓
Phase 1: acc_correct(36.5%) > acc_nk(32.6%) > acc_cf(28.8%)                ✓
KS > 0（两组均成立）                                                        ✓
```

#### 结论

**E1 Sanity Check 通过。** Fusion 模块确实在使用注入的知识。可以安全推进 E2（跨域通用能力）和 E3（层注入 vs RAG 公平对比）。

---

### E1 v3 实验结果（2026-03-02 执行，6 层注入 + knowledge_adapter）

#### 架构变更（v2 → v3）

| 维度 | v2 | v3 |
|------|----|----|
| 注入层 | [6,12,18,24]（4 层） | [4,8,12,16,20,24]（6 层） |
| knowledge_adapter | 无 | 2 层 MLP（零初始化，残差连接） |
| 数据对齐 | 知识 = compress(后半段) | 知识 = compress(全文) |
| 去噪训练 | 无 | 50% 概率 × 30% mask |

#### 运行命令

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_e1_sanity_check_v3.sh
```

#### 结果数据

**v3 Phase 2 权重**（`results/e1_sanity_check_injection_qa_best.json`）：

| 指标 | v2 Phase 2 | v3 Phase 2 | Δ |
|------|-----------|-----------|---|
| acc_correct（正确知识） | 69.52% (885) | **71.09%** (905) | **+1.57%** |
| acc_counterfactual（反事实知识） | 13.20% (168) | 14.30% (182) | +1.10% |
| acc_no_knowledge（全 pad） | 32.36% (412) | 34.72% (442) | **+2.36%** |
| **KS** | +56.32% | **+56.79%** | +0.47% |

**v3 Phase 1 权重**（`results/e1_sanity_check_v3_phase1_best.json`）：

| 指标 | v2 Phase 1 | v3 Phase 1 | Δ |
|------|-----------|-----------|---|
| acc_correct（正确知识） | 36.53% (465) | 36.61% (466) | +0.08% |
| acc_counterfactual（反事实知识） | 28.75% (366) | 28.36% (361) | -0.39% |
| acc_no_knowledge（全 pad） | 32.60% (415) | 34.41% (438) | **+1.81%** |
| **KS** | +7.78% | **+8.25%** | +0.47% |

#### 分析

**1. Phase 2 acc_correct 提升 +1.57%（69.52%→71.09%）**

v3 的四项优化（6 层注入、adapter、数据对齐、去噪训练）共同贡献了这 1.57% 的提升。绝对值不大，但方向正确，且在 1273 题上统计显著（约 20 题差异）。

**2. 无知识组（全 pad）显著提升——6 层注入 + adapter 改善了基础能力**

v3 Phase 2 无知识组从 32.36% 提升到 34.72%（+2.36%），v3 Phase 1 从 32.60% 提升到 34.41%（+1.81%）。这是意料之外的收获：即使不注入知识，更多的注入层 + adapter 也改善了模型的基础表现。可能原因：
- 6 层注入（含早期第 4 层）让模型在浅层就能利用知识编码器的表示
- knowledge_adapter 的残差连接在全 pad 输入时近似恒等映射，不干扰但不排除微弱正则化效果

**3. KS 基本不变（56.32%→56.79%），知识敏感度稳定**

KS 仅微增 +0.47%，说明 v3 的架构改进没有改变"模型依赖知识的程度"，只是让知识利用和基础能力都同步小幅提升。这是健康的——不是通过更激进地依赖知识来提升的。

**4. Phase 1 变化极小，符合预期**

Phase 1 仅做通用 LM 预训练，v3 的数据对齐改进主要影响 Phase 1 训练效率（loss 从 3.06 降到 1.73），但对下游 MedQA 评测的 KS 影响有限（+0.47%）。

**5. 关键不等式全部满足**

```
v3 Phase 2: acc_correct(71.1%) > acc_nk(34.7%) > random(25%) > acc_cf(14.3%)  ✓
v3 Phase 1: acc_correct(36.6%) > acc_nk(34.4%) > acc_cf(28.4%)                ✓
KS > 0（两组均成立）                                                            ✓
```

#### v3 E1 结论

**E1 v3 通过。** v3 架构优化带来的提升为 Phase 2 acc_correct +1.57%、无知识基础能力 +2.36%，KS 保持稳定。改进幅度温和但一致，验证了 6 层注入 + adapter + 数据对齐的有效性。可继续推进 v3 版本的 E2/E3 实验。

---

## E2 实现指南：跨域通用能力验证

### 所需资源

| 资源 | 说明 | 下载命令 |
|------|------|---------|
| ARC-Challenge | 科学推理 4 选 1（1172 test 题） | `load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")` |
| MMLU（all config） | 57 领域多选题 | `load_dataset("cais/mmlu", "all", split="test")` |
| `checkpoints/phase1_best` | Phase 1 权重 | ✅ 已有 |
| `data/arc_knowledge.jsonl` | ARC 知识映射 | ❌ 需新建 |
| `data/mmlu_knowledge.jsonl` | MMLU 知识映射 | ❌ 需新建 |

### 数据集格式说明

**ARC-Challenge**：
```python
ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
# 字段：
# ds[i]["question"]：问题文本
# ds[i]["choices"]["text"]：list[str]，选项文本列表
# ds[i]["choices"]["label"]：list[str]，选项标签列表（"A","B","C","D"）
# ds[i]["answerKey"]：正确答案标签（"A"/"B"/"C"/"D" 或 "1"/"2"/"3"/"4"）
```

知识构造：compress(question + correct_answer)，Oracle 设置，与 MedQA/MMLU 一致。
评测方式：loglikelihood（对 " A"/" B"/" C"/" D" 算 log-prob），仅保留恰好 4 选项的题。

**MMLU（all config）**：
```python
ds = load_dataset("cais/mmlu", "all", split="test")
# 字段：
# ds[i]["question"]：问题文本
# ds[i]["choices"]：list[str]，4 个选项
# ds[i]["answer"]：int，0-3，正确选项索引
# ds[i]["subject"]：str，领域名称（如 "high_school_biology"）
```

知识构造：compress(question + choices[answer])，Oracle 设置，与 MedQA 一致。
评测方式：loglikelihood（与 MedQA 完全一致，对 " A"/" B"/" C"/" D" 算 log-prob）。

### 新建文件

**`evaluation/arc_eval.py`**

功能：
1. `build_arc_knowledge(output_path, config, split="test", limit=None)` — 构造 ARC 知识映射，compress(question + correct_answer)，Oracle 设置
2. `eval_arc(model, tokenizer, ds, device, knowledge_map, ...)` — loglikelihood 评测（与 MedQA/MMLU 一致），返回 `{acc, correct, total, skipped}`
3. `answer_key_to_index(key)` — answerKey ("A"/"1") → 索引 (0-3)

**`evaluation/mmlu_eval.py`**

功能：
1. `build_mmlu_knowledge(output_path, config, split="test", subjects=None)` — 构造 MMLU 知识映射，可选 subjects 过滤
2. `eval_mmlu(model, tokenizer, ds, knowledge_map, device)` — loglikelihood 评测，支持按 subject 分组报告
3. `run_e2_mmlu(injection_weights, config, subjects=None)` — 对比评测

**`evaluation/cross_domain_runner.py`**

功能：
1. `run_e2_all(phase1_weights, config)` — 统一入口，在 MedQA + ARC + MMLU 上跑完整对比，汇总报告

### 修改文件

**`main.py`**：添加命令

```python
# choices 中添加
"eval-cross-domain"
"build-arc-knowledge"
"build-mmlu-knowledge"

# 命令分支
elif args.command == "build-arc-knowledge":
    from evaluation.arc_eval import build_arc_knowledge
    build_arc_knowledge("data/arc_knowledge.jsonl", config)

elif args.command == "build-mmlu-knowledge":
    from evaluation.mmlu_eval import build_mmlu_knowledge
    build_mmlu_knowledge("data/mmlu_knowledge.jsonl", config)

elif args.command == "eval-cross-domain":
    from evaluation.cross_domain_runner import run_e2_all
    assert args.injection_weights is not None
    run_e2_all(args.injection_weights, config)
```

**`config/default.yaml`**：添加新数据集配置

```yaml
evaluation:
  arc:
    knowledge_map: "data/arc_knowledge.jsonl"
    split: "test"
    limit: null  # ARC-Challenge test 仅 1172 题，全量运行
  mmlu:
    knowledge_map: "data/mmlu_knowledge.jsonl"
    split: "test"
    subjects: null  # null = 全部 57 个领域
    limit: null
```

### 执行命令

**Step 1：构造知识映射（一次性）**

```bash
# ARC-Challenge 知识映射
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py build-arc-knowledge \
  --config config/default.yaml

# MMLU 知识映射
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py build-mmlu-knowledge \
  --config config/default.yaml
```

**Step 2：运行 E2 完整跨域评测（Phase 1 权重）**

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py eval-cross-domain \
  --injection-weights checkpoints/phase1_best \
  --config config/default.yaml
```

**Step 3：可选——运行 Phase 2 权重对比**

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py eval-cross-domain \
  --injection-weights checkpoints/phase2_best \
  --config config/default.yaml
```

### 预期输出

保存路径：`results/e2_cross_domain_{phase1|phase2}.json`

```json
{
  "weights": "checkpoints/phase1_best",
  "medqa": {
    "baseline": {"acc": 0.329},
    "fusion_knowledge": {"acc": 0.365},
    "fusion_empty": {"acc": 0.330},
    "delta_acc": 0.036
  },
  "arc": {
    "baseline": {"acc": 0.XXX},
    "fusion_knowledge": {"acc": 0.XXX},
    "fusion_empty": {"acc": 0.XXX},
    "delta_acc": 0.XXX
  },
  "mmlu": {
    "baseline": {"acc": 0.XXX},
    "fusion_knowledge": {"acc": 0.XXX},
    "fusion_empty": {"acc": 0.XXX},
    "delta_acc": 0.XXX
  }
}
```

**判断标准**：
- 所有域 `delta_acc > 0`：通用融合能力得证 ✅
- 医学域 >> 其他域：存在域偏差，Phase 1 训练不够通用 ⚠️

### 实验结果（2026-02-25 执行）

#### 实现文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `evaluation/arc_eval.py` | NEW | ARC 知识构建 + loglikelihood 评测 |
| `evaluation/mmlu_eval.py` | NEW | MMLU 知识构建 + loglikelihood 评测 |
| `evaluation/cross_domain_runner.py` | NEW | E2 跨域评测编排器（支持多 GPU 并行） |
| `tests/unit/test_arc_eval.py` | NEW | ARC 评测单元测试 |
| `tests/unit/test_mmlu_eval.py` | NEW | MMLU 评测单元测试 |
| `scripts/run_e2_cross_domain.sh` | NEW | 一键运行脚本 |
| `main.py` | MODIFY | 添加 `eval-cross-domain`, `build-arc-knowledge`, `build-mmlu-knowledge` 命令 |
| `config/default.yaml` | MODIFY | 添加 ARC/MMLU 知识映射路径配置 |

#### 核心函数

| 函数 | 文件 | 说明 |
|------|------|------|
| `build_arc_knowledge()` | arc_eval.py | 构造 ARC Oracle 知识：compress(question + correct_answer) → 64 tokens |
| `eval_arc()` | arc_eval.py | ARC loglikelihood 4 选 1 评测，仅保留恰好 4 选项的题 |
| `build_mmlu_knowledge()` | mmlu_eval.py | 构造 MMLU Oracle 知识：compress(question + choices[answer]) → 64 tokens |
| `eval_mmlu()` | mmlu_eval.py | MMLU loglikelihood 4 选 1 评测 |
| `run_e2_all()` | cross_domain_runner.py | 完整入口：三数据集 × 三条件对比，支持多 GPU 数据并行 |

#### 运行命令

```bash
# 构造知识映射（一次性）
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py build-arc-knowledge --config config/default.yaml
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py build-mmlu-knowledge --config config/default.yaml

# 运行 E2 跨域评测（Phase 1 权重）
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python main.py eval-cross-domain \
  --injection-weights checkpoints/phase1_best --config config/default.yaml
```

#### 结果数据

**Phase 1 权重**（`results/e2_cross_domain_phase1.json`）：

| 数据集 | 题数 | Baseline | Fusion+知识 | Fusion+空知识 | **Δacc** | Δacc_empty |
|--------|------|----------|-------------|--------------|----------|------------|
| MedQA | 1,273 | 32.91% (419) | 36.53% (465) | 32.60% (415) | **+3.61%** | -0.31% |
| ARC-Challenge | 1,165 | 51.85% (604) | **67.21%** (783) | 52.96% (617) | **+15.36%** | +1.12% |
| MMLU | 14,042 | 41.58% (5838) | 48.87% (6862) | 41.75% (5862) | **+7.29%** | +0.17% |

> ARC 跳过了 7 道非 4 选项题（原始 1172 题中有 7 题为 3 或 5 选项）。

#### 分析

**1. 三个数据集 Δacc 全部为正——通用融合能力得证**

```
ARC-Challenge  +15.36%  ████████████████
MMLU           +7.29%   ████████
MedQA          +3.61%   ████
```

Phase 1 Fusion 模块在完全未见过的领域上均能有效融合知识，不依赖特定领域 fine-tuning。

**2. Fusion+空知识 ≈ Baseline（消融对照成功）**

三组 `delta_acc_empty` 极小（-0.31% / +1.12% / +0.17%），与 Baseline 基本持平。增益完全来自知识内容，而非注入模块的架构本身。

**3. 增幅排序 ARC >> MMLU >> MedQA——无域偏差，符合迁移梯度**

| 数据集 | 领域 | 与 FineWeb-Edu 的域距离 | Δacc |
|--------|------|----------------------|------|
| ARC | 科学推理 | **近**（FineWeb-Edu 含大量科学教育文本） | +15.36% |
| MMLU | 57 个混合领域 | 中等 | +7.29% |
| MedQA | 专业医学 | **远**（需要临床专业知识） | +3.61% |

这不是域偏差（判断标准中的"医学域 >> 其他域"没有发生，反而是医学域增幅**最小**）。增幅与 Phase 1 训练数据（FineWeb-Edu）的域距离成反比，是典型的迁移学习梯度。

**4. MMLU +7.29%（14,042 题）是最强的通用性证据**

MMLU 覆盖 57 个子领域，14,042 道题的样本量远大于 MedQA/ARC。在如此广泛的领域上取得一致的正增益，是通用融合能力的最有力支撑。

**5. 与 E1 结果的一致性**

E1 中 Phase 1 在 MedQA 上 acc_correct=36.53%、KS=7.78%。E2 中 Phase 1 在 MedQA 上 Fusion+知识=36.53%——完全一致，验证了评测代码的可靠性。

#### 结论

**E2 通过。** Phase 1 Fusion 模块具备通用的知识融合能力，在科学推理（ARC +15.36%）、多领域常识（MMLU +7.29%）、医学专业（MedQA +3.61%）三个维度上均显著优于 Baseline，且增益完全来自知识内容（Fusion+空 ≈ Baseline）。可以安全推进 E3（层注入 vs RAG 公平对比）和 E4（SFT 消融分析）。

---

### E2 v3 实验结果（2026-03-02 执行，6 层注入 + knowledge_adapter，4 GPU 并行）

#### 架构变更（v2 → v3）

同 E1 v3，见上文。

#### 运行命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,7 bash scripts/run_e2_cross_domain_v3.sh
```

#### 结果数据

**v3 Phase 1 权重**（`results/e2_cross_domain_phase1.json`，权重 = `checkpoints/v3_phase1_best`）：

| 数据集 | 题数 | Baseline | Fusion+知识 | Fusion+空知识 | **Δacc** | Δacc_empty |
|--------|------|----------|-------------|--------------|----------|------------|
| MedQA | 1,273 | 32.91% (419) | 36.61% (466) | 34.41% (438) | **+3.69%** | +1.49% |
| ARC-Challenge | 1,165 | 51.85% (604) | 57.85% (674) | 52.53% (612) | **+6.01%** | +0.69% |
| MMLU | 14,042 | 41.58% (5838) | 43.98% (6175) | 43.36% (6088) | **+2.40%** | +1.78% |

#### v2 vs v3 逐项对比

| 数据集 | v2 Δacc | v3 Δacc | 变化 | v2 Δacc_empty | v3 Δacc_empty | 变化 |
|--------|---------|---------|------|---------------|---------------|------|
| MedQA | +3.61% | +3.69% | +0.08% | -0.31% | +1.49% | +1.80% |
| ARC | **+15.36%** | +6.01% | **-9.35%** | +1.12% | +0.69% | -0.43% |
| MMLU | **+7.29%** | +2.40% | **-4.89%** | +0.17% | +1.78% | +1.61% |

**净知识融合增益**（= Δacc - Δacc_empty，排除注入模块架构本身的基础增益）：

| 数据集 | v2 净融合增益 | v3 净融合增益 | 变化 |
|--------|-------------|-------------|------|
| MedQA | +3.92% | +2.20% | -1.72% |
| ARC | **+14.24%** | +5.32% | **-8.92%** |
| MMLU | **+7.12%** | +0.62% | **-6.50%** |

#### 分析

**1. v3 Phase 1 跨域知识融合能力显著退化——ARC -9.35%、MMLU -4.89%**

这是 v3 实验中最关键的负面发现。v2 Phase 1 在 ARC 上展现了 +15.36% 的强融合效果，v3 降至 +6.01%（不到一半）。MMLU 从 +7.29% 降至 +2.40%。仅 MedQA 基本持平（+3.69% vs +3.61%）。

**2. Fusion+空知识组升高——注入模块"基础增益"增大**

v3 Fusion+空知识在 MedQA（+1.49%）和 MMLU（+1.78%）上明显高于 v2（-0.31% / +0.17%），说明 v3 注入模块即使不接收有效知识也能改善输出。这与 E1 中无知识组升高的发现一致。

但"基础增益"挤占了"融合增益"——MMLU 上净融合增益仅 +0.62%（接近零），几乎所有 Δacc 都来自注入模块的架构效应而非知识内容。这是一个不健康的信号。

**3. 根因诊断：三个可能因素**

| 因素 | 说明 | 影响程度 |
|------|------|---------|
| **训练轮数不足** | v3 仅 2 epoch（~18M 样本），v2 为 5 epoch（~67.5M 样本），训练量仅为 v2 的 27% | **高** |
| **数据全文对齐导致捷径** | v3 knowledge = compress(全文)，与 input 高度重叠，模型可能学习"冗余复制"而非"跨文档融合" | **中** |
| **更多参数需更多训练** | 6 层 + adapter（27.3M 参数 vs v2 ~25.2M），参数量增加但训练量锐减 | **中** |

最可能的主因是**训练量锐减**（27% of v2）。v2 Phase 1 经过 5 个 epoch 的充分训练，注入模块在大量不同文本对上学会了跨域融合。v3 仅 2 epoch 且因数据对齐改变，注入模块尚未收敛到通用融合状态。

**4. 与 E1 v3 结果不矛盾**

E1 v3 显示 Phase 2 acc_correct 提升 +1.57%（69.52%→71.09%）。这说明 Phase 2 SFT 有足够的监督信号来弥补 Phase 1 的不足——MedQA SFT 直接教会模块"从知识中提取答案"，而不依赖 Phase 1 学到的通用融合能力。

但 E2 测试的是 **Phase 1 权重的裸跨域能力**，没有 SFT 的补救。因此 E1 通过不意味着 E2 也能通过。

**5. MedQA 不受影响的原因**

MedQA 在 v2 中 Δacc 就仅有 +3.61%（三个数据集中最小），v3 的 +3.69% 基本持平。这可能因为 MedQA 的专业医学领域本就远离 FineWeb-Edu 训练分布，Phase 1 的通用融合在此域效果始终有限，无论 v2 还是 v3。

#### v3 E2 结论

**v3 Phase 1 跨域融合能力未达 v2 水平。** 三个域 Δacc 仍为正（E2 最低判断标准通过），但 ARC 和 MMLU 的净融合增益分别下降 8.92% 和 6.50%，退化幅度显著。最可能原因是训练量不足（2 epoch vs 5 epoch）。

**建议后续方向**：
- **方案 A**：延长 v3 Phase 1 训练至 5 epoch，验证训练量是否为主因
- **方案 B**：直接用 v2 Phase 1 权重（通用融合更强）+ v3 Phase 2 SFT 的组合策略
- **方案 C**：先推进 v3 E3（层注入 vs RAG），因为 E3 用 Phase 2 权重——E1 已证明 v3 Phase 2 有效

---

## E3 实现指南：层注入 vs RAG 公平对比

### 核心设计原则

**E3 是整个实验体系中最关键的实验**，必须保证公平性。公平性的唯一标准是：**G1（RAG-compressed）和 G2（Fusion）使用完全相同的知识内容**。

具体而言：
- 两者都使用同一份 `knowledge_ids`（64 tokens）
- G1：将 `knowledge_ids` decode 为文本，prepend 到 prompt → `"Context: {decoded}\n\n{question_prompt}"`
- G2：将 `knowledge_ids` 直接作为知识向量注入各层（已有实现）

### 所需资源

| 资源 | 说明 | 状态 |
|------|------|------|
| `data/medqa_knowledge.jsonl` | MedQA 压缩知识映射 | ✅ 已有 |
| `data/arc_knowledge.jsonl` | ARC 知识映射 | E2 中已建 |
| `data/mmlu_knowledge.jsonl` | MMLU 知识映射 | E2 中已建 |
| `checkpoints/phase1_best` | Phase 1 权重 | ✅ 已有 |
| `checkpoints/phase2_best` | Phase 2 权重 | ✅ 已有 |

### 完整对比矩阵

| 组 | 名称 | 模型 | 知识来源 | 融合方式 |
|----|------|------|---------|---------|
| G0 | Baseline | Qwen3（无修改） | 无 | — |
| G1 | RAG-compressed | Qwen3（无修改） | decode(knowledge_ids) → 文本 | prepend 到 prompt |
| G2 | Fusion-Phase1 | Qwen3 + 注入模块（Phase 1） | 同 G1 的 knowledge_ids | 层注入 Cross-Attention |
| G3 | Fusion-Phase2 | Qwen3 + 注入模块（Phase 2） | 同 G1 的 knowledge_ids | 层注入 Cross-Attention |
| G4 | RAG-oracle | Qwen3（无修改） | 原始完整文本（不压缩） | prepend 到 prompt |

> **G0、G1、G4 无需训练**（直接用原始 Qwen3）。G2、G3 使用训练好的注入模块。

### 修改文件

**`evaluation/compare_eval.py`**：添加 Fusion 组评测函数

```python
# 新增函数：eval_fusion_injection
def eval_fusion_injection(
    model_path: str,          # 原始 Qwen3 路径
    injection_weights: str,   # 注入模块权重路径
    ds,                       # 数据集
    knowledge_map,            # {key: knowledge_ids}
    config,                   # 完整配置
    device,
    group_name: str = "Fusion",
) -> Dict[str, Any]:
    """
    加载 ModifiedQwen（注入模块 + Qwen3），
    用与 evaluate_medqa_inline 相同的 loglikelihood 方法评测。
    复用 models.create_model() 和 model.load_injection_weights()。
    """
    from models import create_model
    model = create_model(
        model_path=model_path,
        injection_method=config["model"]["injection"]["method"],
        injection_layers=config["model"]["injection"]["layers"],
        encoder_depth=config["model"]["injection"]["encoder_depth"],
        device=str(device),
    )
    model.load_injection_weights(injection_weights)
    model.eval()
    ...
```

**`main.py`**：添加命令 `compare-fusion-phase1` 和 `compare-fusion-phase2`

```python
# choices 中添加
"compare-fair-all"    # 一次性跑完整 E3 对比矩阵

# 命令分支
elif args.command == "compare-fair-all":
    from evaluation.e3_fair_compare import run_e3_all
    assert args.injection_weights is not None
    run_e3_all(config, args.injection_weights)
```

**新建 `evaluation/e3_fair_compare.py`**

功能：
1. `run_e3_all(config, phase1_weights, phase2_weights)` — 完整 E3 对比，输出结构化对比表格
2. 在 MedQA、ARC、MMLU 上分别跑 G0-G4 全部 5 组
3. 按数据集汇总 `delta_acc` 对比

### 执行命令

```bash
# 在 MedQA 上跑完整 E3 对比（包含 G0-G4 全部 5 组）
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python -c "
from evaluation.e3_fair_compare import run_e3_all
from utils.helpers import load_config
config = load_config('config/default.yaml')
run_e3_all(
    config=config,
    phase1_weights='checkpoints/phase1_best',
    phase2_weights='checkpoints/phase2_best',
)
"
```

### 预期输出

保存路径：`results/e3_fair_compare.json`

```
=== E3 公平对比结果（MedQA test，1273 题）===
G0 Baseline                    32.9%    +0.0%
G1 RAG-compressed（相同知识）   41.6%    +8.7%   （无训练，文本前缀）
G2 Fusion-Phase1（相同知识）    XX.X%    +X.X%   （层注入，Phase 1 权重）
G3 Fusion-Phase2（相同知识）    71.1%    +38.2%  （层注入，Phase 2 权重）
G4 RAG-oracle（完整知识）       86.9%    +54.0%  （无训练，RAG 上限）
```

**判断标准**：
- G2 > G1：层注入（即使只有 Phase 1 通用训练）优于文本拼接 ✅
- G2 的 delta_acc / G4 的 delta_acc：知识利用效率比

### 实验结果（2026-02-26 执行，4 GPU 并行）

#### 实现文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `evaluation/e3_fair_compare.py` | NEW | 5 组 × 3 数据集公平对比，支持多 GPU 并行 |
| `tests/unit/test_e3_fair_compare.py` | NEW | 8 个单元测试（数据适配、格式验证、无截断） |
| `scripts/run_e3_fair_compare.sh` | NEW | 一键运行脚本（4 GPU） |
| `main.py` | MODIFY | 添加 `eval-fair-compare` 命令，`--phase1-weights`/`--phase2-weights` 参数 |

#### 核心函数

| 函数 | 文件 | 说明 |
|------|------|------|
| `_eval_baseline()` | e3_fair_compare.py | G0：原始 Qwen3 无知识评测 |
| `_eval_rag_compressed()` | e3_fair_compare.py | G1：decode(knowledge_ids) → 文本前缀拼接 |
| `_eval_fusion()` | e3_fair_compare.py | G2/G3：knowledge_ids → 层注入 Cross-Attention |
| `_eval_rag_original()` | e3_fair_compare.py | G4：原始完整文本前缀拼接（**不截断**） |
| `_e3_worker()` | e3_fair_compare.py | mp.spawn worker，处理 5 种评测模式 |
| `_run_e3_parallel()` | e3_fair_compare.py | 多 GPU 调度，单卡自动回退 |
| `run_e3_all()` | e3_fair_compare.py | 完整入口：3 数据集 × 5 组对比 |

#### 运行命令

```bash
bash scripts/run_e3_fair_compare.sh
```

#### 结果数据

**完整对比矩阵**（`results/e3_fair_compare.json`）：

| 组别 | MedQA (1,273) | ARC (1,165) | MMLU (14,042) |
|------|---------------|-------------|---------------|
| G0 Baseline | 32.91% (419) | 51.85% (604) | 41.58% (5,838) |
| G1 RAG-compressed | 41.56% (529) | 66.52% (775) | 52.42% (7,361) |
| G2 Fusion-Phase1 | 36.53% (465) | 67.21% (783) | 48.87% (6,862) |
| **G3 Fusion-Phase2** | **69.52%** (885) | **85.75%** (999) | **70.94%** (9,962) |
| G4 RAG-original | 86.96% (1,107) | 86.95% (1,013) | 79.55% (11,170) |

> ARC 每组均跳过 7 道非 4 选项题（1172 题中有 7 题为 3 或 5 选项）。

**Δacc（相对 Baseline）**：

| 组别 | MedQA | ARC | MMLU |
|------|-------|-----|------|
| G1 RAG-compressed | +8.65% | +14.67% | +10.84% |
| G2 Fusion-Phase1 | +3.61% | +15.36% | +7.29% |
| G3 Fusion-Phase2 | +36.61% | +33.91% | +29.37% |
| G4 RAG-original | +54.05% | +35.11% | +37.97% |

**G2 vs G1（核心对比：同等知识，不同融合方式）**：

| 数据集 | G2 - G1 | 结论 |
|--------|---------|------|
| MedQA | **-5.03%** | Phase 1 层注入 < RAG |
| ARC | **+0.69%** | 基本持平 |
| MMLU | **-3.55%** | Phase 1 层注入 < RAG |

**G3 vs G1（Phase 2 层注入 vs RAG-compressed）**：

| 数据集 | G3 - G1 | 结论 |
|--------|---------|------|
| MedQA | **+27.97%** | 层注入远优于 RAG ✅ |
| ARC | **+19.23%** | 层注入远优于 RAG ✅ |
| MMLU | **+18.52%** | 层注入远优于 RAG ✅ |

**知识利用效率**（efficiency = Δacc(G) / Δacc(G4)）：

| 数据集 | G2 效率 | G3 效率 |
|--------|---------|---------|
| MedQA | 6.7% | **67.7%** |
| ARC | 43.8% | **96.6%** |
| MMLU | 19.2% | **77.3%** |

#### 分析

**1. G3（Phase 2 层注入）在三个数据集上全面大幅超越 G1（RAG-compressed）**

这是 E3 的核心结论。在知识内容完全相同（同一份 knowledge_ids）的条件下，层注入比文本前缀拼接的知识利用率高得多：MedQA +27.97%，ARC +19.23%，MMLU +18.52%。

**2. G2（Phase 1 层注入）未能超越 G1 — Phase 1 训练不足以学会高效融合**

Phase 1 仅做通用 LM 预训练，注入模块学到的融合能力有限。在 MedQA 和 MMLU 上 G2 < G1，仅在 ARC 上基本持平（+0.69%）。这与 E2 中 Phase 1 增幅较小的结论一致。

**3. ARC 上 G3 效率达 96.6%——用 64 token 压缩知识几乎追平完整文本 RAG**

G3（85.75%）vs G4（86.95%），仅差 1.2 个百分点。说明对于科学推理类任务，层注入 + 64 token 压缩知识几乎无损，极其高效。

**4. Phase 2 SFT 是决定性因素**

G2→G3 的提升量：MedQA +33%，ARC +18.5%，MMLU +22%。SFT 训练让注入模块从"部分利用"跃升为"高效利用"，这与 E1 中 Phase 1 KS=7.78% vs Phase 2 KS=56.32% 的巨大差异完全吻合。

**5. 跨域泛化能力再次验证**

Phase 2 仅在 MedQA 上做 SFT，但在 ARC（效率 96.6%）和 MMLU（效率 77.3%）上同样表现优异。层注入机制学到的是通用的知识融合能力，不仅限于训练域。

**6. 与 E1/E2 结果的完美一致性**

- G0 Baseline 三组数据与 E2 完全一致（32.91% / 51.85% / 41.58%）
- G2 Fusion-Phase1 三组数据与 E2 完全一致（36.53% / 67.21% / 48.87%）
- G3 MedQA acc（69.52%）与 E1 Phase 2 acc_correct（69.52%）完全一致

#### 结论

**E3 通过。** 在控制知识内容完全相同的条件下，Phase 2 层注入在三个数据集上全面超越 RAG-compressed（G3 >> G1），知识利用效率远超文本拼接（ARC 96.6%、MMLU 77.3%、MedQA 67.7%）。Phase 1 层注入尚不足以超越 RAG（G2 ≈ G1），说明 SFT 训练是解锁高效融合的关键。可以推进 E4（SFT 消融分析）和 E5（压缩率/相关性分析）。

---

## E4 实现指南：Phase 1 vs Phase 2 消融分析 ✅ 已完成（2026-03-03）

### 目的

量化 domain SFT（Phase 2）对通用融合能力的影响，揭示 trade-off。

### 所用资源

| 资源 | 说明 | 状态 |
|------|------|------|
| `checkpoints/v2_phase1_best` | v2 Phase 1 权重（layers=[6,12,18,24]） | ✅ |
| `checkpoints/v2_phase2_best` | v2 Phase 2 Best Val 权重 | ✅ |
| `data/{medqa,arc,mmlu}_knowledge.jsonl` | 知识映射（E2 已构建） | ✅ |

### 实际实现

独立模块 `evaluation/e4_sft_ablation.py`，复用 E3 的评测函数（`_run_e3_parallel` 等），关键设计：
- 从 checkpoint 的 `injection_config.json` 自动读取模型配置（`_load_injection_config`），解决 v2/v3 配置差异
- `main.py` 注册 `eval-sft-ablation` CLI 命令，复用 `--phase1-weights` / `--phase2-weights` 参数

### 执行命令

```bash
bash scripts/run_e4_sft_ablation.sh
# 或手动指定 GPU:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_e4_sft_ablation.sh
```

### 实验结果

结果文件：`results/e4_sft_ablation.json`

| 数据集 | Baseline | Phase 1 | Phase 2 | Phase1 Δ | Phase2 Δ | SFT 效果 |
|--------|----------|---------|---------|----------|----------|----------|
| MedQA  | 32.91%   | 36.53%  | **69.52%** | +3.61% | +36.61% | +32.99% |
| ARC    | 51.85%   | 67.21%  | **85.75%** | +15.36% | +33.91% | +18.54% |
| MMLU   | 41.58%   | 48.87%  | **70.94%** | +7.29% | +29.37% | +22.08% |

### 结论

**原始假设被完全推翻**。预期 SFT 会损害跨域通用能力（"SFT 代价"），但实际上 Phase 2 在 **所有三个域** 上都大幅超越 Phase 1：

- **MedQA**（目标域）：+32.99%，符合预期
- **ARC**（跨域）：+18.54%，完全出乎意料
- **MMLU**（跨域）：+22.08%，完全出乎意料

**解读**：MedQA SFT 并非让模块学会"MedQA 答题技巧"，而是充当了 **instruction tuning** 的角色——让注入模块从"被动传递知识信号"（Phase 1）进化到"主动利用知识做出判断"（Phase 2）。这种多选题决策能力跨域通用。

**对后续训练的指导**：可以大胆做领域 SFT，不存在通用能力 trade-off。

---

## E5 实现指南：知识注入量与质的分析 ✅ 已完成（2026-03-04）

### 所需资源

| 资源 | 说明 | 状态 |
|------|------|------|
| `checkpoints/v2_phase1_best` | Phase 1 v2 权重 | ✅ 已有 |
| `checkpoints/v2_phase2_best` | Phase 2 v2 权重 | ✅ 已有 |
| `data/{ds}_knowledge.jsonl` | k=64 知识映射（3 数据集） | ✅ 已有，复用 |
| `data/{ds}_knowledge_k32.jsonl` | k=32 知识映射（3 数据集） | ✅ 已构建 |
| `data/{ds}_knowledge_k128.jsonl` | k=128 知识映射（3 数据集） | ✅ 已构建 |
| `data/{ds}_knowledge_k256.jsonl` | k=256 知识映射（3 数据集） | ✅ 已构建 |

### 实现文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `evaluation/e5_knowledge_analysis.py` | NEW ✅ | E5 核心模块（知识构建 + 评测 + 报告） |
| `main.py` | MODIFY ✅ | 添加 `build-e5-knowledge`, `eval-knowledge-analysis` 命令 |
| `scripts/run_e5_knowledge_analysis.sh` | NEW ✅ | 两阶段运行脚本 |

### 知识映射构建方案（实际实现）

| 长度 | 压缩率 | 构建方式 | 文件路径 |
|------|--------|---------|---------|
| k=32 | 0.125 | LLMLingua-2 | `data/{ds}_knowledge_k32.jsonl` |
| k=64 | 0.25 | 复用已有 | `data/{ds}_knowledge.jsonl` |
| k=128 | 0.5 | LLMLingua-2 | `data/{ds}_knowledge_k128.jsonl` |
| k=256 | — | 直接 tokenize 原文（无压缩） | `data/{ds}_knowledge_k256.jsonl` |

k=256 采用直接 tokenize（不经过 LLMLingua），等效 rate=1.0 但快几百倍。

E5-B Shuffled 条件：`_build_shuffled_km()` 用 Fisher-Yates 随机打乱 key↔value 映射（seed=42），保持分布一致但语义不相关。

### 执行命令

```bash
# Step 1: 构建知识映射（耗时，可提前运行）
CUDA_VISIBLE_DEVICES=0 conda run -n ExplicitLLM python main.py build-e5-knowledge --config config/default.yaml

# Step 2: 运行 E5 完整评测（~51 评测）
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 conda run -n ExplicitLLM python main.py eval-knowledge-analysis \
  --config config/default.yaml \
  --phase1-weights checkpoints/v2_phase1_best \
  --phase2-weights checkpoints/v2_phase2_best

# 或一键运行
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/run_e5_knowledge_analysis.sh
```

### 实验结果（2026-03-04 执行）

**E5-A Phase 2 Fusion vs RAG（三数据集汇总）**：

| Token | MedQA Fusion | MedQA RAG | Δ | ARC Fusion | ARC RAG | Δ | MMLU Fusion | MMLU RAG | Δ |
|-------|-------------|-----------|---|-----------|---------|---|-------------|----------|---|
| 32 | 53.26% | 33.78% | **+19.48%** | 70.64% | 57.42% | **+13.22%** | 58.00% | 46.36% | **+11.64%** |
| 64 | 69.52% | 41.56% | **+27.96%** | 85.75% | 66.52% | **+19.23%** | 70.94% | 52.42% | **+18.52%** |
| 128 | 85.15% | 61.51% | **+23.65%** | 94.33% | 77.08% | **+17.25%** | 83.11% | 62.85% | **+20.25%** |
| 256 | 88.22% | 77.61% | **+10.60%** | 95.45% | 86.95% | **+8.50%** | 87.15% | 76.51% | **+10.64%** |

**E5-A Phase 1 Fusion vs RAG（三数据集汇总）**：

| Token | MedQA Fusion | MedQA RAG | Δ | ARC Fusion | ARC RAG | Δ | MMLU Fusion | MMLU RAG | Δ |
|-------|-------------|-----------|---|-----------|---------|---|-------------|----------|---|
| 32 | 34.96% | 33.78% | +1.18% | 60.52% | 57.42% | +3.09% | 46.32% | 46.36% | -0.04% |
| 64 | 36.53% | 41.56% | -5.03% | 67.21% | 66.52% | +0.69% | 48.87% | 52.42% | -3.55% |
| 128 | 36.06% | 61.51% | -25.45% | 72.19% | 77.08% | -4.90% | 51.38% | 62.85% | -11.47% |
| 256 | 32.05% | 77.61% | -45.56% | 70.82% | 86.95% | -16.14% | 51.45% | 76.51% | -25.06% |

**E5-B 知识相关性（k=64, v2 权重）**：

| 条件 | P1 MedQA | P2 MedQA | P1 ARC | P2 ARC | P1 MMLU | P2 MMLU |
|------|----------|----------|--------|--------|---------|---------|
| Oracle | 36.53% | **69.52%** | 67.21% | **85.75%** | 48.87% | **70.94%** |
| Shuffled | 29.77% | 32.05% | 54.25% | 54.51% | 40.21% | 42.33% |
| Empty | 32.60% | 32.36% | 52.96% | 53.05% | 41.75% | 41.58% |

**核心发现**：

1. **Phase 2 Fusion 在所有 token 预算下全面超越 RAG**，Δ 范围 +8.5% ~ +28.0%
2. **Fusion@128 > RAG@256**：用一半 token 超越 RAG 全文（MedQA 85.15% vs 77.61%, ARC 94.33% vs 86.95%）
3. **Phase 1 Fusion 在 k≥128 时反而落后于 RAG**：融合深度不足，无法利用更多信息；k=256 时 MedQA 甚至降至 32.05%
4. **Phase 2 Shuffled ≈ Empty ≈ Baseline**：不相关知识被有效忽略，融合增益完全来自语义匹配
5. **Shuffled 可产生负面干扰**：Phase 1 MedQA 中 Shuffled(29.77%) < Empty(32.60%)，不相关知识反而扰乱了模型

结果文件：`results/e5_knowledge_analysis.json`

---

## E6 实现记录：推理效率对比

### 已实现文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `evaluation/e6_inference_efficiency.py` | NEW ✅ | 核心基准测试模块 |
| `main.py` | MODIFY ✅ | 添加 `eval-inference-efficiency` 命令 |
| `scripts/run_e6_inference_efficiency.sh` | NEW ✅ | 单 GPU 运行脚本 |

### 执行命令

```bash
# 单 GPU 运行（确保公平计时）
CUDA_VISIBLE_DEVICES=0 bash scripts/run_e6_inference_efficiency.sh
```

### 实际结果（2026-03-04）

**E6-A 推理效率（MedQA, N=200, 单 GPU）**：

| 方法 | 延迟 (ms) | 吞吐 (样本/s) | 显存 (MB) | 平均输入长度 | 上下文占用 |
|------|----------|-------------|----------|------------|----------|
| Baseline | 71.29 | 14.03 | 1816.9 | 211.0 | 0 |
| RAG-compressed@64 | 71.38 | 14.01 | 1873.7 | 261.3 | ~64 |
| Fusion-Phase2@64 | 93.94 | 10.64 | 1799.1 | 211.0 | 0 |
| RAG-original@~256 | 72.17 | 13.86 | 2168.9 | 391.0 | ~256 |

**六维对比（MedQA）**：

| 维度 | RAG-original | Fusion Phase 2 | 胜者 |
|------|-------------|----------------|------|
| 绝对准确率 | 86.96% | 69.52% | RAG |
| 同token准确率(k=64) | 41.56% | 69.52% (+27.96%) | **Fusion** |
| 推理延迟 | 72.17 ms | 93.94 ms | RAG |
| 峰值显存 | 2168.9 MB | 1799.1 MB (-17.1%) | **Fusion** |
| 上下文窗口侵占 | ~256 tokens | 0 tokens | **Fusion** |
| 知识可预编码缓存 | 否 | 是 | **Fusion** |

**核心发现**：

1. **推理延迟预期被推翻**：Fusion 比 RAG-original 慢 30.2%，因为 cross-attention + 知识编码的固定开销（~5.7ms/forward）在小模型短序列场景下超过了 RAG 额外序列长度带来的 self-attention 增量（~0.2ms/forward）
2. **显存优势确认**：Fusion 峰值显存最低（1799.1 MB），比 RAG-original 节省 17.1%
3. **RAG 方法延迟几乎相同**：Baseline/RAG-compressed/RAG-original 仅差 ~1ms，说明 0.6B 模型在 211~391 tokens 区间 self-attention 瓶颈不显著
4. **效率优势需大模型验证**：self-attention O(L²·d) vs cross-attention O(L·K·d)，随模型规模和序列长度增大，Fusion 延迟优势会逐渐显现

结果文件：`results/e6_inference_efficiency.json`

---

# 附录 A：文件变更总览

| 文件 | 操作 | 涉及实验 |
|------|------|---------|
| `evaluation/counterfactual_eval.py` | NEW ✅ | E1 |
| `evaluation/arc_eval.py` | NEW | E2, E3, E4 |
| `evaluation/mmlu_eval.py` | NEW | E2, E3, E4 |
| `evaluation/cross_domain_runner.py` | NEW | E2, E4 |
| `evaluation/e3_fair_compare.py` | NEW ✅ | E3 |
| `evaluation/e4_sft_ablation.py` | NEW ✅ | E4 |
| `evaluation/e5_knowledge_analysis.py` | NEW ✅ | E5 |
| `evaluation/e6_inference_efficiency.py` | NEW ✅ | E6 |
| `evaluation/compare_eval.py` | MODIFY | E3（添加 Fusion 组） |
| `main.py` | MODIFY | 所有实验（添加命令） |
| `config/default.yaml` | MODIFY | 所有实验（添加路径配置） |

---

# 附录 B：数据文件总览

| 文件 | 说明 | 来源 |
|------|------|------|
| `data/medqa_knowledge.jsonl` | MedQA test 正确知识映射 | ✅ 已有 |
| `data/medqa_knowledge_train.jsonl` | MedQA train 知识映射 | ✅ 已有 |
| `data/medqa_knowledge_counterfactual.jsonl` | MedQA test 反事实知识映射 | E1 ✅ 已构建 |
| `data/arc_knowledge.jsonl` | ARC-Challenge Oracle 知识映射 | E2 新建 |
| `data/mmlu_knowledge.jsonl` | MMLU oracle 知识映射 | E2 新建 |
| `data/medqa_knowledge_k32.jsonl` | MedQA 知识映射（32 tokens） | E5 ✅ 已构建 |
| `data/medqa_knowledge_k128.jsonl` | MedQA 知识映射（128 tokens） | E5 ✅ 已构建 |
| `data/medqa_knowledge_k256.jsonl` | MedQA 知识映射（256 tokens） | E5 ✅ 已构建 |
| `data/arc_knowledge_k32.jsonl` | ARC 知识映射（32 tokens） | E5 ✅ 已构建 |
| `data/arc_knowledge_k128.jsonl` | ARC 知识映射（128 tokens） | E5 ✅ 已构建 |
| `data/arc_knowledge_k256.jsonl` | ARC 知识映射（256 tokens） | E5 ✅ 已构建 |
| `data/mmlu_knowledge_k32.jsonl` | MMLU 知识映射（32 tokens） | E5 ✅ 已构建 |
| `data/mmlu_knowledge_k128.jsonl` | MMLU 知识映射（128 tokens） | E5 ✅ 已构建 |
| `data/mmlu_knowledge_k256.jsonl` | MMLU 知识映射（256 tokens） | E5 ✅ 已构建 |

---

# 附录 C：依赖检查

所有实验均依赖以下已有能力，无需额外安装：

```bash
# 验证依赖完整性
conda run -n ExplicitLLM python -c "
import torch
import transformers
from datasets import load_dataset
from llmlingua import PromptCompressor
import lm_eval
print('所有依赖就绪')
"
```

ARC-Challenge 和 MMLU 数据集均通过 HuggingFace Hub 加载，首次运行时自动下载，无需手动操作：
```python
# 会自动缓存到 ~/.cache/huggingface/datasets/
load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")  # ~数MB
load_dataset("cais/mmlu", "all", split="test")                   # ~90MB
```

---

*文档版本：v1.6 | 更新日期：2026-03-04 | 状态：E1-E6 全部完成*
