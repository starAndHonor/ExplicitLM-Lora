# ExplicitLM 全系统实验方案与结果

> **状态**：框架已建立，结果待填入。
> **模型矩阵**：Qwen3-0.6B / 4B / 7B（三个基准尺度），14B+ 可选。
> **评测方法**：loglikelihood（对 " A"/" B"/" C"/" D" 取 continuation log-prob 最高者），跨所有实验统一。
> **数据集**：MedQA-USMLE (1,273) / ARC-Challenge (1,165) / MMLU (14,042)
> **参考**：继承 `Reference/Explicit-Lora-fusion/docs/fusion_experiment_plan.md`（E1-E6）的方法论，扩展至全系统。

---

## 0. 核心主张与论证逻辑

本研究需要论证五件事，有严格的依赖关系：

```
主张一（前提 A）：Router 能在百万级知识库中精确检索         ← E1 验证
  ↓ 路由质量是端到端系统的前提
主张二（前提 B）：Fusion 模块具有通用的知识融合能力         ← E2 + E3 验证
  ↓ 融合机制本身有效
主张三（核心贡献）：端到端系统（Router+Fusion）> RAG       ← E4 验证
  ↓ 完整系统优于输入拼接方案
主张四（核心卖点）：知识可动态增删更新，不损害性能          ← E6 验证
  ↓ 持续学习能力得证
主张五（效率 Scaling）：效率优势随模型规模增大而显著        ← E7 验证
  → O(L·K·d) vs O(L²·d) 的理论优势在 4B/7B 上得以验证
```

**失败模式分析**：
- 主张一不成立 → 主张三的端到端结果被路由误差拉低，需加强 Router 训练
- 主张二不成立 → 整个系统假说失败
- 主张四为何必须：项目核心定位是"显式持续学习"，动态更新是区别于 LoRA/RAG 的根本卖点
- 主张五为何必须：旧项目发现 0.6B 上 Fusion 延迟 > RAG（+30%），必须在更大模型上证明 Scaling 优势

### 训练管线与实验的时序关系

```
Phase 0: 知识构建（离线，每个模型独立）
  ├─ LLMLingua 压缩 → Fusion Bank
  ├─ 原文截断 → Anchor Bank
  ├─ 聚类 → Keys + 倒排索引
  └─ ※ 验证点: 聚类质量检查（Cluster 负载均衡）
        ↓
Phase 1: Router 训练
  ├─ CE(粗排) + KL(软标签) + CE(精排)
  └─ ※ 验证点: 【E1】Router 路由质量
        ↓
Phase 2: Fusion 预训练（通用融合能力）
  ├─ FineWeb-Edu 语言建模
  └─ ※ 验证点: 【E2】知识驱动性 + 【E3】跨域通用能力
        ↓
Phase 3: 下游 SFT（任务激活）
  ├─ MedQA SFT（early stopping）
  └─ ※ 验证点: 【E4】端到端 vs RAG + 【E5】消融分析
        ↓
全流程完成后:
  ├─ 【E6】动态更新验证
  └─ 【E7】Scaling & 效率
```

### Oracle 知识协议

所有 Fusion 相关实验（E2-E5）默认使用 Oracle 知识：

```
input (question/text) → [预配对] → knowledge_text → LLMLingua 压缩 → knowledge_ids (64 tokens)
```

端到端实验（E4）同时测试 Oracle 路由和真实路由，量化路由误差对最终性能的影响。

---

## E1：Router 路由质量验证 — 主张一（端到端的前提）

### 设计原理

Router 是端到端系统的入口。如果路由检索质量差，即使 Fusion 模块完美，最终性能也会被拉低。
因此 E1 是所有后续实验的**必要前提**——Phase 1 训练完成后必须立即运行，确认路由可用后再继续。

**实验设计**：

```
知识库: N=1M 条目（双 Bank）
查询: 评测集的每道题
指标: Recall@K（K=1,4,16,64），即真实对应知识是否在 Top-K 候选中

对比条件:
  ① 粗排 only — MemoryGate Top-16 clusters → ~64 候选
  ② 粗排 + 精排 — RefinedSelector 从候选中精选 Top-1
  ③ 随机 baseline — 随机选取（下界参考）

指标解读:
  Recall@1   → 精排后 Top-1 命中率（端到端使用此值）
  Recall@K   → 粗排候选的召回率曲线（K=4,16,64）
  精排提升   → Recall@1(粗排+精排) - Recall@1(粗排 only)
  负载均衡   → max/mean cluster size 比值（>3 则失衡）
```

**判断标准**：
- Recall@64 > 80%：粗排质量合格
- Recall@1 > 50%：端到端可用
- Recall@1 < 30%：Router 训练失败，需诊断

**三模型对比预期**：更大模型的 embedding 质量更好 → Recall 更高。

### E1-1 Recall@K 曲线（MedQA，N=1M）

| 模型 | 方法 | Recall@1 | Recall@4 | Recall@16 | Recall@64 |
|------|------|----------|----------|-----------|-----------|
| 0.6B | 粗排 only | | | | |
| 0.6B | 粗排 + 精排 | | | | |
| 0.6B | 随机 baseline | | | | |
| 4B | 粗排 only | | | | |
| 4B | 粗排 + 精排 | | | | |
| 4B | 随机 baseline | | | | |
| 7B | 粗排 only | | | | |
| 7B | 粗排 + 精排 | | | | |
| 7B | 随机 baseline | | | | |

### E1-2 Cluster 负载均衡

| 模型 | Max Cluster Size | Mean Cluster Size | Max/Mean 比值 | 判定 |
|------|-----------------|-------------------|---------------|------|
| 0.6B | | | | |
| 4B | | | | |
| 7B | | | | |

> 比值 > 3 则失衡，需调整聚类策略。

### E1-3 跨数据集 Recall@1（粗排+精排）

| 模型 | MedQA | ARC | MMLU |
|------|-------|-----|------|
| 0.6B | | | |
| 4B | | | |
| 7B | | | |

---

## E2：Fusion 知识驱动性验证（Sanity Check）— 主张二（前半）

### 设计原理

这是最基础的验证——Fusion 模块**真的在使用**注入的知识，而不是忽略知识仅靠模型本身能力答题。

**继承自**：旧项目 E1（反事实知识测试），方法论完全复用。

**实验设计**：

```
同一问题 Q，三种知识条件:
  ① 正确知识：compress(Q + correct_answer)
  ② 反事实知识：compress(Q + wrong_answer) — 故意给错误答案
  ③ 无知识：knowledge_ids 全为 pad_token_id

核心指标: Knowledge Sensitivity (KS) = acc(正确) - acc(反事实)
关键不等式: acc(正确) > acc(无知识) > random(25%) > acc(反事实)
```

**不等式解读**：
- `acc(正确) > acc(无知识)`：正确知识确实在帮助模型
- `acc(无知识) > random(25%)`：模型本身具备一定能力
- `acc(反事实) < random(25%)`：反事实知识在误导模型 → 证明模型确实在**依赖**注入的知识

**判断标准**：
- KS > 20%：知识注入显著有效
- KS ≈ 0：知识被忽略，训练有问题

**旧项目参考值**（0.6B Phase 2 权重）：KS = 56.32%

### E2-1 Knowledge Sensitivity — MedQA

| 模型 | 权重阶段 | acc(正确知识) | acc(反事实知识) | acc(无知识) | KS | 不等式成立？ |
|------|---------|-------------|---------------|-----------|-----|------------|
| 0.6B | Phase 2 | | | | | |
| 0.6B | Phase 3 | | | | | |
| 4B | Phase 2 | | | | | |
| 4B | Phase 3 | | | | | |
| 7B | Phase 2 | | | | | |
| 7B | Phase 3 | | | | | |

### E2-2 Knowledge Sensitivity — ARC

| 模型 | 权重阶段 | acc(正确知识) | acc(反事实知识) | acc(无知识) | KS | 不等式成立？ |
|------|---------|-------------|---------------|-----------|-----|------------|
| 0.6B | Phase 2 | | | | | |
| 0.6B | Phase 3 | | | | | |
| 4B | Phase 2 | | | | | |
| 4B | Phase 3 | | | | | |
| 7B | Phase 2 | | | | | |
| 7B | Phase 3 | | | | | |

### E2-3 Knowledge Sensitivity — MMLU

| 模型 | 权重阶段 | acc(正确知识) | acc(反事实知识) | acc(无知识) | KS | 不等式成立？ |
|------|---------|-------------|---------------|-----------|-----|------------|
| 0.6B | Phase 2 | | | | | |
| 0.6B | Phase 3 | | | | | |
| 4B | Phase 2 | | | | | |
| 4B | Phase 3 | | | | | |
| 7B | Phase 2 | | | | | |
| 7B | Phase 3 | | | | | |

---

## E3：跨域通用能力验证 — 主张二（后半）

### 设计原理

E2 证明了 Fusion 会使用知识，但仅在训练域（MedQA）内。E3 需要证明 Fusion 的融合能力是**通用的**——
在从未 fine-tune 过的域上也能有效利用知识。这是"通用知识融合模块"定位的核心支撑。

**继承自**：旧项目 E2。

**实验设计**：

```
使用 Phase 2 权重（通用预训练，无 domain SFT），在三个数据集上对比:
  ① Baseline：原始模型，无知识
  ② Fusion + Oracle 知识：Phase 2 权重 + 正确知识注入
  ③ Fusion + 空知识：Phase 2 权重 + 全 pad（消融：验证增益来自知识内容而非模块结构）
```

**判断标准**：
- 三个域 Δacc 全部 > 0：通用融合能力得证
- 仅一个域有效：存在域偏差，需要分析原因

**旧项目参考值**（0.6B Phase 2）：MedQA +3.6%, ARC +15.4%, MMLU +7.3%

**三模型对比预期**：更大模型的基础能力更强，Δacc 可能呈现不同趋势——如果基础模型已经很强，知识增益的绝对值可能更小但相对比例仍有意义。

### E3-1 跨域通用能力（Phase 2 权重）

| 模型 | 条件 | MedQA | ARC | MMLU |
|------|------|-------|-----|------|
| 0.6B | Baseline（无知识） | | | |
| 0.6B | Fusion + Oracle 知识 | | | |
| 0.6B | Fusion + 空知识（消融） | | | |
| 0.6B | **Δacc（Fusion - Baseline）** | | | |
| 4B | Baseline（无知识） | | | |
| 4B | Fusion + Oracle 知识 | | | |
| 4B | Fusion + 空知识（消融） | | | |
| 4B | **Δacc（Fusion - Baseline）** | | | |
| 7B | Baseline（无知识） | | | |
| 7B | Fusion + Oracle 知识 | | | |
| 7B | Fusion + 空知识（消融） | | | |
| 7B | **Δacc（Fusion - Baseline）** | | | |

---

## E4：端到端系统 vs RAG — 主张三（核心贡献）

### 设计原理

这是整个研究的**核心实验**。需要在严格控制变量的条件下，证明完整端到端系统（Router + Fusion）优于传统 RAG（文本前缀拼接）。

**继承自**：旧项目 E3（层注入 vs RAG 对比矩阵）。**关键扩展**：加入真实路由组（G5-G7），旧项目仅有 Oracle 组。

**八组对比矩阵设计**：

```
Oracle 路由组（消除路由误差，聚焦融合机制对比）:
  G0: Baseline — 无知识（下界）
  G1: RAG-compressed — decode(knowledge_ids) → 文本前缀拼接（64 tokens）
  G2: Fusion-Phase2 + Oracle — 层注入（Phase 2 权重 + Oracle 知识）
  G3: Fusion-Phase3 + Oracle — 层注入（Phase 3 权重 + Oracle 知识）
  G4: RAG-original — 原始完整文本前缀拼接（~256 tokens，准确率上限）

真实路由组（端到端系统，含路由误差）:
  G5: Fusion-Phase3 + 真实路由 — Router 检索知识 → 层注入
  G6: RAG-compressed + 真实路由 — Router 检索知识 → decode → 文本拼接
  G7: RAG-original + 真实路由 — Router 检索原文 → 文本拼接
```

**核心对比关系**：

| 对比 | 控制变量 | 论证目标 |
|------|---------|---------|
| G1 vs G2 | 同 Oracle 知识、同 64 tokens | 层注入 > 文本拼接（Fusion 核心论证） |
| G2 vs G3 | 同 Oracle 知识、同注入方式 | Phase 3 SFT 的增益 |
| G3 vs G5 | 同 Phase 3 权重、Oracle vs 真实路由 | 量化路由误差影响 |
| G5 vs G7 | 同真实路由 | 端到端 Fusion vs 端到端 RAG（最终系统对比） |
| G3 vs G4 | Oracle 路由、Fusion@64 vs RAG@256 | 知识利用效率（用更少 token 达到多少效果） |

**预期**：G2 > G1，G3 > G2，G5 接近 G3（路由损失小），G5 > G7

**旧项目参考值**（0.6B）：
- G3(Phase2+Oracle) = MedQA 69.52%, ARC 85.75%, MMLU 70.94%
- G4(RAG-original) = MedQA 86.96%, ARC 86.95%, MMLU 79.55%

### E4-1 八组对比矩阵 — MedQA

| 组 | 方法 | 0.6B acc | 4B acc | 7B acc |
|----|------|----------|--------|--------|
| G0 | Baseline（无知识） | | | |
| G1 | RAG-compressed（64 tokens 前缀拼接） | | | |
| G2 | Fusion-Phase2 + Oracle | | | |
| G3 | Fusion-Phase3 + Oracle | | | |
| G4 | RAG-original（~256 tokens 前缀拼接，上限） | | | |
| G5 | Fusion-Phase3 + 真实路由 | | | |
| G6 | RAG-compressed + 真实路由 | | | |
| G7 | RAG-original + 真实路由 | | | |

### E4-2 八组对比矩阵 — ARC

| 组 | 方法 | 0.6B acc | 4B acc | 7B acc |
|----|------|----------|--------|--------|
| G0 | Baseline（无知识） | | | |
| G1 | RAG-compressed（64 tokens 前缀拼接） | | | |
| G2 | Fusion-Phase2 + Oracle | | | |
| G3 | Fusion-Phase3 + Oracle | | | |
| G4 | RAG-original（~256 tokens 前缀拼接，上限） | | | |
| G5 | Fusion-Phase3 + 真实路由 | | | |
| G6 | RAG-compressed + 真实路由 | | | |
| G7 | RAG-original + 真实路由 | | | |

### E4-3 八组对比矩阵 — MMLU

| 组 | 方法 | 0.6B acc | 4B acc | 7B acc |
|----|------|----------|--------|--------|
| G0 | Baseline（无知识） | | | |
| G1 | RAG-compressed（64 tokens 前缀拼接） | | | |
| G2 | Fusion-Phase2 + Oracle | | | |
| G3 | Fusion-Phase3 + Oracle | | | |
| G4 | RAG-original（~256 tokens 前缀拼接，上限） | | | |
| G5 | Fusion-Phase3 + 真实路由 | | | |
| G6 | RAG-compressed + 真实路由 | | | |
| G7 | RAG-original + 真实路由 | | | |

### E4-4 核心对比指标（MedQA）

| 指标 | 定义 | 0.6B | 4B | 7B |
|------|------|------|----|----|
| **Δacc(G3-G0)** | Fusion-Phase3 Oracle 增益 | | | |
| **Δacc(G4-G0)** | RAG-original 增益（上限） | | | |
| **知识利用效率** | Δacc(G3) / Δacc(G4)，用 64 tokens 达到 256 tokens 多少效果 | | | |
| **路由损失** | acc(G3) - acc(G5)，路由误差的性能代价 | | | |
| **端到端优势** | acc(G5) - acc(G7)，端到端 Fusion vs 端到端 RAG | | | |
| **同 token Fusion vs RAG** | acc(G2) - acc(G1)，同 64 tokens 下层注入 vs 文本拼接 | | | |

> ARC/MMLU 同理计算，此处仅展示 MedQA 模板。

---

## E5：消融与深度分析 — 揭示系统内部机制

### E5-A 训练阶段消融

**设计原理**：量化 Phase 3 (domain SFT) 相对 Phase 2 (通用预训练) 的增益。
旧项目发现 SFT **无跨域 trade-off**（三域全面提升），违反常见预期。需在新架构上验证此结论。

| 模型 | 权重阶段 | MedQA | ARC | MMLU |
|------|---------|-------|-----|------|
| 0.6B | Phase 2 | | | |
| 0.6B | Phase 3 | | | |
| 0.6B | **Δ(Phase3 - Phase2)** | | | |
| 4B | Phase 2 | | | |
| 4B | Phase 3 | | | |
| 4B | **Δ(Phase3 - Phase2)** | | | |
| 7B | Phase 2 | | | |
| 7B | Phase 3 | | | |
| 7B | **Δ(Phase3 - Phase2)** | | | |

> **关注点**：如果 ARC/MMLU 的 Δ < 0，说明 SFT 引入了跨域 trade-off；如果 Δ > 0，说明 SFT 激活了通用能力。

### E5-B 知识 Token 预算分析

**设计原理**：Fusion 的核心卖点之一是"用更少的 token 达到更好的效果"。需要证明：
- Fusion@64 > RAG@64（同 token 预算，层注入更优）
- **关键验证**：Fusion@128 是否仍 > RAG@256（更少 token 仍胜出）

```
4 种 token 预算 (k=32/64/128/256)
× 3 种方法 (Baseline / RAG / Fusion)
× 3 数据集
```

#### E5-B-1 MedQA（Phase 3 权重）

| 模型 | 方法 | k=32 | k=64 | k=128 | k=256 |
|------|------|------|------|-------|-------|
| 0.6B | Baseline | | | | |
| 0.6B | RAG | | | | |
| 0.6B | Fusion | | | | |
| 4B | Baseline | | | | |
| 4B | RAG | | | | |
| 4B | Fusion | | | | |
| 7B | Baseline | | | | |
| 7B | RAG | | | | |
| 7B | Fusion | | | | |

#### E5-B-2 ARC（Phase 3 权重）

| 模型 | 方法 | k=32 | k=64 | k=128 | k=256 |
|------|------|------|------|-------|-------|
| 0.6B | Baseline | | | | |
| 0.6B | RAG | | | | |
| 0.6B | Fusion | | | | |
| 4B | Baseline | | | | |
| 4B | RAG | | | | |
| 4B | Fusion | | | | |
| 7B | Baseline | | | | |
| 7B | RAG | | | | |
| 7B | Fusion | | | | |

#### E5-B-3 MMLU（Phase 3 权重）

| 模型 | 方法 | k=32 | k=64 | k=128 | k=256 |
|------|------|------|------|-------|-------|
| 0.6B | Baseline | | | | |
| 0.6B | RAG | | | | |
| 0.6B | Fusion | | | | |
| 4B | Baseline | | | | |
| 4B | RAG | | | | |
| 4B | Fusion | | | | |
| 7B | Baseline | | | | |
| 7B | RAG | | | | |
| 7B | Fusion | | | | |

### E5-C 知识相关性分析

**设计原理**：排除一种反驳——"Fusion 的增益不是来自语义匹配，只是注入任何信号都能提升"。
通过注入**打乱映射的知识**（同分布但语义不匹配），证明增益完全来自正确的语义对应关系。

```
三种知识条件 (k=64):
  Oracle：正确知识（question → 对应的 compressed answer）
  Shuffled：随机打乱映射（question_i → knowledge_j，j≠i，同分布但不相关）
  Empty：无知识（全 pad）
```

| 模型 | 知识条件 | MedQA | ARC | MMLU |
|------|---------|-------|-----|------|
| 0.6B | Oracle（正确知识） | | | |
| 0.6B | Shuffled（随机打乱映射） | | | |
| 0.6B | Empty（无知识） | | | |
| 4B | Oracle（正确知识） | | | |
| 4B | Shuffled（随机打乱映射） | | | |
| 4B | Empty（无知识） | | | |
| 7B | Oracle（正确知识） | | | |
| 7B | Shuffled（随机打乱映射） | | | |
| 7B | Empty（无知识） | | | |

> **预期不等式**：Oracle >> Empty ≈ Shuffled > random(25%)
> 如果 Shuffled ≈ Oracle → 增益不来自语义匹配，系统假说有问题。

### E5-D 注入方式消融

**设计原理**：`architecture.md` §5.3 定义了三种注入方式，各有不同的参数量和计算复杂度。
需要在**三个模型规模上全量对比**，因为不同规模可能有不同的最优注入方式（小模型可能偏好轻量注入，大模型可能需要更强的融合能力）。

```
三种注入方式（同训练配置、同数据）:
  ① AttentionInjection — Cross-Attention + Null KV（~4.2M/层）
  ② ConcatProjection — mean_pool→concat→MLP（~12.6M/层）
  ③ GatedInjection — per-dim gate × knowledge（~1K/层）

训练要求: 3 注入方式 × 3 模型 = 9 组独立 Phase 2 + Phase 3 训练
```

#### E5-D-1 准确率对比（Phase 3 权重）

| 模型 | 注入方式 | 参数量/层 | MedQA | ARC | MMLU |
|------|---------|----------|-------|-----|------|
| 0.6B | AttentionInjection | ~4.2M | | | |
| 0.6B | ConcatProjection | ~12.6M | | | |
| 0.6B | GatedInjection | ~1K | | | |
| 4B | AttentionInjection | | | | |
| 4B | ConcatProjection | | | | |
| 4B | GatedInjection | | | | |
| 7B | AttentionInjection | | | | |
| 7B | ConcatProjection | | | | |
| 7B | GatedInjection | | | | |

#### E5-D-2 推理效率对比

| 模型 | 注入方式 | 总可训练参数 | 延迟 (ms/sample) | 峰值显存 (MB) |
|------|---------|------------|-----------------|--------------|
| 0.6B | AttentionInjection | | | |
| 0.6B | ConcatProjection | | | |
| 0.6B | GatedInjection | | | |
| 4B | AttentionInjection | | | |
| 4B | ConcatProjection | | | |
| 4B | GatedInjection | | | |
| 7B | AttentionInjection | | | |
| 7B | ConcatProjection | | | |
| 7B | GatedInjection | | | |

---

## E6：动态更新验证 — 主张四（核心卖点）

### 设计原理

项目的核心定位是"显式持续学习"。与 LoRA（需要重训）和 RAG（依赖外部检索、无法预编码）不同，
ExplicitLM 的知识存储在 Bank 中，可以在**部署后动态增删更新**而无需重新训练任何参数。

E6 需要证明三件事：
1. **批量灌入新知识后，旧知识性能不受影响**（E6-A）
2. **增量增删后，系统性能平稳**（E6-B）
3. **近似 cluster 分配的质量衰减可控**（E6-C）

### E6-A 批量灌入（跨域知识灌入后原域性能保持）

**流程**：

```
1. Phase 3 权重 + 原始知识库（N 条，MedQA 训练集）→ 评测 acc_before
2. 批量灌入 M 条 ARC 知识 → 全量 recluster
3. 重新评测 MedQA → acc_after_medqa（不应显著下降）
4. 评测 ARC → acc_after_arc（应有提升，证明新知识可检索可利用）
```

**知识来源**：ARC 知识灌入 MedQA 库（跨域灌入，最严格的测试）

| 模型 | 指标 | 灌入前 | 灌入后(MedQA) | 灌入后(ARC) |
|------|------|--------|-------------|------------|
| 0.6B | acc | | | |
| 0.6B | Recall@1 | | | |
| 4B | acc | | | |
| 4B | Recall@1 | | | |
| 7B | acc | | | |
| 7B | Recall@1 | | | |

> **判断标准**：灌入后 MedQA acc 下降 < 2%，ARC 可检索且有提升。

### E6-B 增量热更新（增删条目后性能变化）

**设计原理**：模拟运行时逐步扩充/删除知识的场景（场景 D）。
增量 add 使用近似 cluster 分配（不 recluster），验证近似分配是否足够好。

**流程**：

```
1. 初始知识库 N 条 → 评测 acc_0
2. 逐步 add 100/500/1000/5000 条（近似 cluster 分配）→ 评测
3. 逐步 delete 100/500/1000 条（逻辑删除）→ 评测
4. compact + recluster → 评测（验证恢复）
```

#### E6-B-1 增量 Add 后性能

| 模型 | 新增条目 | acc | Recall@1 | Recall@64 |
|------|---------|-----|----------|-----------|
| 0.6B | +0（基线） | | | |
| 0.6B | +100 | | | |
| 0.6B | +500 | | | |
| 0.6B | +1000 | | | |
| 0.6B | +5000 | | | |
| 4B | +0（基线） | | | |
| 4B | +100 | | | |
| 4B | +500 | | | |
| 4B | +1000 | | | |
| 4B | +5000 | | | |
| 7B | +0（基线） | | | |
| 7B | +100 | | | |
| 7B | +500 | | | |
| 7B | +1000 | | | |
| 7B | +5000 | | | |

#### E6-B-2 增量 Delete 后性能

| 模型 | 删除条目 | acc | Recall@1 | Recall@64 |
|------|---------|-----|----------|-----------|
| 0.6B | -0（基线） | | | |
| 0.6B | -100 | | | |
| 0.6B | -500 | | | |
| 0.6B | -1000 | | | |
| 4B | -0（基线） | | | |
| 4B | -100 | | | |
| 4B | -500 | | | |
| 4B | -1000 | | | |
| 7B | -0（基线） | | | |
| 7B | -100 | | | |
| 7B | -500 | | | |
| 7B | -1000 | | | |

#### E6-B-3 Compact + Recluster 后恢复

| 模型 | 阶段 | acc | Recall@1 | Recall@64 |
|------|------|-----|----------|-----------|
| 0.6B | 初始 | | | |
| 0.6B | 增删后（compact 前） | | | |
| 0.6B | compact + recluster 后 | | | |
| 4B | 初始 | | | |
| 4B | 增删后（compact 前） | | | |
| 4B | compact + recluster 后 | | | |
| 7B | 初始 | | | |
| 7B | 增删后（compact 前） | | | |
| 7B | compact + recluster 后 | | | |

### E6-C 近似分配质量衰减曲线

**设计原理**：近似 cluster 分配（不 recluster）会随着变更量增大而质量下降。
需要量化衰减曲线，确定合理的 recluster 触发阈值（`change_ratio`）。

```
变量: change_ratio = 自上次 recluster 以来的变更比例 (change_counter / N)
流程:
  1. 全量 recluster → Recall@K 基线
  2. 每次 add 1% 条目（近似分配）→ 测量 Recall@K
  3. 重复至 change_ratio = 0.2 (20%)
  4. 触发 recluster → 验证 Recall@K 恢复

知识来源: 随机生成的知识条目（纯测索引质量，不关心语义）
```

| 模型 | change_ratio | Recall@1 | Recall@4 | Recall@16 | Recall@64 |
|------|-------------|----------|----------|-----------|-----------|
| 0.6B | 0%（recluster 基线） | | | | |
| 0.6B | 1% | | | | |
| 0.6B | 2% | | | | |
| 0.6B | 5% | | | | |
| 0.6B | 10% | | | | |
| 0.6B | 15% | | | | |
| 0.6B | 20% | | | | |
| 0.6B | recluster 后恢复 | | | | |
| 4B | 0%（recluster 基线） | | | | |
| 4B | 5% | | | | |
| 4B | 10% | | | | |
| 4B | 20% | | | | |
| 4B | recluster 后恢复 | | | | |
| 7B | 0%（recluster 基线） | | | | |
| 7B | 5% | | | | |
| 7B | 10% | | | | |
| 7B | 20% | | | | |
| 7B | recluster 后恢复 | | | | |

> **期望**：Recall@K 随 change_ratio 缓慢下降，在 threshold=0.1 (10%) 附近仍可接受。
> recluster 后 Recall@K 应恢复到初始水平。

---

## E7：Scaling & 推理效率 — 主张五（效率论证）

### 设计原理

Fusion 的计算复杂度为 O(L·K·d)（K=知识 token 数，固定 64），而 RAG 前缀拼接的 self-attention 复杂度为 O(L²·d)。
理论上随着模型规模（d）增大，Fusion 的效率优势应该越来越明显。

旧项目发现 0.6B 上 Fusion 延迟 +30%（因为额外的 cross-attention 层和知识编码开销在小模型上占比大）。
**必须在更大模型上验证**——找到 Fusion 延迟 < RAG 的交叉点。

**继承自**：旧项目 E6。**关键扩展**：多模型规模。

### E7-A 推理效率基准测试

> **配置**：单 GPU, batch_size=1, N=200 样本, MedQA

```
4 种方法: Baseline / RAG-compressed@64 / Fusion@64 / RAG-original@256
× 3 种模型: Qwen3-0.6B / 4B / 7B
```

| 模型 | 方法 | 延迟 (ms/sample) | 吞吐 (samples/s) | 峰值显存 (MB) |
|------|------|-----------------|-----------------|--------------|
| 0.6B | Baseline | | | |
| 0.6B | RAG-compressed@64 | | | |
| 0.6B | Fusion@64 | | | |
| 0.6B | RAG-original@256 | | | |
| 4B | Baseline | | | |
| 4B | RAG-compressed@64 | | | |
| 4B | Fusion@64 | | | |
| 4B | RAG-original@256 | | | |
| 7B | Baseline | | | |
| 7B | RAG-compressed@64 | | | |
| 7B | Fusion@64 | | | |
| 7B | RAG-original@256 | | | |

> **旧项目参考**（0.6B）：Fusion 延迟 +30%、显存 -17.1%

### E7-B Scaling 趋势

> **理论预测**：
> - RAG 延迟 ∝ O(L²·d) → input_length 不变但 d 增大 → 延迟随 d 增长
> - Fusion 延迟 ∝ O(L·K·d) → K=64 固定 → 延迟增长更慢
>
> **目标**：找到交叉点（Fusion 开始比 RAG 快的模型规模）。如果 7B 上仍未交叉 → 外推至 14B+。

| 指标 | 0.6B | 4B | 7B | 14B (可选) |
|------|------|----|----|-----------|
| Fusion 延迟 (ms) | | | | |
| RAG-original 延迟 (ms) | | | | |
| Fusion / RAG 延迟比 | | | | |
| Fusion 显存 (MB) | | | | |
| RAG-original 显存 (MB) | | | | |
| **交叉点（延迟比 < 1.0）** | — | — | — | — |

### E7-C 六维对比框架

> 每个模型规模一张综合对比表，回应"为什么不直接用 RAG"。

#### Qwen3-0.6B

| 维度 | RAG-original | Fusion Phase 3 | 胜者 |
|------|-------------|----------------|------|
| 绝对准确率 (MedQA) | | | |
| 同 token 准确率 (k=64) | | | |
| 推理延迟 | | | |
| 峰值显存 | | | |
| 上下文窗口侵占 | ~256 tokens | 0 tokens | Fusion |
| 知识可预编码缓存 | 否 | 是 | Fusion |

#### Qwen3-4B

| 维度 | RAG-original | Fusion Phase 3 | 胜者 |
|------|-------------|----------------|------|
| 绝对准确率 (MedQA) | | | |
| 同 token 准确率 (k=64) | | | |
| 推理延迟 | | | |
| 峰值显存 | | | |
| 上下文窗口侵占 | ~256 tokens | 0 tokens | Fusion |
| 知识可预编码缓存 | 否 | 是 | Fusion |

#### Qwen3-7B

| 维度 | RAG-original | Fusion Phase 3 | 胜者 |
|------|-------------|----------------|------|
| 绝对准确率 (MedQA) | | | |
| 同 token 准确率 (k=64) | | | |
| 推理延迟 | | | |
| 峰值显存 | | | |
| 上下文窗口侵占 | ~256 tokens | 0 tokens | Fusion |
| 知识可预编码缓存 | 否 | 是 | Fusion |

---

## 附录 A：训练进度追踪

| Phase | 模型 | 注入方式 | 状态 | 最优 checkpoint | 备注 |
|-------|------|---------|------|----------------|------|
| Phase 0 | 0.6B | — | ⬜ | — | |
| Phase 1 | 0.6B | — | ⬜ | | |
| Phase 2 | 0.6B | AttentionInjection | ⬜ | | |
| Phase 2 | 0.6B | ConcatProjection | ⬜ | | |
| Phase 2 | 0.6B | GatedInjection | ⬜ | | |
| Phase 3 | 0.6B | AttentionInjection | ⬜ | | |
| Phase 3 | 0.6B | ConcatProjection | ⬜ | | |
| Phase 3 | 0.6B | GatedInjection | ⬜ | | |
| Phase 0 | 4B | — | ⬜ | — | |
| Phase 1 | 4B | — | ⬜ | | |
| Phase 2 | 4B | AttentionInjection | ⬜ | | |
| Phase 2 | 4B | ConcatProjection | ⬜ | | |
| Phase 2 | 4B | GatedInjection | ⬜ | | |
| Phase 3 | 4B | AttentionInjection | ⬜ | | |
| Phase 3 | 4B | ConcatProjection | ⬜ | | |
| Phase 3 | 4B | GatedInjection | ⬜ | | |
| Phase 0 | 7B | — | ⬜ | — | |
| Phase 1 | 7B | — | ⬜ | | |
| Phase 2 | 7B | AttentionInjection | ⬜ | | |
| Phase 2 | 7B | ConcatProjection | ⬜ | | |
| Phase 2 | 7B | GatedInjection | ⬜ | | |
| Phase 3 | 7B | AttentionInjection | ⬜ | | |
| Phase 3 | 7B | ConcatProjection | ⬜ | | |
| Phase 3 | 7B | GatedInjection | ⬜ | | |

> ⬜ 待开始 | 🔄 进行中 | ✅ 完成 | ❌ 失败

## 附录 B：实验执行进度

| 实验 | 0.6B | 4B | 7B |
|------|------|----|----|
| E1 Router 路由质量 | ⬜ | ⬜ | ⬜ |
| E2 知识驱动性验证 | ⬜ | ⬜ | ⬜ |
| E3 跨域通用能力 | ⬜ | ⬜ | ⬜ |
| E4 端到端 vs RAG | ⬜ | ⬜ | ⬜ |
| E5-A 训练阶段消融 | ⬜ | ⬜ | ⬜ |
| E5-B Token 预算分析 | ⬜ | ⬜ | ⬜ |
| E5-C 知识相关性 | ⬜ | ⬜ | ⬜ |
| E5-D 注入方式消融 | ⬜ | ⬜ | ⬜ |
| E6-A 批量灌入 | ⬜ | ⬜ | ⬜ |
| E6-B 增量热更新 | ⬜ | ⬜ | ⬜ |
| E6-C 衰减曲线 | ⬜ | ⬜ | ⬜ |
| E7-A 推理效率 | ⬜ | ⬜ | ⬜ |
| E7-B Scaling 趋势 | ⬜ | ⬜ | ⬜ |

## 附录 C：资源估算

### 主线训练（AttentionInjection）

| Phase | 0.6B | 4B | 7B |
|-------|------|----|----|
| Phase 0 (知识构建) | 1 GPU × 0.5天 | 1 GPU × 1天 | 1 GPU × 1.5天 |
| Phase 1 (Router) | 2 GPU × 1天 | 4 GPU × 2天 | 8 GPU × 3天 |
| Phase 2 (Fusion) | 2 GPU × 2天 | 4 GPU × 4天 | 8 GPU × 5天 |
| Phase 3 (SFT) | 2 GPU × 0.5天 | 4 GPU × 1天 | 8 GPU × 2天 |
| **小计** | **~4天** | **~8天** | **~11.5天** |

### E5-D 消融额外训练（ConcatProjection + GatedInjection × Phase 2+3）

| 额外训练 | 0.6B | 4B | 7B |
|---------|------|----|----|
| 2 注入方式 × (Phase 2 + Phase 3) | +5天 | +10天 | +14天 |

> **总训练量**：0.6B ~9天 + 4B ~18天 + 7B ~25.5天 ≈ **~52.5 GPU·天**
