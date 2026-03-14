# ExplicitLM-LoRA 统一架构设计

## 1. 核心定位

为大语言模型构建 **显式持续学习模块**：结合 Product Key Memory 路由 + Cross-Attention 知识融合，实现动态知识的存储、检索与注入。

```
设计原则:
  - 基础模型完全冻结，仅训练外挂模块（~25M 参数）
  - 知识双存储：Fusion Bank 存压缩 facts（注入用），Anchor Bank 存原文截断（路由索引用）
  - 路由与融合解耦：Router 负责"找什么"，Injection 负责"怎么用"
  - 零初始化策略：初始状态等价于无注入的原始模型
  - 两阶段训练：预训练学融合能力，SFT 激活下游任务性能
```

---

## 2. 系统总览

```
 ┌─────────── 离线：知识构建（初始化 / 可增量更新） ──────────┐
 │                                                        │
 │  原始语料                   双路存储                    │
 │  ┌──────────┐                                          │
 │  │ FineWeb  │──LLMLingua──→ Fusion Bank [N, K_f]       │
 │  │ MedQA    │   压缩(40%)   (压缩 facts token IDs)     │
 │  │          │                                          │
 │  │          │──直接截断───→ Anchor Bank [N, K_a]        │
 │  └──────────┘               (原文截断 token IDs)        │
 │                                       │                │
 │  聚类基于 Anchor Bank:                 │                │
 │  ┌──────────────────────────────────┐ │                │
 │  │ embed(Anchor Bank) →             │ │                │
 │  │ 独立子空间聚类 → 倒排索引         │ │                │
 │  │ Row Keys [√N, D] + Col Keys     │ │                │
 │  └──────────────────────────────────┘ │                │
 │                                       │                │
 └───────────────────────────────────────│────────────────┘
                                         │
 ┌─────────────── 在线：知识检索 ────────│────────────────┐
 │                                       ↓                │
 │  input ──Embed──→ x ──→ MemoryRouter                  │
 │                         │  Product Key Memory (粗排)    │
 │                         │  倒排索引查询                  │
 │                         │  RefinedSelector (精排)        │
 │                         └──→ knowledge_ids [B, K]       │
 │                                │                        │
 └────────────────────────────────│────────────────────────┘
                                  │
 ┌─────────────── 在线：知识融合 ──│────────────────────────┐
 │                                ↓                        │
 │  input_ids ──→ Qwen3 (冻结)                             │
 │                  │                                      │
 │     知识编码:    knowledge_ids → Qwen3前6层 → [B,K,D]   │
 │                  │                                      │
 │     Layer 6  ──Hook──→ CrossAttn(hidden, knowledge)     │
 │     Layer 12 ──Hook──→ CrossAttn(hidden, knowledge)     │
 │     Layer 18 ──Hook──→ CrossAttn(hidden, knowledge)     │
 │     Layer 24 ──Hook──→ CrossAttn(hidden, knowledge)     │
 │                  │                                      │
 │                  ↓                                      │
 │              logits [B, L, V]                           │
 └─────────────────────────────────────────────────────────┘
```

---

## 3. 双存储架构：Fusion Bank + Anchor Bank

知识以 token 序列双路存储，支持动态增删和聚类索引。两个 Bank 服务于不同目的：

- **Fusion Bank**：存 LLMLingua 压缩后的 facts token IDs，供 Fusion 模块编码注入
- **Anchor Bank**：存原始文本截断的 token IDs，供聚类计算 embedding → 更新 Router Keys

```
双存储结构
├── fusion_bank: Tensor [N, K_f]   # N 条知识，每条 K_f=64 tokens（压缩 facts）
├── anchor_bank: Tensor [N, K_a]   # N 条知识，每条 K_a=128 tokens（原文截断）
├── valid_mask: Tensor [N]         # 有效条目标记（共享）
├── inverted_index: Tensor [N]     # 按 cluster 排序的数据 ID（共享）
├── cluster_offsets: Tensor [C+1]  # 每个 cluster 的起始偏移（共享）
├── cluster_counts: Tensor [C]     # 每个 cluster 的条目数（共享）
├── pca_matrix: Tensor [D, D]     # 上次全量聚类的 PCA 旋转矩阵（近似分配用）
├── next_free: int                 # 下一个可写入的空闲槽位
└── change_counter: int            # 自上次 recluster 以来的累计变更数

其中:
  N   = knowledge_num (默认 1024×1024 = 1M)
  K_f = fusion_length (默认 64 tokens，压缩后)
  K_a = anchor_length (默认 128 tokens，原文截断)
  C   = num_keys² (网格索引总数)
  pca_matrix 由全量 recluster 产出，用于新增条目的近似 cluster 分配
```

**为什么需要两个 Bank？**

| 维度 | Fusion Bank | Anchor Bank |
|------|-------------|-------------|
| 内容 | LLMLingua 压缩后的高密度知识 | 原始文本直接截断 |
| 用途 | 知识编码 → Cross-Attention 注入 | 计算 embedding → 聚类 → 更新 Keys |
| 语义空间 | 压缩语义（信息密度高，但词序/表达被改变） | 原文语义（与 query 同一语义空间） |
| 关键原因 | Fusion 需要高密度知识最大化注入效果 | Router 匹配在原文语义空间（query 也是原文），聚类必须基于原文 embedding |

**知识构建管线（双路输出）**：

```
原始文本 (256+ tokens)
     │
     ├──── LLMLingua-2 (XLM-RoBERTa) ────┐
     │     ├─ compression_rate = 0.4      │
     │     ├─ 保留关键 token，删除冗余词   │
     │     └─ 批量处理 (batch_size=196)   │
     │              ↓                     │
     │     compressed_text (~64 tokens)   │
     │              ↓                     │
     │     Tokenize → [K_f] ──→ Fusion Bank
     │
     └──── 直接截断 (前 128 tokens) ──────┐
                    ↓                     │
           Tokenize → [K_a] ──→ Anchor Bank
```

---

## 4. MemoryRouter：知识检索

### 4.1 Product Key Memory（粗排）

两维独立路由，将 N 个知识条目映射到 √N × √N 网格。

```
输入: embedding [B, D]  (D=1024, Qwen3 hidden_dim)

Query Projection:
  q = W_q(embedding)           → [B, 1024]
  q1, q2 = split(q, dim=-1)   → [B, 512], [B, 512]
  q1, q2 = L2_normalize(q1), L2_normalize(q2)

Key Matching:
  scores_1 = q1 @ proj(row_keys).T   → [B, √N]
  scores_2 = q2 @ proj(col_keys).T   → [B, √N]

Candidate Generation:
  top_rows = topk(scores_1, K_COARSE=4)     → 4 个 row 索引
  top_cols = topk(scores_2, K_COARSE=4)     → 4 个 col 索引
  grid_indices = rows × √N + cols           → 16 个候选 cluster

倒排索引查询:
  for each grid_idx:
    start = cluster_offsets[grid_idx]
    count = cluster_counts[grid_idx]
    data_ids = inverted_index[start : start+count]
  → all_candidates [B, ~256]  ← 截断/填充至 num_candidates=256
```

### 4.2 独立子空间聚类

每个 epoch 重新聚类，更新 Keys 和倒排索引。

```
输入: embeddings [N, D]

Step 1: PCA 旋转去相关
  X_centered = X - mean(X)
  U, S, Vt = SVD(cov(X_centered))
  X_rotated = X_centered @ U

Step 2: 交错分割（保证子空间正交）
  subspace_1 = X_rotated[:, 0::2]   # PC[0,2,4,...] → Row
  subspace_2 = X_rotated[:, 1::2]   # PC[1,3,5,...] → Col

Step 3: 平衡聚类（Axis-Aligned Recursive Bisection）
  K 必须为 2 的幂次
  每步选一个维度，按中位数二分
  → 保证每个 cluster 条目数均匀

Step 4: 构造正交 Keys
  row_keys = [center_1, 0]          # 前半维度有值
  col_keys = [0, center_2]          # 后半维度有值
  → row_keys ⊥ col_keys（正交性保证）

Step 5: 构建倒排索引
  grid_idx = row_label × √N + col_label
  → sorted by grid_idx → offsets + counts
```

### 4.3 RefinedSelector（精排）

对粗排候选进行交叉编码精排。

```
┌─────────────────────────────────────────────────┐
│  RefinedSelector（2 层 Transformer 交叉编码器）   │
│                                                   │
│  输入:                                            │
│    query_vec  [B, 512]    ← FeatureAdapter 输出   │
│    cand_vecs  [B, C, 512] ← 候选知识编码          │
│                                                   │
│  拼接: [query; cand_1; ...; cand_C] → [B, 1+C, 512] │
│    ↓                                              │
│  TransformerEncoder (2 层, 8 heads)               │
│    ↓                                              │
│  提取候选部分 [:, 1:, :]                          │
│    ↓                                              │
│  Linear(512, 1) → scores [B, C]                   │
│    ↓                                              │
│  argmax → best_data_id                            │
└─────────────────────────────────────────────────┘
```

### 4.4 FeatureAdapter（特征适配）

将冻结 Embedding 投影到适配空间，防止 Feature Collapse。

```
冻结 Embedding [B, S, D]
    ↓
Centering（去模式偏差）
    ↓
LayerNorm
    ↓
Linear(D, 512) + Tanh
    ↓
Mean Pooling (带 mask) → [B, 512]
    ↓
Output LayerNorm（防止 collapse）
```

---

## 5. 知识融合：Hook 机制注入

### 5.1 ModifiedQwen 前向传播

```
input_ids [B, L]  +  knowledge_ids [B, K]
     ↓
┌──── 知识编码（一次编码，所有注入层复用） ────┐
│ embed_tokens(knowledge_ids) → [B, K, D]     │
│ 通过 Qwen3 前 6 层 → [B, K, D]             │
│ Final RMSNorm → [B, K, D]                  │
│ (可选) knowledge_adapter → [B, K, D]        │
└─────────────────────────────────────────────┘
     ↓
embed_tokens(input_ids) → [B, L, D]
     ↓
Qwen3 Layer 0-5（无注入，正常 forward）
     ↓
Layer 6 ──Hook──→ injection_module(hidden, knowledge)
     ↓
Layer 7-11
     ↓
Layer 12 ──Hook──→ injection_module(hidden, knowledge)
     ↓
Layer 13-17
     ↓
Layer 18 ──Hook──→ injection_module(hidden, knowledge)
     ↓
Layer 19-23
     ↓
Layer 24 ──Hook──→ injection_module(hidden, knowledge)
     ↓
Layer 25-27
     ↓
lm_head → logits [B, L, V]
```

### 5.2 AttentionInjection（Cross-Attention + Null KV）

统一接口：`forward(hidden: [B,L,D], knowledge: [B,K,D], mask: [B,K]) -> [B,L,D]`

```
hidden [B, L, D]                      knowledge [B, K, D]
     ↓                                      ↓
RMSNorm(hidden)                        拼接 Null KV
     ↓                                      ↓
Q = W_q(normed) [B,L,D]           K = [null_k; knowledge] [B, K+1, D]
                                   V = [null_v; knowledge] [B, K+1, D]
     ↓                                      ↓
     └──────── MultiHeadAttention ──────────┘
                    ↓
              attn_out [B, L, D]
                    ↓
        output = hidden + attn_out    ← 干净残差

关键设计:
  - out_proj 零初始化 → 训练初期 attn_out ≈ 0 → 自然退化为原始模型
  - null_k/null_v: 可学习参数，知识全 padding 时 attend to null → 数值稳定
  - PreNorm 残差：fn(norm(x), context) + x
```

### 5.3 三种注入方式对比

| 注入方式 | 原理 | 参数量/层 | 初始化 | 状态 |
|---------|------|---------|-------|------|
| **AttentionInjection** | Cross-Attention + Null KV + 残差 | ~4.2M | out_proj=0 | 主力方案 |
| ConcatProjection | mean_pool→concat→MLP(2D→4D→D)+残差 | ~12.6M | 末层=0 | 备选 |
| GatedInjection | per-dim gate × knowledge + 残差 | ~1K | gate=0 | 轻量备选 |

---

## 6. 训练管线

### 6.1 训练阶段总览

```
Phase 0: 知识构建（离线）
├─ LLMLingua 压缩语料 → Memory Bank
├─ 聚类 → Keys + 倒排索引
└─ 输出: data/compressed/*.parquet, Memory Bank

Phase 1: Router 训练（路由能力）
├─ 目标: 学习从输入精确定位知识条目
├─ 损失: CE(粗排) + KL(软标签) + CE(精排)
├─ 每 epoch 重新聚类 + 更新 Keys
└─ 输出: router_best checkpoint

Phase 2: Fusion 预训练（融合能力）
├─ 目标: 学习将检索到的知识有效注入 LLM
├─ 数据: FineWeb-Edu (通用语料)
├─ 任务: 语言建模（给定压缩知识，预测原文）
├─ 冻结: Qwen3 全量 + Router
├─ 可训练: InjectionModule × 4 + 知识编码器
└─ 输出: phase2_best checkpoint

Phase 3: 下游 SFT（任务激活）
├─ 目标: 在特定任务上精调融合模块
├─ 数据: MedQA / 其他下游数据
├─ 起点: Phase 2 最优权重
├─ 早停: val_loss 监控，patience=3
└─ 输出: phase3_best checkpoint
```

### 6.2 Router 训练细节

```python
# 每个 Epoch 的两阶段流程
for epoch in range(max_epochs):
    # ── Phase A: 数据与索引更新（Main Process） ──
    texts = sample(corpus, N=1M)
    facts = LLMLingua.compress(texts)                # 压缩
    fusion_bank.update(tokenize(facts))              # 压缩 facts → Fusion Bank
    anchor_bank.update(tokenize_truncate(texts))     # 原文截断 → Anchor Bank
    row_keys, col_keys, labels = recluster(          # 基于 Anchor Bank 聚类
        embeddings=embed(anchor_bank),
        num_keys=sqrt(N)
    )
    memory_gate.update_keys(row_keys, col_keys)      # 更新 Keys
    build_inverted_index(labels)                      # 构建倒排索引
    broadcast(labels, fusion_bank, anchor_bank)       # 广播至所有 rank

    # ── Phase B: 路由训练（所有 ranks） ──
    for batch in dataloader:
        s1, s2, teacher_1, teacher_2 = memory_gate(x, mask)
        refined_loss = refined_selector(candidates, target)

        # 损失组合
        ce_loss = CE(s1, target_row) + CE(s2, target_col)
        kl_loss = KL(s1, teacher_1) + KL(s2, teacher_2)
        coarse_loss = (1 - alpha) * ce_loss + alpha * kl_loss  # alpha=0.2
        total_loss = coarse_loss + refined_loss

        total_loss.backward()
        optimizer.step()
```

### 6.3 Fusion 训练细节

| 参数 | Phase 2 (预训练) | Phase 3 (SFT) |
|------|-----------------|---------------|
| **数据** | FineWeb-Edu | MedQA train |
| **任务** | 语言建模 | QA SFT |
| **lr** | 3e-4 | 1e-4 |
| **warmup** | 100 steps | 50 steps |
| **调度器** | Cosine decay | Cosine decay |
| **batch_size** | 32/卡 × 2卡 | 16/卡 × 2卡 |
| **梯度累积** | 4 | 1 |
| **混合精度** | bf16 | bf16 |
| **梯度裁剪** | 1.0 | 1.0 |
| **max_epochs** | 5 | 10 (早停) |
| **可训练** | Injection + Encoder | 同左 |
| **冻结** | Qwen3 + Router | 同左 |

```python
# Fusion 训练核心
for batch in dataloader:
    input_ids = batch["input_ids"]         # [B, 128]
    knowledge_ids = batch["knowledge_ids"] # [B, 64]
    labels = batch["labels"]               # [B, 128], padding=-100

    # 前向传播（Hook 自动注入知识）
    logits = modified_qwen(input_ids, knowledge_ids)

    # 损失（仅对有效 token 计算）
    loss = CrossEntropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)

    accelerator.backward(loss)
    clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    scheduler.step()
```

---

## 7. 端到端推理流程

```
用户输入 query
     ↓
┌─── Step 1: 路由检索 ───────────────────────────────────┐
│ x = qwen.embed(query)                                  │
│ x_pooled = feature_adapter(x)           → [1, 512]     │
│ candidates = memory_gate.coarse_search(x) → [1, ~256]  │
│   (路由基于 Anchor Bank embedding 构建的聚类索引)        │
│ best_id = refined_selector(x_pooled, candidates) → int │
│ knowledge_ids = fusion_bank[best_id]    → [1, K_f]     │
│   (从 Fusion Bank 取压缩 facts，用于注入)               │
└─────────────────────────────────────────────────────────┘
     ↓
┌─── Step 2: 知识融合生成 ───────────────────────────────┐
│ logits = modified_qwen(input_ids, knowledge_ids)        │
│ output = generate(logits)               → text          │
└─────────────────────────────────────────────────────────┘
```

---

## 8. 可训练 vs 冻结组件

```
冻结（不训练）:
  ✗ Qwen3-0.6B 全量参数 (600M)          # 保持原始语言能力
  ✗ LLMLingua-2 (XLM-RoBERTa)           # 知识压缩器
  ✗ Qwen3 Embedding Layer               # 共享词嵌入

可训练（Router，~15M）:
  ✓ MemoryGate.query_proj               # 查询投影 [D→1024]
  ✓ MemoryGate.row_key_proj             # 行键投影 [D→512]
  ✓ MemoryGate.col_key_proj             # 列键投影 [D→512]
  ✓ FeatureAdapter                      # 特征适配 [D→512]
  ✓ RefinedSelector                     # 精排交叉编码器

可训练（Fusion，~20M）:
  ✓ AttentionInjection × 4 层           # Cross-Attention 注入
  ✓ 知识编码器（Qwen3 前 6 层，解冻）    # 知识上下文化
  ✓ knowledge_adapter（可选 2 层 MLP）   # 编码适配

动态更新（非梯度优化）:
  ↻ row_keys / col_keys                 # 每 epoch 基于 Anchor Bank 聚类更新
  ↻ inverted_index                      # 跟随聚类重建
  ↻ Fusion Bank + Anchor Bank           # 知识条目增删（双 Bank 同步更新）
  → 详见 §9 知识动态更新（三种场景 + 操作流程）
```

---

## 9. 知识动态更新

### 9.1 更新场景总览

| 场景 | 触发时机 | 操作 | 索引维护 |
|------|---------|------|---------|
| A. 训练期全量重建 | 每 epoch 开始 | 重采样 N 条 → 双 Bank 全量替换 | 全量 recluster（见 §6.2） |
| B. 部署后批量灌入 | 新领域知识到位 | 批量 add_entries → 触发 recluster | 全量 recluster |
| D. 推理期热更新 | 运行时随时 | add/delete 单条或少量 → 近似分配 | 惰性 recluster |

### 9.2 核心操作

#### add_entries(texts: List[str])

伪代码：

```
1. facts = LLMLingua.compress(texts)
2. 对每条 text:
   - fusion_bank[next_free] = tokenize(fact)[:K_f]
   - anchor_bank[next_free] = tokenize(text)[:K_a]
   - valid_mask[next_free] = True
   - 近似 cluster 分配（见 §9.3）→ 追加到倒排索引
   - next_free += 1; change_counter += 1
3. 若 next_free >= N → 先 compact（见 §9.4），若仍满则报错
```

#### delete_entries(ids: List[int])

伪代码：

```
1. 对每个 id:
   - valid_mask[id] = False
   - change_counter += 1
2. 路由检索时自动跳过 valid_mask=False 的条目
3. 数据不立即清除（逻辑删除），物理清理由 compact 处理
```

#### should_recluster() → bool

触发条件：`change_counter / N_valid > recluster_threshold`（默认 0.1）

### 9.3 近似 Cluster 分配

新增条目不触发全量聚类，而是复用上次 recluster 保存的 PCA 旋转矩阵：

```
1. anchor_emb = embed(anchor_bank[new_id])
2. x_rotated = (anchor_emb - pca_mean) @ pca_matrix
3. sub1, sub2 = x_rotated[0::2], x_rotated[1::2]
4. row_label = argmin(||sub1 - row_centroids||)
5. col_label = argmin(||sub2 - col_centroids||)
6. grid_idx = row_label × √N + col_label
7. 追加 new_id 到 inverted_index 中 grid_idx 对应的段
```

注意：近似分配不修改 Keys，仅更新倒排索引。cluster 会渐进失衡，由周期性 recluster 恢复。

### 9.4 物理压缩（Compaction）

当逻辑删除条目累积过多（或 next_free 接近 N 上限）时执行：

```
1. 收集所有 valid_mask=True 的条目
2. 紧凑排列到 bank[0..N_valid-1]
3. 重置 next_free = N_valid
4. 全量 recluster（因为所有 ID 都变了）
5. 重置 change_counter = 0
```

Compaction 天然搭配 recluster，两者一起执行。

### 9.5 推理期热更新的并发安全

采用 `threading.Lock` 保护双 Bank + 倒排索引的读写：

- 推理（读）和更新（写）不会同时访问同一条目
- Lock 粒度：整个 add/delete 操作持锁，推理的路由阶段持锁
- 研究场景并发量低，粗粒度锁足够，无需读写锁优化

---

## 10. 核心数据结构

```python
# ── 配置 ──
@dataclass
class ModelConfig:
    base_model: str              # "Qwen3-0.6B"
    hidden_dim: int              # 1024
    num_layers: int              # 28
    injection_method: str        # "attention" | "concat" | "gated"
    injection_layers: List[int]  # [6, 12, 18, 24]
    encoder_depth: int           # 6
    fusion_length: int           # 64 (K_f, 压缩 facts 长度)
    anchor_length: int           # 128 (K_a, 原文截断长度)

@dataclass
class RouterConfig:
    knowledge_num: int           # 1024 * 1024
    dim: int                     # 1024 (backbone hidden_dim)
    query_dim: int               # 1024
    key_proj_dim: int            # 512
    adapter_dim: int             # 512
    num_candidates: int          # 32
    temperature: float           # 0.1

# ── 双存储 ──
fusion_bank:  Tensor[N, K_f]    # K_f=64, 压缩 facts token IDs
anchor_bank:  Tensor[N, K_a]    # K_a=128, 原文截断 token IDs
valid_mask:   Tensor[N]         # 有效条目标记（共享）

# ── 更新管理 ──
pca_matrix:       Tensor[D, D]      # 上次 recluster 的 PCA 旋转矩阵
pca_mean:         Tensor[D]         # 上次 recluster 的均值向量
row_centroids:    Tensor[√N, D//2]  # 行聚类中心（PCA 子空间）
col_centroids:    Tensor[√N, D//2]  # 列聚类中心（PCA 子空间）
next_free:        int               # 下一个可写入槽位
change_counter:   int               # 自上次 recluster 以来的变更计数
recluster_threshold: float          # 默认 0.1

# ── 训练批次 ──
TrainingBatch = {
    "input_ids":      Tensor[B, L],     # 主文本 token IDs
    "labels":         Tensor[B, L],     # 目标（padding=-100）
    "knowledge_ids":  Tensor[B, K_f],   # 知识 token IDs（来自 Fusion Bank）
    "attention_mask": Tensor[B, L],     # 注意力掩码
}

# ── Router 输出 ──
RouterOutput = {
    "best_id":        Tensor[B],        # 最佳知识条目索引
    "candidates":     Tensor[B, C],     # 候选索引
    "scores":         Tensor[B, C],     # 候选分数
    "knowledge_ids":  Tensor[B, K_f],   # 检索到的知识 token IDs（来自 Fusion Bank）
}
```

---

## 11. 与参考项目的关系

```
参考代码                              新架构                         来源
──────────────────────────────────────────────────────────────────────────
[Fusion] modified_model.py          → ModifiedQwen                 保留
[Fusion] injection_modules.py       → AttentionInjection           保留
[Fusion] qwen_wrapper.py            → QwenWrapper                  保留
[Fusion] trainer.py                 → FusionTrainer                适配
[Fusion] dataset.py                 → ExplicitDataset              保留
[Fusion] medqa_dataset.py           → MedQADataset                 保留
[Fusion] compressor.py              → KnowledgeCompressor          保留
[Fusion] parallel_pipeline.py       → ParallelBuildPipeline        保留
[Router] MemoryGate.py              → MemoryGate                   保留
[Router] clustering.py              → SubspaceClustering           保留
[Router] refined_selector.py        → RefinedSelector              保留
[Router] feature_adapter.py         → FeatureAdapter               保留
[Router] train_router.py            → RouterTrainer                重构
[Router] ParquetDataLoader.py       → DataLoader                   保留
[Router] FactExtractor.py          → KnowledgeCompressor           合并

新增:
  + router/memory_bank.py           Memory Bank 管理（从 model.py 拆出）
  + pipeline.py                     端到端推理管线
  + main.py                         统一 CLI 入口
  + config/default.yaml             全量配置
```

---

## 12. 文件结构

```
ExplicitLM-LoRA/
├── main.py                              # CLI 入口
├── config/
│   └── default.yaml                     # 全量配置
├── models/                              # 融合模块
│   ├── qwen_wrapper.py                  # Qwen3 封装
│   ├── injection_modules.py             # 注入模块（3 种）
│   └── modified_model.py               # Hook 注入核心
├── router/                              # 路由模块
│   ├── memory_bank.py                   # 知识存储
│   ├── memory_gate.py                   # Product Key Memory
│   ├── clustering.py                    # 独立子空间聚类
│   ├── feature_adapter.py              # 特征适配
│   ├── refined_selector.py             # 精排选择器
│   └── model.py                         # Router 整合
├── data_builder/                        # 数据构建
│   ├── compressor.py                    # LLMLingua 压缩
│   ├── data_loader.py                   # 流式数据加载
│   └── parallel_pipeline.py            # 多 GPU 并行
├── training/                            # 训练
│   ├── router_trainer.py               # Router 训练
│   ├── fusion_trainer.py               # Fusion 训练
│   └── dataset.py                       # 数据集定义
├── evaluation/                          # 评测
│   ├── compare_eval.py                  # 对比实验
│   └── run_eval.py                      # 评测入口
├── utils/
│   ├── logger_system.py                 # 日志系统
│   └── helpers.py                       # 工具函数
├── tests/                               # 测试
├── docs/                                # 文档
├── checkpoints/                         # 权重
├── data/                                # 数据
└── logs/                                # 日志
```
