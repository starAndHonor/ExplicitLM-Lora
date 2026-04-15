# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> [!URGENT]
> **研究性项目 (Research Project)**
> 1. 本项目为 MVP（最小可行性产品），严禁过度工程化。
> 2. 你的所有思考过程和回复必须使用 **简体中文**。

## 1. 项目概述

**核心目标**: 为大语言模型构建内置持续学习模块，使模型具备显式的动态知识更新能力。核心思路是：将知识存储在外部索引中，通过 dense retrieval 检索，再利用 forward hook 将知识注入冻结的 LLM，规避参数重训的算力成本与遗忘风险。

**基座模型**: Qwen3-0.6B（冻结，不更新参数）

**当前主线**: dense retrieval + qwen3/trainable 知识编码模式

**三层训练管线**:
- **Phase 1**: Router 训练 — 在 FineWeb-Edu 上训练路由器（PKM 粗排 + RefinedSelector 精排，旧方案）
- **Phase 2**: Fusion 预训练 — 训练 KnowledgeEncoder + AttentionInjection 模块（4层 hook）
- **Phase 3**: 下游 SFT — 在 MedQA 上微调，支持接 dense index 作为知识源，带 early stopping

**实验矩阵**: E1-E8，覆盖路由质量、跨域评估、RAG 对比、消融、效率、dense retrieval 扩展、editable memory。

## 2. 环境与命令

### 2.1 环境

- **Conda 环境**: `ExplicitLLM`（Python 3.11）
- **GPU 规范**: 必须且只能使用 GPU 6 — 所有 GPU 命令加 `CUDA_VISIBLE_DEVICES=6`
- **LLM**: 可通过 `.env` 环境变量配置

```bash
# 激活环境
conda activate ExplicitLLM

# 推荐执行方式
conda run -n ExplicitLLM python xxx
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python xxx
```

### 2.2 常用命令

**训练**:
```bash
# Phase 2 Fusion
MODEL_PATH=Qwen3-0.6B ENC_MODE=qwen3 NUM_GPUS=1 GPU_IDS=6 bash scripts/run_phase2_fusion.sh

# Phase 3 SFT（依赖 Phase 2 checkpoint）
MODEL_PATH=Qwen3-0.6B ENC_MODE=qwen3 FROM_PHASE2=checkpoints/p2_qwen3_10ep/phase2_best NUM_GPUS=1 GPU_IDS=6 bash scripts/run_phase3_sft.sh

# Dense Phase 3
MODEL_PATH=Qwen3-0.6B ENC_MODE=qwen3 FROM_PHASE2=... FROM_DENSE_INDEX=... KNOWLEDGE_SOURCE=dense_retriever NUM_GPUS=1 GPU_IDS=6 bash scripts/run_phase3_sft.sh
```

**推理**:
```bash
# Scheme2 推理
bash scripts/run_scheme2_p1_p3_infer.sh --question "..." --option-a "..." --option-b "..." --option-c "..." --option-d "..."

# Dense 检索推理
CUDA_VISIBLE_DEVICES=6 conda run -n ExplicitLLM python scripts/run_dense_phase3_infer.py --config config/default.yaml ...
```

**实验**:
```bash
bash scripts/run_experiment_auto.sh <e1|e2|e3|e3_multik|e4|e5|e6|e7|e7_full>
bash scripts/run_experiment_suite.sh  # 顺序运行 E1-E7
```

**评测与结果**:
```bash
conda run -n ExplicitLLM python scripts/collect_results.py  # 汇总所有实验结果
```

**代码质量**:
```bash
conda run -n ExplicitLLM ruff format xxx
conda run -n ExplicitLLM ruff check xxx --fix
```

**测试**:
```bash
conda run -n ExplicitLLM pytest tests/unit/ --cov=models --cov-report=term-missing
conda run -n ExplicitLLM pytest tests/integration/
conda run -n ExplicitLLM pytest tests/unit/test_injection_modules.py  # 单个测试
```

### 2.3 main.py CLI 入口

`main.py` 是生产 CLI，子命令：`build-knowledge`、`train --phase {0,1,2,3}`、`eval`、`answer`

用法：`conda run -n ExplicitLLM python main.py [--config path] [--device dev] [--override key=value ...] {子命令}`

## 3. 架构

### 3.1 核心数据流

```
用户输入 → DenseRetriever(dense向量检索) → 知识检索
         → KnowledgeEncoder(双向编码) → 知识向量
         → ModifiedQwen(hook注入4层) → 生成回答
```

### 3.2 模块职责

| 模块 | 核心类 | 职责 |
|------|--------|------|
| `models/qwen_wrapper.py` | `KnowledgeEncoder` | Qwen3 前 N 层 + 双向 attention 编码知识 |
| `models/modified_model.py` | `ModifiedQwen` | 冻结 Qwen3 + 在 [6,12,18,24] 层注册 hook 注入知识 |
| `models/injection_modules.py` | `AttentionInjection` | Cross-Attention + Null KV + zero-init 残差（主方法） |
| `router/model.py` | `MemoryRouter` | PKM 粗排 + RefinedSelector 精排的完整路由器 |
| `router/memory_bank.py` | `DualKnowledgeStore` | FusionBank[压缩事实,64tok] + AnchorBank[原文,128tok] |
| `router/memory_gate.py` | `ProductKeyMemory` | 二维 √N×√N 网格粗排，~256 候选 |
| `router/clustering.py` | `SubspaceClustering` | PCA 去相关 + 平衡递归二分 |
| `router/refined_selector.py` | `RefinedSelector` | 2 层 Transformer cross-encoder 精排 |
| `retrieval/dense_index.py` | `DenseKnowledgeIndex` | Dense 检索索引（Flat/HNSW 后端） |
| `pipeline.py` | `ExplicitLMPipeline` | 端到端：检索 → 编码 → 生成 |

### 3.3 知识双库设计

- **FusionBank** `[N, 64]`: LLMLingua 压缩后的知识，用于注入 LLM
- **AnchorBank** `[N, 128]`: 截断原文，用于路由和聚类
- 支持动态增删、compact + recluster

### 3.4 注入机制

在冻结 Qwen3 的第 6, 12, 18, 24 层注册 forward hook，通过 `AttentionInjection`（Cross-Attention）将编码后的知识与 hidden states 交互。zero-init 的 output projection 确保未训练时退化为原始模型。

## 4. 配置

**优先级**: CLI args > `.env` > `config/default.yaml`，统一归口到 `config.py` 中的 dataclass。

关键配置项（`config/default.yaml`）:
- 模型: hidden_dim=1024, 28 层, 注入层 [6,12,18,24], fusion_encoder_depth=6
- 路由: 1M 知识条目, 256 候选, adapter_dim=512
- 训练: Phase1 lr=5e-4/72batch/20ep, Phase2 lr=3e-4/32batch/10ep, Phase3 lr=1e-4/16batch/10ep
- 编码模式: `qwen3`（推荐）或 `trainable`

## 5. 开发规范

### 5.1 工作流程

1. **规划**: 阅读 `docs/` 下对应文档，理解设计意图，与用户讨论不明确之处
2. **计划**: 使用 plan 模式输出开发计划（摘要 → 审查点 → 变更列表 → 验证计划）
3. **等待用户批准**
4. **编码与验证**: 编码后运行测试确认通过

### 5.2 代码风格

- 函数签名必须包含完整类型注解
- 所有模块/类/方法必须有**中文 Docstring**
- 使用阶段化注释（`# Phase 1`, `# Phase 2`）组织复杂逻辑
- 类名 `PascalCase`，私有变量前缀 `_`
- 导入顺序：标准库 → 第三方库 → 项目内部
- **不考虑向后兼容**，直接修改原文件
- 禁用 `print()`，使用 `log_msg()` / `log_json()` / `ensure()`

### 5.3 测试

- 目录: `tests/unit/` 和 `tests/integration/`
- Agent 测试输出: `tests/outputs/<test_module>/<test_name>_<timestamp>.md`（结构化 Markdown）
- 严禁在 `main.py` 中使用 MagicMock/玩具参数

## 6. 上下文获取

> 注意 `Reference/` 下的代码来自参考项目，不是本项目代码。

| 需求 | 文档路径 |
|------|----------|
| 项目背景与使用说明 | `README.md` |
| 系统架构设计 | `docs/architecture.md` |
| 技术设计文档（模块规格） | `docs/TD.md` |
| 实验计划 | `docs/experiment_plan.md` |
| 参考项目（最主要） | `Reference/Tree-TRM/` |

## 7. GitNexus MCP

本项目由 GitNexus 索引为 **Explicit-Lora**。使用前先读 `gitnexus://repo/Explicit-Lora/context` 检查索引新鲜度。

| 任务 | 技能文件 |
|------|----------|
| 理解架构 | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| 影响分析 | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| 调试追踪 | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| 重构 | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
