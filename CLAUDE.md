# CLAUDE.md

> [!URGENT]
> **研究性项目 (Research Project)**
> 1. 本项目为 MVP（最小可行性产品），严禁过度工程化。
> 2. 你的所有思考过程和回复必须使用 **简体中文**。

## 1. 项目元数据 (Metadata)
- **核心目标**: 在本项目中，我们将主要致力于构建一种兼容各类大语言模型的内置持续学习模块。该模块旨在挑战现有大模型的知识存储与调用范式，使模型具备显式的动态知识更新能力。此方案不仅规避了参数重训的高昂算力成本与遗忘风险，也克服了传统外挂检索方案的局限，从而为打造高透明度、高可信度及可持续演进的新一代大语言模型架构提供创新性解法。
- **项目类型**: MVP / 研究性项目
- **后端架构**: Python 3.11
- **版本管理**: Git
- **Conda 环境**: ExplicitLLM (Python 3.11)

## 2. 常用命令 (Commands)

### 2.1 Conda 环境管理

> [!CRITICAL]
> **所有 Python 相关命令必须在 ExplicitLLM 环境中执行**
> - 使用 `conda run -n ExplicitLLM <command>` 确保命令在正确环境中运行
> - 或在命令前显式添加 `source activate ExplicitLLM &&`
> - 如果需要使用llm可以依据 '.env'环境变量文件使用

```bash
# 激活项目环境（交互式 shell）
conda activate ExplicitLLM

# 推荐：使用 conda run 执行命令（自动使用正确环境）
conda run -n ExplicitLLM pip install xxx
conda run -n ExplicitLLM python -m pytest xxx #注意不能conda run -n ExplicitLLM pytest xxx，因为这样子pytest不会调用ExplicitLLM
conda run -n ExplicitLLM python xxx

# 或者：在命令前激活环境
source activate ExplicitLLM && xxx
```

### 2.2 代码质量检查
```bash
# 代码格式化
conda run -n ExplicitLLM ruff format xxx

# 代码检查并自动修复
conda run -n ExplicitLLM ruff check xxx --fix
```


### 2.3 main.py CLI 入口（生产命令）

| 子命令 | 功能 | 前置模块 |
|--------|------|----------|
| `build-knowledge` | Phase 0 知识构建 | §1.2+§1.3+§1.5 |
| `train --phase {0,1,2,3}` | 训练管线 | §1.7+§1.9+§1.10 |
| `eval` | 评测入口 | §1.10 |
| `answer` | 端到端 QA | §1.10 |

用法：`conda run -n ExplicitLLM python main.py [--config path] [--device dev] [--override key=value ...] {子命令}`

模块验证不在 main.py 中，统一通过 `tests/integration/` 执行。

## 3. 标准作业程序 (Standard Operating Procedure)
> **Agent 必须严格遵守以下生命周期执行任务：**

### Phase 1: 规划与设计 (Planning)
1. **查阅规格 (Read Specs)&讨论**: 在撰写计划前，**必须**仔细阅读 `docs/` 下对应的文档与`.report`下的项目整理架构，并使用GitNexus MCP来了解整个项目的最新情况。对于不理解的地方请与人类进行多轮讨论，确保理解人类的设计意图。
2. **计划 (Plan)**: 正式编码前，**必须**使用plan模式输出开发计划，内容必须严格包含：
   - **1.1 摘要 (Summary)**: 1-2句话的简单总结。
   - **1.2 审查点 (User Review Required)**: 明确列出整个计划中不清楚、需要用户审查和确认的部分。若无，请注明"无"。
   - **1.3 拟议变更 (Proposed Changes)**:
     - 以 **文件名 + 修改内容** 的形式列出。
     - 修改内容必须精确到 **函数/方法级别 (Function-level)**。
     - 明确标识 `[NEW]`, `[MODIFY]`, `[DELETE]`。
   - **1.4 验证计划 (Verification Plan)**: 具体描述如何验证修改是否成功（如具体的测试命令、预期日志输出等）。
4. **等待 (Wait)**: **必须** 暂停并等待用户审核开发计划。用户批准后方可进入下一阶段。

### Phase 2: 执行与验证 (Execution & Verification)
1. **编码 (Coding)**: 审核通过后，开始编写代码。
2. **验证 (Verify)**:
   - **环境检查**: 确保所有命令在 ExplicitLLM 环境中执行（使用 `conda run -n ExplicitLLM`）
   - **运行验证命令**:
     - *失败*: 回到编码阶段修复，直到通过。
     - *成功*: 进入下一步。

## 4. 核心规则 (Rules)

### 4.1 代码开发规范 (Code Style)
- **类型系统**: 强制所有函数签名包含完整类型注解 (`Union`, `Dict`, `Optional` 等)。
- **文档**: 所有模块、类、方法必须包含 **中文 Docstring** (功能、参数、返回值、关键实现细节)。
- **MVP原则**:
  - **必须** 必须在`tests/`目录下编写测试代码。
  - **严禁** 使用默认参数掩盖仅需逻辑（必须显式传递关键参数）。
  - **必需** 运行时检查：关键维度、设备一致性必须通过 assertion 或 if 验证。
- **代码组织**:
  - 使用阶段化注释 (`# Phase 1`, `# Phase 2`) 组织复杂逻辑。
  - 接口返回值需包含完整诊断信息（输出、损失、统计），使用条件标志控制。
- **命名与依赖**:
  - 类名 `PascalCase`，变量描述性命名，私有变量前缀 `_`。
  - 导入顺序：标准库 → 第三方库 → 项目内部。
- **日志与错误处理**: 使用 `utils/logger_system.py` 的 `log_msg()`, `log_json()`, `ensure()`, `log_exception()`
  - 禁用 `print()`，`log_msg("ERROR")` 不自动抛出异常，输出到 `logs/system.log` + `logs/metrics.json`
- **功能修改**:
  - **必须** 不考虑向后兼容，直接修改原文件。代码简洁性优先。

### 4.2 配置管理规范
- **优先级**: CLI args > `.env` > YAML，三者统一归口到 dataclass
- **文件**: `config/default.yaml`（全量非敏感配置，必须写全）, `.env`（敏感信息，不提交）, `.env.example`（模板）

### 4.3 测试组织规范
- **目录**: `tests/{unit,integration,e2e}/test_*.py`，最低覆盖率 80%
- **运行**: `conda run -n ExplicitLLM pytest tests/unit/ --cov=utils --cov-report=term-missing`

#### Agent 测试输出规范

> **main.py 是生产 CLI 入口**（build-knowledge / train / eval / answer），不含 demo 或验证逻辑。
> 模块验证通过 `tests/integration/test_{module}_flow.py` + Markdown 报告完成。
> 严禁在 main.py 中使用 MagicMock/玩具参数。

| 要素 | 规范 |
|------|------|
| **输出位置** | `tests/outputs/<test_module>/<test_name>_<timestamp>.md` |
| **触发时机** | 所有涉及 Agent 执行的测试 |
| **内容要求** | 任务描述、每步 Agent 输入/输出/推理过程、工具调用、最终结果 |
| **格式要求** | 结构化 Markdown（标题、代码块、列表），人类可读 |
| **分析方式** | Claude Code 读取 MD 文件，评估推理质量、任务完成度、代码正确性 |

**示例结构**:
```markdown
# Agent 测试: <test_name>
## 任务: <task>
## Step 1: <AgentName>
- 输入: ...
- 输出: ...
- 推理: ...
## Step 2: ...
## 最终结果: ...
```

**pytest 集成**: 使用 fixture 或工具类自动保存，测试结束后输出文件路径。


## 5. 上下文获取与迷途指南 (Context & Navigation)
！注意 `Reference` 文件夹下的所有代码和文件都来自于参考项目，不是本项目的代码。

| 需求 | 文档路径 | 说明 |
|------|----------|------|
| 项目目标与背景 | `README.md` | 核心业务逻辑与项目定性 |
| 架构与模块设计 | `.report/CODEMAPS/{architecture,backend,data}.md` | 整体架构、分层设计、模块依赖 |
| 该项目最主要的参考项目 | `Reference/Tree-TRM`| 特定模块的详细设计 |
| 其他参考项目 | `Reference/PageIndex_ Next-Generation Vectorless, Reasoning-based RAG.md`和 `Reference/Tree-TRM`|  |
| 模块构建状态 | `docs/TD.md` 各 §1.x 节顶部 | 实现状态、依赖、验证 checkpoint |

## 6. 输出规范

### 6.1 语言要求
- 所有输出语言: **中文**

### 6.2 信息密度原则
- **优先使用**:
  - 简洁文本描述
  - 伪代码（而非完整代码）
  - 表格（对比、配置、参数说明）
  - 流程图（Mermaid）
  - 项目符号列表
- **避免使用**:
  - 大段完整代码（信息密度低，可读性差）
  - 冗长的自然语言解释
- **核心原则**: 用最少的字符传递最多的信息
`

<!-- gitnexus:start -->
# GitNexus MCP

This project is indexed by GitNexus as **Explicit-Lora** (1960 symbols, 4034 relationships, 116 execution flows).

## Always Start Here

1. **Read `gitnexus://repo/{name}/context`** — codebase overview + check index freshness
2. **Match your task to a skill below** and **read that skill file**
3. **Follow the skill's workflow and checklist**

> If step 1 warns the index is stale, run `npx gitnexus analyze` in the terminal first.

## Skills

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
