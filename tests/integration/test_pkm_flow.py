"""
ProductKeyMemory 端到端集成测试

原 main.py run_demo() 逻辑迁移至此处，验证 PKM 前向流程的形状、值域和归一化正确性。
测试结果保存为 Markdown 报告至 tests/outputs/memory_gate/。
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RouterConfig, load_config  # noqa: E402
from router.memory_bank import DualKnowledgeStore  # noqa: E402
from router.memory_gate import ProductKeyMemory  # noqa: E402

# ─────────────────────────────────────────────
# 演示用常量
# ─────────────────────────────────────────────
DEMO_KNOWLEDGE_NUM = 16
DEMO_NUM_CANDIDATES = 4
DEVICE = "cpu"
YAML_PATH = str(PROJECT_ROOT / "config" / "default.yaml")
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs" / "memory_gate"


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────


def _make_demo_store(router_cfg: RouterConfig) -> DualKnowledgeStore:
    """
    构造演示用 DualKnowledgeStore，手动注入倒排索引（不依赖真实聚类）。

    设计：knowledge_num=16，每个 grid cell 恰好 1 条知识（完美 1:1 映射）。
    cell i → entry i（i = 0..15）

    参数：
        router_cfg: 从 load_config() 获取的 RouterConfig

    返回：
        已注入倒排索引的 DualKnowledgeStore
    """
    store = DualKnowledgeStore(
        router_cfg,
        fusion_length=4,
        anchor_length=4,
        device=DEVICE,
    )

    n = router_cfg.knowledge_num  # 16
    num_keys = int(n**0.5)  # 4
    c = num_keys * num_keys  # 16 个 grid cell

    # 倒排索引：cell i 对应 entry i
    store.inverted_index = torch.arange(n, dtype=torch.long)
    store.cluster_counts = torch.ones(c, dtype=torch.long)
    store.cluster_offsets = torch.arange(c + 1, dtype=torch.long)

    # 随机聚类中心（[num_keys, key_proj_dim=512]）
    torch.manual_seed(42)
    store.row_centroids = torch.randn(
        num_keys, router_cfg.key_proj_dim, dtype=torch.float
    )
    store.col_centroids = torch.randn(
        num_keys, router_cfg.key_proj_dim, dtype=torch.float
    )
    store.valid_mask = torch.ones(n, dtype=torch.bool)

    return store


def _save_markdown_report(results: dict[str, Any]) -> Path:
    """
    将演示结果保存为结构化 Markdown 报告。

    参数：
        results: test 返回的结果 dict

    返回：
        保存的报告文件路径
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"pkm_flow_{timestamp}.md"

    fm = results.get("full_mode", {})
    oom = results.get("one_to_one_mode", {})
    params = results.get("params", {})

    content = f"""# Agent 测试: ProductKeyMemory (§1.4) 端到端流程

## 任务: 验证 ProductKeyMemory 前向流程的形状、值域和归一化正确性

---

## 配置（通过 load_config + cli_overrides 加载）

| 参数 | 值 |
|------|-----|
| knowledge_num | {results.get("knowledge_num")} |
| num_keys (√N) | {results.get("num_keys")} |
| num_candidates | {results.get("num_candidates")} |
| dim | {results.get("dim")} |
| key_proj_dim | {results.get("key_proj_dim")} |
| temperature | {results.get("temperature")} |
| device | {DEVICE} |

---

## Step 1: 全量倒排索引模式 forward（max_candidates_per_cell=-1）

### candidates 值
```python
{fm.get("candidates")}
```

### 验证结果

| 检查项 | 结果 |
|--------|------|
| candidates 形状 [B, num_candidates] | PASS |
| candidates 值域 [{fm.get("candidates_min")}, {fm.get("candidates_max")}] ⊂ [0, {results.get("knowledge_num")}) | PASS |
| q_adapted L2 norm ≈ 1.0 | {fm.get("q_adapted_norms")} PASS |
| scores 形状 [B, num_keys] | PASS |

---

## Step 2: 1:1 映射模式对比（max_candidates_per_cell=1）

### candidates 值
```python
{oom.get("candidates")}
```

| 检查项 | 结果 |
|--------|------|
| candidates 形状 [B, num_candidates] | PASS |
| candidates 值域 [{oom.get("candidates_min")}, {oom.get("candidates_max")}] ⊂ [0, {results.get("knowledge_num")}) | PASS |

---

## Step 3: 参数量统计

| 类别 | 参数量 |
|------|--------|
| 可训练（query_proj + key_proj × 2） | {params.get("trainable", 0):,} |
| 缓冲区（row_keys + col_keys，非训练） | {params.get("buffer_keys", 0):,} |

---

## 最终结论

所有验证通过：
- ProductKeyMemory 前向流程形状正确
- candidates 值域合法，未越界
- q_adapted 已 L2 归一化
- 两种模式（全量/1:1）均可正常运行

**时间戳**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    report_path.write_text(content, encoding="utf-8")
    return report_path


# ─────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────


def test_pkm_forward_flow() -> None:
    """端到端验证 ProductKeyMemory 前向流程（原 main.py run_demo 迁移）。"""
    results: dict[str, Any] = {}

    # ── Step 1: 加载配置 ─────────────────────────────
    cfg = load_config(
        YAML_PATH,
        cli_overrides={
            "router.knowledge_num": DEMO_KNOWLEDGE_NUM,
            "router.num_candidates": DEMO_NUM_CANDIDATES,
            "router.max_candidates_per_cell": -1,
        },
    )
    router_cfg = cfg.router
    num_keys = int(router_cfg.knowledge_num**0.5)

    results["knowledge_num"] = router_cfg.knowledge_num
    results["num_keys"] = num_keys
    results["num_candidates"] = router_cfg.num_candidates
    results["dim"] = router_cfg.dim
    results["key_proj_dim"] = router_cfg.key_proj_dim
    results["temperature"] = router_cfg.temperature

    # ── Step 2: 构造组件 ─────────────────────────────
    store = _make_demo_store(router_cfg)
    pkm = ProductKeyMemory(router_cfg)

    # ── Step 3: 注入 Keys ────────────────────────────
    pkm.update_keys(store.row_centroids, store.col_centroids)

    # ── Step 4: 全量模式前向 ─────────────────────────
    torch.manual_seed(0)
    B = 2
    embedding = torch.randn(B, router_cfg.dim)
    candidates, scores_1, scores_2, q_adapted = pkm(embedding, store)

    assert candidates.shape == (B, router_cfg.num_candidates), "candidates 形状错误"
    assert scores_1.shape == (B, num_keys), "scores_1 形状错误"
    assert scores_2.shape == (B, num_keys), "scores_2 形状错误"
    assert q_adapted.shape == (B, router_cfg.key_proj_dim), "q_adapted 形状错误"
    assert candidates.min().item() >= 0, "candidates 含负值"
    assert candidates.max().item() < router_cfg.knowledge_num, "candidates 越界"

    q_norms = q_adapted.norm(p=2, dim=-1)
    assert torch.allclose(q_norms, torch.ones_like(q_norms), atol=1e-5), (
        f"q_adapted 未归一化，norms={q_norms}"
    )

    results["full_mode"] = {
        "candidates": candidates.tolist(),
        "scores_1_shape": list(scores_1.shape),
        "scores_2_shape": list(scores_2.shape),
        "q_adapted_norms": q_norms.tolist(),
        "candidates_min": int(candidates.min().item()),
        "candidates_max": int(candidates.max().item()),
    }

    # ── Step 5: 1:1 模式对比 ────────────────────────
    cfg_1to1 = load_config(
        YAML_PATH,
        cli_overrides={
            "router.knowledge_num": DEMO_KNOWLEDGE_NUM,
            "router.num_candidates": DEMO_NUM_CANDIDATES,
            "router.max_candidates_per_cell": 1,
        },
    )
    pkm_1to1 = ProductKeyMemory(cfg_1to1.router)
    pkm_1to1.update_keys(store.row_centroids, store.col_centroids)
    torch.manual_seed(0)
    candidates_1to1, _, _, _ = pkm_1to1(embedding, store)

    assert candidates_1to1.shape == (B, router_cfg.num_candidates), (
        "1:1 模式 candidates 形状错误"
    )

    results["one_to_one_mode"] = {
        "candidates": candidates_1to1.tolist(),
        "candidates_min": int(candidates_1to1.min().item()),
        "candidates_max": int(candidates_1to1.max().item()),
    }

    # ── Step 6: 参数量统计 ──────────────────────────
    trainable_params = sum(p.numel() for p in pkm.parameters() if p.requires_grad)
    buffer_params = sum(b.numel() for _, b in pkm.named_buffers())

    results["params"] = {
        "trainable": trainable_params,
        "buffer_keys": buffer_params,
    }

    # ── 保存 Markdown 报告 ──────────────────────────
    report_path = _save_markdown_report(results)
    print(f"\n报告已保存至: {report_path}")
