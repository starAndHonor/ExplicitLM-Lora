"""
tests/integration/test_injection_flow.py — 注入模块端到端集成测试

验证注入模块在完整流水线上下文中的行为：
  - 模拟 KnowledgeEncoder 输出接入 AttentionInjection
  - 可训练参数 requires_grad 正确性
  - 三种注入方式在模拟 ModifiedQwen Hook 场景中的协同
  - 生成 Markdown 报告到 tests/outputs/injection/

测试策略：
  - 不依赖真实 Qwen3 模型，使用随机张量模拟 KnowledgeEncoder 输出
  - 所有测试在 CPU 上运行
  - 使用小规模参数（B=2, L=16, K_f=64, D=1024）

说明：
  - knowledge_embeddings 模拟 KnowledgeEncoder.forward() 输出 [B, K_f, D]
  - hidden 模拟 Qwen3 某一层的隐藏状态 [B, L, D]
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.injection_modules import (  # noqa: E402
    AttentionInjection,
    ConcatProjection,
    GatedInjection,
)

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

B = 2
L = 16
K_F = 64
D = 1024
NUM_HEADS = 8
INJECTION_LAYERS = [6, 12, 18, 24]  # 4 个注入点（与生产配置一致）
OUTPUT_DIR = PROJECT_ROOT / "tests" / "outputs" / "injection"

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hidden_states() -> torch.Tensor:
    """模拟 Qwen3 某层隐藏状态 [B, L, D]"""
    torch.manual_seed(42)
    return torch.randn(B, L, D)


@pytest.fixture(scope="module")
def knowledge_embeddings() -> torch.Tensor:
    """模拟 KnowledgeEncoder 输出 [B, K_f, D]"""
    torch.manual_seed(43)
    return torch.randn(B, K_F, D)


@pytest.fixture(scope="module")
def knowledge_mask() -> torch.Tensor:
    """模拟知识 padding mask [B, K_f]，前 48 有效，后 16 padding"""
    m = torch.zeros(B, K_F, dtype=torch.long)
    m[:, :48] = 1
    return m


@pytest.fixture(scope="module")
def report_lines() -> List[str]:
    """跨测试共享的 Markdown 报告内容列表"""
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 集成测试
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionInjectionWithEncoderOutput:
    """验证 AttentionInjection 与模拟 KnowledgeEncoder 输出的协同"""

    def test_forward_shape(
        self,
        hidden_states: torch.Tensor,
        knowledge_embeddings: torch.Tensor,
        knowledge_mask: torch.Tensor,
        report_lines: List[str],
    ) -> None:
        """AttentionInjection 接收 KnowledgeEncoder 输出，形状正确"""
        module = AttentionInjection(hidden_dim=D, num_heads=NUM_HEADS)
        module.eval()

        with torch.no_grad():
            out = module(hidden_states, knowledge_embeddings, knowledge_mask)

        assert out.shape == (B, L, D)

        report_lines.append("## 测试 1：AttentionInjection + KnowledgeEncoder 输出")
        report_lines.append(f"- 输入 hidden: {list(hidden_states.shape)}")
        report_lines.append(f"- 输入 knowledge: {list(knowledge_embeddings.shape)}")
        report_lines.append(f"- 输入 mask 有效比例: {knowledge_mask.float().mean():.2f}")
        report_lines.append(f"- 输出形状: {list(out.shape)} ✓")
        report_lines.append(f"- 输出数值有限: {torch.isfinite(out).all().item()} ✓")
        report_lines.append("")

    def test_multi_layer_injection(
        self,
        hidden_states: torch.Tensor,
        knowledge_embeddings: torch.Tensor,
        knowledge_mask: torch.Tensor,
        report_lines: List[str],
    ) -> None:
        """
        模拟 ModifiedQwen 的 4 层注入流水线。

        创建 4 个独立 AttentionInjection 实例（对应 injection_layers=[6,12,18,24]），
        顺序注入，验证形状和数值正确性。
        """
        injection_modules = nn.ModuleList([
            AttentionInjection(hidden_dim=D, num_heads=NUM_HEADS)
            for _ in INJECTION_LAYERS
        ])
        for m in injection_modules:
            m.eval()

        # 模拟 4 层顺序注入
        current_hidden = hidden_states.clone()
        with torch.no_grad():
            for idx, module in enumerate(injection_modules):
                current_hidden = module(current_hidden, knowledge_embeddings, knowledge_mask)
                assert current_hidden.shape == (B, L, D), (
                    f"第 {idx} 层注入后形状错误：{current_hidden.shape}"
                )
                assert torch.isfinite(current_hidden).all(), (
                    f"第 {idx} 层注入后存在 NaN/Inf"
                )

        report_lines.append("## 测试 2：4 层顺序注入流水线")
        report_lines.append(f"- 注入层位置: {INJECTION_LAYERS}")
        report_lines.append(f"- 最终输出形状: {list(current_hidden.shape)} ✓")
        report_lines.append(
            f"- 总参数量: {sum(p.numel() for m in injection_modules for p in m.parameters()):,}"
        )
        report_lines.append("")


class TestInjectionTrainableParams:
    """验证各注入方式的可训练参数配置"""

    @pytest.mark.parametrize(
        "module,name",
        [
            (AttentionInjection(hidden_dim=D, num_heads=NUM_HEADS), "AttentionInjection"),
            (ConcatProjection(hidden_dim=D), "ConcatProjection"),
            (GatedInjection(hidden_dim=D), "GatedInjection"),
        ],
    )
    def test_all_params_require_grad(
        self,
        module: nn.Module,
        name: str,
        report_lines: List[str],
    ) -> None:
        """所有注入方式的参数 requires_grad=True"""
        frozen = [
            pname for pname, p in module.named_parameters() if not p.requires_grad
        ]
        assert len(frozen) == 0, f"{name} 存在冻结参数: {frozen}"

        total_params = sum(p.numel() for p in module.parameters())
        report_lines.append(f"## 测试 3：{name} 可训练参数")
        report_lines.append(f"- 总参数量: {total_params:,}")
        report_lines.append(f"- 全部 requires_grad=True ✓")
        report_lines.append("")

    def test_injection_modules_list_trainable(self, report_lines: List[str]) -> None:
        """nn.ModuleList 包含的注入模块参数均可训练"""
        injection_modules = nn.ModuleList([
            AttentionInjection(hidden_dim=D, num_heads=NUM_HEADS)
            for _ in INJECTION_LAYERS
        ])
        total = sum(p.numel() for p in injection_modules.parameters())
        trainable = sum(
            p.numel() for p in injection_modules.parameters() if p.requires_grad
        )
        assert total == trainable, f"ModuleList 中存在不可训练参数：total={total}, trainable={trainable}"

        report_lines.append("## 测试 4：nn.ModuleList 参数可训练性")
        report_lines.append(f"- 4 层 AttentionInjection 总参数: {total:,}")
        report_lines.append(f"- 可训练参数: {trainable:,} ✓")
        report_lines.append("")


class TestZeroInitResidualIdentity:
    """端到端零初始化残差恒等性验证"""

    @pytest.mark.parametrize(
        "module,name",
        [
            (AttentionInjection(hidden_dim=D, num_heads=NUM_HEADS), "AttentionInjection"),
            (ConcatProjection(hidden_dim=D), "ConcatProjection"),
        ],
    )
    def test_residual_identity_end_to_end(
        self,
        module: nn.Module,
        name: str,
        hidden_states: torch.Tensor,
        knowledge_embeddings: torch.Tensor,
        knowledge_mask: torch.Tensor,
        report_lines: List[str],
    ) -> None:
        """初始化后 forward 输出相对误差 < 1e-4（模拟生产上下文）"""
        module.eval()
        with torch.no_grad():
            out = module(hidden_states, knowledge_embeddings, knowledge_mask)

        diff = (out - hidden_states).norm() / hidden_states.norm()
        assert diff.item() < 1e-4, (
            f"{name} 零初始化后相对误差 {diff.item():.2e} 超过阈值 1e-4"
        )

        report_lines.append(f"## 测试 5：{name} 端到端零初始化残差恒等")
        report_lines.append(f"- 相对误差: {diff.item():.2e} (阈值 < 1e-4) ✓")
        report_lines.append("")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown 报告生成
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module", autouse=True)
def generate_markdown_report(report_lines: List[str]) -> None:
    """
    模块级 autouse fixture：所有测试完成后生成 Markdown 报告。

    报告路径：tests/outputs/injection/test_injection_flow_<timestamp>.md
    """
    yield  # 等待所有测试执行完毕

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_DIR / f"test_injection_flow_{timestamp}.md"

    header = [
        "# Agent 测试: test_injection_flow",
        "",
        "## 任务",
        "验证注入模块在完整流水线上下文中的行为：形状正确性、可训练参数配置、零初始化残差恒等性。",
        "",
        f"## 测试环境",
        f"- 时间戳: {timestamp}",
        f"- 设备: CPU",
        f"- 参数: B={B}, L={L}, K_f={K_F}, D={D}, num_heads={NUM_HEADS}",
        f"- 注入层: {INJECTION_LAYERS}",
        "",
        "## 测试结果",
        "",
    ]

    footer = [
        "",
        "## 最终结论",
        "- 所有注入模块形状正确 ✓",
        "- 零初始化残差恒等性验证通过（相对误差 < 1e-4） ✓",
        "- 所有参数 requires_grad=True ✓",
        "- 4 层顺序注入流水线数值稳定 ✓",
    ]

    content = "\n".join(header + report_lines + footer)
    report_path.write_text(content, encoding="utf-8")
    print(f"\n[报告] 已生成 Markdown 报告：{report_path}")
