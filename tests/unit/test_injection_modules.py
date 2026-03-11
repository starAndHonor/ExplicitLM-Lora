"""
tests/unit/test_injection_modules.py — 注入模块单元测试

使用随机张量（不依赖真实 Qwen3 模型），验证三种注入方式的接口正确性、
形状一致性、零初始化残差恒等性及反向传播正确性。

测试覆盖：
  - RMSNorm 形状不变性
  - masked_mean_pool 池化正确性（含全 padding 边界情况）
  - AttentionInjection：形状、零初始化（output ≈ hidden）、全 padding 稳定性
  - ConcatProjection：形状、零初始化（output ≈ hidden）
  - GatedInjection：形状、gate 初始值为 0
  - 全部三种方式：反向传播正确性（参数有梯度）

常量：
  - B=2, L=10, K_f=64, D=1024（与生产配置一致）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.injection_modules import (  # noqa: E402
    AttentionInjection,
    ConcatProjection,
    GatedInjection,
    RMSNorm,
    masked_mean_pool,
)

# ─────────────────────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────────────────────

B = 2
L = 10
K_F = 64
D = 1024
NUM_HEADS = 8
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def hidden() -> torch.Tensor:
    """随机生成 hidden states [B, L, D]"""
    torch.manual_seed(SEED)
    return torch.randn(B, L, D)


@pytest.fixture
def knowledge() -> torch.Tensor:
    """随机生成知识编码器输出 [B, K_f, D]"""
    torch.manual_seed(SEED + 1)
    return torch.randn(B, K_F, D)


@pytest.fixture
def mask_full() -> torch.Tensor:
    """全有效 mask [B, K_f]，所有 token 均有效"""
    return torch.ones(B, K_F, dtype=torch.long)


@pytest.fixture
def mask_partial() -> torch.Tensor:
    """部分有效 mask [B, K_f]：前 32 个有效，后 32 个 padding"""
    m = torch.zeros(B, K_F, dtype=torch.long)
    m[:, :32] = 1
    return m


@pytest.fixture
def mask_all_pad() -> torch.Tensor:
    """全 padding mask [B, K_f]，所有位置均为 padding"""
    return torch.zeros(B, K_F, dtype=torch.long)


@pytest.fixture
def attn_inj() -> AttentionInjection:
    """AttentionInjection 实例（eval 模式）"""
    module = AttentionInjection(hidden_dim=D, num_heads=NUM_HEADS)
    module.eval()
    return module


@pytest.fixture
def concat_proj() -> ConcatProjection:
    """ConcatProjection 实例（eval 模式）"""
    module = ConcatProjection(hidden_dim=D)
    module.eval()
    return module


@pytest.fixture
def gated_inj() -> GatedInjection:
    """GatedInjection 实例（eval 模式）"""
    module = GatedInjection(hidden_dim=D)
    module.eval()
    return module


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm 测试
# ─────────────────────────────────────────────────────────────────────────────


class TestRMSNorm:
    def test_output_shape_unchanged(self) -> None:
        """RMSNorm 输出形状与输入相同"""
        norm = RMSNorm(D)
        x = torch.randn(B, L, D)
        out = norm(x)
        assert out.shape == (B, L, D), f"期望 {(B, L, D)}，得到 {out.shape}"

    def test_output_shape_2d(self) -> None:
        """RMSNorm 支持 2D 输入"""
        norm = RMSNorm(D)
        x = torch.randn(B, D)
        out = norm(x)
        assert out.shape == (B, D)

    def test_gamma_initialized_to_ones(self) -> None:
        """gamma 参数初始化为全 1"""
        norm = RMSNorm(D)
        assert torch.allclose(norm.gamma, torch.ones(D)), "gamma 应初始化为全 1"

    def test_output_finite(self) -> None:
        """输出不包含 NaN 或 Inf"""
        norm = RMSNorm(D)
        x = torch.randn(B, L, D)
        out = norm(x)
        assert torch.isfinite(out).all(), "RMSNorm 输出存在 NaN/Inf"

    def test_zero_input_stability(self) -> None:
        """全零输入时输出数值稳定（不出现 NaN）"""
        norm = RMSNorm(D)
        x = torch.zeros(B, L, D)
        out = norm(x)
        assert torch.isfinite(out).all(), "全零输入时 RMSNorm 输出不稳定"


# ─────────────────────────────────────────────────────────────────────────────
# masked_mean_pool 测试
# ─────────────────────────────────────────────────────────────────────────────


class TestMaskedMeanPool:
    def test_output_shape(self, knowledge: torch.Tensor, mask_full: torch.Tensor) -> None:
        """输出形状为 [B, 1, D]"""
        out = masked_mean_pool(knowledge, mask_full)
        assert out.shape == (B, 1, D), f"期望 {(B, 1, D)}，得到 {out.shape}"

    def test_no_mask(self, knowledge: torch.Tensor) -> None:
        """无 mask 时输出形状正确，且等价于 mean(dim=1)"""
        out = masked_mean_pool(knowledge, None)
        expected = knowledge.mean(dim=1, keepdim=True)
        assert out.shape == (B, 1, D)
        assert torch.allclose(out, expected, atol=1e-5), "无 mask 时结果应等价于 mean"

    def test_all_valid_mask(self, knowledge: torch.Tensor, mask_full: torch.Tensor) -> None:
        """全有效 mask 时输出等价于直接均值"""
        out = masked_mean_pool(knowledge, mask_full)
        expected = knowledge.mean(dim=1, keepdim=True)
        assert torch.allclose(out, expected, atol=1e-5), "全有效 mask 时应等价于 mean"

    def test_partial_mask(self, knowledge: torch.Tensor, mask_partial: torch.Tensor) -> None:
        """部分 mask 时只对有效 token 做均值"""
        out = masked_mean_pool(knowledge, mask_partial)
        # 手动计算前 32 个 token 的均值
        expected = knowledge[:, :32, :].mean(dim=1, keepdim=True)
        assert torch.allclose(out, expected, atol=1e-5), "部分 mask 时应仅对有效 token 均值"

    def test_all_pad_returns_zero(
        self, knowledge: torch.Tensor, mask_all_pad: torch.Tensor
    ) -> None:
        """全 padding 时输出接近零向量（count clamp 防除零）"""
        out = masked_mean_pool(knowledge, mask_all_pad)
        assert out.shape == (B, 1, D)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6), (
            "全 padding 时应返回零向量"
        )


# ─────────────────────────────────────────────────────────────────────────────
# AttentionInjection 测试
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionInjection:
    def test_output_shape(
        self,
        attn_inj: AttentionInjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_full: torch.Tensor,
    ) -> None:
        """forward 输出形状为 [B, L, D]"""
        with torch.no_grad():
            out = attn_inj(hidden, knowledge, mask_full)
        assert out.shape == (B, L, D), f"期望 {(B, L, D)}，得到 {out.shape}"

    def test_zero_init_residual_identity(
        self,
        attn_inj: AttentionInjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_full: torch.Tensor,
    ) -> None:
        """零初始化后 output ≈ hidden，残差相对误差 < 1e-4"""
        with torch.no_grad():
            out = attn_inj(hidden, knowledge, mask_full)
        diff = (out - hidden).norm() / hidden.norm()
        assert diff.item() < 1e-4, (
            f"零初始化后相对误差应 < 1e-4，实际: {diff.item():.2e}"
        )

    def test_all_pad_mask_no_crash(
        self,
        attn_inj: AttentionInjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_all_pad: torch.Tensor,
    ) -> None:
        """全 padding mask 时 Null KV 机制保证数值稳定，不崩溃"""
        with torch.no_grad():
            out = attn_inj(hidden, knowledge, mask_all_pad)
        assert out.shape == (B, L, D)
        assert torch.isfinite(out).all(), "全 padding 时输出存在 NaN/Inf"

    def test_out_proj_weight_zero_init(self, attn_inj: AttentionInjection) -> None:
        """out_proj.weight 初始化为全零"""
        assert torch.allclose(
            attn_inj.out_proj.weight, torch.zeros_like(attn_inj.out_proj.weight)
        ), "out_proj.weight 应初始化为全零"

    def test_out_proj_bias_zero_init(self, attn_inj: AttentionInjection) -> None:
        """out_proj.bias 初始化为全零"""
        assert torch.allclose(
            attn_inj.out_proj.bias, torch.zeros_like(attn_inj.out_proj.bias)
        ), "out_proj.bias 应初始化为全零"

    def test_get_out_proj_norm_near_zero(self, attn_inj: AttentionInjection) -> None:
        """初始 out_proj norm 接近 0"""
        norm_val = attn_inj.get_out_proj_norm()
        assert norm_val < 1e-7, f"初始 out_proj norm 应接近 0，实际: {norm_val}"

    def test_output_finite(
        self,
        attn_inj: AttentionInjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_partial: torch.Tensor,
    ) -> None:
        """partial mask 时输出数值有限"""
        with torch.no_grad():
            out = attn_inj(hidden, knowledge, mask_partial)
        assert torch.isfinite(out).all(), "输出存在 NaN/Inf"


# ─────────────────────────────────────────────────────────────────────────────
# ConcatProjection 测试
# ─────────────────────────────────────────────────────────────────────────────


class TestConcatProjection:
    def test_output_shape(
        self,
        concat_proj: ConcatProjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_full: torch.Tensor,
    ) -> None:
        """forward 输出形状为 [B, L, D]"""
        with torch.no_grad():
            out = concat_proj(hidden, knowledge, mask_full)
        assert out.shape == (B, L, D), f"期望 {(B, L, D)}，得到 {out.shape}"

    def test_zero_init_residual_identity(
        self,
        concat_proj: ConcatProjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_full: torch.Tensor,
    ) -> None:
        """末层零初始化后 output ≈ hidden，残差相对误差 < 1e-4"""
        with torch.no_grad():
            out = concat_proj(hidden, knowledge, mask_full)
        diff = (out - hidden).norm() / hidden.norm()
        assert diff.item() < 1e-4, (
            f"零初始化后相对误差应 < 1e-4，实际: {diff.item():.2e}"
        )

    def test_proj_out_weight_zero_init(self, concat_proj: ConcatProjection) -> None:
        """proj_out.weight 初始化为全零"""
        assert torch.allclose(
            concat_proj.proj_out.weight, torch.zeros_like(concat_proj.proj_out.weight)
        ), "proj_out.weight 应初始化为全零"

    def test_proj_out_bias_zero_init(self, concat_proj: ConcatProjection) -> None:
        """proj_out.bias 初始化为全零"""
        assert torch.allclose(
            concat_proj.proj_out.bias, torch.zeros_like(concat_proj.proj_out.bias)
        ), "proj_out.bias 应初始化为全零"

    def test_output_finite(
        self,
        concat_proj: ConcatProjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_partial: torch.Tensor,
    ) -> None:
        """partial mask 时输出数值有限"""
        with torch.no_grad():
            out = concat_proj(hidden, knowledge, mask_partial)
        assert torch.isfinite(out).all(), "输出存在 NaN/Inf"


# ─────────────────────────────────────────────────────────────────────────────
# GatedInjection 测试
# ─────────────────────────────────────────────────────────────────────────────


class TestGatedInjection:
    def test_output_shape(
        self,
        gated_inj: GatedInjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_full: torch.Tensor,
    ) -> None:
        """forward 输出形状为 [B, L, D]"""
        with torch.no_grad():
            out = gated_inj(hidden, knowledge, mask_full)
        assert out.shape == (B, L, D), f"期望 {(B, L, D)}，得到 {out.shape}"

    def test_gate_zero_init(self, gated_inj: GatedInjection) -> None:
        """gate 参数初始化为全零"""
        assert torch.allclose(
            gated_inj.gate, torch.zeros(D)
        ), "gate 应初始化为全零"

    def test_get_gate_stats(self, gated_inj: GatedInjection) -> None:
        """初始 gate_stats 接近 0"""
        stats = gated_inj.get_gate_stats()
        assert stats < 1e-7, f"初始 gate 绝对值均值应接近 0，实际: {stats}"

    def test_output_finite(
        self,
        gated_inj: GatedInjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_partial: torch.Tensor,
    ) -> None:
        """partial mask 时输出数值有限"""
        with torch.no_grad():
            out = gated_inj(hidden, knowledge, mask_partial)
        assert torch.isfinite(out).all(), "输出存在 NaN/Inf"

    def test_all_pad_no_crash(
        self,
        gated_inj: GatedInjection,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_all_pad: torch.Tensor,
    ) -> None:
        """全 padding mask 时不崩溃"""
        with torch.no_grad():
            out = gated_inj(hidden, knowledge, mask_all_pad)
        assert out.shape == (B, L, D)
        assert torch.isfinite(out).all()


# ─────────────────────────────────────────────────────────────────────────────
# 反向传播测试（train 模式）
# ─────────────────────────────────────────────────────────────────────────────


class TestBackward:
    @pytest.mark.parametrize(
        "module_cls,kwargs",
        [
            (AttentionInjection, {"hidden_dim": D, "num_heads": NUM_HEADS}),
            (ConcatProjection, {"hidden_dim": D}),
            (GatedInjection, {"hidden_dim": D}),
        ],
    )
    def test_backward_no_error(
        self,
        module_cls: type,
        kwargs: dict,
        hidden: torch.Tensor,
        knowledge: torch.Tensor,
        mask_partial: torch.Tensor,
    ) -> None:
        """三种注入方式均可完成反向传播，参数获得梯度"""
        module = module_cls(**kwargs)
        module.train()

        h = hidden.clone().requires_grad_(True)
        k = knowledge.clone().requires_grad_(True)

        out = module(h, k, mask_partial)
        loss = out.sum()
        loss.backward()

        # 验证输出参数有梯度
        params_with_grad = [
            p for p in module.parameters() if p.grad is not None
        ]
        assert len(params_with_grad) > 0, (
            f"{module_cls.__name__} 反向传播后无参数获得梯度"
        )

    def test_attention_injection_trainable_params(self) -> None:
        """AttentionInjection 所有参数 requires_grad=True"""
        module = AttentionInjection(hidden_dim=D, num_heads=NUM_HEADS)
        for name, p in module.named_parameters():
            assert p.requires_grad, f"参数 {name} 的 requires_grad=False"
