"""
router/clustering.py 集成测试

测试覆盖：
    - SubspaceClustering.fit 与 DualKnowledgeStore.compact_and_recluster 端到端联动
    - compact_and_recluster 后 pca 状态、倒排索引的完整性验证
    - compact_and_recluster 后 add_entries 的高精度分配路径
    - ProductKeyMemory.update_keys 接收 store 中心点后 shape 正确

测试输出：
    tests/outputs/clustering/<test_name>_<timestamp>.md
"""

from __future__ import annotations

import datetime
import math
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from router.memory_bank import DualKnowledgeStore
from router.memory_gate import ProductKeyMemory

# ─────────────────────────────────────────────
# 测试参数常量
# ─────────────────────────────────────────────

# 最小完全平方数，2^2=4 cluster，保证 num_keys=2 为 2 的幂次
KNOWLEDGE_NUM = 16  # 4×4
FUSION_LENGTH = 8
ANCHOR_LENGTH = 12
# encoder 输出维度（偶数，用于 PCA 子空间分割）
EMBED_DIM = 8
DEVICE = "cpu"
OUTPUT_DIR = Path("tests/outputs/clustering")


# ─────────────────────────────────────────────
# 共用辅助函数
# ─────────────────────────────────────────────

def _make_router_config(
    knowledge_num: int = KNOWLEDGE_NUM,
    recluster_threshold: float = 0.1,
) -> Any:
    """构造最小 RouterConfig mock。"""
    cfg = MagicMock()
    cfg.knowledge_num = knowledge_num
    cfg.recluster_threshold = recluster_threshold
    return cfg


def _make_store(knowledge_num: int = KNOWLEDGE_NUM) -> DualKnowledgeStore:
    """构造空 DualKnowledgeStore。"""
    cfg = _make_router_config(knowledge_num=knowledge_num)
    return DualKnowledgeStore(cfg, FUSION_LENGTH, ANCHOR_LENGTH, DEVICE)


def _make_mock_encoder(embed_dim: int = EMBED_DIM) -> MagicMock:
    """
    构造 mock KnowledgeEncoder，encode_mean 返回随机 [B, embed_dim] float 张量。
    输出维度与 pca_matrix 维度一致。
    """
    encoder = MagicMock()
    encoder.device = DEVICE

    rng = np.random.default_rng(42)

    def fake_encode_mean(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = ids.shape[0]
        arr = rng.standard_normal((b, embed_dim)).astype(np.float32)
        return torch.from_numpy(arr)

    encoder.encode_mean.side_effect = fake_encode_mean
    return encoder


def _fill_store_and_recluster(
    store: DualKnowledgeStore,
    n_entries: int,
    encoder: MagicMock,
) -> None:
    """
    向 store 写入 n_entries 条随机数据并执行 compact_and_recluster。
    使用 update_all 批量写入（绕过 pca_matrix=None 的 add_entries 限制）。
    """
    fusion_ids = torch.randint(1, 500, (n_entries, FUSION_LENGTH))
    anchor_ids = torch.randint(1, 500, (n_entries, ANCHOR_LENGTH))

    # 直接写入 bank 数据（模拟 Phase 0 离线构建，绕过热更新路径）
    store.fusion_bank.data[:n_entries].copy_(fusion_ids)
    store.anchor_bank.data[:n_entries].copy_(anchor_ids)
    store.valid_mask[:n_entries] = True
    store.next_free = n_entries

    store.compact_and_recluster(encoder)


def _save_md_report(test_name: str, content: str) -> str:
    """保存测试结果到 Markdown 文件，返回文件路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{test_name}_{ts}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


# ─────────────────────────────────────────────
# 集成测试
# ─────────────────────────────────────────────

class TestClusteringFlow:
    """SubspaceClustering 与 DualKnowledgeStore 的端到端集成测试。"""

    def test_compact_and_recluster_sets_pca_state(self) -> None:
        """
        compact_and_recluster 后 store 的 pca 状态均被正确初始化（非 None），
        且 shape 与 embed_dim 和 num_keys 一致。
        """
        store = _make_store()
        encoder = _make_mock_encoder()
        n_entries = KNOWLEDGE_NUM // 2  # 写入半满

        _fill_store_and_recluster(store, n_entries, encoder)

        num_keys = int(KNOWLEDGE_NUM ** 0.5)

        # 验证 pca 状态
        assert store.pca_matrix is not None, "pca_matrix 应在 compact_and_recluster 后初始化"
        assert store.pca_mean is not None
        assert store.row_centroids is not None
        assert store.col_centroids is not None

        assert store.pca_matrix.shape == (EMBED_DIM, EMBED_DIM)
        assert store.pca_mean.shape == (EMBED_DIM,)
        assert store.row_centroids.shape == (num_keys, EMBED_DIM // 2)
        assert store.col_centroids.shape == (num_keys, EMBED_DIM // 2)

        # 保存报告
        report = f"""# Agent 测试: test_compact_and_recluster_sets_pca_state
## 任务: 验证 compact_and_recluster 后 pca 状态正确初始化
## 输入: n_entries={n_entries}, KNOWLEDGE_NUM={KNOWLEDGE_NUM}, EMBED_DIM={EMBED_DIM}
## 验证点:
- pca_matrix shape: {tuple(store.pca_matrix.shape)} ✓
- pca_mean shape: {tuple(store.pca_mean.shape)} ✓
- row_centroids shape: {tuple(store.row_centroids.shape)} ✓
- col_centroids shape: {tuple(store.col_centroids.shape)} ✓
## 最终结果: PASSED
"""
        path = _save_md_report("test_compact_and_recluster_sets_pca_state", report)
        print(f"\n报告保存至: {path}")

    def test_compact_and_recluster_inverted_index_valid(self) -> None:
        """
        compact_and_recluster 后倒排索引完整性验证：
        - sum(cluster_counts) == N_valid
        - inverted_index[:N_valid] 无 -1（均有效）
        - inverted_index[:N_valid] 无重复值
        """
        store = _make_store()
        encoder = _make_mock_encoder()
        n_entries = 8  # < KNOWLEDGE_NUM，留有余量

        _fill_store_and_recluster(store, n_entries, encoder)

        # sum(cluster_counts) == N_valid
        total = int(store.cluster_counts.sum().item())
        assert total == n_entries, (
            f"cluster_counts 总和应为 N_valid={n_entries}，实际 {total}"
        )

        # inverted_index[:n_entries] 无 -1
        valid_slice = store.inverted_index[:n_entries]
        assert (valid_slice >= 0).all(), "inverted_index 有效段不应含 -1"

        # 无重复 ID
        unique_count = len(valid_slice.unique())
        assert unique_count == n_entries, (
            f"inverted_index[:n_entries] 不应有重复 ID，unique={unique_count}"
        )

        report = f"""# Agent 测试: test_compact_and_recluster_inverted_index_valid
## 任务: 验证 compact_and_recluster 后倒排索引完整性
## 输入: n_entries={n_entries}
## 验证点:
- cluster_counts 总和: {total} == {n_entries} ✓
- inverted_index 无 -1: ✓
- 无重复 ID: unique={unique_count} ✓
## 最终结果: PASSED
"""
        path = _save_md_report("test_compact_and_recluster_inverted_index_valid", report)
        print(f"\n报告保存至: {path}")

    def test_add_entries_after_recluster(self) -> None:
        """
        compact_and_recluster 后可正常调用 add_entries，
        新条目写入 bank 且倒排索引更新（cluster_counts 增加）。
        """
        store = _make_store()
        encoder = _make_mock_encoder()
        n_init = 6

        _fill_store_and_recluster(store, n_init, encoder)

        # 记录 recluster 后状态
        counts_before = store.cluster_counts.clone()
        next_free_before = store.next_free

        # 热更新：添加 2 条新知识
        b = 2
        fusion_ids = torch.randint(1, 500, (b, FUSION_LENGTH))
        anchor_ids = torch.randint(1, 500, (b, ANCHOR_LENGTH))
        store.add_entries(fusion_ids, anchor_ids, encoder)

        # next_free 增加
        assert store.next_free == next_free_before + b
        assert store.change_counter == b  # recluster 后 change_counter 重置，再加 b
        assert store.valid_mask[next_free_before:next_free_before + b].all()

        # 倒排索引总和增加 b
        total_after = int(store.cluster_counts.sum().item())
        assert total_after == n_init + b, (
            f"热更新后 cluster_counts 总和应为 {n_init + b}，实际 {total_after}"
        )

        report = f"""# Agent 测试: test_add_entries_after_recluster
## 任务: 验证 compact_and_recluster 后 add_entries 正常工作
## 输入: n_init={n_init}, 热更新 b={b}
## 验证点:
- next_free: {next_free_before} → {store.next_free} ✓
- cluster_counts 总和: {n_init} → {total_after} ✓
- valid_mask 新条目全 True ✓
## 最终结果: PASSED
"""
        path = _save_md_report("test_add_entries_after_recluster", report)
        print(f"\n报告保存至: {path}")

    def test_pkm_update_keys_shape(self) -> None:
        """
        compact_and_recluster 后 store.row_centroids/col_centroids 传入
        ProductKeyMemory.update_keys 不报错，且 keys 被正确更新（shape 合法）。

        ProductKeyMemory 期望 [num_keys, key_proj_dim]，其中 key_proj_dim = EMBED_DIM // 2。
        """
        # 构造 PKM 配置：dim=EMBED_DIM，key_proj_dim=EMBED_DIM//2，knowledge_num=KNOWLEDGE_NUM
        pkm_config = MagicMock()
        pkm_config.knowledge_num = KNOWLEDGE_NUM
        pkm_config.dim = EMBED_DIM
        pkm_config.query_dim = EMBED_DIM
        pkm_config.key_proj_dim = EMBED_DIM // 2
        pkm_config.num_candidates = 4
        pkm_config.temperature = 0.1
        pkm_config.max_candidates_per_cell = -1

        pkm = ProductKeyMemory(pkm_config)

        store = _make_store()
        encoder = _make_mock_encoder()

        _fill_store_and_recluster(store, KNOWLEDGE_NUM // 2, encoder)

        # update_keys 接收 row_centroids/col_centroids（shape [num_keys, EMBED_DIM//2]）
        pkm.update_keys(store.row_centroids, store.col_centroids)

        num_keys = int(KNOWLEDGE_NUM ** 0.5)
        expected_shape = (num_keys, EMBED_DIM // 2)
        assert tuple(pkm.row_keys.shape) == expected_shape, (
            f"row_keys shape 不匹配: 期望 {expected_shape}，实际 {tuple(pkm.row_keys.shape)}"
        )
        assert tuple(pkm.col_keys.shape) == expected_shape

        report = f"""# Agent 测试: test_pkm_update_keys_shape
## 任务: 验证 compact_and_recluster 后 PKM.update_keys 接收中心点不报错
## 输入: KNOWLEDGE_NUM={KNOWLEDGE_NUM}, EMBED_DIM={EMBED_DIM}
## 验证点:
- row_keys shape: {tuple(pkm.row_keys.shape)} == {expected_shape} ✓
- col_keys shape: {tuple(pkm.col_keys.shape)} == {expected_shape} ✓
## 最终结果: PASSED
"""
        path = _save_md_report("test_pkm_update_keys_shape", report)
        print(f"\n报告保存至: {path}")
