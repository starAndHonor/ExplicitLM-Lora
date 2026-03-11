"""
router/memory_bank.py 单元测试

测试覆盖：
    - FusionBank：初始化、全量写入、批量索引
    - AnchorBank：初始化、全量写入
    - DualKnowledgeStore：初始化、增删、recluster 触发、溢出/未初始化错误、序列化
"""

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from router.memory_bank import AnchorBank, DualKnowledgeStore, FusionBank


# ─────────────────────────────────────────────
# 共用 fixtures
# ─────────────────────────────────────────────

KNOWLEDGE_NUM = 16   # 4×4，最小完全平方数，测试用
FUSION_LENGTH = 8
ANCHOR_LENGTH = 12
DEVICE = "cpu"


def _make_router_config(
    knowledge_num: int = KNOWLEDGE_NUM,
    recluster_threshold: float = 0.1,
) -> Any:
    """构造最小 RouterConfig mock（仅含 DualKnowledgeStore 需要的字段）。"""
    cfg = MagicMock()
    cfg.knowledge_num = knowledge_num
    cfg.recluster_threshold = recluster_threshold
    return cfg


def _make_store(knowledge_num: int = KNOWLEDGE_NUM) -> DualKnowledgeStore:
    """构造测试用 DualKnowledgeStore。"""
    cfg = _make_router_config(knowledge_num=knowledge_num)
    return DualKnowledgeStore(cfg, FUSION_LENGTH, ANCHOR_LENGTH, DEVICE)


def _make_store_with_pca(knowledge_num: int = KNOWLEDGE_NUM) -> DualKnowledgeStore:
    """
    构造已初始化 pca_matrix 的 DualKnowledgeStore（跳过真实聚类，直接注入合法状态）。
    用于测试 add_entries 的近似分配路径。
    """
    store = _make_store(knowledge_num)
    d = 4  # 极小维度，用于 pca_matrix 构造
    num_keys = int(knowledge_num ** 0.5)

    store.pca_matrix = torch.eye(d, dtype=torch.float)
    store.pca_mean = torch.zeros(d, dtype=torch.float)
    store.row_centroids = torch.zeros(num_keys, d // 2, dtype=torch.float)
    store.col_centroids = torch.zeros(num_keys, d // 2, dtype=torch.float)
    return store


# ─────────────────────────────────────────────
# FusionBank 测试
# ─────────────────────────────────────────────

class TestFusionBank:
    """FusionBank 单元测试。"""

    def test_fusion_bank_init(self) -> None:
        """初始化后 shape、dtype、device 均正确，数据全零。"""
        bank = FusionBank(KNOWLEDGE_NUM, FUSION_LENGTH, DEVICE)
        assert bank.data.shape == (KNOWLEDGE_NUM, FUSION_LENGTH)
        assert bank.data.dtype == torch.long
        assert bank.knowledge_num == KNOWLEDGE_NUM
        assert bank.fusion_length == FUSION_LENGTH
        assert bank.data.sum().item() == 0

    def test_fusion_bank_update_all(self) -> None:
        """全量写入后读取数据与输入一致。"""
        bank = FusionBank(KNOWLEDGE_NUM, FUSION_LENGTH, DEVICE)
        token_ids = torch.randint(1, 1000, (KNOWLEDGE_NUM, FUSION_LENGTH))
        bank.update_all(token_ids)
        assert torch.equal(bank.data, token_ids)

    def test_fusion_bank_update_all_wrong_shape_raises(self) -> None:
        """形状不匹配时 update_all 应 AssertionError。"""
        bank = FusionBank(KNOWLEDGE_NUM, FUSION_LENGTH, DEVICE)
        bad_ids = torch.randint(0, 100, (KNOWLEDGE_NUM + 1, FUSION_LENGTH))
        with pytest.raises(AssertionError, match="形状不匹配"):
            bank.update_all(bad_ids)

    def test_fusion_bank_getitem(self) -> None:
        """批量索引返回正确的 [B, K_f] 子集。"""
        bank = FusionBank(KNOWLEDGE_NUM, FUSION_LENGTH, DEVICE)
        token_ids = torch.arange(KNOWLEDGE_NUM * FUSION_LENGTH).reshape(KNOWLEDGE_NUM, FUSION_LENGTH)
        bank.update_all(token_ids)

        ids = torch.tensor([0, 3, 7], dtype=torch.long)
        result = bank[ids]
        assert result.shape == (3, FUSION_LENGTH)
        assert torch.equal(result[0], token_ids[0])
        assert torch.equal(result[1], token_ids[3])
        assert torch.equal(result[2], token_ids[7])

    def test_fusion_bank_getitem_out_of_range_raises(self) -> None:
        """索引超出 knowledge_num 时应 AssertionError。"""
        bank = FusionBank(KNOWLEDGE_NUM, FUSION_LENGTH, DEVICE)
        bad_ids = torch.tensor([KNOWLEDGE_NUM], dtype=torch.long)
        with pytest.raises(AssertionError, match="索引越界"):
            _ = bank[bad_ids]


# ─────────────────────────────────────────────
# AnchorBank 测试
# ─────────────────────────────────────────────

class TestAnchorBank:
    """AnchorBank 单元测试。"""

    def test_anchor_bank_init(self) -> None:
        """初始化后 shape、dtype 均正确。"""
        bank = AnchorBank(KNOWLEDGE_NUM, ANCHOR_LENGTH, DEVICE)
        assert bank.data.shape == (KNOWLEDGE_NUM, ANCHOR_LENGTH)
        assert bank.data.dtype == torch.long
        assert bank.knowledge_num == KNOWLEDGE_NUM
        assert bank.anchor_length == ANCHOR_LENGTH

    def test_anchor_bank_update_all(self) -> None:
        """全量替换后数据与输入一致。"""
        bank = AnchorBank(KNOWLEDGE_NUM, ANCHOR_LENGTH, DEVICE)
        token_ids = torch.randint(1, 2000, (KNOWLEDGE_NUM, ANCHOR_LENGTH))
        bank.update_all(token_ids)
        assert torch.equal(bank.data, token_ids)

    def test_anchor_bank_update_all_wrong_shape_raises(self) -> None:
        """形状不匹配时应 AssertionError。"""
        bank = AnchorBank(KNOWLEDGE_NUM, ANCHOR_LENGTH, DEVICE)
        bad_ids = torch.randint(0, 100, (KNOWLEDGE_NUM, ANCHOR_LENGTH + 1))
        with pytest.raises(AssertionError, match="形状不匹配"):
            bank.update_all(bad_ids)


# ─────────────────────────────────────────────
# DualKnowledgeStore 测试
# ─────────────────────────────────────────────

class TestDualKnowledgeStore:
    """DualKnowledgeStore 单元测试。"""

    def test_dual_store_init(self) -> None:
        """所有 buffer 初始化正确：valid_mask 全 False，next_free=0，change_counter=0。"""
        store = _make_store()
        assert store.fusion_bank.data.shape == (KNOWLEDGE_NUM, FUSION_LENGTH)
        assert store.anchor_bank.data.shape == (KNOWLEDGE_NUM, ANCHOR_LENGTH)
        assert store.valid_mask.shape == (KNOWLEDGE_NUM,)
        assert store.valid_mask.dtype == torch.bool
        assert not store.valid_mask.any()
        assert store.inverted_index.shape == (KNOWLEDGE_NUM,)
        assert (store.inverted_index == -1).all()
        num_keys = int(KNOWLEDGE_NUM ** 0.5)
        c = num_keys * num_keys
        assert store.cluster_offsets.shape == (c + 1,)
        assert store.cluster_counts.shape == (c,)
        assert store.pca_matrix is None
        assert store.pca_mean is None
        assert store.row_centroids is None
        assert store.col_centroids is None
        assert store.next_free == 0
        assert store.change_counter == 0

    def test_dual_store_init_non_square_raises(self) -> None:
        """knowledge_num 不是完全平方数时应 AssertionError。"""
        cfg = _make_router_config(knowledge_num=15)
        with pytest.raises(AssertionError, match="完全平方数"):
            DualKnowledgeStore(cfg, FUSION_LENGTH, ANCHOR_LENGTH, DEVICE)

    def test_dual_store_add_entries(self) -> None:
        """写入 B 条后 valid_mask、next_free、change_counter 正确更新。"""
        store = _make_store_with_pca()
        b = 3
        fusion_ids = torch.randint(1, 500, (b, FUSION_LENGTH))
        anchor_ids = torch.randint(1, 500, (b, ANCHOR_LENGTH))
        store.add_entries(fusion_ids, anchor_ids)

        assert store.next_free == b
        assert store.change_counter == b
        assert store.valid_mask[:b].all()
        assert not store.valid_mask[b:].any()
        # 验证写入内容
        assert torch.equal(store.fusion_bank.data[:b], fusion_ids)
        assert torch.equal(store.anchor_bank.data[:b], anchor_ids)

    def test_dual_store_delete_entries(self) -> None:
        """逻辑删除后指定条目 valid_mask=False，change_counter 增加。"""
        store = _make_store_with_pca()
        b = 4
        fusion_ids = torch.randint(1, 500, (b, FUSION_LENGTH))
        anchor_ids = torch.randint(1, 500, (b, ANCHOR_LENGTH))
        store.add_entries(fusion_ids, anchor_ids)
        assert store.change_counter == b

        store.delete_entries([1, 3])
        assert not store.valid_mask[1]
        assert not store.valid_mask[3]
        assert store.valid_mask[0]
        assert store.valid_mask[2]
        assert store.change_counter == b + 2

    def test_dual_store_should_recluster(self) -> None:
        """change_counter / N_valid > threshold 时 should_recluster 返回 True。"""
        store = _make_store_with_pca()
        # 写入 KNOWLEDGE_NUM 条（threshold=0.1，需 >10% 变更触发）
        b = KNOWLEDGE_NUM
        fusion_ids = torch.randint(1, 500, (b, FUSION_LENGTH))
        anchor_ids = torch.randint(1, 500, (b, ANCHOR_LENGTH))
        store.add_entries(fusion_ids, anchor_ids)
        # 刚写入时 change_counter=N_valid，比例=1.0 > 0.1
        assert store.should_recluster()

    def test_dual_store_should_recluster_false_when_empty(self) -> None:
        """无有效条目时 should_recluster 返回 False。"""
        store = _make_store()
        assert not store.should_recluster()

    def test_dual_store_next_free_overflow_raises(self) -> None:
        """超出 knowledge_num 时 add_entries 应 RuntimeError。"""
        store = _make_store_with_pca(knowledge_num=KNOWLEDGE_NUM)
        # 先填满
        fusion_ids = torch.randint(1, 500, (KNOWLEDGE_NUM, FUSION_LENGTH))
        anchor_ids = torch.randint(1, 500, (KNOWLEDGE_NUM, ANCHOR_LENGTH))
        store.add_entries(fusion_ids, anchor_ids)
        assert store.next_free == KNOWLEDGE_NUM

        # 再添加 1 条，应溢出
        with pytest.raises(RuntimeError, match="knowledge_num 已满"):
            store.add_entries(
                torch.randint(1, 500, (1, FUSION_LENGTH)),
                torch.randint(1, 500, (1, ANCHOR_LENGTH)),
            )

    def test_dual_store_add_entries_before_recluster_raises(self) -> None:
        """pca_matrix 为 None 时 add_entries 应 RuntimeError，提示先调用 compact_and_recluster。"""
        store = _make_store()  # pca_matrix=None
        fusion_ids = torch.randint(1, 500, (2, FUSION_LENGTH))
        anchor_ids = torch.randint(1, 500, (2, ANCHOR_LENGTH))
        with pytest.raises(RuntimeError, match="compact_and_recluster"):
            store.add_entries(fusion_ids, anchor_ids)

    def test_dual_store_delete_empty_ids_noop(self) -> None:
        """空列表删除是 no-op，不改变状态。"""
        store = _make_store()
        store.delete_entries([])
        assert store.change_counter == 0

    def test_dual_store_save_load_state(self) -> None:
        """序列化后反序列化，所有状态与原始一致。"""
        store = _make_store_with_pca()
        b = 3
        fusion_ids = torch.randint(1, 500, (b, FUSION_LENGTH))
        anchor_ids = torch.randint(1, 500, (b, ANCHOR_LENGTH))
        store.add_entries(fusion_ids, anchor_ids)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            store.save_state(tmp_path)

            # 构造新 store 并加载
            store2 = _make_store_with_pca()
            store2.load_state(tmp_path)

            assert store2.next_free == store.next_free
            assert store2.change_counter == store.change_counter
            assert torch.equal(store2.valid_mask, store.valid_mask)
            assert torch.equal(store2.fusion_bank.data, store.fusion_bank.data)
            assert torch.equal(store2.anchor_bank.data, store.anchor_bank.data)
            assert torch.equal(store2.inverted_index, store.inverted_index)
            assert store2.pca_matrix is not None
            assert torch.equal(store2.pca_matrix, store.pca_matrix)
        finally:
            os.unlink(tmp_path)

    def test_dual_store_save_load_state_no_pca(self) -> None:
        """pca_matrix 为 None 时序列化/反序列化正确处理 None。"""
        store = _make_store()  # pca_matrix=None

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name
        try:
            store.save_state(tmp_path)
            store2 = _make_store()
            store2.load_state(tmp_path)
            assert store2.pca_matrix is None
            assert store2.next_free == 0
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────
# AnchorBank.get_embeddings 测试（mock encoder）
# ─────────────────────────────────────────────

class TestAnchorBankGetEmbeddings:
    """AnchorBank.get_embeddings 的单元测试，使用 mock encoder。"""

    def _make_mock_encoder(self, d: int = 16) -> MagicMock:
        """构造 mock KnowledgeEncoder，encode_mean 返回随机 [B, D] 张量。"""
        encoder = MagicMock()
        encoder.device = DEVICE

        def fake_encode_mean(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            b = ids.shape[0]
            return torch.randn(b, d)

        encoder.encode_mean.side_effect = fake_encode_mean
        return encoder

    def test_get_embeddings_basic(self) -> None:
        """有效条目被正确编码，返回 [N_valid, D]。"""
        bank = AnchorBank(KNOWLEDGE_NUM, ANCHOR_LENGTH, DEVICE)
        token_ids = torch.randint(1, 500, (KNOWLEDGE_NUM, ANCHOR_LENGTH))
        bank.update_all(token_ids)

        # valid_mask：前 4 条有效
        valid_mask = torch.zeros(KNOWLEDGE_NUM, dtype=torch.bool)
        valid_mask[:4] = True

        encoder = self._make_mock_encoder(d=16)
        result = bank.get_embeddings(encoder, valid_mask, chunk_size=2)

        assert result.shape == (4, 16)
        # encode_mean 被调用 2 次（4 条 / chunk_size=2）
        assert encoder.encode_mean.call_count == 2

    def test_get_embeddings_empty_valid_mask(self) -> None:
        """valid_mask 全 False 时返回空张量。"""
        bank = AnchorBank(KNOWLEDGE_NUM, ANCHOR_LENGTH, DEVICE)
        valid_mask = torch.zeros(KNOWLEDGE_NUM, dtype=torch.bool)
        encoder = self._make_mock_encoder()
        result = bank.get_embeddings(encoder, valid_mask, chunk_size=4)
        assert result.shape[0] == 0

    def test_get_embeddings_all_valid(self) -> None:
        """全部有效时返回 [N, D]。"""
        bank = AnchorBank(KNOWLEDGE_NUM, ANCHOR_LENGTH, DEVICE)
        token_ids = torch.randint(1, 500, (KNOWLEDGE_NUM, ANCHOR_LENGTH))
        bank.update_all(token_ids)

        valid_mask = torch.ones(KNOWLEDGE_NUM, dtype=torch.bool)
        encoder = self._make_mock_encoder(d=8)
        result = bank.get_embeddings(encoder, valid_mask, chunk_size=4)
        assert result.shape == (KNOWLEDGE_NUM, 8)


# ─────────────────────────────────────────────
# DualKnowledgeStore._rebuild_inverted_index 测试
# ─────────────────────────────────────────────

class TestRebuildInvertedIndex:
    """_rebuild_inverted_index 内部逻辑的单元测试。"""

    def test_rebuild_inverted_index_basic(self) -> None:
        """重建后 inverted_index 按 grid_indices 排序，offsets/counts 正确。"""
        store = _make_store()
        num_keys = int(KNOWLEDGE_NUM ** 0.5)

        # 4 条数据，分配到 4 个不同 cluster（每个 1 条）
        data_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        grid_indices = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        store._rebuild_inverted_index(data_indices, grid_indices)

        # cluster 0 应有 1 条
        assert store.cluster_counts[0].item() == 1
        assert store.cluster_offsets[0].item() == 0
        assert store.cluster_offsets[1].item() == 1

    def test_rebuild_inverted_index_same_cluster(self) -> None:
        """多条数据分配到同一 cluster，counts 累加正确。"""
        store = _make_store()

        # 全部分配到 cluster 0
        data_indices = torch.tensor([0, 1, 2], dtype=torch.long)
        grid_indices = torch.tensor([0, 0, 0], dtype=torch.long)

        store._rebuild_inverted_index(data_indices, grid_indices)

        assert store.cluster_counts[0].item() == 3
        assert store.cluster_offsets[1].item() == 3
        # 其余 cluster 为空
        assert store.cluster_counts[1:].sum().item() == 0
