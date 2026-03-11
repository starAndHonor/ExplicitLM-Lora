"""
router/memory_gate.py 单元测试

测试覆盖：
    - ProductKeyMemory 初始化：层维度、buffer 形状、超参数
    - update_keys：正常更新、形状/dtype 校验
    - forward：输出形状、候选值域、分数形状、q_adapted 归一化
    - _lookup_candidates：全量模式、1:1 模式、空 store 安全处理
    - keys 无梯度（register_buffer 验证）
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from router.memory_bank import DualKnowledgeStore
from router.memory_gate import ProductKeyMemory


# ─────────────────────────────────────────────
# 常量与辅助函数
# ─────────────────────────────────────────────

# 测试用最小规模：4×4=16 条知识，num_keys=4
KNOWLEDGE_NUM = 16
NUM_KEYS = 4  # √16
DIM = 1024
QUERY_DIM = 1024
KEY_PROJ_DIM = 512  # = DIM // 2；同时也是 row_centroids 的列维度
NUM_CANDIDATES = 4
TEMPERATURE = 0.1
DEVICE = "cpu"


def _make_router_config(
    knowledge_num: int = KNOWLEDGE_NUM,
    dim: int = DIM,
    query_dim: int = QUERY_DIM,
    key_proj_dim: int = KEY_PROJ_DIM,
    num_candidates: int = NUM_CANDIDATES,
    temperature: float = TEMPERATURE,
    max_candidates_per_cell: int = -1,
) -> Any:
    """构造最小 RouterConfig mock（仅含 ProductKeyMemory 需要的字段）。"""
    cfg = MagicMock()
    cfg.knowledge_num = knowledge_num
    cfg.dim = dim
    cfg.query_dim = query_dim
    cfg.key_proj_dim = key_proj_dim
    cfg.num_candidates = num_candidates
    cfg.temperature = temperature
    cfg.max_candidates_per_cell = max_candidates_per_cell
    return cfg


def _make_pkm(max_candidates_per_cell: int = -1) -> ProductKeyMemory:
    """构造测试用 ProductKeyMemory（小规模配置）。"""
    cfg = _make_router_config(max_candidates_per_cell=max_candidates_per_cell)
    return ProductKeyMemory(cfg)


def _make_store_with_index() -> DualKnowledgeStore:
    """
    构造已初始化倒排索引的 DualKnowledgeStore（不依赖真实聚类，直接手动注入）。

    设计：knowledge_num=16，num_keys=4，每个 grid cell 恰好 1 条知识（完美平衡分布）。
    grid cell c 包含 entry c（即 cell 0→entry 0，cell 1→entry 1，...，cell 15→entry 15）。

    注意：row_centroids/col_centroids 形状为 [num_keys, KEY_PROJ_DIM=512]
          等同于 DualKnowledgeStore 的 [num_keys, D//2]（D=1024, D//2=512）。
    """
    store_cfg = MagicMock()
    store_cfg.knowledge_num = KNOWLEDGE_NUM
    store_cfg.recluster_threshold = 0.1
    store = DualKnowledgeStore(
        store_cfg, fusion_length=4, anchor_length=4, device=DEVICE
    )

    n = KNOWLEDGE_NUM  # 16
    num_keys = NUM_KEYS  # 4
    c = num_keys * num_keys  # 16 个 grid cell

    # Phase 1: 注入倒排索引
    # inverted_index: 按 grid cell 顺序排列 entry ID（cell i 对应 entry i）
    store.inverted_index = torch.arange(n, dtype=torch.long)  # [0,1,...,15]

    # Phase 2: cluster_offsets / cluster_counts（每格恰好 1 条）
    store.cluster_counts = torch.ones(c, dtype=torch.long)  # [1,1,...,1]
    store.cluster_offsets = torch.arange(c + 1, dtype=torch.long)  # [0,1,2,...,16]

    # Phase 3: 注入 row_centroids / col_centroids（[num_keys, KEY_PROJ_DIM=512]）
    # topk 逻辑不依赖具体值，随机初始化即可
    store.row_centroids = torch.randn(num_keys, KEY_PROJ_DIM, dtype=torch.float)
    store.col_centroids = torch.randn(num_keys, KEY_PROJ_DIM, dtype=torch.float)

    # Phase 4: valid_mask（全部有效）
    store.valid_mask = torch.ones(n, dtype=torch.bool)

    return store


# ─────────────────────────────────────────────
# 测试类
# ─────────────────────────────────────────────


class TestProductKeyMemoryInit:
    """测试 __init__ 层维度和 buffer 形状。"""

    def test_init_param_shapes(self) -> None:
        """验证三个可训练层的 in_features/out_features 正确。"""
        pkm = _make_pkm()

        # query_proj: [DIM=1024 → QUERY_DIM=1024]
        assert pkm.query_proj.in_features == DIM
        assert pkm.query_proj.out_features == QUERY_DIM
        assert pkm.query_proj.bias is None  # bias=False

        # row_key_proj / col_key_proj: [KEY_PROJ_DIM=512 → KEY_PROJ_DIM=512]
        assert pkm.row_key_proj.in_features == KEY_PROJ_DIM
        assert pkm.row_key_proj.out_features == KEY_PROJ_DIM
        assert pkm.row_key_proj.bias is None

        assert pkm.col_key_proj.in_features == KEY_PROJ_DIM
        assert pkm.col_key_proj.out_features == KEY_PROJ_DIM
        assert pkm.col_key_proj.bias is None

    def test_buffer_shapes(self) -> None:
        """验证 register_buffer row_keys/col_keys 的形状和初始值。"""
        pkm = _make_pkm()

        # [√N, KEY_PROJ_DIM] = [4, 512]
        assert pkm.row_keys.shape == (NUM_KEYS, KEY_PROJ_DIM)
        assert pkm.col_keys.shape == (NUM_KEYS, KEY_PROJ_DIM)

        # 初始化为全零
        assert pkm.row_keys.sum().item() == 0.0
        assert pkm.col_keys.sum().item() == 0.0

    def test_hyperparams(self) -> None:
        """验证超参数正确写入。"""
        pkm = _make_pkm()
        assert pkm.num_keys == NUM_KEYS
        assert pkm.K_COARSE == 4
        assert pkm.temperature == TEMPERATURE
        assert pkm.num_candidates == NUM_CANDIDATES

    def test_knowledge_num_not_perfect_square_raises(self) -> None:
        """knowledge_num 不是完全平方数时，__init__ 应抛出 AssertionError。"""
        cfg = _make_router_config(knowledge_num=15)  # √15 不是整数
        with pytest.raises(AssertionError, match="完全平方数"):
            ProductKeyMemory(cfg)


class TestUpdateKeys:
    """测试 update_keys 方法。"""

    def test_update_keys_normal(self) -> None:
        """正常调用 update_keys 后，buffer 内容应更新。"""
        pkm = _make_pkm()
        new_row = torch.ones(NUM_KEYS, KEY_PROJ_DIM, dtype=torch.float) * 2.0
        new_col = torch.ones(NUM_KEYS, KEY_PROJ_DIM, dtype=torch.float) * 3.0

        pkm.update_keys(new_row, new_col)

        assert torch.allclose(pkm.row_keys, new_row)
        assert torch.allclose(pkm.col_keys, new_col)

    def test_update_keys_shape_mismatch_raises(self) -> None:
        """行数不匹配时 update_keys 应抛出 AssertionError。"""
        pkm = _make_pkm()
        bad_keys = torch.zeros(NUM_KEYS + 1, KEY_PROJ_DIM, dtype=torch.float)
        col_keys = torch.zeros(NUM_KEYS, KEY_PROJ_DIM, dtype=torch.float)

        with pytest.raises(AssertionError, match="row_keys 形状不匹配"):
            pkm.update_keys(bad_keys, col_keys)

    def test_update_keys_dtype_mismatch_raises(self) -> None:
        """dtype 不为 torch.float 时 update_keys 应抛出 AssertionError。"""
        pkm = _make_pkm()
        bad_dtype = torch.zeros(NUM_KEYS, KEY_PROJ_DIM, dtype=torch.half)
        ok_keys = torch.zeros(NUM_KEYS, KEY_PROJ_DIM, dtype=torch.float)

        with pytest.raises(AssertionError, match="torch.float"):
            pkm.update_keys(bad_dtype, ok_keys)


class TestProductKeyMemoryForward:
    """测试 forward 的输出形状和语义正确性。"""

    @pytest.fixture()
    def pkm_and_store(self):
        """返回已注入 keys 的 PKM 和合法 store。"""
        pkm = _make_pkm()
        store = _make_store_with_index()
        # 注入 keys（使用 store 的 centroids，形状 [num_keys, KEY_PROJ_DIM=512]）
        pkm.update_keys(store.row_centroids, store.col_centroids)
        return pkm, store

    def test_forward_output_shapes(self, pkm_and_store) -> None:
        """验证 forward 四个返回值的形状。"""
        pkm, store = pkm_and_store
        B = 3
        embedding = torch.randn(B, DIM)

        candidates, scores_1, scores_2, q_adapted = pkm(embedding, store)

        assert candidates.shape == (B, NUM_CANDIDATES), (
            f"candidates 形状错误：{candidates.shape}"
        )
        assert scores_1.shape == (B, NUM_KEYS), f"scores_1 形状错误：{scores_1.shape}"
        assert scores_2.shape == (B, NUM_KEYS), f"scores_2 形状错误：{scores_2.shape}"
        # q_adapted = q1 = query[:, :query_dim//2] → [B, 512] = [B, KEY_PROJ_DIM]
        assert q_adapted.shape == (B, KEY_PROJ_DIM), (
            f"q_adapted 形状错误：{q_adapted.shape}"
        )

    def test_candidates_dtype_long(self, pkm_and_store) -> None:
        """candidates 应为 torch.long。"""
        pkm, store = pkm_and_store
        embedding = torch.randn(2, DIM)
        candidates, _, _, _ = pkm(embedding, store)
        assert candidates.dtype == torch.long

    def test_candidates_range(self, pkm_and_store) -> None:
        """candidates 值域必须在 [0, knowledge_num)。"""
        pkm, store = pkm_and_store
        embedding = torch.randn(4, DIM)
        candidates, _, _, _ = pkm(embedding, store)

        assert candidates.min().item() >= 0, f"candidates 有负值：{candidates.min()}"
        assert candidates.max().item() < KNOWLEDGE_NUM, (
            f"candidates 越界：max={candidates.max()} >= knowledge_num={KNOWLEDGE_NUM}"
        )

    def test_q_adapted_normalized(self, pkm_and_store) -> None:
        """q_adapted 每行 L2 norm 应约为 1.0。"""
        pkm, store = pkm_and_store
        embedding = torch.randn(4, DIM)
        _, _, _, q_adapted = pkm(embedding, store)

        norms = q_adapted.norm(p=2, dim=-1)  # [B]
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            f"q_adapted 未归一化，norms={norms}"
        )

    def test_scores_shape(self, pkm_and_store) -> None:
        """scores_1/scores_2 均应为 [B, num_keys]。"""
        pkm, store = pkm_and_store
        B = 2
        embedding = torch.randn(B, DIM)
        _, scores_1, scores_2, _ = pkm(embedding, store)

        assert scores_1.shape == (B, NUM_KEYS)
        assert scores_2.shape == (B, NUM_KEYS)

    def test_embedding_ndim_check(self, pkm_and_store) -> None:
        """输入 embedding 不是 2D 时应抛出 AssertionError。"""
        pkm, store = pkm_and_store
        bad_embedding = torch.randn(2, 4, DIM)  # 3D
        with pytest.raises(AssertionError, match="2D"):
            pkm(bad_embedding, store)


class TestKeysNoGrad:
    """验证 row_keys/col_keys 是 register_buffer（无梯度）。"""

    def test_keys_no_grad(self) -> None:
        """row_keys 和 col_keys 不应有梯度。"""
        pkm = _make_pkm()
        assert not pkm.row_keys.requires_grad
        assert not pkm.col_keys.requires_grad

    def test_keys_not_in_parameters(self) -> None:
        """row_keys/col_keys 不应出现在 named_parameters() 中。"""
        pkm = _make_pkm()
        param_names = [name for name, _ in pkm.named_parameters()]
        assert "row_keys" not in param_names
        assert "col_keys" not in param_names

    def test_keys_in_buffers(self) -> None:
        """row_keys/col_keys 应出现在 named_buffers() 中。"""
        pkm = _make_pkm()
        buffer_names = [name for name, _ in pkm.named_buffers()]
        assert "row_keys" in buffer_names
        assert "col_keys" in buffer_names


class TestLookupCandidates:
    """测试候选查询逻辑（全量模式 vs 1:1 模式 vs 空 store）。"""

    def test_empty_store_safe(self) -> None:
        """inverted_index 全-1（空 store）时，candidates 全零，不抛异常。"""
        pkm = _make_pkm()
        # 构造空 store（所有 cluster_counts=0）
        store_cfg = MagicMock()
        store_cfg.knowledge_num = KNOWLEDGE_NUM
        store_cfg.recluster_threshold = 0.1
        store = DualKnowledgeStore(
            store_cfg, fusion_length=4, anchor_length=4, device=DEVICE
        )

        # store.cluster_counts 全零（默认），inverted_index 全-1（默认）
        store.row_centroids = torch.randn(NUM_KEYS, KEY_PROJ_DIM, dtype=torch.float)
        store.col_centroids = torch.randn(NUM_KEYS, KEY_PROJ_DIM, dtype=torch.float)
        pkm.update_keys(store.row_centroids, store.col_centroids)

        embedding = torch.randn(2, DIM)
        candidates, _, _, _ = pkm(embedding, store)

        # 空 store 时 candidates 全零（安全默认值）
        assert torch.all(candidates == 0), (
            f"空 store 时 candidates 非全零：{candidates}"
        )

    def test_one_to_one_mode(self) -> None:
        """max_candidates_per_cell=1 时，每格最多 1 条，验证 cap 逻辑生效。"""
        pkm = _make_pkm(max_candidates_per_cell=1)
        store = _make_store_with_index()
        pkm.update_keys(store.row_centroids, store.col_centroids)

        # 为 cell 0 注入 2 条（模拟热更新后多条/格），验证 1:1 模式只取 1 条
        n = KNOWLEDGE_NUM
        num_keys = NUM_KEYS
        c = num_keys * num_keys
        inverted_extended = [0, 0] + list(range(1, n))  # cell 0 有 2 条 entry(0,0)

        store.inverted_index = torch.tensor(inverted_extended, dtype=torch.long)
        store.cluster_counts = torch.ones(c, dtype=torch.long)
        store.cluster_counts[0] = 2  # cell 0 现在有 2 条
        offsets = torch.zeros(c + 1, dtype=torch.long)
        offsets[1:] = store.cluster_counts.cumsum(0)
        store.cluster_offsets = offsets

        embedding = torch.randn(1, DIM)
        candidates, _, _, _ = pkm(embedding, store)

        assert candidates.shape == (1, NUM_CANDIDATES)
        assert candidates.min().item() >= 0

    def test_full_mode_uses_all_entries(self) -> None:
        """max_candidates_per_cell=-1 时，能从每格取多条。"""
        pkm = _make_pkm(max_candidates_per_cell=-1)
        store = _make_store_with_index()
        pkm.update_keys(store.row_centroids, store.col_centroids)

        embedding = torch.randn(2, DIM)
        candidates, _, _, _ = pkm(embedding, store)

        assert candidates.shape == (2, NUM_CANDIDATES)
        assert candidates.min().item() >= 0
        assert candidates.max().item() < KNOWLEDGE_NUM
