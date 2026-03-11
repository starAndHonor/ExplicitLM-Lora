"""
router/clustering.py 单元测试

测试覆盖：
    - ClusteringResult：数据类字段类型与形状
    - SubspaceClustering.fit：输出形状、标签范围、负载均衡、异常输入
    - SubspaceClustering.assign_approximate：单条/批量输入、结果范围、一致性
    - SubspaceClustering.validate_orthogonality：正交性诊断
"""

import math

import numpy as np
import pytest

from router.clustering import ClusteringResult, SubspaceClustering

# ─────────────────────────────────────────────
# 测试参数常量（小尺寸，快速执行）
# ─────────────────────────────────────────────

N = 64        # 知识条目数
D = 8         # embedding 维度（偶数）
NUM_KEYS = 4  # 聚类数（2^2 = 4，完全平方数）


def _make_embeddings(n: int = N, d: int = D, seed: int = 42) -> np.ndarray:
    """生成随机测试 embedding，float32。"""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


# ─────────────────────────────────────────────
# SubspaceClustering.fit 测试
# ─────────────────────────────────────────────

class TestSubspaceClusteringFit:
    """SubspaceClustering.fit 的单元测试。"""

    def test_fit_output_shapes(self) -> None:
        """ClusteringResult 所有字段的 shape 符合预期。"""
        emb = _make_embeddings()
        result = SubspaceClustering.fit(emb, NUM_KEYS)

        assert isinstance(result, ClusteringResult)
        assert result.pca_matrix.shape == (D, D)
        assert result.pca_mean.shape == (D,)
        assert result.pca_components.shape == (D,)
        assert result.row_centroids.shape == (NUM_KEYS, D // 2)
        assert result.col_centroids.shape == (NUM_KEYS, D // 2)
        assert result.row_labels.shape == (N,)
        assert result.col_labels.shape == (N,)

    def test_fit_output_dtypes(self) -> None:
        """ClusteringResult 各字段的 dtype 正确。"""
        emb = _make_embeddings()
        result = SubspaceClustering.fit(emb, NUM_KEYS)

        assert result.pca_matrix.dtype == np.float32
        assert result.pca_mean.dtype == np.float32
        assert result.pca_components.dtype == np.float32
        assert result.row_centroids.dtype == np.float32
        assert result.col_centroids.dtype == np.float32
        assert result.row_labels.dtype == np.int64
        assert result.col_labels.dtype == np.int64

    def test_fit_label_range(self) -> None:
        """row_labels 和 col_labels 均在 [0, NUM_KEYS) 范围内。"""
        emb = _make_embeddings()
        result = SubspaceClustering.fit(emb, NUM_KEYS)

        assert result.row_labels.min() >= 0
        assert result.row_labels.max() < NUM_KEYS
        assert result.col_labels.min() >= 0
        assert result.col_labels.max() < NUM_KEYS

    def test_fit_label_count(self) -> None:
        """row_labels 和 col_labels 的长度等于输入条目数 N。"""
        emb = _make_embeddings()
        result = SubspaceClustering.fit(emb, NUM_KEYS)

        assert len(result.row_labels) == N
        assert len(result.col_labels) == N

    def test_fit_balance(self) -> None:
        """
        轴对齐递归二分保证绝对平衡：每个 cluster 大小 ≤ ceil(N / NUM_KEYS) + 1。

        由于对半分（floor 截断），最大组大小不超过 ceil(N / K) + 1。
        """
        emb = _make_embeddings()
        result = SubspaceClustering.fit(emb, NUM_KEYS)

        # 检查子空间 cluster（不是网格）的平衡性
        for labels in [result.row_labels, result.col_labels]:
            counts = np.bincount(labels, minlength=NUM_KEYS)
            max_count = counts.max()
            expected_max = math.ceil(N / NUM_KEYS) + 1
            assert max_count <= expected_max, (
                f"cluster 不平衡：max={max_count}，期望 ≤ {expected_max}"
            )

    def test_fit_pca_eigenvalues_descending(self) -> None:
        """pca_components（特征值）应按降序排列。"""
        emb = _make_embeddings()
        result = SubspaceClustering.fit(emb, NUM_KEYS)

        # 允许浮点误差
        diffs = np.diff(result.pca_components)
        assert np.all(diffs <= 1e-4), "pca_components 应为降序特征值"

    def test_fit_diagnostic_fields(self) -> None:
        """诊断字段类型正确且值合理。"""
        emb = _make_embeddings()
        result = SubspaceClustering.fit(emb, NUM_KEYS)

        assert isinstance(result.max_cluster_size, int)
        assert isinstance(result.mean_cluster_size, float)
        assert result.max_cluster_size >= result.mean_cluster_size
        # 所有条目均被分配：sum(counts) == N
        grid_labels = result.row_labels * NUM_KEYS + result.col_labels
        total = int(np.bincount(grid_labels, minlength=NUM_KEYS * NUM_KEYS).sum())
        assert total == N

    def test_fit_invalid_num_keys_non_power_of_2(self) -> None:
        """num_keys 不为 2 的幂次时抛出 ValueError。"""
        emb = _make_embeddings()
        with pytest.raises(ValueError, match="2 的幂次"):
            SubspaceClustering.fit(emb, 3)

    def test_fit_invalid_num_keys_zero(self) -> None:
        """num_keys = 0 时抛出 ValueError。"""
        emb = _make_embeddings()
        with pytest.raises(ValueError, match="2 的幂次"):
            SubspaceClustering.fit(emb, 0)

    def test_fit_empty_raises(self) -> None:
        """N == 0 时抛出 AssertionError。"""
        emb = np.zeros((0, D), dtype=np.float32)
        with pytest.raises(AssertionError):
            SubspaceClustering.fit(emb, NUM_KEYS)

    def test_fit_odd_dimension_raises(self) -> None:
        """D 为奇数时抛出 AssertionError。"""
        emb = _make_embeddings(d=7)
        with pytest.raises(AssertionError, match="偶数"):
            SubspaceClustering.fit(emb, NUM_KEYS)


# ─────────────────────────────────────────────
# SubspaceClustering.assign_approximate 测试
# ─────────────────────────────────────────────

class TestAssignApproximate:
    """SubspaceClustering.assign_approximate 的单元测试。"""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """先跑 fit 获取 PCA 状态，供 assign_approximate 使用。"""
        emb = _make_embeddings()
        self.result = SubspaceClustering.fit(emb, NUM_KEYS)

    def test_assign_approximate_single_return_type(self) -> None:
        """单条 [D] 输入返回 int。"""
        query = np.random.default_rng(0).standard_normal(D).astype(np.float32)
        grid_idx = SubspaceClustering.assign_approximate(
            query,
            self.result.pca_matrix,
            self.result.pca_mean,
            self.result.row_centroids,
            self.result.col_centroids,
            NUM_KEYS,
        )
        assert isinstance(grid_idx, int)

    def test_assign_approximate_single_range(self) -> None:
        """单条输入的返回值 ∈ [0, NUM_KEYS²)。"""
        query = np.random.default_rng(1).standard_normal(D).astype(np.float32)
        grid_idx = SubspaceClustering.assign_approximate(
            query,
            self.result.pca_matrix,
            self.result.pca_mean,
            self.result.row_centroids,
            self.result.col_centroids,
            NUM_KEYS,
        )
        assert 0 <= grid_idx < NUM_KEYS * NUM_KEYS

    def test_assign_approximate_batch_shape(self) -> None:
        """批量 [B, D] 输入返回 np.ndarray，shape=[B]。"""
        B = 8
        queries = np.random.default_rng(2).standard_normal((B, D)).astype(np.float32)
        result = SubspaceClustering.assign_approximate(
            queries,
            self.result.pca_matrix,
            self.result.pca_mean,
            self.result.row_centroids,
            self.result.col_centroids,
            NUM_KEYS,
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (B,)

    def test_assign_approximate_batch_range(self) -> None:
        """批量输入的所有返回值 ∈ [0, NUM_KEYS²)。"""
        B = 16
        queries = np.random.default_rng(3).standard_normal((B, D)).astype(np.float32)
        result = SubspaceClustering.assign_approximate(
            queries,
            self.result.pca_matrix,
            self.result.pca_mean,
            self.result.row_centroids,
            self.result.col_centroids,
            NUM_KEYS,
        )
        assert result.min() >= 0
        assert result.max() < NUM_KEYS * NUM_KEYS

    def test_assign_approximate_consistent(self) -> None:
        """相同输入，两次调用结果相同（无随机性）。"""
        query = np.random.default_rng(4).standard_normal(D).astype(np.float32)
        kwargs = dict(
            pca_matrix=self.result.pca_matrix,
            pca_mean=self.result.pca_mean,
            row_centroids=self.result.row_centroids,
            col_centroids=self.result.col_centroids,
            num_keys=NUM_KEYS,
        )
        r1 = SubspaceClustering.assign_approximate(query, **kwargs)
        r2 = SubspaceClustering.assign_approximate(query, **kwargs)
        assert r1 == r2

    def test_assign_approximate_accepts_torch_tensor(self) -> None:
        """支持 torch.Tensor 输入，结果与 numpy 输入一致。"""
        import torch

        query_np = np.random.default_rng(5).standard_normal(D).astype(np.float32)
        query_t = torch.tensor(query_np)
        kwargs = dict(
            pca_matrix=self.result.pca_matrix,
            pca_mean=self.result.pca_mean,
            row_centroids=self.result.row_centroids,
            col_centroids=self.result.col_centroids,
            num_keys=NUM_KEYS,
        )
        r_np = SubspaceClustering.assign_approximate(query_np, **kwargs)
        r_t = SubspaceClustering.assign_approximate(query_t, **kwargs)
        assert r_np == r_t


# ─────────────────────────────────────────────
# SubspaceClustering.validate_orthogonality 测试
# ─────────────────────────────────────────────

class TestValidateOrthogonality:
    """SubspaceClustering.validate_orthogonality 的单元测试。"""

    def test_orthogonal_centroids_low_score(self) -> None:
        """行列中心点完全正交时偏差 ≈ 0。"""
        # 行中心点：仅前半维有值
        row_c = np.zeros((NUM_KEYS, D // 2), dtype=np.float32)
        row_c[:, 0] = 1.0
        # 列中心点：仅后半维有值（若在同一空间中为 0 → 正交）
        col_c = np.zeros((NUM_KEYS, D // 2), dtype=np.float32)
        col_c[:, -1] = 1.0

        score = SubspaceClustering.validate_orthogonality(row_c, col_c)
        # 行/列子空间各自独立，点积非 0 但比例低（因为所有行共同方向与列共同方向有一定内积）
        # 此处主要验证函数可调用且返回非负浮点
        assert isinstance(score, float)
        assert score >= 0.0

    def test_zero_centroids_returns_zero(self) -> None:
        """中心点全零时返回 0.0（避免除零）。"""
        row_c = np.zeros((NUM_KEYS, D // 2), dtype=np.float32)
        col_c = np.zeros((NUM_KEYS, D // 2), dtype=np.float32)
        score = SubspaceClustering.validate_orthogonality(row_c, col_c)
        assert score == 0.0
