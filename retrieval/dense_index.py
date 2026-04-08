"""
retrieval/dense_index.py — 动态可更新的稠密检索索引

第一版提供：
    - Flat backend（纯 PyTorch，便于当前环境直接运行）
    - 动态 add / replace / logical delete / compact
    - save / load 持久化

第二版预留：
    - HNSW backend（推荐使用 faiss.IndexHNSWFlat）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - 当前环境无 faiss
    faiss = None


@dataclass
class DenseSearchOutput:
    """一次稠密检索的输出。"""

    indices: torch.Tensor
    scores: torch.Tensor
    valid_mask: torch.Tensor
    fusion_ids: torch.Tensor


@dataclass
class DenseAnchorStore:
    """
    显式的检索侧 store。

    角色上对应原 Phase1 中的 AnchorBank，但这里保存的是：
        - 检索 embedding
        - key / text 元信息
        - valid_mask
    """

    embeddings: torch.Tensor
    keys: List[str]
    texts: List[str]
    valid_mask: torch.Tensor


@dataclass
class DenseFusionStore:
    """
    显式的注入侧 store。

    角色上对应原 Phase1 中的 FusionBank，保存最终注入 LLM 的 fusion token ids。
    """

    fusion_ids: torch.Tensor


class _TorchFlatBackend:
    """基于 PyTorch 的 Flat 相似度检索后端。"""

    def __init__(self, dim: int, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize
        self.embeddings = torch.empty((0, dim), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    def add(self, embeddings: torch.Tensor) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embeddings shape mismatch: expected [N, {self.dim}], got {tuple(embeddings.shape)}"
            )
        emb = embeddings.detach().cpu().float()
        if self.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        self.embeddings = torch.cat([self.embeddings, emb], dim=0)

    def rebuild(self, embeddings: torch.Tensor) -> None:
        self.embeddings = torch.empty((0, self.dim), dtype=torch.float32)
        if embeddings.numel() > 0:
            self.add(embeddings)

    def search(
        self,
        queries: torch.Tensor,
        top_k: int,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if len(self) == 0:
            scores = torch.full((queries.shape[0], top_k), float("-inf"), dtype=torch.float32)
            indices = torch.full((queries.shape[0], top_k), -1, dtype=torch.long)
            return scores, indices

        q = queries.detach().cpu().float()
        if self.normalize:
            q = F.normalize(q, p=2, dim=-1)

        scores = q @ self.embeddings.T
        if valid_mask is not None:
            if valid_mask.shape != (len(self),):
                raise ValueError(
                    f"valid_mask shape mismatch: expected ({len(self)},), got {tuple(valid_mask.shape)}"
                )
            invalid = ~valid_mask.detach().cpu().bool()
            if invalid.any():
                scores[:, invalid] = float("-inf")

        k = min(top_k, scores.shape[1])
        top_scores, top_indices = torch.topk(scores, k=k, dim=-1)
        if k < top_k:
            pad_scores = torch.full((scores.shape[0], top_k - k), float("-inf"), dtype=top_scores.dtype)
            pad_indices = torch.full((scores.shape[0], top_k - k), -1, dtype=top_indices.dtype)
            top_scores = torch.cat([top_scores, pad_scores], dim=-1)
            top_indices = torch.cat([top_indices, pad_indices], dim=-1)
        return top_scores, top_indices

    def state_dict(self) -> Dict[str, object]:
        return {
            "backend_type": "flat",
            "dim": self.dim,
            "normalize": self.normalize,
            "embeddings": self.embeddings,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "_TorchFlatBackend":
        backend = cls(dim=int(state["dim"]), normalize=bool(state["normalize"]))
        backend.embeddings = torch.as_tensor(state["embeddings"], dtype=torch.float32).cpu()
        return backend


class _FaissHNSWBackend:
    """
    预留的 HNSW 后端。

    当前实现已具备 save/load 接口，但只有在安装 faiss 后才能使用。
    """

    def __init__(
        self,
        dim: int,
        normalize: bool = True,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        if faiss is None:  # pragma: no cover
            raise ImportError(
                "faiss is required for HNSW backend. Install faiss-cpu/faiss-gpu before using index_type='hnsw'."
            )
        self.dim = dim
        self.normalize = normalize
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, embeddings: torch.Tensor) -> None:
        emb = embeddings.detach().cpu().float()
        if emb.ndim != 2 or emb.shape[1] != self.dim:
            raise ValueError(
                f"embeddings shape mismatch: expected [N, {self.dim}], got {tuple(emb.shape)}"
            )
        if self.normalize and emb.numel() > 0:
            emb = F.normalize(emb, p=2, dim=-1)
        self.index.add(emb.numpy())
        self._size += int(emb.shape[0])

    def rebuild(self, embeddings: torch.Tensor) -> None:
        self.index = faiss.IndexHNSWFlat(self.dim, self.m)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        self._size = 0
        if embeddings.numel() > 0:
            self.add(embeddings)

    def search(
        self,
        queries: torch.Tensor,
        top_k: int,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_mask is not None and not bool(valid_mask.all().item()):
            raise RuntimeError(
                "HNSW backend currently requires compact() after logical deletes; invalid entries are not masked online."
            )
        q = queries.detach().cpu().float()
        if self.normalize and q.numel() > 0:
            q = F.normalize(q, p=2, dim=-1)
        scores_np, indices_np = self.index.search(q.numpy(), top_k)
        return (
            torch.from_numpy(scores_np).float(),
            torch.from_numpy(indices_np).long(),
        )

    def state_dict(self) -> Dict[str, object]:
        if faiss is None:  # pragma: no cover
            raise RuntimeError("faiss unavailable while exporting HNSW state")
        return {
            "backend_type": "hnsw",
            "dim": self.dim,
            "normalize": self.normalize,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "index_bytes": faiss.serialize_index(self.index),
            "size": self._size,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "_FaissHNSWBackend":
        if faiss is None:  # pragma: no cover
            raise ImportError("faiss is required to load a saved HNSW index")
        backend = cls(
            dim=int(state["dim"]),
            normalize=bool(state["normalize"]),
            m=int(state["m"]),
            ef_construction=int(state["ef_construction"]),
            ef_search=int(state["ef_search"]),
        )
        backend.index = faiss.deserialize_index(state["index_bytes"])
        backend._size = int(state["size"])
        return backend


class DenseKnowledgeIndex:
    """
    动态可更新的稠密检索知识索引。

    逻辑层保存：
        - 文档 embedding 的 ANN / Flat backend
        - anchor_store：检索侧显式 store
        - fusion_store：注入侧显式 store
        - backend：ANN / Flat 后端
    """

    def __init__(
        self,
        dim: int,
        fusion_length: int,
        index_type: str = "flat",
        normalize: bool = True,
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ) -> None:
        self.dim = dim
        self.fusion_length = fusion_length
        self.index_type = index_type.lower()
        self.normalize = normalize
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        if self.index_type == "flat":
            self._backend = _TorchFlatBackend(dim=dim, normalize=normalize)
        elif self.index_type == "hnsw":
            self._backend = _FaissHNSWBackend(
                dim=dim,
                normalize=normalize,
                m=hnsw_m,
                ef_construction=hnsw_ef_construction,
                ef_search=hnsw_ef_search,
            )
        else:
            raise ValueError(f"unsupported dense index type: {index_type}")

        self.anchor_store = DenseAnchorStore(
            embeddings=torch.empty((0, dim), dtype=torch.float32),
            keys=[],
            texts=[],
            valid_mask=torch.empty((0,), dtype=torch.bool),
        )
        self.fusion_store = DenseFusionStore(
            fusion_ids=torch.empty((0, fusion_length), dtype=torch.long)
        )
        self._key_to_latest: Dict[str, int] = {}

    def __len__(self) -> int:
        return int(self.anchor_store.valid_mask.shape[0])

    @property
    def num_active(self) -> int:
        return int(self.anchor_store.valid_mask.sum().item())

    @property
    def embeddings(self) -> torch.Tensor:
        return self.anchor_store.embeddings

    @property
    def keys(self) -> List[str]:
        return self.anchor_store.keys

    @keys.setter
    def keys(self, value: List[str]) -> None:
        self.anchor_store.keys = value

    @property
    def texts(self) -> List[str]:
        return self.anchor_store.texts

    @texts.setter
    def texts(self, value: List[str]) -> None:
        self.anchor_store.texts = value

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.anchor_store.valid_mask

    @valid_mask.setter
    def valid_mask(self, value: torch.Tensor) -> None:
        self.anchor_store.valid_mask = value

    @property
    def fusion_ids(self) -> torch.Tensor:
        return self.fusion_store.fusion_ids

    @fusion_ids.setter
    def fusion_ids(self, value: torch.Tensor) -> None:
        self.fusion_store.fusion_ids = value

    def add_entries(
        self,
        embeddings: torch.Tensor,
        fusion_ids: torch.Tensor,
        keys: Optional[Sequence[str]] = None,
        texts: Optional[Sequence[str]] = None,
        replace_existing: bool = False,
    ) -> None:
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(
                f"embeddings shape mismatch: expected [N, {self.dim}], got {tuple(embeddings.shape)}"
            )
        if fusion_ids.shape != (embeddings.shape[0], self.fusion_length):
            raise ValueError(
                f"fusion_ids shape mismatch: expected {(embeddings.shape[0], self.fusion_length)}, got {tuple(fusion_ids.shape)}"
            )

        n = int(embeddings.shape[0])
        keys_list = list(keys) if keys is not None else [str(len(self) + i) for i in range(n)]
        texts_list = list(texts) if texts is not None else [""] * n
        if len(keys_list) != n or len(texts_list) != n:
            raise ValueError("keys/texts length must match embeddings")

        if replace_existing:
            for key in keys_list:
                old_idx = self._key_to_latest.get(key)
                if old_idx is not None and 0 <= old_idx < len(self.valid_mask):
                    self.valid_mask[old_idx] = False

        self._backend.add(embeddings)
        backend_state = self._backend.state_dict()
        self.anchor_store.embeddings = torch.as_tensor(
            backend_state.get("embeddings", torch.empty((0, self.dim))),
            dtype=torch.float32,
        ).cpu()
        self.fusion_ids = torch.cat([self.fusion_ids, fusion_ids.detach().cpu().long()], dim=0)
        self.keys.extend(keys_list)
        self.texts.extend(texts_list)
        self.valid_mask = torch.cat([self.valid_mask, torch.ones(n, dtype=torch.bool)], dim=0)

        start = len(self.keys) - n
        for offset, key in enumerate(keys_list):
            self._key_to_latest[key] = start + offset

    def delete_by_keys(self, keys: Iterable[str]) -> int:
        deleted = 0
        for key in keys:
            idx = self._key_to_latest.get(str(key))
            if idx is not None and self.valid_mask[idx]:
                self.valid_mask[idx] = False
                deleted += 1
        return deleted

    def compact(self) -> None:
        """物理清理失效条目并重建后端索引。"""
        if len(self) == self.num_active:
            return
        keep = torch.where(self.valid_mask)[0]
        backend_state = self._backend.state_dict()
        embeddings = torch.as_tensor(
            backend_state.get("embeddings", torch.empty((0, self.dim))),
            dtype=torch.float32,
        )
        if self.index_type == "hnsw":
            raise RuntimeError("compact() for HNSW is not yet implemented; rebuild from source instead.")

        kept_embeddings = embeddings[keep] if keep.numel() > 0 else torch.empty((0, self.dim), dtype=torch.float32)
        self.fusion_ids = self.fusion_ids[keep]
        self.keys = [self.keys[int(i)] for i in keep.tolist()]
        self.texts = [self.texts[int(i)] for i in keep.tolist()]
        self.valid_mask = torch.ones(len(self.keys), dtype=torch.bool)
        self._key_to_latest = {key: idx for idx, key in enumerate(self.keys)}
        self._backend.rebuild(kept_embeddings)
        self.anchor_store.embeddings = kept_embeddings.cpu()

    def search(self, query_embeddings: torch.Tensor, top_k: int) -> DenseSearchOutput:
        scores, indices = self._backend.search(query_embeddings, top_k=top_k, valid_mask=self.valid_mask)
        valid = indices >= 0
        if valid.any():
            selected_valid = self.valid_mask[indices.clamp(min=0)]
            valid = valid & selected_valid
        fusion = torch.zeros(
            (indices.shape[0], indices.shape[1], self.fusion_length),
            dtype=torch.long,
        )
        if valid.any():
            flat_valid = indices[valid]
            fusion[valid] = self.fusion_ids[flat_valid]
        if (~valid).any():
            scores = scores.clone()
            indices = indices.clone()
            scores[~valid] = float("-inf")
            indices[~valid] = -1
        return DenseSearchOutput(
            indices=indices,
            scores=scores,
            valid_mask=valid,
            fusion_ids=fusion,
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": 1,
            "dim": self.dim,
            "fusion_length": self.fusion_length,
            "index_type": self.index_type,
            "normalize": self.normalize,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "backend": self._backend.state_dict(),
            "anchor_store": {
                "embeddings": self.anchor_store.embeddings,
                "keys": self.anchor_store.keys,
                "texts": self.anchor_store.texts,
                "valid_mask": self.anchor_store.valid_mask,
            },
            "fusion_store": {
                "fusion_ids": self.fusion_store.fusion_ids,
            },
        }

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), Path(path))

    @classmethod
    def from_state_dict(cls, state: Dict[str, object]) -> "DenseKnowledgeIndex":
        index = cls(
            dim=int(state["dim"]),
            fusion_length=int(state["fusion_length"]),
            index_type=str(state["index_type"]),
            normalize=bool(state["normalize"]),
            hnsw_m=int(state.get("hnsw_m", 32)),
            hnsw_ef_construction=int(state.get("hnsw_ef_construction", 200)),
            hnsw_ef_search=int(state.get("hnsw_ef_search", 64)),
        )
        backend_state = state["backend"]
        if index.index_type == "flat":
            index._backend = _TorchFlatBackend.from_state_dict(backend_state)
        else:
            index._backend = _FaissHNSWBackend.from_state_dict(backend_state)

        anchor_state = state.get("anchor_store")
        fusion_state = state.get("fusion_store")

        if anchor_state is not None and fusion_state is not None:
            index.anchor_store = DenseAnchorStore(
                embeddings=torch.as_tensor(anchor_state["embeddings"], dtype=torch.float32).cpu(),
                keys=[str(x) for x in anchor_state["keys"]],
                texts=[str(x) for x in anchor_state["texts"]],
                valid_mask=torch.as_tensor(anchor_state["valid_mask"], dtype=torch.bool).cpu(),
            )
            index.fusion_store = DenseFusionStore(
                fusion_ids=torch.as_tensor(fusion_state["fusion_ids"], dtype=torch.long).cpu()
            )
        else:
            backend_embeddings = torch.as_tensor(
                backend_state.get("embeddings", torch.empty((0, index.dim))),
                dtype=torch.float32,
            ).cpu()
            index.anchor_store = DenseAnchorStore(
                embeddings=backend_embeddings,
                keys=[str(x) for x in state["keys"]],
                texts=[str(x) for x in state["texts"]],
                valid_mask=torch.as_tensor(state["valid_mask"], dtype=torch.bool).cpu(),
            )
            index.fusion_store = DenseFusionStore(
                fusion_ids=torch.as_tensor(state["fusion_ids"], dtype=torch.long).cpu()
            )
        index._key_to_latest = {
            key: idx for idx, key in enumerate(index.keys) if index.valid_mask[idx]
        }
        return index

    @classmethod
    def load(cls, path: str | Path) -> "DenseKnowledgeIndex":
        state = torch.load(Path(path), map_location="cpu")
        return cls.from_state_dict(state)
