"""
training/phase1_retriever.py — 冻结版 Phase 1 Router 检索封装

用途：
    为 Phase 2 / Phase 3 提供统一的 frozen retrieval 接口。
    输入一批文本 query，内部完成：

        texts
          -> tokenize(anchor_length)
          -> KnowledgeEncoder.encode_mean()
          -> MemoryRouter.forward()
          -> FusionBank[best_id]
          -> knowledge_ids [B, K_f]

设计约束：
    - Phase 1 参数与 Store 全部冻结，只做推理检索
    - 默认将 Store 保留在 CPU，避免额外显存占用
    - 返回的 knowledge_ids 会移动到调用设备，便于直接喂给 ModifiedQwen
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoTokenizer

from config import Config
from models.qwen_wrapper import KnowledgeEncoder, load_base_model
from router.memory_bank import DualKnowledgeStore
from router.model import MemoryRouter

logger = logging.getLogger(__name__)


class Phase1Retriever:
    """冻结版 Phase 1 Router 检索器。"""

    def __init__(
        self,
        cfg: Config,
        phase1_ckpt: str,
        device: torch.device | str,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.ckpt_dir = Path(phase1_ckpt)
        if not self.ckpt_dir.exists():
            raise FileNotFoundError(f"Phase 1 checkpoint 目录不存在: {self.ckpt_dir}")

        router_path = self.ckpt_dir / "router.pt"
        store_path = self.ckpt_dir / "store.pt"
        if not router_path.exists():
            raise FileNotFoundError(f"router checkpoint 不存在: {router_path}")
        if not store_path.exists():
            raise FileNotFoundError(f"store checkpoint 不存在: {store_path}")

        model_path = os.environ.get("MODEL_PATH", cfg.paths.model_dir)
        use_bf16 = cfg.train.bf16 and self.device.type != "cpu"
        base_model = load_base_model(model_path, bf16=use_bf16)
        self.encoder = KnowledgeEncoder(
            base_model=base_model,
            encoder_depth=cfg.model.retrieval_encoder_depth,
            hidden_dim=cfg.model.hidden_dim,
            mode=cfg.model.knowledge_encoder_mode,
        )
        self.encoder.requires_grad_(False)
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        self.router = MemoryRouter(cfg.router, self.encoder)
        router_state = torch.load(router_path, map_location="cpu", weights_only=True)
        self.router.load_state_dict(router_state)
        self.router.requires_grad_(False)
        router_dtype = next(self.encoder.parameters()).dtype
        self.router = self.router.to(device=self.device, dtype=router_dtype)
        self.router.eval()

        self.store = DualKnowledgeStore(
            cfg.router,
            cfg.model.fusion_length,
            cfg.model.anchor_length,
            device="cpu",
        )
        self.store.load_state(str(store_path))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.tokenizer = tokenizer

        logger.info(
            "[Phase1Retriever] 加载完成: ckpt=%s, device=%s, dtype=%s, store_next_free=%d",
            self.ckpt_dir,
            self.device,
            router_dtype,
            self.store.next_free,
        )

    @torch.no_grad()
    def encode_queries(self, query_ids: Tensor, query_mask: Tensor) -> Tensor:
        """将 query token 编码为 [B, D] 检索向量。"""
        return self.encoder.encode_mean(
            query_ids.to(self.device),
            query_mask.to(self.device),
        )

    @torch.no_grad()
    def retrieve_from_tokens(self, query_ids: Tensor, query_mask: Tensor) -> Tensor:
        """输入 [B, L] query tokens，返回 [B, K_f] knowledge_ids。"""
        q_emb = self.encode_queries(query_ids, query_mask)
        q_emb = q_emb.to(dtype=self.router.adapter.proj.weight.dtype)
        out = self.router(q_emb, self.store)
        knowledge_ids = self.store.fusion_bank[out.best_id.cpu()]
        return knowledge_ids.to(self.device)

    @torch.no_grad()
    def tokenize_queries(self, texts: list[str]) -> tuple[Tensor, Tensor]:
        """将文本 query 编码为 anchor_length 长度的 token IDs。"""
        encoded = self.tokenizer(
            texts,
            max_length=self.cfg.model.anchor_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )
        query_ids = encoded["input_ids"]
        query_mask = encoded["attention_mask"].long()
        return query_ids, query_mask

    @torch.no_grad()
    def retrieve_from_texts(self, texts: list[str]) -> Tensor:
        """输入文本列表，返回 [B, K_f] knowledge_ids。"""
        query_ids, query_mask = self.tokenize_queries(texts)
        return self.retrieve_from_tokens(query_ids, query_mask)
