"""
training/dense_retriever.py — 双塔稠密检索封装

第一版：
    - 共享 KnowledgeEncoder 作为 query/doc encoder
    - 动态知识库索引来自 DenseKnowledgeIndex
    - Flat backend 默认可用；HNSW backend 预留
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from transformers import AutoTokenizer

from config import Config
from models.qwen_wrapper import KnowledgeEncoder, load_base_model
from retrieval.dense_index import DenseKnowledgeIndex, DenseSearchOutput

logger = logging.getLogger(__name__)


class DenseRetriever:
    """冻结版双塔稠密检索器。"""

    def __init__(
        self,
        cfg: Config,
        index_path: str,
        device: torch.device | str,
        tokenizer: Optional[AutoTokenizer] = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.index_path = Path(index_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f"dense index file does not exist: {self.index_path}")

        model_path = os.environ.get("MODEL_PATH", cfg.paths.model_dir)
        use_bf16 = cfg.train.bf16 and self.device.type != "cpu"
        logger.info(
            "[DenseRetriever] 初始化开始 | index=%s | device=%s | encoder_mode=%s | model_path=%s",
            self.index_path,
            self.device,
            self.cfg.model.knowledge_encoder_mode,
            model_path,
        )

        model_t0 = time.time()
        logger.info("[DenseRetriever] 开始加载基础模型（独立于 Phase3 主模型）")
        base_model = load_base_model(model_path, bf16=use_bf16)
        logger.info(
            "[DenseRetriever] 基础模型加载完成，用时 %.1fs",
            time.time() - model_t0,
        )

        encoder_t0 = time.time()
        self.encoder = KnowledgeEncoder(
            base_model=base_model,
            encoder_depth=cfg.model.retrieval_encoder_depth,
            hidden_dim=cfg.model.hidden_dim,
            mode=cfg.model.knowledge_encoder_mode,
        )
        self.encoder.requires_grad_(False)
        self.encoder = self.encoder.to(self.device).eval()
        logger.info(
            "[DenseRetriever] KnowledgeEncoder 就绪，用时 %.1fs",
            time.time() - encoder_t0,
        )

        if tokenizer is None:
            tok_t0 = time.time()
            logger.info("[DenseRetriever] 未复用外部 tokenizer，开始单独加载 tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            logger.info(
                "[DenseRetriever] tokenizer 加载完成，用时 %.1fs",
                time.time() - tok_t0,
            )
        else:
            self.tokenizer = tokenizer
            logger.info("[DenseRetriever] 复用 Phase3 已加载 tokenizer")

        index_t0 = time.time()
        logger.info("[DenseRetriever] 开始加载 dense index: %s", self.index_path)
        self.index = DenseKnowledgeIndex.load(self.index_path)
        logger.info(
            "[DenseRetriever] dense index 反序列化完成，用时 %.1fs",
            time.time() - index_t0,
        )
        logger.info(
            "[DenseRetriever] loaded index=%s | device=%s | total=%d | active=%d | index_type=%s",
            self.index_path,
            self.device,
            len(self.index),
            self.index.num_active,
            self.index.index_type,
        )

    @torch.no_grad()
    def encode_queries(self, query_ids: Tensor, query_mask: Tensor) -> Tensor:
        return self.encoder.encode_mean(
            query_ids.to(self.device),
            query_mask.to(self.device),
        )

    @torch.no_grad()
    def tokenize_queries(self, texts: list[str]) -> tuple[Tensor, Tensor]:
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
    def search_from_tokens(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        top_k: int = 1,
    ) -> DenseSearchOutput:
        q_emb = self.encode_queries(query_ids, query_mask)
        return self.index.search(q_emb, top_k=top_k)

    @torch.no_grad()
    def retrieve_from_tokens(self, query_ids: Tensor, query_mask: Tensor) -> Tensor:
        out = self.search_from_tokens(query_ids, query_mask, top_k=1)
        return out.fusion_ids[:, 0].to(self.device)

    @torch.no_grad()
    def search_from_texts(self, texts: list[str], top_k: int = 1) -> DenseSearchOutput:
        query_ids, query_mask = self.tokenize_queries(texts)
        return self.search_from_tokens(query_ids, query_mask, top_k=top_k)

    @torch.no_grad()
    def retrieve_from_texts(self, texts: list[str]) -> Tensor:
        query_ids, query_mask = self.tokenize_queries(texts)
        return self.retrieve_from_tokens(query_ids, query_mask)
