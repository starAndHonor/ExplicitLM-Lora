from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoTokenizer

from config import Config
from experiments.e2.common import get_model_path
from models import KnowledgeEncoder, load_base_model
from router.memory_bank import DualKnowledgeStore
from router.model import MemoryRouter

logger = logging.getLogger(__name__)


class Phase1Retriever:
    """Load a frozen Phase1 router and expose online retrieval for prompt texts."""

    def __init__(
        self,
        cfg: Config,
        checkpoint_dir: str,
        device: torch.device | str,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"phase1 checkpoint not found: {self.checkpoint_dir}")

        model_path = get_model_path(cfg)
        use_bf16 = bool(cfg.train.bf16 and self.device.type == "cuda")
        base_model = load_base_model(model_path, bf16=use_bf16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.encoder = KnowledgeEncoder(
            base_model=base_model,
            encoder_depth=cfg.model.retrieval_encoder_depth,
            hidden_dim=cfg.model.hidden_dim,
            mode=cfg.model.knowledge_encoder_mode,
        )
        self.encoder.requires_grad_(False)
        self.encoder = self.encoder.to(self.device).eval()

        self.store = DualKnowledgeStore(
            cfg.router,
            cfg.model.fusion_length,
            cfg.model.anchor_length,
            device="cpu",
        )
        self.store.load_state(str(self.checkpoint_dir / "store.pt"))

        self.router = MemoryRouter(cfg.router, self.encoder)
        state = torch.load(self.checkpoint_dir / "router.pt", map_location="cpu", weights_only=True)
        self.router.load_state_dict(state, strict=True)
        self.router = self.router.to(self.device).eval()

        if self.device.type == "cpu":
            dtype = next(self.encoder.parameters()).dtype
            self.router = self.router.to(dtype=dtype)

        logger.info(
            "Phase1Retriever ready | ckpt=%s | device=%s | next_free=%d",
            self.checkpoint_dir,
            self.device,
            self.store.next_free,
        )

    def tokenize_texts(self, texts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.cfg.model.anchor_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)
        return input_ids, attention_mask

    @torch.no_grad()
    def retrieve_from_tokens(
        self,
        query_ids: torch.Tensor,
        query_mask: torch.Tensor,
    ) -> torch.Tensor:
        q_emb = self.encoder.encode_mean(query_ids, query_mask)
        return self.router.retrieve(q_emb, self.store)

    @torch.no_grad()
    def retrieve_from_texts(self, texts: Sequence[str]) -> torch.Tensor:
        query_ids, query_mask = self.tokenize_texts(texts)
        return self.retrieve_from_tokens(query_ids, query_mask)
