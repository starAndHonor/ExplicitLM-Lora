from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from logger_system import log_msg
from models import (
    AttentionInjection,
    ConcatProjection,
    GatedInjection,
    KnowledgeEncoder,
    ModifiedQwen,
    load_base_model,
)


def _resolve_model_path(model_path: str) -> str:
    return os.environ.get("MODEL_PATH", model_path)


def _load_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


class ReferenceCompatModel(nn.Module):
    """Compatibility wrapper exposing the old Reference E3 eval interface."""

    def __init__(
        self,
        model_path: str,
        injection_method: str,
        injection_layers: Optional[List[int]],
        device: str,
        encoder_depth: int = 6,
        knowledge_adapter: bool = False,
    ) -> None:
        super().__init__()
        if knowledge_adapter:
            log_msg("WARNING", "knowledge_adapter=True is ignored in e3-ref compat loader")

        model_path = _resolve_model_path(model_path)
        base_model = load_base_model(model_path, bf16=True)
        tokenizer = _load_tokenizer(model_path)

        encoder = KnowledgeEncoder(
            base_model=base_model,
            encoder_depth=encoder_depth,
            hidden_dim=base_model.config.hidden_size,
        )

        if injection_layers is None:
            injection_layers = [6, 12, 18, 24]

        method = injection_method.lower()
        if method == "attention":
            factory = AttentionInjection
        elif method == "concat":
            factory = ConcatProjection
        elif method == "gated":
            factory = GatedInjection
        else:
            raise ValueError(f"unknown injection_method: {injection_method}")

        injection_modules = nn.ModuleList(
            [factory(base_model.config.hidden_size) for _ in injection_layers]
        )
        self.model = ModifiedQwen(
            base_model=base_model,
            knowledge_encoder=encoder,
            injection_modules=injection_modules,
            injection_layers=injection_layers,
            pad_token_id=tokenizer.pad_token_id,
        )
        self.pad_token_id = tokenizer.pad_token_id
        self._device = device
        self.model = self.model.to(device)

    def load_injection_weights(self, load_dir: str) -> None:
        load_path = Path(load_dir)
        weights_path = load_path / "injection_modules.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"missing injection checkpoint: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.injection_modules.load_state_dict(state_dict)
        if (load_path / "encoder_layers.pt").exists() or (load_path / "encoder_norm.pt").exists():
            log_msg(
                "INFO",
                f"e3-ref compat loader only restores injection_modules.pt from {load_path}",
            )

    def to(self, device: str):
        self._device = device
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def forward(
        self,
        input_ids: torch.LongTensor,
        knowledge_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.model(
            input_ids=input_ids,
            knowledge_ids=knowledge_ids,
            attention_mask=attention_mask,
        )
        return outputs.logits


def create_model(
    model_path: str = "Qwen3-0.6B",
    injection_method: str = "attention",
    injection_layers: Optional[List[int]] = None,
    device: str = "cuda",
    encoder_depth: int = 6,
    knowledge_adapter: bool = False,
    **_: object,
) -> ReferenceCompatModel:
    return ReferenceCompatModel(
        model_path=model_path,
        injection_method=injection_method,
        injection_layers=injection_layers,
        device=device,
        encoder_depth=encoder_depth,
        knowledge_adapter=knowledge_adapter,
    )
