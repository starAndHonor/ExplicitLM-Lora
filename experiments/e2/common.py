from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import Config
from models import (
    AttentionInjection,
    ConcatProjection,
    GatedInjection,
    KnowledgeEncoder,
    ModifiedQwen,
    load_base_model,
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def load_knowledge_map(jsonl_path: str) -> Dict[str, List[int]]:
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"knowledge map not found: {jsonl_path}")

    result: Dict[str, List[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            result[obj["key"]] = obj["knowledge_ids"]
    return result


def get_model_path(cfg: Config) -> str:
    return os.environ.get("MODEL_PATH", cfg.paths.model_dir)


def load_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_baseline_model(cfg: Config, device: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model_path = get_model_path(cfg)
    model = load_base_model(model_path, bf16=cfg.train.bf16).to(device).eval()
    tokenizer = load_tokenizer(model_path)
    return model, tokenizer


def build_injection_model(
    cfg: Config,
    fusion_ckpt: str,
    device: str,
    log_prefix: str = "E2Load",
) -> tuple[ModifiedQwen, AutoTokenizer]:
    """Build E2 fusion model from the current repo's `models/` implementation.

    Unlike the earlier Reference-compatible E2 path, this loader restores every
    checkpoint component that matches the current local model structure:
    - `injection_modules.pt`
    - `encoder_layers.pt`
    - `encoder_norm.pt`

    This makes E2 evaluate the full model represented by the checkpoint rather
    than only the injection-module subgraph.
    """
    model_path = get_model_path(cfg)
    base_model = load_base_model(model_path, bf16=cfg.train.bf16)
    tokenizer = load_tokenizer(model_path)

    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.encoder_depth,
        hidden_dim=cfg.model.hidden_dim,
        mode=cfg.model.knowledge_encoder_mode,
    )
    injection_method = cfg.model.injection_method.lower()
    if injection_method == "attention":
        factory = AttentionInjection
    elif injection_method == "concat":
        factory = ConcatProjection
    elif injection_method == "gated":
        factory = GatedInjection
    else:
        raise ValueError(f"unsupported injection_method: {cfg.model.injection_method}")

    injection_modules = nn.ModuleList(
        [factory(cfg.model.hidden_dim) for _ in cfg.model.injection_layers]
    )
    model = ModifiedQwen(
        base_model=base_model,
        knowledge_encoder=encoder,
        injection_modules=injection_modules,
        injection_layers=cfg.model.injection_layers,
        pad_token_id=tokenizer.pad_token_id,
    )

    ckpt_dir = Path(fusion_ckpt)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"fusion checkpoint not found: {fusion_ckpt}")

    state_specs = [
        (model.injection_modules, ckpt_dir / "injection_modules.pt", "injection_modules"),
    ]
    if not model.knowledge_encoder.uses_qwen3_mode:
        state_specs.extend(
            [
                (model.knowledge_encoder.layers, ckpt_dir / "encoder_layers.pt", "knowledge_encoder.layers"),
                (model.knowledge_encoder.norm, ckpt_dir / "encoder_norm.pt", "knowledge_encoder.norm"),
            ]
        )
    else:
        print(
            f"[{log_prefix}] qwen3 mode active | skip loading knowledge_encoder.layers / knowledge_encoder.norm",
            flush=True,
        )
    for module, path, label in state_specs:
        if path.exists():
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=True)
            logger.info(
                "Loaded %s from %s | keys=%d | missing=%d | unexpected=%d",
                label,
                path,
                len(state_dict),
                len(missing_keys),
                len(unexpected_keys),
            )
            print(
                f"[{log_prefix}] loaded {label} from {path} | "
                f"keys={len(state_dict)} missing={len(missing_keys)} unexpected={len(unexpected_keys)}",
                flush=True,
            )
        else:
            logger.warning("Missing %s checkpoint: %s", label, path)
            print(f"[{log_prefix}] missing {label}: {path}", flush=True)

    model = model.to(device).eval()
    return model, tokenizer


def prepare_knowledge_tensor(
    token_ids: Sequence[int] | None,
    knowledge_length: int,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    if token_ids is None:
        ids = [pad_token_id] * knowledge_length
    else:
        ids = list(token_ids[:knowledge_length])
        if len(ids) < knowledge_length:
            ids.extend([pad_token_id] * (knowledge_length - len(ids)))
    return torch.tensor([ids], dtype=torch.long, device=device)
