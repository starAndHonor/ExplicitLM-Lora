#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from router.memory_bank import DualKnowledgeStore  # noqa: E402
from training.phase1_retriever import Phase1Retriever  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on old knowledge entries that were not overwritten")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--old-ckpt", default=str(PROJECT_ROOT / "checkpoints/phase1_best"))
    parser.add_argument("--overlay-ckpt", default=str(PROJECT_ROOT / "checkpoints/phase1_medqa_overlay"))
    parser.add_argument("--limit", type=int, default=20, help="Number of uncovered old entries to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser.parse_args()


def _load_store(cfg, ckpt_dir: Path) -> DualKnowledgeStore:
    store = DualKnowledgeStore(
        cfg.router,
        cfg.model.fusion_length,
        cfg.model.anchor_length,
        device="cpu",
    )
    store.load_state(str(ckpt_dir / "store.pt"))
    return store


def _trim_pad(ids: List[int], pad_token_id: int) -> List[int]:
    return [x for x in ids if x != pad_token_id]


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg.train.bf16 = args.device != "cpu"

    old_ckpt = Path(args.old_ckpt)
    overlay_ckpt = Path(args.overlay_ckpt)

    model_path = cfg.paths.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    old_store = _load_store(cfg, old_ckpt)
    overlay_store = _load_store(cfg, overlay_ckpt)

    uncovered_ids = []
    for idx in range(cfg.router.knowledge_num):
        old_anchor = old_store.anchor_bank.data[idx]
        overlay_anchor = overlay_store.anchor_bank.data[idx]
        old_fusion = old_store.fusion_bank.data[idx]
        overlay_fusion = overlay_store.fusion_bank.data[idx]
        if torch.equal(old_anchor, overlay_anchor) and torch.equal(old_fusion, overlay_fusion):
            uncovered_ids.append(idx)

    rng = random.Random(args.seed)
    rng.shuffle(uncovered_ids)
    selected = uncovered_ids[: args.limit]

    old_retriever = Phase1Retriever(cfg=cfg, phase1_ckpt=str(old_ckpt), device=args.device)
    overlay_retriever = Phase1Retriever(cfg=cfg, phase1_ckpt=str(overlay_ckpt), device=args.device)

    details = []
    old_self_hit = 0
    overlay_same_text_hit = 0
    overlay_old_id_hit = 0

    for idx in selected:
        query_ids = old_store.anchor_bank.data[idx].tolist()
        query_trim = _trim_pad(query_ids, tokenizer.pad_token_id)
        query_text = tokenizer.decode(query_trim, skip_special_tokens=True)

        old_pred = old_retriever.retrieve_from_texts([query_text])[0].detach().cpu().tolist()
        overlay_pred = overlay_retriever.retrieve_from_texts([query_text])[0].detach().cpu().tolist()
        old_gold = old_store.fusion_bank.data[idx].tolist()

        old_pred_trim = _trim_pad(old_pred, tokenizer.pad_token_id)
        overlay_pred_trim = _trim_pad(overlay_pred, tokenizer.pad_token_id)
        old_gold_trim = _trim_pad(old_gold, tokenizer.pad_token_id)

        old_exact = old_pred_trim == old_gold_trim
        if old_exact:
            old_self_hit += 1

        overlay_exact_same_text = overlay_pred_trim == old_gold_trim
        if overlay_exact_same_text:
            overlay_same_text_hit += 1

        overlay_output = overlay_retriever.retrieve_from_texts([query_text])
        overlay_q_ids, overlay_q_mask = overlay_retriever.tokenize_queries([query_text])
        overlay_q_emb = overlay_retriever.encode_queries(overlay_q_ids, overlay_q_mask)
        overlay_router_out = overlay_retriever.router(overlay_q_emb.to(dtype=overlay_retriever.router.adapter.proj.weight.dtype), overlay_retriever.store)
        overlay_best_id = int(overlay_router_out.best_id[0].item())
        if overlay_best_id == idx:
            overlay_old_id_hit += 1

        details.append(
            {
                "id": idx,
                "old_exact_match": old_exact,
                "overlay_exact_match_old_fusion": overlay_exact_same_text,
                "overlay_best_id_same_slot": overlay_best_id == idx,
                "query_text": query_text,
                "gold_text": tokenizer.decode(old_gold_trim, skip_special_tokens=True),
                "old_pred_text": tokenizer.decode(old_pred_trim, skip_special_tokens=True),
                "overlay_pred_text": tokenizer.decode(overlay_pred_trim, skip_special_tokens=True),
                "overlay_best_id": overlay_best_id,
            }
        )

    tested = len(selected)
    result = {
        "old_ckpt": str(old_ckpt),
        "overlay_ckpt": str(overlay_ckpt),
        "uncovered_total": len(uncovered_ids),
        "tested": tested,
        "old_self_hit_rate": old_self_hit / tested if tested else 0.0,
        "overlay_exact_match_old_fusion_rate": overlay_same_text_hit / tested if tested else 0.0,
        "overlay_same_slot_hit_rate": overlay_old_id_hit / tested if tested else 0.0,
        "examples": details,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
