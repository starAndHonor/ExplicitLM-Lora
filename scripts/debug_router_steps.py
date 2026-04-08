#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from experiments.e2.common import load_knowledge_map  # noqa: E402
from experiments.e2.scoring import build_multiple_choice_prompt  # noqa: E402
from experiments.e3.data_loading import load_medqa_rows  # noqa: E402
from training.phase1_retriever import Phase1Retriever  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect step-by-step router retrieval behavior")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--phase1-ckpt", default=str(PROJECT_ROOT / "checkpoints/phase1_best"))
    parser.add_argument("--gold-map", default=str(PROJECT_ROOT / "data/medqa_knowledge.jsonl"))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=5, help="How many MedQA test examples to inspect")
    parser.add_argument("--topk-candidates", type=int, default=5, help="How many router candidates to print")
    return parser.parse_args()


def _trim_pad(ids: List[int], pad_token_id: int) -> List[int]:
    return [x for x in ids if x != pad_token_id]


def _decode_ids(tokenizer, ids: List[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def _format_topk(scores: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    topk = torch.topk(scores, k=min(k, scores.numel()))
    return [
        {"index": int(idx), "score": round(float(score), 4)}
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist())
    ]


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg.train.bf16 = args.device != "cpu"

    retriever = Phase1Retriever(cfg=cfg, phase1_ckpt=args.phase1_ckpt, device=args.device)
    gold_map = load_knowledge_map(args.gold_map)
    rows = load_medqa_rows(limit=args.limit)

    tokenizer = retriever.tokenizer

    for idx, row in enumerate(rows):
        prompt = build_multiple_choice_prompt(row["question"], row["choices"])
        query_ids, query_mask = retriever.tokenize_queries([prompt])
        q_emb = retriever.encode_queries(query_ids, query_mask)
        q_emb = q_emb.to(dtype=retriever.router.adapter.proj.weight.dtype)
        router_out = retriever.router(q_emb, retriever.store)

        candidates = router_out.candidates[0].detach().cpu().tolist()
        cand_mask = router_out.cand_mask[0].detach().cpu().tolist()
        best_id = int(router_out.best_id[0].item())

        coarse_row, coarse_col = router_out.coarse_scores
        row_top = _format_topk(coarse_row[0].detach().cpu(), 5)
        col_top = _format_topk(coarse_col[0].detach().cpu(), 5)

        gold_ids = gold_map.get(row["key"])
        gold_trim = _trim_pad(gold_ids, tokenizer.pad_token_id) if gold_ids is not None else []

        best_fusion_ids = retriever.store.fusion_bank.data[best_id].detach().cpu().tolist()
        best_fusion_trim = _trim_pad(best_fusion_ids, tokenizer.pad_token_id)

        candidate_infos = []
        for cand_id, valid in zip(candidates[: args.topk_candidates], cand_mask[: args.topk_candidates]):
            fusion_ids = retriever.store.fusion_bank.data[cand_id].detach().cpu().tolist()
            fusion_trim = _trim_pad(fusion_ids, tokenizer.pad_token_id)
            candidate_infos.append(
                {
                    "id": int(cand_id),
                    "valid": bool(valid),
                    "text": _decode_ids(tokenizer, fusion_trim),
                }
            )

        result = {
            "idx": idx,
            "question": row["question"],
            "choices": row["choices"],
            "gold_answer": ["A", "B", "C", "D"][row["label"]],
            "query_token_count": int(query_mask.sum().item()),
            "q_emb_norm": round(float(torch.norm(q_emb[0]).item()), 4),
            "top_row_scores": row_top,
            "top_col_scores": col_top,
            "best_id": best_id,
            "best_text": _decode_ids(tokenizer, best_fusion_trim),
            "gold_text": _decode_ids(tokenizer, gold_trim) if gold_trim else "",
            "best_matches_gold_exactly": best_fusion_trim == gold_trim if gold_trim else False,
            "top_candidates": candidate_infos,
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("=" * 120)


if __name__ == "__main__":
    main()
