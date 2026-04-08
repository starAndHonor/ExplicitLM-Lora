#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from experiments.e2.common import load_knowledge_map  # noqa: E402
from experiments.e2.scoring import build_multiple_choice_prompt  # noqa: E402
from experiments.e3.data_loading import load_medqa_rows  # noqa: E402
from training.phase1_retriever import Phase1Retriever  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase1 retrieval against a gold knowledge map")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument(
        "--phase1-ckpt",
        default=str(PROJECT_ROOT / "checkpoints/phase1_medqa_padded"),
        help="Phase1 checkpoint directory to evaluate",
    )
    parser.add_argument(
        "--gold-map",
        default=str(PROJECT_ROOT / "data/medqa_knowledge.jsonl"),
        help="Gold key -> knowledge_ids map",
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path",
    )
    return parser.parse_args()


def _trim_pad(ids: List[int], pad_token_id: int) -> List[int]:
    return [x for x in ids if x != pad_token_id]


def _prefix_match_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    count = 0
    for i in range(n):
        if a[i] != b[i]:
            break
        count += 1
    return count


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg.train.bf16 = args.device != "cpu"
    device = torch.device(args.device)

    retriever = Phase1Retriever(cfg=cfg, phase1_ckpt=args.phase1_ckpt, device=device)
    gold_map: Dict[str, List[int]] = load_knowledge_map(args.gold_map)
    rows = load_medqa_rows(limit=args.limit)

    tested = 0
    exact_match = 0
    prefix_full_match = 0
    token_overlap_sum = 0.0
    prefix_ratio_sum = 0.0
    details = []

    for idx, row in enumerate(rows):
        gold = gold_map.get(row["key"])
        if gold is None:
            continue

        prompt = build_multiple_choice_prompt(row["question"], row["choices"])
        pred_tensor = retriever.retrieve_from_texts([prompt])[0].detach().cpu().tolist()

        pred = _trim_pad(pred_tensor, retriever.tokenizer.pad_token_id)
        gold_trim = _trim_pad(gold, retriever.tokenizer.pad_token_id)

        tested += 1
        is_exact = pred == gold_trim
        if is_exact:
            exact_match += 1

        prefix_len = _prefix_match_len(pred, gold_trim)
        max_gold_len = max(len(gold_trim), 1)
        prefix_ratio = prefix_len / max_gold_len
        prefix_ratio_sum += prefix_ratio
        if prefix_len == len(gold_trim):
            prefix_full_match += 1

        pred_set = set(pred)
        gold_set = set(gold_trim)
        overlap = len(pred_set & gold_set) / max(len(gold_set), 1)
        token_overlap_sum += overlap

        if idx < 10:
            details.append(
                {
                    "idx": idx,
                    "key": row["key"],
                    "gold_len": len(gold_trim),
                    "pred_len": len(pred),
                    "exact_match": is_exact,
                    "prefix_ratio": round(prefix_ratio, 4),
                    "token_overlap": round(overlap, 4),
                    "gold_text": retriever.tokenizer.decode(gold_trim, skip_special_tokens=True),
                    "pred_text": retriever.tokenizer.decode(pred, skip_special_tokens=True),
                }
            )

    if tested == 0:
        raise RuntimeError("No matched samples between dataset keys and gold map")

    summary = {
        "phase1_ckpt": args.phase1_ckpt,
        "gold_map": args.gold_map,
        "tested": tested,
        "exact_match_rate": exact_match / tested,
        "full_prefix_match_rate": prefix_full_match / tested,
        "avg_prefix_ratio": prefix_ratio_sum / tested,
        "avg_token_overlap": token_overlap_sum / tested,
        "examples": details,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
