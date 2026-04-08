#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices_injection  # noqa: E402
from experiments.e3.data_loading import load_medqa_rows  # noqa: E402
from training.phase1_retriever import Phase1Retriever  # noqa: E402
from training.phase3_sft import _build_modified_qwen_phase3  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Scheme2 on MedQA samples")
    parser.add_argument("--limit", type=int, default=10, help="Number of MedQA test examples to run")
    parser.add_argument(
        "--phase1-ckpt",
        default=str(PROJECT_ROOT / "checkpoints/phase1_medqa_padded"),
        help="Phase1 checkpoint directory",
    )
    parser.add_argument(
        "--phase3-ckpt",
        default=str(PROJECT_ROOT / "checkpoints/phase3_best"),
        help="Phase3 checkpoint directory",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(str(PROJECT_ROOT / "config/default.yaml"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.train.bf16 = device.type == "cuda"

    phase1_ckpt = Path(args.phase1_ckpt)
    phase3_ckpt = Path(args.phase3_ckpt)

    rows = load_medqa_rows(limit=args.limit)
    label_map = ["A", "B", "C", "D"]

    retriever = Phase1Retriever(cfg=cfg, phase1_ckpt=str(phase1_ckpt), device=device)
    model, tokenizer = _build_modified_qwen_phase3(cfg, str(phase3_ckpt))
    model = model.to(device).eval()

    results = []
    with torch.no_grad():
        for idx, row in enumerate(rows):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            knowledge_ids = retriever.retrieve_from_texts([prompt])
            pred = score_choices_injection(
                model,
                tokenizer,
                tokenizer.encode(prompt, add_special_tokens=False),
                knowledge_ids.to(device),
                device,
            )
            knowledge_text = retriever.tokenizer.decode(
                knowledge_ids[0][knowledge_ids[0] != retriever.tokenizer.pad_token_id].tolist(),
                skip_special_tokens=True,
            )
            item = {
                "idx": idx,
                "question": row["question"],
                "choices": row["choices"],
                "gold": label_map[row["label"]],
                "pred": label_map[pred],
                "correct": pred == row["label"],
                "retrieved_knowledge": knowledge_text,
            }
            results.append(item)
            print(json.dumps(item, ensure_ascii=False, indent=2))
            print("=" * 100)

    acc = sum(int(x["correct"]) for x in results) / len(results)
    print(json.dumps({"n": len(results), "acc": acc}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
