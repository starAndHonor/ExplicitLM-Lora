#!/usr/bin/env python
"""
方案二推理：Retriever -> P3-FusionInference

支持两种检索来源：
    - phase1_router: 旧版 Phase1 Router
    - dense_retriever: 新版 Dense 检索

输入一个问题与选项，先检索知识，再用 Phase 3 融合推理输出答案。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from training.dense_retriever import DenseRetriever  # noqa: E402
from training.phase1_retriever import Phase1Retriever  # noqa: E402
from training.phase3_sft import _build_modified_qwen_phase3  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retriever -> Phase3 frozen inference")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--phase1-ckpt", default=str(PROJECT_ROOT / "checkpoints/phase1_best"))
    parser.add_argument("--dense-index", default="", help="Dense index path for knowledge_source=dense_retriever")
    parser.add_argument("--phase3-ckpt", default=str(PROJECT_ROOT / "checkpoints/phase3_best"))
    parser.add_argument(
        "--knowledge-source",
        choices=["phase1_router", "dense_retriever"],
        default="phase1_router",
        help="Retriever backend used before Phase3 fusion inference",
    )
    parser.add_argument(
        "--query-mode",
        choices=["question_only", "question_choices"],
        default="question_choices",
        help="Retrieval query formulation; model input prompt remains full multiple-choice prompt",
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--option-a", required=True)
    parser.add_argument("--option-b", required=True)
    parser.add_argument("--option-c", required=True)
    parser.add_argument("--option-d", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    return parser.parse_args()


def _build_retrieval_query(args: argparse.Namespace) -> str:
    if args.query_mode == "question_only":
        return args.question
    return (
        f"Question: {args.question}\n"
        f"A. {args.option_a}\n"
        f"B. {args.option_b}\n"
        f"C. {args.option_c}\n"
        f"D. {args.option_d}\n"
        f"Answer:"
    )


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg.train.bf16 = args.device != "cpu"

    prompt = (
        f"Question: {args.question}\n"
        f"A. {args.option_a}\n"
        f"B. {args.option_b}\n"
        f"C. {args.option_c}\n"
        f"D. {args.option_d}\n"
        f"Answer:"
    )
    retrieval_query = _build_retrieval_query(args)

    if args.knowledge_source == "phase1_router":
        retriever = Phase1Retriever(
            cfg=cfg,
            phase1_ckpt=args.phase1_ckpt,
            device=args.device,
        )
    else:
        if not args.dense_index:
            raise ValueError("--dense-index is required when --knowledge-source dense_retriever")
        retriever = DenseRetriever(
            cfg=cfg,
            index_path=args.dense_index,
            device=args.device,
        )

    knowledge_ids = retriever.retrieve_from_texts([retrieval_query])
    knowledge_text = retriever.tokenizer.decode(
        knowledge_ids[0][knowledge_ids[0] != retriever.tokenizer.pad_token_id].tolist(),
        skip_special_tokens=True,
    )

    modified_qwen, tokenizer = _build_modified_qwen_phase3(cfg, args.phase3_ckpt)
    modified_qwen = modified_qwen.to(args.device)
    modified_qwen.eval()

    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded["input_ids"].to(args.device)
    attention_mask = encoded["attention_mask"].to(args.device)

    with torch.no_grad():
        out = modified_qwen(
            input_ids=input_ids,
            knowledge_ids=knowledge_ids.to(args.device),
            attention_mask=attention_mask,
            labels=None,
        )
    last_idx = int(attention_mask.sum(dim=1).item() - 1)
    next_logits = out.logits[0, last_idx]
    topk = torch.topk(next_logits, k=5)
    top5 = []
    for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
        top5.append(
            {
                "token_id": idx,
                "token": tokenizer.decode([idx]).strip(),
                "score": round(float(score), 4),
            }
        )

    result = {
        "prompt": prompt,
        "retrieval_query": retrieval_query,
        "knowledge_source": args.knowledge_source,
        "query_mode": args.query_mode,
        "knowledge_text": knowledge_text,
        "pred_token": tokenizer.decode([int(topk.indices[0].item())]).strip(),
        "top5": top5,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("[Scheme2] Prompt:")
        print(prompt)
        print("\n[Scheme2] Retrieved Knowledge:")
        print(knowledge_text)
        print("\n[Scheme2] Predicted Next Token:")
        print(result["pred_token"])
        print("\n[Scheme2] Top-5:")
        for item in top5:
            print(item)


if __name__ == "__main__":
    main()
