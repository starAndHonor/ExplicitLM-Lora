#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e2.common import load_knowledge_map
from experiments.e2.scoring import build_multiple_choice_prompt
from experiments.e3.data_loading import load_arc_rows, load_medqa_rows, load_mmlu_rows
from training.dense_retriever import DenseRetriever


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval against a gold knowledge map")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument(
        "--dataset",
        choices=["medqa", "arc", "mmlu"],
        default="medqa",
        help="Benchmark dataset used for retrieval evaluation",
    )
    parser.add_argument("--index", required=True, help="Dense index path")
    parser.add_argument("--gold-map", default="", help="Gold knowledge map jsonl; defaults follow --dataset")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--query-mode",
        choices=["question_only", "question_choices", "both"],
        default="both",
        help="Query formulation used for retrieval evaluation",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _default_gold_map(dataset: str) -> str:
    mapping = {
        "medqa": str(PROJECT_ROOT / "data/medqa_knowledge.jsonl"),
        "arc": str(PROJECT_ROOT / "data/arc_knowledge.jsonl"),
        "mmlu": str(PROJECT_ROOT / "data/mmlu_knowledge.jsonl"),
    }
    return mapping[dataset]


def _load_rows(dataset: str, limit: int) -> List[Dict[str, object]]:
    if dataset == "medqa":
        return load_medqa_rows(limit=limit)
    if dataset == "arc":
        return load_arc_rows(limit=limit)
    if dataset == "mmlu":
        return load_mmlu_rows(limit=limit)
    raise ValueError(f"unsupported dataset: {dataset}")


def _trim_pad(ids: List[int], pad_token_id: int) -> List[int]:
    return [x for x in ids if x != pad_token_id]


def _build_query(row: Dict[str, object], mode: str) -> str:
    question = str(row["question"])
    if mode == "question_only":
        return question
    if mode == "question_choices":
        return build_multiple_choice_prompt(question, list(row["choices"]))
    raise ValueError(f"unsupported query mode: {mode}")


def _evaluate_mode(
    retriever: DenseRetriever,
    rows: List[Dict[str, object]],
    gold_map: Dict[str, List[int]],
    top_k: int,
    query_mode: str,
) -> Dict[str, object]:
    tested = 0
    top1_exact = 0
    topk_hit = 0
    details = []

    for idx, row in enumerate(rows):
        gold = gold_map.get(str(row["key"]))
        if gold is None:
            continue

        query = _build_query(row, query_mode)
        search = retriever.search_from_texts([query], top_k=top_k)
        candidates = search.fusion_ids[0].detach().cpu().tolist()
        pred_top1 = candidates[0] if candidates else []

        top1_exact += int(pred_top1 == gold)
        hit = any(candidate == gold for candidate in candidates)
        topk_hit += int(hit)
        tested += 1

        if idx < 10:
            details.append(
                {
                    "idx": idx,
                    "key": row["key"],
                    "query_mode": query_mode,
                    "top1_exact": pred_top1 == gold,
                    "topk_hit": hit,
                    "query_preview": query[:200],
                    "gold_text": retriever.tokenizer.decode(
                        _trim_pad(gold, retriever.tokenizer.pad_token_id),
                        skip_special_tokens=True,
                    ),
                    "pred_text": retriever.tokenizer.decode(
                        _trim_pad(pred_top1, retriever.tokenizer.pad_token_id),
                        skip_special_tokens=True,
                    ),
                }
            )

    if tested == 0:
        raise RuntimeError(f"No matched samples between dataset keys and gold map for mode={query_mode}")

    return {
        "tested": tested,
        "top1_exact_rate": top1_exact / tested,
        "topk_hit_rate": topk_hit / tested,
        "top_k": top_k,
        "examples": details,
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    cfg.train.bf16 = args.device != "cpu"
    retriever = DenseRetriever(cfg=cfg, index_path=args.index, device=args.device)
    gold_map_path = args.gold_map or _default_gold_map(args.dataset)
    gold_map: Dict[str, List[int]] = load_knowledge_map(gold_map_path)
    rows = _load_rows(args.dataset, limit=args.limit)

    modes = ["question_only", "question_choices"] if args.query_mode == "both" else [args.query_mode]
    results = {
        mode: _evaluate_mode(
            retriever=retriever,
            rows=rows,
            gold_map=gold_map,
            top_k=args.top_k,
            query_mode=mode,
        )
        for mode in modes
    }

    summary = {
        "dataset": args.dataset,
        "index": args.index,
        "gold_map": gold_map_path,
        "limit": args.limit,
        "query_mode": args.query_mode,
        "results": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
