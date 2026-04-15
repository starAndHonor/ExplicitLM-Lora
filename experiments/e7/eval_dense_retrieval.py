#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e2.common import load_knowledge_map
from experiments.e2.scoring import build_multiple_choice_prompt
from experiments.e3.data_loading import load_arc_rows, load_medqa_rows, load_mmlu_rows
from training.dense_retriever import DenseRetriever


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dense retrieval correctness across MedQA / ARC / MMLU")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--override", nargs="?", action="append", help="config overrides")
    parser.add_argument("--dense-index-medqa", required=True)
    parser.add_argument("--dense-index-arc", required=True)
    parser.add_argument("--dense-index-mmlu", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--query-mode",
        choices=["question_only", "question_choices", "both"],
        default="question_only",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    flat: list[str] = []
    for item in overrides:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    result: dict[str, Any] = {}
    for item in flat:
        if "=" not in item:
            raise ValueError(f"invalid override (missing '='): {item}")
        key, value = item.split("=", 1)
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


def _trim_pad(ids: List[int], pad_token_id: int) -> List[int]:
    return [x for x in ids if x != pad_token_id]


def _build_query(row: Dict[str, Any], mode: str) -> str:
    if mode == "question_only":
        return str(row["question"])
    if mode == "question_choices":
        return build_multiple_choice_prompt(str(row["question"]), list(row["choices"]))
    raise ValueError(f"unsupported query mode: {mode}")


def _load_rows(dataset: str, limit: int) -> List[Dict[str, Any]]:
    if dataset == "medqa":
        return load_medqa_rows(limit=None if limit < 0 else limit)
    if dataset == "arc":
        return load_arc_rows(limit=None if limit < 0 else limit)
    if dataset == "mmlu":
        return load_mmlu_rows(limit=None if limit < 0 else limit)
    raise ValueError(f"unsupported dataset: {dataset}")


def _gold_map_path(dataset: str) -> str:
    return str(PROJECT_ROOT / "data" / f"{dataset}_knowledge.jsonl")


def _index_path(args: argparse.Namespace, dataset: str) -> str:
    return {
        "medqa": args.dense_index_medqa,
        "arc": args.dense_index_arc,
        "mmlu": args.dense_index_mmlu,
    }[dataset]


def _evaluate_mode(
    retriever: DenseRetriever,
    rows: List[Dict[str, Any]],
    gold_map: Dict[str, List[int]],
    top_k: int,
    query_mode: str,
) -> Dict[str, Any]:
    tested = 0
    top1_exact = 0
    topk_hit = 0
    examples = []

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

        if idx < 5:
            examples.append(
                {
                    "idx": idx,
                    "key": row["key"],
                    "query_preview": query[:160],
                    "top1_exact": pred_top1 == gold,
                    "topk_hit": hit,
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
        raise RuntimeError(f"no matched samples for query_mode={query_mode}")

    return {
        "tested": tested,
        "top1_exact_rate": top1_exact / tested,
        "topk_hit_rate": topk_hit / tested,
        "top_k": top_k,
        "examples": examples,
    }


def _evaluate_dataset(
    cfg,
    dataset: str,
    index_path: str,
    limit: int,
    top_k: int,
    query_mode: str,
    device: str,
) -> Dict[str, Any]:
    rows = _load_rows(dataset, limit)
    gold_map = load_knowledge_map(_gold_map_path(dataset))
    retriever = DenseRetriever(cfg=cfg, index_path=index_path, device=device)
    modes = ["question_only", "question_choices"] if query_mode == "both" else [query_mode]
    return {
        "dataset": dataset,
        "index": index_path,
        "gold_map": _gold_map_path(dataset),
        "results": {
            mode: _evaluate_mode(retriever, rows, gold_map, top_k=top_k, query_mode=mode)
            for mode in modes
        },
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    cfg.train.bf16 = args.device != "cpu"

    summary = {
        ds: _evaluate_dataset(
            cfg=cfg,
            dataset=ds,
            index_path=_index_path(args, ds),
            limit=args.limit,
            top_k=args.top_k,
            query_mode=args.query_mode,
            device=args.device,
        )
        for ds in ("medqa", "arc", "mmlu")
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
