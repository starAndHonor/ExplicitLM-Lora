#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e2.scoring import build_multiple_choice_prompt
from experiments.e3.data_loading import load_arc_rows, load_medqa_rows, load_mmlu_rows
from training.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


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


def _load_indexed_keys(dataset: str, fusion_length: int) -> Set[str]:
    """从 k-specific knowledge.jsonl 加载已注入索引的 key 集合。"""
    if fusion_length == 64:
        path = PROJECT_ROOT / "data" / f"{dataset}_knowledge.jsonl"
    else:
        path = PROJECT_ROOT / "data" / f"{dataset}_knowledge_k{fusion_length}.jsonl"
    keys: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = obj.get("key")
            if key is not None:
                keys.add(str(key))
    return keys


def _index_path(args: argparse.Namespace, dataset: str) -> str:
    return {
        "medqa": args.dense_index_medqa,
        "arc": args.dense_index_arc,
        "mmlu": args.dense_index_mmlu,
    }[dataset]


def _evaluate_mode(
    retriever: DenseRetriever,
    rows: List[Dict[str, Any]],
    indexed_keys: Set[str],
    top_k: int,
    query_mode: str,
) -> Dict[str, Any]:
    """按 key 匹配评测检索正确率（单视图模式）。"""
    tested = 0
    top1_hit = 0
    topk_hit = 0
    examples = []

    for idx, row in enumerate(rows):
        gold_key = str(row["key"])
        # 只评测确实在索引中的文档
        if gold_key not in indexed_keys:
            continue

        query = _build_query(row, query_mode)
        search = retriever.search_from_texts([query], top_k=top_k)
        # indices shape: [1, top_k]；valid_mask shape: [1, top_k]
        raw_indices = search.indices[0].tolist()
        valid_mask = search.valid_mask[0].tolist()
        retrieved_keys: List[str] = [
            retriever.index.keys[int(i)]
            for i, v in zip(raw_indices, valid_mask)
            if v and int(i) >= 0
        ]

        hit_top1 = bool(retrieved_keys) and retrieved_keys[0] == gold_key
        hit_topk = gold_key in retrieved_keys
        top1_hit += int(hit_top1)
        topk_hit += int(hit_topk)
        tested += 1

        if idx < 5:
            examples.append(
                {
                    "idx": idx,
                    "gold_key": gold_key[:120],
                    "query_preview": query[:160],
                    "top1_hit": hit_top1,
                    "topk_hit": hit_topk,
                    "pred_top1_key": retrieved_keys[0][:120] if retrieved_keys else "",
                }
            )

    if tested == 0:
        raise RuntimeError(f"no indexed samples matched for query_mode={query_mode}")

    logger.info(
        "[%s] tested=%d top1_hit_rate=%.4f topk(%d)_hit_rate=%.4f",
        query_mode,
        tested,
        top1_hit / tested,
        top_k,
        topk_hit / tested,
    )

    return {
        "tested": tested,
        "top1_hit_rate": top1_hit / tested,
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
    indexed_keys = _load_indexed_keys(dataset, cfg.model.fusion_length)
    logger.info("[%s] rows=%d indexed_keys=%d", dataset, len(rows), len(indexed_keys))
    retriever = DenseRetriever(cfg=cfg, index_path=index_path, device=device)
    modes = ["question_only", "question_choices"] if query_mode == "both" else [query_mode]
    return {
        "dataset": dataset,
        "index": index_path,
        "indexed_keys": len(indexed_keys),
        "results": {
            mode: _evaluate_mode(retriever, rows, indexed_keys, top_k=top_k, query_mode=mode)
            for mode in modes
        },
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
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
