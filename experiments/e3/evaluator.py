from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from experiments.e2.common import prepare_knowledge_tensor
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices, score_choices_injection

logger = logging.getLogger(__name__)


def _iter_rows(
    rows: List[Dict[str, Any]],
    desc: str,
    show_progress: bool,
):
    return tqdm(rows, total=len(rows), desc=desc, leave=True, disable=not show_progress)


def eval_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    total = len(rows)
    correct = 0
    logger.info("📌 %s | G0 Baseline | start | samples=%d", dataset_name.upper(), total)
    with torch.no_grad():
        progress = _iter_rows(rows, f"📌 {dataset_name.upper()} / G0", show_progress)
        for i, row in enumerate(progress, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pred = score_choices(model, tokenizer, context_ids, device)
            if pred == row["label"]:
                correct += 1
            progress.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}")
    acc = correct / total if total else 0.0
    logger.info("✅ %s | G0 Baseline | done | acc=%.4f | correct=%d/%d", dataset_name.upper(), acc, correct, total)
    return {"acc": acc, "correct": correct, "total": total}


def eval_rag_compressed(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    knowledge_map: Dict[str, List[int]],
    show_progress: bool = True,
) -> Dict[str, Any]:
    total = len(rows)
    correct = 0
    missing_knowledge = 0
    pad_id = tokenizer.pad_token_id
    logger.info("📌 %s | G1 RAG-compressed | start | samples=%d", dataset_name.upper(), total)
    with torch.no_grad():
        progress = _iter_rows(rows, f"📌 {dataset_name.upper()} / G1", show_progress)
        for i, row in enumerate(progress, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            k_ids = knowledge_map.get(row["key"])
            if k_ids is None:
                context = prompt
                missing_knowledge += 1
            else:
                clean_ids = [t for t in k_ids if t != pad_id]
                compressed_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
                context = f"Context: {compressed_text}\n\n{prompt}"
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            pred = score_choices(model, tokenizer, context_ids, device)
            if pred == row["label"]:
                correct += 1
            progress.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}", miss=missing_knowledge)
    acc = correct / total if total else 0.0
    logger.info(
        "✅ %s | G1 RAG-compressed | done | acc=%.4f | correct=%d/%d | missing=%d",
        dataset_name.upper(),
        acc,
        correct,
        total,
        missing_knowledge,
    )
    return {"acc": acc, "correct": correct, "total": total, "missing_knowledge": missing_knowledge}


def eval_fusion(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    knowledge_map: Dict[str, List[int]],
    group_name: str,
    knowledge_length: int,
    show_progress: bool = True,
) -> Dict[str, Any]:
    total = len(rows)
    correct = 0
    misses = 0
    logger.info("📌 %s | %s Fusion | start | samples=%d", dataset_name.upper(), group_name, total)
    with torch.no_grad():
        progress = _iter_rows(rows, f"📌 {dataset_name.upper()} / {group_name}", show_progress)
        for i, row in enumerate(progress, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            token_ids = knowledge_map.get(row["key"])
            if token_ids is None:
                misses += 1
            knowledge_tensor = prepare_knowledge_tensor(
                token_ids,
                knowledge_length,
                tokenizer.pad_token_id,
                device,
            )
            pred = score_choices_injection(model, tokenizer, context_ids, knowledge_tensor, device)
            if pred == row["label"]:
                correct += 1
            progress.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}", miss=misses)
    acc = correct / total if total else 0.0
    logger.info(
        "✅ %s | %s Fusion | done | acc=%.4f | correct=%d/%d | misses=%d",
        dataset_name.upper(),
        group_name,
        acc,
        correct,
        total,
        misses,
    )
    return {"acc": acc, "correct": correct, "total": total, "knowledge_miss_count": misses}


def eval_rag_original(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    total = len(rows)
    correct = 0
    logger.info("📌 %s | G4 RAG-original | start | samples=%d", dataset_name.upper(), total)
    with torch.no_grad():
        progress = _iter_rows(rows, f"📌 {dataset_name.upper()} / G4", show_progress)
        for i, row in enumerate(progress, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context = f"Context: {row['original_text']}\n\n{prompt}"
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            pred = score_choices(model, tokenizer, context_ids, device)
            if pred == row["label"]:
                correct += 1
            progress.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}")
    acc = correct / total if total else 0.0
    logger.info("✅ %s | G4 RAG-original | done | acc=%.4f | correct=%d/%d", dataset_name.upper(), acc, correct, total)
    return {"acc": acc, "correct": correct, "total": total}
