from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from experiments.e2.common import prepare_knowledge_tensor
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices, score_choices_injection

logger = logging.getLogger(__name__)


def load_medqa_examples(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    rows: List[Dict[str, Any]] = []
    for row in ds:
        rows.append(
            {
                "key": row["sent1"][:200].strip(),
                "question": row["sent1"].strip(),
                "choices": [row["ending0"], row["ending1"], row["ending2"], row["ending3"]],
                "label": int(row["label"]),
            }
        )
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def eval_medqa_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    show_progress: bool = True,
) -> Dict[str, Any]:
    total = len(rows)
    correct = 0
    logger.info("🩺 MedQA | baseline | start | samples=%d", total)
    with torch.no_grad():
        progress = tqdm(
            rows,
            total=total,
            desc="🩺 MedQA / baseline",
            leave=True,
            disable=not show_progress,
        )
        for i, row in enumerate(progress, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pred = score_choices(model, tokenizer, context_ids, device)
            if pred == row["label"]:
                correct += 1
            progress.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}")
            if i % 200 == 0 or i == total:
                logger.info(
                    "🩺 MedQA | baseline | progress %d/%d | acc=%.4f | correct=%d",
                    i,
                    total,
                    correct / i,
                    correct,
                )
    acc = correct / total if total else 0.0
    logger.info("✅ MedQA | baseline | done | acc=%.4f | correct=%d/%d", acc, correct, total)
    return {"acc": acc, "correct": correct, "total": total}


def eval_medqa_injection(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    knowledge_map: Optional[Dict[str, List[int]]],
    knowledge_length: int,
    show_progress: bool = True,
) -> Dict[str, Any]:
    total = len(rows)
    correct = 0
    misses = 0
    use_knowledge = knowledge_map is not None
    mode = "Fusion+knowledge" if use_knowledge else "Fusion+empty"
    logger.info("🩺 MedQA | %s | start | samples=%d", mode, total)
    with torch.no_grad():
        progress = tqdm(
            rows,
            total=total,
            desc=f"🩺 MedQA / {mode}",
            leave=True,
            disable=not show_progress,
        )
        for i, row in enumerate(progress, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            token_ids = knowledge_map.get(row["key"]) if knowledge_map is not None else None
            if use_knowledge and token_ids is None:
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
            if i % 200 == 0 or i == total:
                logger.info(
                    "🩺 MedQA | %s | progress %d/%d | acc=%.4f | correct=%d | misses=%d",
                    mode,
                    i,
                    total,
                    correct / i,
                    correct,
                    misses,
                )
    acc = correct / total if total else 0.0
    logger.info(
        "✅ MedQA | %s | done | acc=%.4f | correct=%d/%d | misses=%d",
        mode,
        acc,
        correct,
        total,
        misses,
    )
    return {
        "acc": acc,
        "correct": correct,
        "total": total,
        "knowledge_miss_count": misses,
    }
