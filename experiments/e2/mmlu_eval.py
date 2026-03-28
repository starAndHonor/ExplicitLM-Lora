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


def load_mmlu_examples(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows: List[Dict[str, Any]] = []
    for row in ds:
        choices = row["choices"]
        if len(choices) != 4:
            continue
        rows.append(
            {
                "key": row["question"][:200].strip(),
                "question": row["question"].strip(),
                "choices": list(choices),
                "label": int(row["answer"]),
            }
        )
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def eval_mmlu(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    knowledge_map: Optional[Dict[str, List[int]]] = None,
    knowledge_length: int = 64,
    is_injection: bool = False,
    show_progress: bool = True,
) -> Dict[str, Any]:
    total = len(rows)
    correct = 0
    misses = 0
    logger.info(
        "📚 MMLU | %s | start | samples=%d | knowledge=%s",
        "fusion" if is_injection else "baseline",
        total,
        knowledge_map is not None,
    )
    with torch.no_grad():
        progress = tqdm(
            rows,
            total=total,
            desc=f"📚 MMLU / {'fusion' if is_injection else 'baseline'}",
            leave=True,
            disable=not show_progress,
        )
        for i, row in enumerate(progress, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            if is_injection:
                token_ids = knowledge_map.get(row["key"]) if knowledge_map is not None else None
                if knowledge_map is not None and token_ids is None:
                    misses += 1
                knowledge_tensor = prepare_knowledge_tensor(
                    token_ids,
                    knowledge_length,
                    tokenizer.pad_token_id,
                    device,
                )
                pred = score_choices_injection(model, tokenizer, context_ids, knowledge_tensor, device)
            else:
                pred = score_choices(model, tokenizer, context_ids, device)
            if pred == row["label"]:
                correct += 1
            progress.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}", miss=misses)
            if i % 500 == 0 or i == total:
                logger.info(
                    "📚 MMLU | %s | progress %d/%d | acc=%.4f | correct=%d | misses=%d",
                    "fusion" if is_injection else "baseline",
                    i,
                    total,
                    correct / i,
                    correct,
                    misses,
                )
    acc = correct / total if total else 0.0
    logger.info(
        "✅ MMLU | %s | done | acc=%.4f | correct=%d/%d | misses=%d",
        "fusion" if is_injection else "baseline",
        acc,
        correct,
        total,
        misses,
    )
    result = {"acc": acc, "correct": correct, "total": total}
    if is_injection:
        result["knowledge_miss_count"] = misses
    return result
