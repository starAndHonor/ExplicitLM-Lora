from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import load_dataset, load_from_disk

from experiments.e2.arc_eval import ANSWER_KEY_TO_INDEX


def load_medqa_rows(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_from_disk("data/medqa/hf_dataset")["test"]
    rows: List[Dict[str, Any]] = []
    for row in ds:
        options = row["options"]
        choices = [options[k] for k in ("A", "B", "C", "D")]
        label = ["A", "B", "C", "D"].index(row["answer_idx"])
        rows.append(
            {
                "question": row["question"].strip(),
                "choices": choices,
                "label": label,
                "key": row["question"][:200].strip(),
                "original_text": f"{row['question'].strip()} {choices[label]}",
            }
        )
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def load_arc_rows(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows: List[Dict[str, Any]] = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        if len(labels) != 4 or len(texts) != 4:
            continue
        answer_key = str(row["answerKey"]).strip()
        if answer_key not in ANSWER_KEY_TO_INDEX:
            continue
        label = ANSWER_KEY_TO_INDEX[answer_key]
        rows.append(
            {
                "question": row["question"].strip(),
                "choices": list(texts),
                "label": label,
                "key": row["question"][:200].strip(),
                "original_text": f"{row['question'].strip()} {texts[label]}",
            }
        )
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def load_mmlu_rows(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows: List[Dict[str, Any]] = []
    for row in ds:
        choices = row["choices"]
        if len(choices) != 4:
            continue
        label = int(row["answer"])
        rows.append(
            {
                "question": row["question"].strip(),
                "choices": list(choices),
                "label": label,
                "key": row["question"][:200].strip(),
                "original_text": f"{row['question'].strip()} {choices[label]}",
            }
        )
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows
