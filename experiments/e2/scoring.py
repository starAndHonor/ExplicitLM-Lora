from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from models import ModifiedQwen


CHOICE_LABELS = ["A", "B", "C", "D"]


def build_multiple_choice_prompt(question: str, choices: Sequence[str]) -> str:
    answers_str = "".join(f"{label}. {choice}\n" for label, choice in zip(CHOICE_LABELS, choices))
    return f"Question: {question}\n{answers_str}Answer:"


def score_choices(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context_ids: List[int],
    device: torch.device,
) -> int:
    scores = []
    for choice in [" A", " B", " C", " D"]:
        cont_ids = tokenizer.encode(choice, add_special_tokens=False)
        input_ids = context_ids + cont_ids
        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        outputs = model(input_ids=input_t)
        logits = outputs.logits

        cont_start = len(context_ids) - 1
        cont_end = len(input_ids) - 1
        cont_logits = logits[0, cont_start:cont_end, :]
        cont_tokens = torch.tensor(cont_ids, dtype=torch.long, device=device)
        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_ll = log_probs.gather(1, cont_tokens.unsqueeze(-1)).squeeze(-1)
        scores.append(token_ll.sum().item())
    return scores.index(max(scores))


def score_choices_injection(
    model: ModifiedQwen,
    tokenizer: AutoTokenizer,
    context_ids: List[int],
    knowledge_ids: torch.LongTensor,
    device: torch.device,
) -> int:
    scores = []
    for choice in [" A", " B", " C", " D"]:
        cont_ids = tokenizer.encode(choice, add_special_tokens=False)
        input_ids = context_ids + cont_ids
        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_t)
        outputs: CausalLMOutputWithPast = model(
            input_ids=input_t,
            knowledge_ids=knowledge_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        cont_start = len(context_ids) - 1
        cont_end = len(input_ids) - 1
        cont_logits = logits[0, cont_start:cont_end, :]
        cont_tokens = torch.tensor(cont_ids, dtype=torch.long, device=device)
        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_ll = log_probs.gather(1, cont_tokens.unsqueeze(-1)).squeeze(-1)
        scores.append(token_ll.sum().item())
    return scores.index(max(scores))
