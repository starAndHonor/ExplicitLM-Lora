from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from experiments.e2.arc_eval import load_arc_examples  # noqa: E402
from experiments.e2.common import (  # noqa: E402
    build_baseline_model,
    build_injection_model,
    load_knowledge_map,
    prepare_knowledge_tensor,
    setup_logging,
)
from experiments.e2.medqa_eval import load_medqa_examples  # noqa: E402
from experiments.e2.mmlu_eval import load_mmlu_examples  # noqa: E402
from experiments.e2.scoring import CHOICE_LABELS, build_multiple_choice_prompt  # noqa: E402
from models import ModifiedQwen  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a single E2 example")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fusion-ckpt", default="checkpoints/phase2_best")
    parser.add_argument("--dataset", choices=["medqa", "arc", "mmlu"], required=True)
    parser.add_argument("--index", type=int, default=0, help="0-based example index")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _load_rows(dataset: str) -> List[Dict[str, object]]:
    if dataset == "medqa":
        return load_medqa_examples()
    if dataset == "arc":
        return load_arc_examples()
    if dataset == "mmlu":
        return load_mmlu_examples()
    raise ValueError(f"unsupported dataset: {dataset}")


def _load_map(cfg, dataset: str) -> Dict[str, List[int]]:
    if dataset == "medqa":
        return load_knowledge_map(str(PROJECT_ROOT / cfg.eval.medqa_knowledge_map))
    if dataset == "arc":
        return load_knowledge_map(str(PROJECT_ROOT / cfg.eval.arc_knowledge_map))
    if dataset == "mmlu":
        return load_knowledge_map(str(PROJECT_ROOT / cfg.eval.mmlu_knowledge_map))
    raise ValueError(f"unsupported dataset: {dataset}")


def _choice_scores_baseline(
    model: torch.nn.Module,
    tokenizer,
    context_ids: List[int],
    device: torch.device,
) -> List[float]:
    scores: List[float] = []
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
        scores.append(float(token_ll.sum().item()))
    return scores


def _choice_scores_injection(
    model: ModifiedQwen,
    tokenizer,
    context_ids: List[int],
    knowledge_ids: torch.LongTensor,
    device: torch.device,
    triggered_layers: List[int],
) -> List[float]:
    handles = []
    base_layers = model.base_model.model.layers
    for layer_idx in model.injection_layers:
        def _observer(_module, _inputs, _output, idx=layer_idx):
            triggered_layers.append(idx)
            return None

        handles.append(base_layers[layer_idx].register_forward_hook(_observer))

    try:
        scores: List[float] = []
        for choice in [" A", " B", " C", " D"]:
            cont_ids = tokenizer.encode(choice, add_special_tokens=False)
            input_ids = context_ids + cont_ids
            input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_t)
            outputs = model(
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
            scores.append(float(token_ll.sum().item()))
        return scores
    finally:
        for handle in handles:
            handle.remove()


def _print_scores(title: str, scores: Sequence[float]) -> None:
    print(f"\n[{title}]")
    for label, score in zip(CHOICE_LABELS, scores):
        print(f"  {label}: {score:.6f}")
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    print(f"  pred: {CHOICE_LABELS[best_idx]}")


def main() -> None:
    args = _parse_args()
    setup_logging()

    cfg = load_config(args.config)
    device = torch.device(args.device)

    rows = _load_rows(args.dataset)
    if args.index < 0 or args.index >= len(rows):
        raise IndexError(f"index out of range: {args.index}, dataset size={len(rows)}")
    row = rows[args.index]
    knowledge_map = _load_map(cfg, args.dataset)

    prompt = build_multiple_choice_prompt(row["question"], row["choices"])
    key = row["key"]
    token_ids = knowledge_map.get(key)
    knowledge_tensor = prepare_knowledge_tensor(
        token_ids,
        cfg.model.fusion_length,
        0,
        device,
    )

    baseline_model, baseline_tokenizer = build_baseline_model(cfg, device=str(device))
    fusion_model, fusion_tokenizer = build_injection_model(cfg, args.fusion_ckpt, device=str(device))

    # Use the fusion tokenizer pad id to reflect the actual injection model path.
    knowledge_tensor = prepare_knowledge_tensor(
        token_ids,
        cfg.model.fusion_length,
        fusion_tokenizer.pad_token_id,
        device,
    )
    empty_tensor = prepare_knowledge_tensor(
        None,
        cfg.model.fusion_length,
        fusion_tokenizer.pad_token_id,
        device,
    )
    knowledge_mask = (knowledge_tensor != fusion_tokenizer.pad_token_id).long()
    empty_mask = (empty_tensor != fusion_tokenizer.pad_token_id).long()

    print(f"dataset: {args.dataset}")
    print(f"index: {args.index}")
    print(f"question key: {key}")
    print(f"gold: {CHOICE_LABELS[int(row['label'])]}")
    print(f"prompt:\n{prompt}")
    print(f"\nknowledge found: {token_ids is not None}")
    print(f"knowledge ids (first 16): {knowledge_tensor[0, :16].tolist()}")
    print(f"knowledge mask sum: {int(knowledge_mask.sum().item())}")
    print(f"knowledge mask (first 16): {knowledge_mask[0, :16].tolist()}")
    print(f"empty mask sum: {int(empty_mask.sum().item())}")
    print(f"empty mask (first 16): {empty_mask[0, :16].tolist()}")

    baseline_context_ids = baseline_tokenizer.encode(prompt, add_special_tokens=False)
    fusion_context_ids = fusion_tokenizer.encode(prompt, add_special_tokens=False)

    baseline_scores = _choice_scores_baseline(
        baseline_model,
        baseline_tokenizer,
        baseline_context_ids,
        device,
    )
    triggered_knowledge: List[int] = []
    fusion_scores = _choice_scores_injection(
        fusion_model,
        fusion_tokenizer,
        fusion_context_ids,
        knowledge_tensor,
        device,
        triggered_knowledge,
    )
    triggered_empty: List[int] = []
    empty_scores = _choice_scores_injection(
        fusion_model,
        fusion_tokenizer,
        fusion_context_ids,
        empty_tensor,
        device,
        triggered_empty,
    )

    print(f"\ntriggered layers (fusion+knowledge): {sorted(set(triggered_knowledge))}")
    print(f"hook call count (fusion+knowledge): {len(triggered_knowledge)}")
    print(f"triggered layers (fusion+empty): {sorted(set(triggered_empty))}")
    print(f"hook call count (fusion+empty): {len(triggered_empty)}")

    _print_scores("baseline", baseline_scores)
    _print_scores("fusion+knowledge", fusion_scores)
    _print_scores("fusion+empty", empty_scores)


if __name__ == "__main__":
    main()
