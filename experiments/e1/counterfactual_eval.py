from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import Config
from experiments.e2.common import build_injection_model, load_knowledge_map, prepare_knowledge_tensor
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices_injection

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COUNTERFACTUAL_MAP = PROJECT_ROOT / "data" / "medqa_knowledge_counterfactual.jsonl"


class KnowledgeCompressor:
    """Minimal LLMLingua-2 wrapper matching the reference E1 behavior."""

    def __init__(
        self,
        model_name: str,
        compression_rate: float = 0.25,
        gpu_id: Optional[int] = None,
        use_llmlingua2: bool = True,
    ) -> None:
        try:
            from llmlingua import PromptCompressor
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ImportError(
                "llmlingua is required to build counterfactual knowledge. "
                "Install it in the ExplicitLLM environment or prebuild "
                f"{DEFAULT_COUNTERFACTUAL_MAP}."
            ) from exc

        if gpu_id is not None and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            device_map = "cuda"
        else:
            device_map = "cpu"

        self.compressor = PromptCompressor(
            model_name=model_name,
            use_llmlingua2=use_llmlingua2,
            device_map=device_map,
        )
        self.compression_rate = compression_rate

    def compress_text(self, text: str, target_token: Optional[int] = None) -> Optional[str]:
        """压缩文本。

        参数：
            target_token: 若指定，使用固定目标 token 数（对齐 FineWeb 预处理方式）；
                          否则使用 compression_rate 按比例压缩。
        """
        kwargs: Dict[str, Any] = dict(
            force_tokens=[],
            use_token_level_filter=True,
            use_context_level_filter=False,
            use_sentence_level_filter=False,
        )
        if target_token is not None:
            kwargs["target_token"] = target_token
        else:
            kwargs["rate"] = self.compression_rate
        try:
            result = self.compressor.compress_prompt(text, **kwargs)
        except Exception:  # pragma: no cover - compressor/runtime dependent
            logger.exception("Counterfactual compression failed")
            return None
        return result["compressed_prompt"]


def load_medqa_rows(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_from_disk(str(PROJECT_ROOT / "data" / "medqa" / "hf_dataset"))["test"]
    rows: List[Dict[str, Any]] = []
    for row in ds:
        choices = [row["options"][k] for k in ("A", "B", "C", "D")]
        label = ["A", "B", "C", "D"].index(row["answer_idx"])
        wrong_idx = (label + 1) % 4
        rows.append(
            {
                "key": row["question"][:200].strip(),
                "question": row["question"].strip(),
                "choices": choices,
                "label": label,
                "correct_answer": choices[label],
                "wrong_answer": choices[wrong_idx],
            }
        )
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def build_counterfactual_knowledge(
    cfg: Config,
    output_path: str | Path,
    limit: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Build `question + wrong_answer` knowledge ids, mirroring the reference E1."""
    model_path = os.environ.get("MODEL_PATH", cfg.paths.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compressor_gpu = int(os.environ.get("E1_CF_GPU", "0"))
    compressor = KnowledgeCompressor(
        model_name=cfg.paths.llmlingua_model_dir,
        compression_rate=0.25,
        gpu_id=compressor_gpu,
    )

    rows = load_medqa_rows(limit=limit)
    knowledge_map: Dict[str, List[int]] = {}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building counterfactual MedQA knowledge | samples=%d | output=%s", len(rows), output)
    for row in tqdm(rows, total=len(rows), desc="E1 / build cf knowledge", leave=True):
        source_text = f"{row['question']} {row['wrong_answer']}"
        compressed = compressor.compress_text(source_text)
        if compressed is None:
            compressed = row["question"][:100]

        tokens = tokenizer.encode(compressed, add_special_tokens=False)
        tokens = tokens[: cfg.model.fusion_length]
        if len(tokens) < cfg.model.fusion_length:
            tokens.extend([tokenizer.pad_token_id] * (cfg.model.fusion_length - len(tokens)))
        knowledge_map[row["key"]] = tokens

    with output.open("w", encoding="utf-8") as f:
        for key, ids in knowledge_map.items():
            f.write(json.dumps({"key": key, "knowledge_ids": ids}, ensure_ascii=False) + "\n")

    logger.info("Built counterfactual knowledge map | count=%d | path=%s", len(knowledge_map), output)
    return knowledge_map


def eval_e1_sanity_check(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    rows: List[Dict[str, Any]],
    correct_km: Dict[str, List[int]],
    cf_km: Dict[str, List[int]],
    device: torch.device,
    knowledge_length: int,
) -> Dict[str, Any]:
    pad_id = tokenizer.pad_token_id
    no_knowledge = [pad_id] * knowledge_length

    correct_a = 0
    correct_cf = 0
    correct_nk = 0

    model.eval()
    progress = tqdm(rows, total=len(rows), desc="E1 / sanity check", leave=True)
    with torch.no_grad():
        for i, row in enumerate(progress, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)

            k_ids_correct = correct_km.get(row["key"], no_knowledge)
            k_ids_cf = cf_km.get(row["key"], no_knowledge)

            k_correct = prepare_knowledge_tensor(k_ids_correct, knowledge_length, pad_id, device)
            k_cf = prepare_knowledge_tensor(k_ids_cf, knowledge_length, pad_id, device)
            k_nk = prepare_knowledge_tensor(None, knowledge_length, pad_id, device)

            pred_correct = score_choices_injection(model, tokenizer, context_ids, k_correct, device)
            pred_cf = score_choices_injection(model, tokenizer, context_ids, k_cf, device)
            pred_nk = score_choices_injection(model, tokenizer, context_ids, k_nk, device)

            if pred_correct == row["label"]:
                correct_a += 1
            if pred_cf == row["label"]:
                correct_cf += 1
            if pred_nk == row["label"]:
                correct_nk += 1

            progress.set_postfix(
                acc_correct=f"{correct_a / i:.4f}",
                acc_cf=f"{correct_cf / i:.4f}",
                acc_nk=f"{correct_nk / i:.4f}",
            )

    total = len(rows)
    acc_correct = correct_a / total if total else 0.0
    acc_cf = correct_cf / total if total else 0.0
    acc_nk = correct_nk / total if total else 0.0
    ks = acc_correct - acc_cf

    return {
        "acc_correct": acc_correct,
        "acc_counterfactual": acc_cf,
        "acc_no_knowledge": acc_nk,
        "knowledge_sensitivity": ks,
        "correct_correct": correct_a,
        "correct_counterfactual": correct_cf,
        "correct_no_knowledge": correct_nk,
        "total": total,
    }


def run_e1_sanity_check(
    cfg: Config,
    fusion_ckpt: str,
    output_path: Optional[str] = None,
    max_samples: int = -1,
    counterfactual_map_path: str | Path = DEFAULT_COUNTERFACTUAL_MAP,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = build_injection_model(cfg, fusion_ckpt=fusion_ckpt, device=str(device), log_prefix="E1Load")

    cf_path = Path(counterfactual_map_path)
    if not cf_path.exists():
        logger.info("Counterfactual map not found, building: %s", cf_path)
        cf_km = build_counterfactual_knowledge(cfg, cf_path, limit=None if max_samples < 0 else max_samples)
    else:
        cf_km = load_knowledge_map(str(cf_path))

    correct_km = load_knowledge_map(str(PROJECT_ROOT / cfg.eval.medqa_knowledge_map))
    rows = load_medqa_rows(limit=None if max_samples < 0 else max_samples)

    results = eval_e1_sanity_check(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        correct_km=correct_km,
        cf_km=cf_km,
        device=device,
        knowledge_length=cfg.model.fusion_length,
    )

    if output_path is None:
        tag = Path(fusion_ckpt).name
        out = PROJECT_ROOT / cfg.paths.results_dir / "e1" / f"e1_sanity_check_{tag}.json"
    else:
        out = Path(output_path)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": str(Path(fusion_ckpt).resolve()),
        "device": str(device),
        "max_samples": max_samples,
        **results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("E1 saved to %s", out)
    return payload
