from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config import Config
from experiments.e2.common import (
    PROJECT_ROOT,
    build_baseline_model,
    build_injection_model,
    load_knowledge_map,
    setup_logging,
)
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices_injection
from experiments.e3.evaluator import eval_baseline, eval_rag_compressed
from experiments.e7.phase1_retrieval import Phase1Retriever

logger = logging.getLogger(__name__)


def _log_section(title: str) -> None:
    logger.info("%s %s %s", "=" * 16, title, "=" * 16)


def _resolve_output(cfg: Config, output_path: Optional[str]) -> Path:
    if output_path is not None:
        return Path(output_path)
    return PROJECT_ROOT / cfg.paths.results_dir / "e7" / "e7_benchmark_compare.json"


def _load_datasets(max_samples: int) -> Dict[str, List[Dict[str, Any]]]:
    from experiments.e3.data_loading import load_arc_rows, load_medqa_rows, load_mmlu_rows

    limit = None if max_samples < 0 else max_samples
    return {
        "medqa": load_medqa_rows(limit=limit),
        "arc": load_arc_rows(limit=limit),
        "mmlu": load_mmlu_rows(limit=limit),
    }


def _load_knowledge_maps(cfg: Config) -> Dict[str, Dict[str, List[int]]]:
    return {
        "medqa": load_knowledge_map(str(PROJECT_ROOT / cfg.eval.medqa_knowledge_map)),
        "arc": load_knowledge_map(str(PROJECT_ROOT / cfg.eval.arc_knowledge_map)),
        "mmlu": load_knowledge_map(str(PROJECT_ROOT / cfg.eval.mmlu_knowledge_map)),
    }


def eval_phase1_router_fusion(
    model: torch.nn.Module,
    tokenizer: Any,
    retriever: Phase1Retriever,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    group_name: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    from tqdm.auto import tqdm

    total = len(rows)
    correct = 0
    iterator = tqdm(
        rows,
        total=total,
        desc=f"📌 {dataset_name.upper()} / {group_name}",
        leave=True,
        disable=not show_progress,
    )
    logger.info("📌 %s | %s | start | samples=%d", dataset_name.upper(), group_name, total)
    with torch.no_grad():
        for i, row in enumerate(iterator, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            knowledge_ids = retriever.retrieve_from_texts([prompt])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pred = score_choices_injection(model, tokenizer, context_ids, knowledge_ids, device)
            if pred == row["label"]:
                correct += 1
            iterator.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}")
    acc = correct / total if total else 0.0
    logger.info("✅ %s | %s | done | acc=%.4f | correct=%d/%d", dataset_name.upper(), group_name, acc, correct, total)
    return {"acc": acc, "correct": correct, "total": total}


def _print_report(results: Dict[str, Any]) -> None:
    _log_section("📊 E7 SUMMARY")
    logger.info("%-24s %10s %10s %10s", "group", "MEDQA", "ARC", "MMLU")
    logger.info("%s", "-" * 64)
    for key, label in [
        ("B0_qwen3_base", "B0 Qwen3-0.6B"),
        ("B1_qwen3_rag", "B1 Qwen3-0.6B+RAG"),
        ("S1_p1_p2_p3", "S1 P1->P2->P3"),
        ("S2_p1_p3_infer", "S2 P1->P3 Infer"),
        ("S3_p2oracle_p1_p3", "S3 P2oracle->P1->P3"),
    ]:
        logger.info(
            "%-24s %10.2f%% %10.2f%% %10.2f%%",
            label,
            results["medqa"][key]["acc"] * 100,
            results["arc"][key]["acc"] * 100,
            results["mmlu"][key]["acc"] * 100,
        )
    logger.info("%s", "-" * 64)
    logger.info("%-24s %10s %10s %10s", "delta", "MEDQA", "ARC", "MMLU")
    logger.info("%s", "-" * 64)
    for key, label in [
        ("S1_minus_B0", "S1 - B0"),
        ("S1_minus_B1", "S1 - B1"),
        ("S2_minus_B0", "S2 - B0"),
        ("S2_minus_B1", "S2 - B1"),
        ("S3_minus_B0", "S3 - B0"),
        ("S3_minus_B1", "S3 - B1"),
    ]:
        logger.info(
            "%-24s %10.2f%% %10.2f%% %10.2f%%",
            label,
            results["summary"]["medqa"][key] * 100,
            results["summary"]["arc"][key] * 100,
            results["summary"]["mmlu"][key] * 100,
        )


def run_e7_all(
    cfg: Config,
    phase1_weights: str,
    scheme1_weights: str,
    scheme2_weights: str,
    scheme3_weights: str,
    device: str = "cuda:0",
    max_samples: int = -1,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    setup_logging()
    started = time.time()
    device_obj = torch.device(device)

    _log_section("🌍 E7 BENCHMARK COMPARE")
    logger.info(
        "E7 start | phase1=%s | s1=%s | s2=%s | s3=%s | device=%s | max_samples=%d",
        phase1_weights,
        scheme1_weights,
        scheme2_weights,
        scheme3_weights,
        device,
        max_samples,
    )

    datasets = _load_datasets(max_samples=max_samples)
    knowledge_maps = _load_knowledge_maps(cfg)
    logger.info(
        "Datasets loaded | MedQA=%d | ARC=%d | MMLU=%d",
        len(datasets["medqa"]),
        len(datasets["arc"]),
        len(datasets["mmlu"]),
    )

    results: Dict[str, Any] = {
        "phase1_weights": phase1_weights,
        "scheme1_weights": scheme1_weights,
        "scheme2_weights": scheme2_weights,
        "scheme3_weights": scheme3_weights,
        "device": device,
        "max_samples": max_samples,
        "medqa": {},
        "arc": {},
        "mmlu": {},
    }

    _log_section("📌 B0 / B1")
    baseline_model, baseline_tokenizer = build_baseline_model(cfg, device=device)
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            rows = datasets[ds_name]
            km = knowledge_maps[ds_name]
            results[ds_name]["B0_qwen3_base"] = eval_baseline(
                baseline_model,
                baseline_tokenizer,
                rows,
                device_obj,
                ds_name,
            )
            results[ds_name]["B1_qwen3_rag"] = eval_rag_compressed(
                baseline_model,
                baseline_tokenizer,
                rows,
                device_obj,
                ds_name,
                knowledge_map=km,
            )
    finally:
        del baseline_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    retriever = Phase1Retriever(cfg, checkpoint_dir=phase1_weights, device=device_obj)

    _log_section("📌 S1")
    scheme1_model, scheme1_tokenizer = build_injection_model(cfg, scheme1_weights, device=device, log_prefix="E7LoadS1")
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            results[ds_name]["S1_p1_p2_p3"] = eval_phase1_router_fusion(
                scheme1_model,
                scheme1_tokenizer,
                retriever,
                datasets[ds_name],
                device_obj,
                ds_name,
                "S1",
            )
    finally:
        del scheme1_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    _log_section("📌 S2")
    scheme2_model, scheme2_tokenizer = build_injection_model(cfg, scheme2_weights, device=device, log_prefix="E7LoadS2")
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            results[ds_name]["S2_p1_p3_infer"] = eval_phase1_router_fusion(
                scheme2_model,
                scheme2_tokenizer,
                retriever,
                datasets[ds_name],
                device_obj,
                ds_name,
                "S2",
            )
    finally:
        del scheme2_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    _log_section("📌 S3")
    scheme3_model, scheme3_tokenizer = build_injection_model(cfg, scheme3_weights, device=device, log_prefix="E7LoadS3")
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            results[ds_name]["S3_p2oracle_p1_p3"] = eval_phase1_router_fusion(
                scheme3_model,
                scheme3_tokenizer,
                retriever,
                datasets[ds_name],
                device_obj,
                ds_name,
                "S3",
            )
    finally:
        del scheme3_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    del retriever
    if torch.cuda.is_available() and device_obj.type == "cuda":
        torch.cuda.empty_cache()

    results["summary"] = {}
    for ds_name in ("medqa", "arc", "mmlu"):
        b0 = results[ds_name]["B0_qwen3_base"]["acc"]
        b1 = results[ds_name]["B1_qwen3_rag"]["acc"]
        s1 = results[ds_name]["S1_p1_p2_p3"]["acc"]
        s2 = results[ds_name]["S2_p1_p3_infer"]["acc"]
        s3 = results[ds_name]["S3_p2oracle_p1_p3"]["acc"]
        results["summary"][ds_name] = {
            "S1_minus_B0": s1 - b0,
            "S1_minus_B1": s1 - b1,
            "S2_minus_B0": s2 - b0,
            "S2_minus_B1": s2 - b1,
            "S3_minus_B0": s3 - b0,
            "S3_minus_B1": s3 - b1,
            "best_acc": max(b0, b1, s1, s2, s3),
            "best_group": max(
                [
                    ("B0_qwen3_base", b0),
                    ("B1_qwen3_rag", b1),
                    ("S1_p1_p2_p3", s1),
                    ("S2_p1_p3_infer", s2),
                    ("S3_p2oracle_p1_p3", s3),
                ],
                key=lambda x: x[1],
            )[0],
        }

    results["elapsed_sec"] = time.time() - started
    _print_report(results)

    output_file = _resolve_output(cfg, output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("E7 finished | elapsed_sec=%.2f | output=%s", results["elapsed_sec"], output_file)
    return results
