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
    setup_logging,
)
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices_injection
from experiments.e3.evaluator import eval_baseline
from training.dense_retriever import DenseRetriever

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


def _build_dense_query(row: Dict[str, Any], query_mode: str) -> str:
    if query_mode == "question_only":
        return row["question"]
    if query_mode == "question_choices":
        return build_multiple_choice_prompt(row["question"], row["choices"])
    raise ValueError(f"unsupported query_mode: {query_mode}")


def eval_dense_fusion(
    model: torch.nn.Module,
    tokenizer: Any,
    retriever: DenseRetriever,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    group_name: str,
    query_mode: str,
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
            retrieval_query = _build_dense_query(row, query_mode)
            knowledge_ids = retriever.retrieve_from_texts([retrieval_query])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pred = score_choices_injection(model, tokenizer, context_ids, knowledge_ids, device)
            if pred == row["label"]:
                correct += 1
            iterator.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}")
    acc = correct / total if total else 0.0
    logger.info("✅ %s | %s | done | acc=%.4f | correct=%d/%d", dataset_name.upper(), group_name, acc, correct, total)
    return {"acc": acc, "correct": correct, "total": total}


def eval_dense_rag(
    model: torch.nn.Module,
    tokenizer: Any,
    retriever: DenseRetriever,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    query_mode: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    from tqdm.auto import tqdm

    total = len(rows)
    correct = 0
    iterator = tqdm(
        rows,
        total=total,
        desc=f"📌 {dataset_name.upper()} / RAG",
        leave=True,
        disable=not show_progress,
    )
    logger.info("📌 %s | Dense RAG | start | samples=%d", dataset_name.upper(), total)
    with torch.no_grad():
        for i, row in enumerate(iterator, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            retrieval_query = _build_dense_query(row, query_mode)
            knowledge_ids = retriever.retrieve_from_texts([retrieval_query])
            pad_id = tokenizer.pad_token_id
            clean_ids = [t for t in knowledge_ids[0].tolist() if t != pad_id]
            compressed_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
            context = f"Context: {compressed_text}\n\n{prompt}"
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            from experiments.e2.scoring import score_choices

            pred = score_choices(model, tokenizer, context_ids, device)
            if pred == row["label"]:
                correct += 1
            iterator.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}")
    acc = correct / total if total else 0.0
    logger.info("✅ %s | Dense RAG | done | acc=%.4f | correct=%d/%d", dataset_name.upper(), acc, correct, total)
    return {"acc": acc, "correct": correct, "total": total}


def _print_report(results: Dict[str, Any]) -> None:
    _log_section("📊 E7 SUMMARY")
    logger.info("%-24s %10s %10s %10s", "group", "MEDQA", "ARC", "MMLU")
    logger.info("%s", "-" * 64)
    for key, label in [
        ("B0_qwen3_base", "B0 Qwen3-0.6B"),
        ("TF_dense_p3_infer", "TF Dense->P3"),
        ("RAG_dense", "Dense RAG"),
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
        ("TF_minus_B0", "TF - B0"),
        ("RAG_minus_B0", "RAG - B0"),
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
    dense_indices: Dict[str, str],
    training_free_weights: str,
    device: str = "cuda:0",
    max_samples: int = -1,
    output_path: Optional[str] = None,
    query_mode: str = "question_only",
) -> Dict[str, Any]:
    setup_logging()
    started = time.time()
    device_obj = torch.device(device)

    _log_section("🌍 E7 BENCHMARK COMPARE")
    logger.info(
        "E7 start | dense_indices=%s | tf=%s | device=%s | max_samples=%d | query_mode=%s",
        dense_indices,
        training_free_weights,
        device,
        max_samples,
        query_mode,
    )

    datasets = _load_datasets(max_samples=max_samples)
    logger.info(
        "Datasets loaded | MedQA=%d | ARC=%d | MMLU=%d",
        len(datasets["medqa"]),
        len(datasets["arc"]),
        len(datasets["mmlu"]),
    )

    results: Dict[str, Any] = {
        "dense_indices": dense_indices,
        "training_free_weights": training_free_weights,
        "device": device,
        "max_samples": max_samples,
        "query_mode": query_mode,
        "medqa": {},
        "arc": {},
        "mmlu": {},
    }

    _log_section("📌 B0")
    baseline_model, baseline_tokenizer = build_baseline_model(cfg, device=device)
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            rows = datasets[ds_name]
            results[ds_name]["B0_qwen3_base"] = eval_baseline(
                baseline_model,
                baseline_tokenizer,
                rows,
                device_obj,
                ds_name,
            )
    finally:
        del baseline_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    _log_section("📌 TF")
    training_free_model, training_free_tokenizer = build_injection_model(
        cfg,
        training_free_weights,
        device=device,
        log_prefix="E7LoadTF",
    )
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            retriever = DenseRetriever(cfg=cfg, index_path=dense_indices[ds_name], device=device_obj, tokenizer=training_free_tokenizer)
            results[ds_name]["TF_dense_p3_infer"] = eval_dense_fusion(
                training_free_model,
                training_free_tokenizer,
                retriever,
                datasets[ds_name],
                device_obj,
                ds_name,
                "TF",
                query_mode=query_mode,
            )
            del retriever
            if torch.cuda.is_available() and device_obj.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del training_free_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    _log_section("📌 RAG")
    rag_model, rag_tokenizer = build_baseline_model(cfg, device=device)
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            retriever = DenseRetriever(cfg=cfg, index_path=dense_indices[ds_name], device=device_obj, tokenizer=rag_tokenizer)
            results[ds_name]["RAG_dense"] = eval_dense_rag(
                rag_model,
                rag_tokenizer,
                retriever,
                datasets[ds_name],
                device_obj,
                ds_name,
                query_mode=query_mode,
            )
            del retriever
            if torch.cuda.is_available() and device_obj.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del rag_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    results["summary"] = {}
    for ds_name in ("medqa", "arc", "mmlu"):
        b0 = results[ds_name]["B0_qwen3_base"]["acc"]
        tf = results[ds_name]["TF_dense_p3_infer"]["acc"]
        rag = results[ds_name]["RAG_dense"]["acc"]
        results["summary"][ds_name] = {
            "TF_minus_B0": tf - b0,
            "RAG_minus_B0": rag - b0,
            "best_acc": max(b0, tf, rag),
            "best_group": max(
                [
                    ("B0_qwen3_base", b0),
                    ("TF_dense_p3_infer", tf),
                    ("RAG_dense", rag),
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
