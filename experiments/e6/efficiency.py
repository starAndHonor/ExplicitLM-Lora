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
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices, score_choices_injection
from experiments.e2.common import prepare_knowledge_tensor
from experiments.e3.data_loading import load_medqa_rows

logger = logging.getLogger(__name__)

_METHOD_LABELS = {
    "baseline": "Baseline",
    "rag_compressed": "RAG-compressed@64",
    "rag_original": "RAG-original@~256",
    "fusion": "Fusion-Phase2@64",
}

_CONTEXT_TOKENS = {
    "baseline": 0,
    "rag_compressed": 64,
    "rag_original": 256,
    "fusion": 0,
}


def _log_section(title: str) -> None:
    line = "=" * 18
    logger.info("%s %s %s", line, title, line)


def _benchmark_method(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: List[Dict[str, Any]],
    device: torch.device,
    method: str,
    knowledge_map: Optional[Dict[str, List[int]]] = None,
    knowledge_length: int = 64,
    n_warmup: int = 10,
    n_measure: int = 200,
) -> Dict[str, Any]:
    assert len(rows) >= n_warmup + n_measure, f"not enough rows: {len(rows)} < {n_warmup + n_measure}"
    label = _METHOD_LABELS[method]

    def process_one(row: Dict[str, Any]) -> int:
        prompt = build_multiple_choice_prompt(row["question"], row["choices"])

        if method == "baseline":
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            score_choices(model, tokenizer, context_ids, device)
            return len(context_ids)

        if method == "rag_compressed":
            assert knowledge_map is not None
            k_ids = knowledge_map.get(row["key"])
            if k_ids is None:
                context = prompt
            else:
                clean_ids = [t for t in k_ids if t != tokenizer.pad_token_id]
                compressed_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
                context = f"Context: {compressed_text}\n\n{prompt}"
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            score_choices(model, tokenizer, context_ids, device)
            return len(context_ids)

        if method == "rag_original":
            context = f"Context: {row['original_text']}\n\n{prompt}"
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            score_choices(model, tokenizer, context_ids, device)
            return len(context_ids)

        if method == "fusion":
            assert knowledge_map is not None
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            knowledge_tensor = prepare_knowledge_tensor(
                knowledge_map.get(row["key"]),
                knowledge_length,
                tokenizer.pad_token_id,
                device,
            )
            score_choices_injection(model, tokenizer, context_ids, knowledge_tensor, device)
            return len(context_ids)

        raise ValueError(f"unknown method: {method}")

    logger.info("[E6] %s warmup (%d samples)...", label, n_warmup)
    with torch.no_grad():
        for row in rows[:n_warmup]:
            process_one(row)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    logger.info("[E6] %s benchmark (%d samples)...", label, n_measure)
    input_lengths: List[int] = []
    start = time.perf_counter()
    with torch.no_grad():
        for i, row in enumerate(rows[n_warmup : n_warmup + n_measure], start=1):
            input_lengths.append(process_one(row))
            if i % 50 == 0:
                logger.info("[E6] %s progress %d/%d", label, i, n_measure)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - start
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    avg_input_len = sum(input_lengths) / len(input_lengths)

    result = {
        "method": method,
        "label": label,
        "n_samples": n_measure,
        "total_time_s": round(total_time, 3),
        "latency_ms": round(total_time / n_measure * 1000, 2),
        "throughput": round(n_measure / total_time, 2),
        "peak_memory_mb": round(peak_memory, 1),
        "avg_input_len": round(avg_input_len, 1),
        "context_tokens": _CONTEXT_TOKENS[method],
    }
    logger.info(
        "[E6] %s done | latency=%sms | throughput=%s/s | peak_memory=%sMB | avg_input_len=%s",
        label,
        result["latency_ms"],
        result["throughput"],
        result["peak_memory_mb"],
        result["avg_input_len"],
    )
    return result


def _load_accuracy_data(
    results_dir: Path,
    e5_path: Optional[Path] = None,
    e3_path: Optional[Path] = None,
) -> Optional[Dict[str, Dict[str, float]]]:
    accuracy: Dict[str, Dict[str, float]] = {}
    loaded_any = False

    e5_candidates = [e5_path] if e5_path is not None else [
        results_dir / "e5" / "e5_knowledge_analysis.json",
        results_dir / "e5_knowledge_analysis.json",
    ]
    e5_result_path = next((p for p in e5_candidates if p is not None and p.exists()), None)
    if e5_result_path is not None:
        data = json.loads(e5_result_path.read_text(encoding="utf-8"))
        e5a = data.get("e5a", {})
        for ds_name in ("medqa", "arc", "mmlu"):
            ds_data = e5a.get(ds_name, {})
            accuracy[ds_name] = {
                "baseline": ds_data.get("baseline", {}).get("acc"),
                "rag_k64": ds_data.get("rag_k64", {}).get("acc"),
                "rag_k256": ds_data.get("rag_k256", {}).get("acc"),
                "fusion_p2_k64": ds_data.get("fusion_p2_k64", {}).get("acc"),
            }
        loaded_any = True
        logger.info("E5 accuracy loaded: %s", e5_result_path)
    else:
        logger.warning("E5 result missing in expected locations")

    e3_candidates = [e3_path] if e3_path is not None else [
        results_dir / "e3" / "e3_fair_compare.json",
        results_dir / "e3_fair_compare.json",
    ]
    e3_result_path = next((p for p in e3_candidates if p is not None and p.exists()), None)
    if e3_result_path is not None:
        data = json.loads(e3_result_path.read_text(encoding="utf-8"))
        for ds_name in ("medqa", "arc", "mmlu"):
            ds_data = data.get(ds_name, {})
            if ds_name not in accuracy:
                accuracy[ds_name] = {}
            g4 = ds_data.get("G4_rag_original", {})
            g3 = ds_data.get("G3_fusion_phase2", {})
            if g4.get("acc") is not None:
                accuracy[ds_name]["rag_original"] = g4["acc"]
            if g3.get("acc") is not None:
                accuracy[ds_name]["fusion_phase2"] = g3["acc"]
        loaded_any = True
        logger.info("E3 accuracy loaded: %s", e3_result_path)
    else:
        logger.warning("E3 result missing in expected locations")

    return accuracy if loaded_any else None


def _print_e6_report(benchmarks: Dict[str, Dict[str, Any]], acc_data: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    sep = "=" * 95
    logger.info("\n%s", sep)
    logger.info("E6-A Inference Efficiency (MedQA, N=%d, single GPU)", benchmarks["baseline"]["n_samples"])
    logger.info("%s", sep)
    logger.info(
        "%-25s %14s %14s %10s %14s %14s",
        "Method",
        "Latency(ms)",
        "Throughput",
        "Memory(MB)",
        "Avg Input Len",
        "Context Used",
    )
    logger.info("%s", "-" * 95)
    for method in ("baseline", "rag_compressed", "fusion", "rag_original"):
        if method not in benchmarks:
            continue
        b = benchmarks[method]
        ctx = f"~{b['context_tokens']} tokens" if b["context_tokens"] > 0 else "0 tokens"
        logger.info(
            "%-25s %14.2f %14.2f %10.1f %14.1f %14s",
            b["label"],
            b["latency_ms"],
            b["throughput"],
            b["peak_memory_mb"],
            b["avg_input_len"],
            ctx,
        )

    if "rag_original" in benchmarks and "fusion" in benchmarks:
        rag_lat = benchmarks["rag_original"]["latency_ms"]
        fusion_lat = benchmarks["fusion"]["latency_ms"]
        speedup = rag_lat / fusion_lat if fusion_lat > 0 else float("inf")
        mem_rag = benchmarks["rag_original"]["peak_memory_mb"]
        mem_fusion = benchmarks["fusion"]["peak_memory_mb"]
        mem_save = mem_rag - mem_fusion
        logger.info("")
        logger.info(
            "Fusion vs RAG-original: speed %.2fx, memory saved %.0fMB (%.1f%%)",
            speedup,
            mem_save,
            (mem_save / mem_rag * 100) if mem_rag > 0 else 0.0,
        )

    if "rag_compressed" in benchmarks and "fusion" in benchmarks:
        rag_c_lat = benchmarks["rag_compressed"]["latency_ms"]
        fusion_lat = benchmarks["fusion"]["latency_ms"]
        speedup_c = rag_c_lat / fusion_lat if fusion_lat > 0 else float("inf")
        logger.info("Fusion vs RAG-compressed: speed %.2fx", speedup_c)

    if acc_data is None:
        return

    medqa_acc = acc_data.get("medqa", {})
    rag_abs_acc = medqa_acc.get("rag_original", medqa_acc.get("rag_k256"))
    fusion_abs_acc = medqa_acc.get("fusion_phase2", medqa_acc.get("fusion_p2_k64"))
    rag_acc64 = medqa_acc.get("rag_k64")
    fusion_acc64 = medqa_acc.get("fusion_p2_k64")
    if rag_abs_acc is None or fusion_abs_acc is None:
        return

    logger.info("\n%s", sep)
    logger.info("E6 Six-Dimension Comparison (MedQA, Fusion@64 vs RAG-original)")
    logger.info("%s", sep)
    logger.info("%-28s %18s %18s %18s", "Dimension", "RAG-original", "Fusion Phase 2", "Winner")
    logger.info("%s", "-" * 95)
    winner_abs = "RAG" if rag_abs_acc > fusion_abs_acc else "Fusion"
    logger.info("%-28s %17.2f%% %17.2f%% %18s", "Absolute Accuracy", rag_abs_acc * 100, fusion_abs_acc * 100, winner_abs)
    if rag_acc64 is not None and fusion_acc64 is not None:
        logger.info(
            "%-28s %17.2f%% %17.2f%% %18s",
            "Equal-Token Accuracy(k=64)",
            rag_acc64 * 100,
            fusion_acc64 * 100,
            f"Fusion ({(fusion_acc64 - rag_acc64) * 100:+.2f}%)",
        )
    rag_b = benchmarks["rag_original"]
    fusion_b = benchmarks["fusion"]
    logger.info(
        "%-28s %18s %18s %18s",
        "Latency(ms/sample)",
        f"{rag_b['latency_ms']:.2f} ms",
        f"{fusion_b['latency_ms']:.2f} ms",
        "Fusion" if fusion_b["latency_ms"] < rag_b["latency_ms"] else "RAG",
    )
    logger.info(
        "%-28s %18s %18s %18s",
        "Peak Memory(MB)",
        f"{rag_b['peak_memory_mb']:.0f} MB",
        f"{fusion_b['peak_memory_mb']:.0f} MB",
        "Fusion" if fusion_b["peak_memory_mb"] < rag_b["peak_memory_mb"] else "RAG",
    )
    logger.info("%-28s %18s %18s %18s", "Context Window Occupancy", "~256 tokens", "0 tokens", "Fusion")
    logger.info("%-28s %18s %18s %18s", "Pre-encodable Knowledge Cache", "No", "Yes", "Fusion")


def run_e6_all(
    cfg: Config,
    phase2_weights: str,
    device: str = "cuda:0",
    n_warmup: int = 10,
    n_measure: int = 200,
    output_path: Optional[str] = None,
    e5_result_path: Optional[str] = None,
    e3_result_path: Optional[str] = None,
) -> Dict[str, Any]:
    setup_logging()
    if str(device) != "cuda:0" and not str(device).startswith("cpu"):
        logger.warning("E6 is intended for single-GPU timing; got device=%s", device)
    rows = load_medqa_rows(limit=n_warmup + n_measure)
    if len(rows) < n_warmup + n_measure:
        raise ValueError(f"not enough MedQA rows: {len(rows)} < {n_warmup + n_measure}")

    results_dir = PROJECT_ROOT / cfg.paths.results_dir
    knowledge_map = load_knowledge_map(str(PROJECT_ROOT / cfg.eval.medqa_knowledge_map))
    torch_device = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")

    _log_section("📊 E6 INFERENCE EFFICIENCY")
    logger.info("Phase2 weights: %s", phase2_weights)
    logger.info("Dataset: MedQA | N=%d | warmup=%d", n_measure, n_warmup)
    logger.info("Device: %s", torch_device)

    benchmarks: Dict[str, Dict[str, Any]] = {}

    _log_section("Phase B: Baseline + RAG")
    base_model, tokenizer = build_baseline_model(cfg, device=str(torch_device))
    for method in ("baseline", "rag_compressed", "rag_original"):
        benchmarks[method] = _benchmark_method(
            base_model,
            tokenizer,
            rows,
            torch_device,
            method,
            knowledge_map=knowledge_map,
            knowledge_length=cfg.model.fusion_length,
            n_warmup=n_warmup,
            n_measure=n_measure,
        )
    del base_model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    _log_section("Phase C: Fusion")
    fusion_model, fusion_tokenizer = build_injection_model(
        cfg,
        fusion_ckpt=phase2_weights,
        device=str(torch_device),
        log_prefix="E6Load",
    )
    benchmarks["fusion"] = _benchmark_method(
        fusion_model,
        fusion_tokenizer,
        rows,
        torch_device,
        "fusion",
        knowledge_map=knowledge_map,
        knowledge_length=cfg.model.fusion_length,
        n_warmup=n_warmup,
        n_measure=n_measure,
    )
    del fusion_model
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    acc_data = _load_accuracy_data(
        results_dir,
        Path(e5_result_path) if e5_result_path else None,
        Path(e3_result_path) if e3_result_path else None,
    )
    _print_e6_report(benchmarks, acc_data)

    results = {
        "experiment": "E6_inference_efficiency",
        "phase2_weights": phase2_weights,
        "dataset": "medqa",
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "benchmarks": benchmarks,
    }
    if acc_data is not None:
        results["accuracy_data"] = acc_data

    if output_path is None:
        out = results_dir / "e6" / "e6_inference_efficiency.json"
    else:
        out = Path(output_path)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("E6 finished | output=%s", out)
    return results
