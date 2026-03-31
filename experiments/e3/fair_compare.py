from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp

from config import Config
from experiments.e2.common import (
    PROJECT_ROOT,
    build_baseline_model,
    build_injection_model,
    load_knowledge_map,
    setup_logging,
)
from experiments.e3.data_loading import load_arc_rows, load_medqa_rows, load_mmlu_rows
from experiments.e3.evaluator import eval_baseline, eval_fusion, eval_rag_compressed, eval_rag_original

logger = logging.getLogger(__name__)


def _knowledge_map_path(default_path: str, k: int) -> str:
    if k == 64:
        return str(PROJECT_ROOT / default_path)
    default = Path(default_path)
    return str(PROJECT_ROOT / default.with_name(f"{default.stem}_k{k}{default.suffix}"))


def _log_section(title: str) -> None:
    line = "=" * 16
    logger.info("%s %s %s", line, title, line)


def _run_group(
    task_spec: Dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, Any]:
    mode = task_spec["eval_mode"]
    rows = task_spec["rows"]
    dataset_name = task_spec["dataset_name"]
    show_progress = task_spec.get("show_progress", True)

    if mode == "G0":
        return eval_baseline(model, tokenizer, rows, device, dataset_name, show_progress=show_progress)
    if mode == "G1":
        return eval_rag_compressed(
            model,
            tokenizer,
            rows,
            device,
            dataset_name,
            knowledge_map=task_spec["knowledge_map"],
            show_progress=show_progress,
        )
    if mode in {"G2", "G3"}:
        return eval_fusion(
            model,
            tokenizer,
            rows,
            device,
            dataset_name,
            knowledge_map=task_spec["knowledge_map"],
            group_name=mode,
            knowledge_length=task_spec["knowledge_length"],
            show_progress=show_progress,
        )
    if mode == "G4":
        return eval_rag_original(model, tokenizer, rows, device, dataset_name, show_progress=show_progress)
    raise ValueError(f"unsupported eval mode: {mode}")


def _load_model(
    cfg: Config,
    task_spec: Dict[str, Any],
    device: str,
):
    if task_spec["model_type"] == "baseline":
        return build_baseline_model(cfg, device=device)
    return build_injection_model(
        cfg,
        fusion_ckpt=task_spec["weights"],
        device=device,
        log_prefix="E3Load",
    )


def _worker(rank: int, world_size: int, cfg: Config, task_spec: Dict[str, Any], tmp_dir: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    rows = task_spec["rows"]
    per_shard = len(rows) // world_size
    start = rank * per_shard
    end = start + per_shard if rank < world_size - 1 else len(rows)
    shard_spec = {**task_spec, "rows": rows[start:end], "show_progress": rank == 0}

    logger.info(
        "[GPU %d/%d] %s | %s | shard [%d:%d] | rows=%d",
        rank,
        world_size,
        task_spec["dataset_name"],
        task_spec["eval_mode"],
        start,
        end,
        len(shard_spec["rows"]),
    )

    model, tokenizer = _load_model(cfg, task_spec, str(device))
    result = _run_group(shard_spec, model, tokenizer, device)

    shard_path = Path(tmp_dir) / f"shard_{rank}.json"
    shard_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    logger.info("[GPU %d/%d] %s | %s | done | %s", rank, world_size, task_spec["dataset_name"], task_spec["eval_mode"], result)

    del model
    torch.cuda.empty_cache()


def _merge_shards(tmp_dir: str, num_shards: int) -> Dict[str, Any]:
    shards = []
    for i in range(num_shards):
        shard_path = Path(tmp_dir) / f"shard_{i}.json"
        if not shard_path.exists():
            raise FileNotFoundError(f"missing shard result: {shard_path}")
        shards.append(json.loads(shard_path.read_text(encoding="utf-8")))

    total = sum(s["total"] for s in shards)
    correct = sum(s["correct"] for s in shards)
    result: Dict[str, Any] = {
        "acc": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }
    if any("missing_knowledge" in s for s in shards):
        result["missing_knowledge"] = sum(s.get("missing_knowledge", 0) for s in shards)
    if any("knowledge_miss_count" in s for s in shards):
        result["knowledge_miss_count"] = sum(s.get("knowledge_miss_count", 0) for s in shards)
    return result


def _run_parallel(cfg: Config, task_spec: Dict[str, Any], num_gpus: int) -> Dict[str, Any]:
    if num_gpus <= 1:
        device = torch.device(task_spec["device"])
        _log_section(f"🚀 {task_spec['dataset_name'].upper()} / {task_spec['eval_mode']} / single-process")
        model, tokenizer = _load_model(cfg, task_spec, str(device))
        try:
            return _run_group(task_spec, model, tokenizer, device)
        finally:
            del model
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

    tmp_dir = tempfile.mkdtemp(prefix=f"e3_{task_spec['dataset_name']}_{task_spec['eval_mode']}_")
    _log_section(f"🚀 {task_spec['dataset_name'].upper()} / {task_spec['eval_mode']} / multi-gpu")
    logger.info(
        "%s | %s | multi-GPU run | gpus=%d | rows=%d | tmp=%s",
        task_spec["dataset_name"],
        task_spec["eval_mode"],
        num_gpus,
        len(task_spec["rows"]),
        tmp_dir,
    )
    mp.spawn(_worker, args=(num_gpus, cfg, task_spec, tmp_dir), nprocs=num_gpus, join=True)
    result = _merge_shards(tmp_dir, num_gpus)
    logger.info("%s | %s | merged result=%s", task_spec["dataset_name"], task_spec["eval_mode"], result)
    return result


def _print_report(results: Dict[str, Any]) -> None:
    _log_section("📊 E3 SUMMARY")
    logger.info("%-22s %10s %10s %10s", "group", "MEDQA", "ARC", "MMLU")
    logger.info("%s", "-" * 60)
    for key, label in [
        ("G0_baseline", "G0 Baseline"),
        ("G1_rag_compressed", "G1 RAG-compressed"),
        ("G2_fusion_phase1", "G2 Fusion-Phase1"),
        ("G3_fusion_phase2", "G3 Fusion-Phase2"),
        ("G4_rag_original", "G4 RAG-original"),
    ]:
        logger.info(
            "%-22s %10.2f%% %10.2f%% %10.2f%%",
            label,
            results["medqa"][key]["acc"] * 100,
            results["arc"][key]["acc"] * 100,
            results["mmlu"][key]["acc"] * 100,
        )

    logger.info("%s", "-" * 60)
    logger.info("%-22s %10s %10s %10s", "delta", "MEDQA", "ARC", "MMLU")
    logger.info("%s", "-" * 60)
    logger.info(
        "%-22s %10.2f%% %10.2f%% %10.2f%%",
        "G2 - G1",
        results["summary"]["medqa"]["G2_vs_G1"] * 100,
        results["summary"]["arc"]["G2_vs_G1"] * 100,
        results["summary"]["mmlu"]["G2_vs_G1"] * 100,
    )
    logger.info(
        "%-22s %10.2f%% %10.2f%% %10.2f%%",
        "G3 - G1",
        results["summary"]["medqa"]["G3_vs_G1"] * 100,
        results["summary"]["arc"]["G3_vs_G1"] * 100,
        results["summary"]["mmlu"]["G3_vs_G1"] * 100,
    )
    logger.info(
        "%-22s %10.2f%% %10.2f%% %10.2f%%",
        "G4 - G0",
        results["summary"]["medqa"]["G4_delta_over_baseline"] * 100,
        results["summary"]["arc"]["G4_delta_over_baseline"] * 100,
        results["summary"]["mmlu"]["G4_delta_over_baseline"] * 100,
    )

    logger.info("%s", "-" * 60)
    logger.info("%-22s %10s %10s %10s", "efficiency", "MEDQA", "ARC", "MMLU")
    logger.info("%s", "-" * 60)
    logger.info(
        "%-22s %10.1f%% %10.1f%% %10.1f%%",
        "eff_G2",
        results["summary"]["medqa"]["efficiency_G2"] * 100,
        results["summary"]["arc"]["efficiency_G2"] * 100,
        results["summary"]["mmlu"]["efficiency_G2"] * 100,
    )
    logger.info(
        "%-22s %10.1f%% %10.1f%% %10.1f%%",
        "eff_G3",
        results["summary"]["medqa"]["efficiency_G3"] * 100,
        results["summary"]["arc"]["efficiency_G3"] * 100,
        results["summary"]["mmlu"]["efficiency_G3"] * 100,
    )

    logger.info("%s", "-" * 60)
    for ds_name in ("medqa", "arc", "mmlu"):
        summary = results["summary"][ds_name]
        logger.info(
            "%s | G2-G1=%+.2f%% | G3-G1=%+.2f%% | eff_G2=%.1f%% | eff_G3=%.1f%%",
            ds_name.upper(),
            summary["G2_vs_G1"] * 100,
            summary["G3_vs_G1"] * 100,
            summary["efficiency_G2"] * 100,
            summary["efficiency_G3"] * 100,
        )


def run_e3_all(
    cfg: Config,
    phase1_weights: str,
    phase2_weights: str,
    k: int = 64,
    device: str = "cuda:0",
    max_samples: int = -1,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    setup_logging()
    if k not in {32, 64, 128, 256}:
        raise ValueError(f"unsupported k: {k}")
    _log_section("🌍 E3 FAIR COMPARE")
    num_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    logger.info(
        "E3 start | phase1=%s | phase2=%s | k=%d | device=%s | visible_gpus=%d",
        phase1_weights,
        phase2_weights,
        k,
        device,
        num_gpus,
    )

    datasets = {
        "medqa": load_medqa_rows(limit=max_samples),
        "arc": load_arc_rows(limit=max_samples),
        "mmlu": load_mmlu_rows(limit=max_samples),
    }
    knowledge_maps = {
        "medqa": load_knowledge_map(_knowledge_map_path(cfg.eval.medqa_knowledge_map, k)),
        "arc": load_knowledge_map(_knowledge_map_path(cfg.eval.arc_knowledge_map, k)),
        "mmlu": load_knowledge_map(_knowledge_map_path(cfg.eval.mmlu_knowledge_map, k)),
    }
    logger.info(
        "Datasets loaded | MedQA=%d | ARC=%d | MMLU=%d",
        len(datasets["medqa"]),
        len(datasets["arc"]),
        len(datasets["mmlu"]),
    )

    started = time.time()
    results: Dict[str, Any] = {
        "phase1_weights": phase1_weights,
        "phase2_weights": phase2_weights,
        "device": device,
        "num_gpus": num_gpus,
        "max_samples": max_samples,
        "k": k,
        "medqa": {},
        "arc": {},
        "mmlu": {},
    }

    base_task = {"device": device, "knowledge_length": k}

    _log_section("📌 PHASE A: G0 / G1 / G4")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        km = knowledge_maps[ds_name]
        results[ds_name]["G0_baseline"] = _run_parallel(
            cfg,
            {**base_task, "model_type": "baseline", "eval_mode": "G0", "dataset_name": ds_name, "rows": rows},
            num_gpus,
        )
        results[ds_name]["G1_rag_compressed"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "baseline",
                "eval_mode": "G1",
                "dataset_name": ds_name,
                "rows": rows,
                "knowledge_map": km,
            },
            num_gpus,
        )
        results[ds_name]["G4_rag_original"] = _run_parallel(
            cfg,
            {**base_task, "model_type": "baseline", "eval_mode": "G4", "dataset_name": ds_name, "rows": rows},
            num_gpus,
        )

    _log_section("📌 PHASE B: G2")
    for ds_name in ("medqa", "arc", "mmlu"):
        results[ds_name]["G2_fusion_phase1"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "injection",
                "eval_mode": "G2",
                "dataset_name": ds_name,
                "rows": datasets[ds_name],
                "knowledge_map": knowledge_maps[ds_name],
                "weights": phase1_weights,
            },
            num_gpus,
        )

    _log_section("📌 PHASE C: G3")
    for ds_name in ("medqa", "arc", "mmlu"):
        results[ds_name]["G3_fusion_phase2"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "injection",
                "eval_mode": "G3",
                "dataset_name": ds_name,
                "rows": datasets[ds_name],
                "knowledge_map": knowledge_maps[ds_name],
                "weights": phase2_weights,
            },
            num_gpus,
        )

    results["summary"] = {}
    for ds_name in ("medqa", "arc", "mmlu"):
        g0 = results[ds_name]["G0_baseline"]["acc"]
        g1 = results[ds_name]["G1_rag_compressed"]["acc"]
        g2 = results[ds_name]["G2_fusion_phase1"]["acc"]
        g3 = results[ds_name]["G3_fusion_phase2"]["acc"]
        g4 = results[ds_name]["G4_rag_original"]["acc"]
        delta_g4 = g4 - g0
        results["summary"][ds_name] = {
            "G2_vs_G1": g2 - g1,
            "G3_vs_G1": g3 - g1,
            "G2_delta_over_baseline": g2 - g0,
            "G3_delta_over_baseline": g3 - g0,
            "G4_delta_over_baseline": delta_g4,
            "efficiency_G2": (g2 - g0) / delta_g4 if delta_g4 > 0 else 0.0,
            "efficiency_G3": (g3 - g0) / delta_g4 if delta_g4 > 0 else 0.0,
        }

    results["elapsed_sec"] = time.time() - started
    _print_report(results)

    if output_path is None:
        output_path = str(PROJECT_ROOT / cfg.paths.results_dir / "e3" / f"e3_fair_compare_k{k}.json")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("E3 finished | elapsed_sec=%.2f | output=%s", results["elapsed_sec"], output_file)
    return results
