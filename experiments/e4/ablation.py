from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

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
from experiments.e3.evaluator import eval_baseline, eval_fusion

logger = logging.getLogger(__name__)


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

    if mode == "baseline":
        return eval_baseline(model, tokenizer, rows, device, dataset_name, show_progress=show_progress)
    if mode in {"phase1", "phase2"}:
        return eval_fusion(
            model,
            tokenizer,
            rows,
            device,
            dataset_name,
            knowledge_map=task_spec["knowledge_map"],
            group_name=mode.upper(),
            knowledge_length=task_spec["knowledge_length"],
            show_progress=show_progress,
        )
    raise ValueError(f"unsupported eval mode: {mode}")


def _load_model(cfg: Config, task_spec: Dict[str, Any], device: str):
    if task_spec["model_type"] == "baseline":
        return build_baseline_model(cfg, device=device)
    return build_injection_model(
        cfg,
        fusion_ckpt=task_spec["weights"],
        device=device,
        log_prefix="E4Load",
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
    if any("knowledge_miss_count" in s for s in shards):
        result["knowledge_miss_count"] = sum(s.get("knowledge_miss_count", 0) for s in shards)
    return result


def _run_parallel(cfg: Config, task_spec: Dict[str, Any], num_gpus: int) -> Dict[str, Any]:
    if num_gpus <= 1:
        device = torch.device(task_spec["device"])
        model, tokenizer = _load_model(cfg, task_spec, str(device))
        try:
            return _run_group(task_spec, model, tokenizer, device)
        finally:
            del model
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

    tmp_dir = tempfile.mkdtemp(prefix=f"e4_{task_spec['dataset_name']}_{task_spec['eval_mode']}_")
    mp.spawn(_worker, args=(num_gpus, cfg, task_spec, tmp_dir), nprocs=num_gpus, join=True)
    return _merge_shards(tmp_dir, num_gpus)


def _build_ablation(results: Dict[str, Any]) -> Dict[str, Any]:
    ablation: Dict[str, Any] = {}
    for ds_name in ("medqa", "arc", "mmlu"):
        base_acc = results[ds_name]["baseline"]["acc"]
        p1_acc = results[ds_name]["phase1"]["acc"]
        p2_acc = results[ds_name]["phase2"]["acc"]
        ablation[ds_name] = {
            "baseline_acc": base_acc,
            "phase1_acc": p1_acc,
            "phase2_acc": p2_acc,
            "phase1_delta": p1_acc - base_acc,
            "phase2_delta": p2_acc - base_acc,
            "sft_effect": p2_acc - p1_acc,
        }
    return ablation


def _print_report(results: Dict[str, Any]) -> None:
    _log_section("📊 E4 SUMMARY")
    logger.info(
        "%-10s %10s %10s %10s %12s %12s %10s",
        "dataset",
        "Baseline",
        "Phase1",
        "Phase2",
        "Δ(P1-Base)",
        "Δ(P2-Base)",
        "SFT effect",
    )
    logger.info("%s", "-" * 84)
    for ds_name, label in (("medqa", "MedQA"), ("arc", "ARC"), ("mmlu", "MMLU")):
        ab = results["ablation"][ds_name]
        logger.info(
            "%-10s %10.2f%% %10.2f%% %10.2f%% %12+.2f%% %12+.2f%% %10+.2f%%",
            label,
            ab["baseline_acc"] * 100,
            ab["phase1_acc"] * 100,
            ab["phase2_acc"] * 100,
            ab["phase1_delta"] * 100,
            ab["phase2_delta"] * 100,
            ab["sft_effect"] * 100,
        )


def run_e4_all(
    cfg: Config,
    phase1_weights: str,
    phase2_weights: str,
    device: str = "cuda:0",
    max_samples: int = -1,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    setup_logging()
    num_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    started = time.time()

    datasets = {
        "medqa": load_medqa_rows(limit=max_samples),
        "arc": load_arc_rows(limit=max_samples),
        "mmlu": load_mmlu_rows(limit=max_samples),
    }
    knowledge_maps = {
        "medqa": load_knowledge_map(str(PROJECT_ROOT / cfg.eval.medqa_knowledge_map)),
        "arc": load_knowledge_map(str(PROJECT_ROOT / cfg.eval.arc_knowledge_map)),
        "mmlu": load_knowledge_map(str(PROJECT_ROOT / cfg.eval.mmlu_knowledge_map)),
    }

    results: Dict[str, Any] = {
        "phase1_weights": phase1_weights,
        "phase2_weights": phase2_weights,
        "device": device,
        "num_gpus": num_gpus,
        "max_samples": max_samples,
        "medqa": {},
        "arc": {},
        "mmlu": {},
    }

    base_task = {"device": device, "knowledge_length": cfg.model.fusion_length}

    _log_section("📌 E4 Baseline")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        results[ds_name]["baseline"] = _run_parallel(
            cfg,
            {**base_task, "model_type": "baseline", "eval_mode": "baseline", "dataset_name": ds_name, "rows": rows},
            num_gpus,
        )

    _log_section("📌 E4 Phase1")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        km = knowledge_maps[ds_name]
        results[ds_name]["phase1"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "injection",
                "weights": phase1_weights,
                "eval_mode": "phase1",
                "dataset_name": ds_name,
                "rows": rows,
                "knowledge_map": km,
            },
            num_gpus,
        )

    _log_section("📌 E4 Phase2")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        km = knowledge_maps[ds_name]
        results[ds_name]["phase2"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "injection",
                "weights": phase2_weights,
                "eval_mode": "phase2",
                "dataset_name": ds_name,
                "rows": rows,
                "knowledge_map": km,
            },
            num_gpus,
        )

    results["ablation"] = _build_ablation(results)
    results["elapsed_sec"] = time.time() - started
    _print_report(results)

    if output_path is None:
        out = PROJECT_ROOT / cfg.paths.results_dir / "e4" / "e4_sft_ablation.json"
    else:
        out = Path(output_path)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("E4 finished | elapsed_sec=%.2f | output=%s", results["elapsed_sec"], out)
    return results
