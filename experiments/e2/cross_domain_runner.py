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
from experiments.e2.arc_eval import eval_arc, load_arc_examples
from experiments.e2.common import (
    PROJECT_ROOT,
    build_baseline_model,
    build_injection_model,
    load_knowledge_map,
    setup_logging,
)
from experiments.e2.medqa_eval import eval_medqa_baseline, eval_medqa_injection, load_medqa_examples
from experiments.e2.mmlu_eval import eval_mmlu, load_mmlu_examples

logger = logging.getLogger(__name__)


def _log_section(title: str) -> None:
    line = "=" * 18
    logger.info("%s %s %s", line, title, line)


def _run_task_with_model(
    task_spec: Dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    device: torch.device,
) -> Dict[str, Any]:
    dataset_name = task_spec["dataset_name"]
    rows = task_spec["rows"]
    knowledge_map = task_spec.get("knowledge_map")
    knowledge_length = task_spec["knowledge_length"]
    is_injection = task_spec["model_type"] == "injection"
    show_progress = task_spec.get("show_progress", True)

    if dataset_name == "medqa":
        if is_injection:
            return eval_medqa_injection(
                model,
                tokenizer,
                rows,
                device,
                knowledge_map=knowledge_map,
                knowledge_length=knowledge_length,
                show_progress=show_progress,
            )
        return eval_medqa_baseline(model, tokenizer, rows, device, show_progress=show_progress)

    if dataset_name == "arc":
        return eval_arc(
            model,
            tokenizer,
            rows,
            device,
            knowledge_map=knowledge_map,
            knowledge_length=knowledge_length,
            is_injection=is_injection,
            show_progress=show_progress,
        )

    if dataset_name == "mmlu":
        return eval_mmlu(
            model,
            tokenizer,
            rows,
            device,
            knowledge_map=knowledge_map,
            knowledge_length=knowledge_length,
            is_injection=is_injection,
            show_progress=show_progress,
        )

    raise ValueError(f"unsupported dataset: {dataset_name}")


def _load_task_model(
    cfg: Config,
    task_spec: Dict[str, Any],
    device: str,
) -> tuple[torch.nn.Module, Any]:
    if task_spec["model_type"] == "baseline":
        return build_baseline_model(cfg, device=device)
    return build_injection_model(cfg, fusion_ckpt=task_spec["fusion_ckpt"], device=device)


def _eval_worker(
    rank: int,
    world_size: int,
    cfg: Config,
    task_spec: Dict[str, Any],
    tmp_dir: str,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    rows = task_spec["rows"]
    total = len(rows)
    per_shard = total // world_size
    start = rank * per_shard
    end = start + per_shard if rank < world_size - 1 else total
    shard_rows = rows[start:end]
    shard_spec = {**task_spec, "rows": shard_rows, "show_progress": rank == 0}

    logger.info(
        "[GPU %d/%d] %s | %s | shard [%d:%d] | rows=%d",
        rank,
        world_size,
        task_spec["dataset_name"],
        task_spec["model_type"],
        start,
        end,
        len(shard_rows),
    )

    if task_spec.get("knowledge_map") is None:
        logger.info(
            "[GPU %d/%d] %s | %s | using empty knowledge",
            rank,
            world_size,
            task_spec["dataset_name"],
            task_spec["model_type"],
        )
    else:
        logger.info(
            "[GPU %d/%d] %s | %s | knowledge map loaded (%d entries)",
            rank,
            world_size,
            task_spec["dataset_name"],
            task_spec["model_type"],
            len(task_spec["knowledge_map"]),
        )

    logger.info(
        "[GPU %d/%d] %s | %s | loading model on %s",
        rank,
        world_size,
        task_spec["dataset_name"],
        task_spec["model_type"],
        device,
    )

    model, tokenizer = _load_task_model(cfg, task_spec, device=str(device))
    result = _run_task_with_model(shard_spec, model, tokenizer, device)

    shard_path = Path(tmp_dir) / f"shard_{rank}.json"
    shard_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "[GPU %d/%d] %s | %s | shard done | result=%s",
        rank,
        world_size,
        task_spec["dataset_name"],
        task_spec["model_type"],
        result,
    )

    del model
    torch.cuda.empty_cache()


def _merge_shard_results(
    tmp_dir: str,
    num_shards: int,
) -> Dict[str, Any]:
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
    if any("skipped" in s for s in shards):
        result["skipped"] = sum(s.get("skipped", 0) for s in shards)
    logger.info("Merged %d shard results -> %s", num_shards, result)
    return result


def _run_parallel(
    cfg: Config,
    task_spec: Dict[str, Any],
    num_gpus: int,
) -> Dict[str, Any]:
    if num_gpus <= 1:
        device = torch.device(task_spec["device"])
        _log_section(f"🚀 {task_spec['dataset_name'].upper()} / {task_spec['model_type']} / single-process")
        logger.info(
            "%s | %s | single-process run on %s | rows=%d",
            task_spec["dataset_name"],
            task_spec["model_type"],
            device,
            len(task_spec["rows"]),
        )
        model, tokenizer = _load_task_model(cfg, task_spec, device=str(device))
        try:
            result = _run_task_with_model(task_spec, model, tokenizer, device)
            logger.info(
                "%s | %s | completed on %s | result=%s",
                task_spec["dataset_name"],
                task_spec["model_type"],
                device,
                result,
            )
            return result
        finally:
            del model
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()

    tmp_dir = tempfile.mkdtemp(prefix=f"e2_{task_spec['dataset_name']}_")
    _log_section(f"🚀 {task_spec['dataset_name'].upper()} / {task_spec['model_type']} / multi-gpu")
    logger.info(
        "%s | %s | multi-GPU run | gpus=%d | rows=%d | tmp=%s",
        task_spec["dataset_name"],
        task_spec["model_type"],
        num_gpus,
        len(task_spec["rows"]),
        tmp_dir,
    )
    mp.spawn(
        _eval_worker,
        args=(num_gpus, cfg, task_spec, tmp_dir),
        nprocs=num_gpus,
        join=True,
    )
    return _merge_shard_results(tmp_dir, num_gpus)


def run_e2_all(
    cfg: Config,
    phase2_ckpt: str,
    phase3_ckpt: str,
    device: str = "cuda:0",
    max_samples: int = -1,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    setup_logging()
    _log_section("🌍 E2 CROSS-DOMAIN EVALUATION")
    num_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0
    logger.info(
        "E2 start | phase2=%s | phase3=%s | device=%s | visible_gpus=%d",
        phase2_ckpt,
        phase3_ckpt,
        device,
        num_gpus,
    )

    medqa_rows = load_medqa_examples(limit=max_samples)
    arc_rows = load_arc_examples(limit=max_samples)
    mmlu_rows = load_mmlu_examples(limit=max_samples)

    medqa_km = load_knowledge_map(str(PROJECT_ROOT / cfg.eval.medqa_knowledge_map))
    arc_km = load_knowledge_map(str(PROJECT_ROOT / cfg.eval.arc_knowledge_map))
    mmlu_km = load_knowledge_map(str(PROJECT_ROOT / cfg.eval.mmlu_knowledge_map))

    logger.info(
        "Datasets loaded | MedQA=%d | ARC=%d | MMLU=%d",
        len(medqa_rows),
        len(arc_rows),
        len(mmlu_rows),
    )
    logger.info(
        "Knowledge maps loaded | MedQA=%d | ARC=%d | MMLU=%d",
        len(medqa_km),
        len(arc_km),
        len(mmlu_km),
    )

    started = time.time()
    results: Dict[str, Any] = {
        "phase2_weights": phase2_ckpt,
        "phase3_weights": phase3_ckpt,
        "device": device,
        "num_gpus": num_gpus,
        "max_samples": max_samples,
    }

    def make_base_task(fusion_ckpt: str) -> Dict[str, Any]:
        return {
            "device": device,
            "fusion_ckpt": fusion_ckpt,
            "knowledge_length": cfg.model.fusion_length,
        }

    def run_baseline_and_fusions(
        dataset_name: str,
        rows: List[Dict[str, Any]],
        knowledge_map: Dict[str, List[int]],
    ) -> Dict[str, Any]:
        baseline = _run_parallel(
            cfg,
            {
                **make_base_task(phase2_ckpt),
                "model_type": "baseline",
                "dataset_name": dataset_name,
                "rows": rows,
                "knowledge_map": None,
            },
            num_gpus,
        )

        phase2 = {
            "fusion_knowledge": _run_parallel(
                cfg,
                {
                    **make_base_task(phase2_ckpt),
                    "model_type": "injection",
                    "dataset_name": dataset_name,
                    "rows": rows,
                    "knowledge_map": knowledge_map,
                },
                num_gpus,
            ),
            "fusion_empty": _run_parallel(
                cfg,
                {
                    **make_base_task(phase2_ckpt),
                    "model_type": "injection",
                    "dataset_name": dataset_name,
                    "rows": rows,
                    "knowledge_map": None,
                },
                num_gpus,
            ),
        }

        phase3 = {
            "fusion_knowledge": _run_parallel(
                cfg,
                {
                    **make_base_task(phase3_ckpt),
                    "model_type": "injection",
                    "dataset_name": dataset_name,
                    "rows": rows,
                    "knowledge_map": knowledge_map,
                },
                num_gpus,
            ),
            "fusion_empty": _run_parallel(
                cfg,
                {
                    **make_base_task(phase3_ckpt),
                    "model_type": "injection",
                    "dataset_name": dataset_name,
                    "rows": rows,
                    "knowledge_map": None,
                },
                num_gpus,
            ),
        }

        for phase_result in (phase2, phase3):
            baseline_acc = baseline["acc"]
            fusion_acc = phase_result["fusion_knowledge"]["acc"]
            empty_acc = phase_result["fusion_empty"]["acc"]
            phase_result["delta_acc"] = fusion_acc - baseline_acc
            phase_result["delta_acc_empty"] = empty_acc - baseline_acc

        return {
            "baseline": baseline,
            "phase2": phase2,
            "phase3": phase3,
            "phase3_vs_phase2": phase3["fusion_knowledge"]["acc"] - phase2["fusion_knowledge"]["acc"],
        }

    _log_section("🩺 MEDQA")
    results["medqa"] = run_baseline_and_fusions("medqa", medqa_rows, medqa_km)

    _log_section("🧪 ARC")
    results["arc"] = run_baseline_and_fusions("arc", arc_rows, arc_km)

    _log_section("📚 MMLU")
    results["mmlu"] = run_baseline_and_fusions("mmlu", mmlu_rows, mmlu_km)

    for ds_name in ("medqa", "arc", "mmlu"):
        phase2 = results[ds_name]["phase2"]
        phase3 = results[ds_name]["phase3"]
        logger.info(
            "%s phase2 | baseline=%.4f | fusion=%.4f | empty=%.4f | delta=%+.4f | delta_empty=%+.4f",
            ds_name.upper(),
            results[ds_name]["baseline"]["acc"],
            phase2["fusion_knowledge"]["acc"],
            phase2["fusion_empty"]["acc"],
            phase2["delta_acc"],
            phase2["delta_acc_empty"],
        )
        logger.info(
            "%s phase3 | baseline=%.4f | fusion=%.4f | empty=%.4f | delta=%+.4f | delta_empty=%+.4f | vs_phase2=%+.4f",
            ds_name.upper(),
            results[ds_name]["baseline"]["acc"],
            phase3["fusion_knowledge"]["acc"],
            phase3["fusion_empty"]["acc"],
            phase3["delta_acc"],
            phase3["delta_acc_empty"],
            results[ds_name]["phase3_vs_phase2"],
        )

    results["elapsed_sec"] = time.time() - started
    _log_section("📊 E2 SUMMARY")

    if output_path is None:
        phase2_tag = Path(phase2_ckpt).parent.name + "_" + Path(phase2_ckpt).name
        phase3_tag = Path(phase3_ckpt).parent.name + "_" + Path(phase3_ckpt).name
        output_path = str(
            PROJECT_ROOT / cfg.paths.results_dir / "e2" / f"e2_cross_domain_{phase2_tag}__{phase3_tag}.json"
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("E2 finished | elapsed_sec=%.2f | output=%s", results["elapsed_sec"], output_file)
    return results
