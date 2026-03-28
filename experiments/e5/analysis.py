from __future__ import annotations

import json
import logging
import random
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from config import Config
from experiments.e1.counterfactual_eval import KnowledgeCompressor
from experiments.e2.common import (
    PROJECT_ROOT,
    build_baseline_model,
    build_injection_model,
    get_model_path,
    load_knowledge_map,
    setup_logging,
)
from experiments.e3.data_loading import load_arc_rows, load_medqa_rows, load_mmlu_rows
from experiments.e3.evaluator import eval_baseline, eval_fusion, eval_rag_compressed

logger = logging.getLogger(__name__)

_E5_TOKEN_BUDGETS = {
    32: (32, 0.125, True),
    64: (64, 0.25, True),
    128: (128, 0.5, True),
    256: (256, None, False),
}


def _log_section(title: str) -> None:
    line = "=" * 16
    logger.info("%s %s %s", line, title, line)


def _km_rel_path(dataset_name: str, k: int) -> Path:
    if k == 64:
        return Path("data") / f"{dataset_name}_knowledge.jsonl"
    return Path("data") / f"{dataset_name}_knowledge_k{k}.jsonl"


def _load_rows(dataset_name: str, limit: int = -1) -> List[Dict[str, Any]]:
    if dataset_name == "medqa":
        return load_medqa_rows(limit=None if limit < 0 else limit)
    if dataset_name == "arc":
        return load_arc_rows(limit=None if limit < 0 else limit)
    if dataset_name == "mmlu":
        return load_mmlu_rows(limit=None if limit < 0 else limit)
    raise ValueError(f"unsupported dataset: {dataset_name}")


def _load_model(cfg: Config, task_spec: Dict[str, Any], device: str):
    if task_spec["model_type"] == "baseline":
        return build_baseline_model(cfg, device=device)
    return build_injection_model(
        cfg,
        fusion_ckpt=task_spec["weights"],
        device=device,
        log_prefix="E5Load",
    )


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
    if mode == "rag":
        return eval_rag_compressed(
            model,
            tokenizer,
            rows,
            device,
            dataset_name,
            knowledge_map=task_spec["knowledge_map"],
            show_progress=show_progress,
        )
    if mode in {"phase1", "phase2", "oracle_p1", "oracle_p2", "shuffled_p1", "shuffled_p2", "empty_p1", "empty_p2"}:
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
    if any("missing_knowledge" in s for s in shards):
        result["missing_knowledge"] = sum(s.get("missing_knowledge", 0) for s in shards)
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

    tmp_dir = tempfile.mkdtemp(prefix=f"e5_{task_spec['dataset_name']}_{task_spec['eval_mode']}_")
    mp.spawn(_worker, args=(num_gpus, cfg, task_spec, tmp_dir), nprocs=num_gpus, join=True)
    return _merge_shards(tmp_dir, num_gpus)


def _build_shuffled_km(knowledge_map: Dict[str, List[int]], seed: int = 42) -> Dict[str, List[int]]:
    keys = list(knowledge_map.keys())
    values = list(knowledge_map.values())
    rng = random.Random(seed)
    rng.shuffle(values)
    return dict(zip(keys, values))


def _build_knowledge_at_rate(
    cfg: Config,
    dataset_name: str,
    knowledge_length: int,
    compression_rate: float,
    output_path: Path,
    max_samples: int = -1,
) -> Dict[str, List[int]]:
    model_path = get_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compressor_gpu = int(torch.cuda.current_device()) if torch.cuda.is_available() else None
    compressor = KnowledgeCompressor(
        model_name=cfg.paths.llmlingua_model_dir,
        compression_rate=compression_rate,
        gpu_id=compressor_gpu,
    )

    rows = _load_rows(dataset_name, limit=max_samples)
    result: Dict[str, List[int]] = {}
    logger.info(
        "E5 build | %s | k=%d | compression_rate=%.3f | rows=%d",
        dataset_name,
        knowledge_length,
        compression_rate,
        len(rows),
    )
    for row in rows:
        source_text = row["original_text"]
        compressed = compressor.compress_text(source_text)
        if compressed is None:
            compressed = row["question"][:100]

        token_ids = tokenizer.encode(compressed, add_special_tokens=False)[:knowledge_length]
        if len(token_ids) < knowledge_length:
            token_ids.extend([tokenizer.pad_token_id] * (knowledge_length - len(token_ids)))
        result[row["key"]] = token_ids

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for key, ids in result.items():
            f.write(json.dumps({"key": key, "knowledge_ids": ids}, ensure_ascii=False) + "\n")
    logger.info("E5 build done | %s | count=%d", output_path, len(result))
    return result


def _build_knowledge_tokenize_only(
    cfg: Config,
    dataset_name: str,
    knowledge_length: int,
    output_path: Path,
    max_samples: int = -1,
) -> Dict[str, List[int]]:
    model_path = get_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows = _load_rows(dataset_name, limit=max_samples)
    result: Dict[str, List[int]] = {}
    logger.info("E5 build | %s | k=%d | tokenize-only | rows=%d", dataset_name, knowledge_length, len(rows))
    for row in rows:
        token_ids = tokenizer.encode(row["original_text"], add_special_tokens=False)[:knowledge_length]
        if len(token_ids) < knowledge_length:
            token_ids.extend([tokenizer.pad_token_id] * (knowledge_length - len(token_ids)))
        result[row["key"]] = token_ids

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for key, ids in result.items():
            f.write(json.dumps({"key": key, "knowledge_ids": ids}, ensure_ascii=False) + "\n")
    logger.info("E5 build done | %s | count=%d", output_path, len(result))
    return result


def build_e5_knowledge_all(cfg: Config, max_samples: int = -1, rebuild: bool = False) -> None:
    setup_logging()
    _log_section("🧱 E5 Build Knowledge")
    for dataset_name in ("medqa", "arc", "mmlu"):
        for k, (knowledge_length, compression_rate, use_compressor) in _E5_TOKEN_BUDGETS.items():
            rel_path = _km_rel_path(dataset_name, k)
            output_path = PROJECT_ROOT / rel_path
            if output_path.exists() and not rebuild:
                logger.info("skip existing knowledge map: %s", output_path)
                continue
            if use_compressor:
                _build_knowledge_at_rate(
                    cfg,
                    dataset_name,
                    knowledge_length=knowledge_length,
                    compression_rate=float(compression_rate),
                    output_path=output_path,
                    max_samples=max_samples,
                )
            else:
                _build_knowledge_tokenize_only(
                    cfg,
                    dataset_name,
                    knowledge_length=knowledge_length,
                    output_path=output_path,
                    max_samples=max_samples,
                )


def _run_e5a_token_budget(
    cfg: Config,
    phase1_weights: str,
    phase2_weights: str,
    device: str,
    max_samples: int,
    num_gpus: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    datasets = {ds: _load_rows(ds, limit=max_samples) for ds in ("medqa", "arc", "mmlu")}
    all_kms = {
        ds: {k: load_knowledge_map(str(PROJECT_ROOT / _km_rel_path(ds, k))) for k in (32, 64, 128, 256)}
        for ds in ("medqa", "arc", "mmlu")
    }

    base_task = {"device": device}

    _log_section("📌 E5-A Baseline + RAG")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        results.setdefault(ds_name, {})
        results[ds_name]["baseline"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "baseline",
                "eval_mode": "baseline",
                "dataset_name": ds_name,
                "rows": rows,
            },
            num_gpus,
        )
        for k in (32, 64, 128, 256):
            results[ds_name][f"rag_k{k}"] = _run_parallel(
                cfg,
                {
                    **base_task,
                    "model_type": "baseline",
                    "eval_mode": "rag",
                    "dataset_name": ds_name,
                    "rows": rows,
                    "knowledge_map": all_kms[ds_name][k],
                },
                num_gpus,
            )

    _log_section("📌 E5-A Fusion Phase1")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        for k in (32, 64, 128, 256):
            results[ds_name][f"fusion_p1_k{k}"] = _run_parallel(
                cfg,
                {
                    **base_task,
                    "model_type": "injection",
                    "weights": phase1_weights,
                    "eval_mode": "phase1",
                    "dataset_name": ds_name,
                    "rows": rows,
                    "knowledge_map": all_kms[ds_name][k],
                    "knowledge_length": k,
                },
                num_gpus,
            )

    _log_section("📌 E5-A Fusion Phase2")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        for k in (32, 64, 128, 256):
            results[ds_name][f"fusion_p2_k{k}"] = _run_parallel(
                cfg,
                {
                    **base_task,
                    "model_type": "injection",
                    "weights": phase2_weights,
                    "eval_mode": "phase2",
                    "dataset_name": ds_name,
                    "rows": rows,
                    "knowledge_map": all_kms[ds_name][k],
                    "knowledge_length": k,
                },
                num_gpus,
            )
    return results


def _run_e5b_relevance(
    cfg: Config,
    phase1_weights: str,
    phase2_weights: str,
    e5a_results: Dict[str, Any],
    device: str,
    max_samples: int,
    num_gpus: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    datasets = {ds: _load_rows(ds, limit=max_samples) for ds in ("medqa", "arc", "mmlu")}
    kms_64 = {ds: load_knowledge_map(str(PROJECT_ROOT / _km_rel_path(ds, 64))) for ds in ("medqa", "arc", "mmlu")}
    kms_shuffled = {ds: _build_shuffled_km(kms_64[ds]) for ds in ("medqa", "arc", "mmlu")}

    base_task = {"device": device, "knowledge_length": 64}
    for ds_name in ("medqa", "arc", "mmlu"):
        results.setdefault(ds_name, {})
        results[ds_name]["oracle_p1"] = e5a_results[ds_name]["fusion_p1_k64"]
        results[ds_name]["oracle_p2"] = e5a_results[ds_name]["fusion_p2_k64"]

    _log_section("📌 E5-B Phase1")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        results[ds_name]["shuffled_p1"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "injection",
                "weights": phase1_weights,
                "eval_mode": "shuffled_p1",
                "dataset_name": ds_name,
                "rows": rows,
                "knowledge_map": kms_shuffled[ds_name],
            },
            num_gpus,
        )
        results[ds_name]["empty_p1"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "injection",
                "weights": phase1_weights,
                "eval_mode": "empty_p1",
                "dataset_name": ds_name,
                "rows": rows,
                "knowledge_map": {},
            },
            num_gpus,
        )

    _log_section("📌 E5-B Phase2")
    for ds_name in ("medqa", "arc", "mmlu"):
        rows = datasets[ds_name]
        results[ds_name]["shuffled_p2"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "injection",
                "weights": phase2_weights,
                "eval_mode": "shuffled_p2",
                "dataset_name": ds_name,
                "rows": rows,
                "knowledge_map": kms_shuffled[ds_name],
            },
            num_gpus,
        )
        results[ds_name]["empty_p2"] = _run_parallel(
            cfg,
            {
                **base_task,
                "model_type": "injection",
                "weights": phase2_weights,
                "eval_mode": "empty_p2",
                "dataset_name": ds_name,
                "rows": rows,
                "knowledge_map": {},
            },
            num_gpus,
        )
    return results


def _print_e5_report(e5a: Dict[str, Any], e5b: Dict[str, Any]) -> None:
    sep = "=" * 80
    for phase, label in (("p1", "Phase 1"), ("p2", "Phase 2")):
        for ds_name in ("medqa", "arc", "mmlu"):
            logger.info("\n%s", sep)
            logger.info("E5-A Token Budget | %s | %s", label, ds_name.upper())
            logger.info("%s", sep)
            baseline_acc = e5a[ds_name]["baseline"]["acc"]
            logger.info("%-8s %10s %10s %10s %15s", "Token", "Baseline", "Fusion", "RAG", "Δ(Fusion-RAG)")
            logger.info("%s", "-" * 60)
            for k in (32, 64, 128, 256):
                fusion_acc = e5a[ds_name][f"fusion_{phase}_k{k}"]["acc"]
                rag_acc = e5a[ds_name][f"rag_k{k}"]["acc"]
                logger.info(
                    "%-8d %10.2f%% %10.2f%% %10.2f%% %15s",
                    k,
                    baseline_acc * 100,
                    fusion_acc * 100,
                    rag_acc * 100,
                    f"{(fusion_acc - rag_acc) * 100:+.2f}%",
                )

    for phase, label in (("p1", "Phase 1"), ("p2", "Phase 2")):
        for ds_name in ("medqa", "arc", "mmlu"):
            logger.info("\n%s", sep)
            logger.info("E5-B Relevance | %s | %s | k=64", label, ds_name.upper())
            logger.info("%s", sep)
            oracle_acc = e5b[ds_name][f"oracle_{phase}"]["acc"]
            logger.info("%-12s %10s %15s", "Condition", "Acc", "Δ(vs Oracle)")
            logger.info("%s", "-" * 40)
            for cond, cond_label in (
                (f"oracle_{phase}", "Oracle"),
                (f"shuffled_{phase}", "Shuffled"),
                (f"empty_{phase}", "Empty"),
            ):
                acc = e5b[ds_name][cond]["acc"]
                if cond.startswith("oracle"):
                    delta_str = f"{'—':>15}"
                else:
                    delta_str = f"{(acc - oracle_acc):>+15.2%}"
                logger.info("%-12s %10.2f%% %s", cond_label, acc * 100, delta_str)


def run_e5_all(
    cfg: Config,
    phase1_weights: str,
    phase2_weights: str,
    device: str = "cuda:0",
    max_samples: int = -1,
    output_path: Optional[str] = None,
    build_missing: bool = False,
) -> Dict[str, Any]:
    setup_logging()
    _log_section("📊 E5 KNOWLEDGE ANALYSIS")
    started = time.time()
    num_gpus = torch.cuda.device_count() if str(device).startswith("cuda") else 0

    if build_missing:
        build_e5_knowledge_all(cfg, max_samples=max_samples, rebuild=False)

    for dataset_name in ("medqa", "arc", "mmlu"):
        for k in (32, 64, 128, 256):
            path = PROJECT_ROOT / _km_rel_path(dataset_name, k)
            if not path.exists():
                raise FileNotFoundError(f"knowledge map missing: {path}")

    e5a_results = _run_e5a_token_budget(
        cfg=cfg,
        phase1_weights=phase1_weights,
        phase2_weights=phase2_weights,
        device=device,
        max_samples=max_samples,
        num_gpus=num_gpus,
    )
    e5b_results = _run_e5b_relevance(
        cfg=cfg,
        phase1_weights=phase1_weights,
        phase2_weights=phase2_weights,
        e5a_results=e5a_results,
        device=device,
        max_samples=max_samples,
        num_gpus=num_gpus,
    )
    _print_e5_report(e5a_results, e5b_results)

    results = {
        "experiment": "E5_knowledge_analysis",
        "phase1_weights": phase1_weights,
        "phase2_weights": phase2_weights,
        "device": device,
        "num_gpus": num_gpus,
        "max_samples": max_samples,
        "e5a": e5a_results,
        "e5b": e5b_results,
        "elapsed_sec": time.time() - started,
    }

    if output_path is None:
        out = PROJECT_ROOT / cfg.paths.results_dir / "e5" / "e5_knowledge_analysis.json"
    else:
        out = Path(output_path)
        if not out.is_absolute():
            out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("E5 finished | elapsed_sec=%.2f | output=%s", results["elapsed_sec"], out)
    return results
