"""E2 跨域通用能力验证 — 评测编排器。

在 MedQA / ARC-Challenge / MMLU 三个数据集上运行三组对比：
  A) Baseline（原始 Qwen3-0.6B，无注入模块）
  B) Fusion + 知识（ModifiedQwen + injection_weights + 对应知识）
  C) Fusion + 空知识（ModifiedQwen + injection_weights + 全 pad）

并行化设计: 每个评测步骤内部通过 mp.spawn 利用多块 GPU 数据并行。
单 GPU 时自动退化为串行执行（与旧版行为一致）。
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compare_eval import _build_question_prompt, _score_choices
from counterfactual_eval import _score_choices_injection
from medqa_knowledge_builder import MedQAKnowledgeBuilder
from arc_eval import eval_arc
from mmlu_eval import eval_mmlu
from logger_system import log_msg


# ═══════════════════════════════════════════════════════════════════
# 单数据集评测函数（不含并行逻辑，worker 内部调用）
# ═══════════════════════════════════════════════════════════════════


def _eval_medqa_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """MedQA baseline 评测（原始 Qwen3，无知识）。

    Args:
        model: AutoModelForCausalLM
        tokenizer: tokenizer
        ds: MedQA test dataset
        device: 计算设备

    Returns:
        {"acc": float, "correct": int, "total": int}
    """
    total = len(ds)
    correct = 0

    log_msg("INFO", f"MedQA baseline 评测开始 | 总数={total}")

    with torch.no_grad():
        for i, row in enumerate(ds):
            context = _build_question_prompt(row)
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            pred = _score_choices(model, tokenizer, context_ids, device)
            if pred == row["label"]:
                correct += 1
            if (i + 1) % 200 == 0:
                log_msg("INFO", f"  MedQA baseline 进度: {i+1}/{total}, acc={correct/(i+1):.4f}")

    acc = correct / total if total > 0 else 0.0
    log_msg("INFO", f"MedQA baseline 完成 | acc={acc:.4f} ({correct}/{total})")
    return {"acc": acc, "correct": correct, "total": total}


def _eval_medqa_injection(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    device: torch.device,
    knowledge_map: Dict[str, List[int]],
    knowledge_length: int = 64,
    use_knowledge: bool = True,
) -> Dict[str, Any]:
    """MedQA injection 评测（ModifiedQwen + 知识/空知识）。

    Args:
        model: ModifiedQwen
        tokenizer: tokenizer
        ds: MedQA test dataset
        device: 计算设备
        knowledge_map: 知识映射
        knowledge_length: 知识 token 长度
        use_knowledge: True=使用知识映射, False=全 pad

    Returns:
        {"acc": float, "correct": int, "total": int}
    """
    pad_id = tokenizer.pad_token_id
    pad_knowledge = [pad_id] * knowledge_length
    total = len(ds)
    correct = 0
    mode_str = "Fusion+知识" if use_knowledge else "Fusion+空知识"

    log_msg("INFO", f"MedQA {mode_str} 评测开始 | 总数={total}")

    with torch.no_grad():
        for i, row in enumerate(ds):
            key = row["sent1"][:200].strip()
            context = _build_question_prompt(row)
            context_ids = tokenizer.encode(context, add_special_tokens=False)

            if use_knowledge:
                k_ids = knowledge_map.get(key, pad_knowledge)
            else:
                k_ids = pad_knowledge

            k_tensor = torch.tensor([k_ids], dtype=torch.long, device=device)
            pred = _score_choices_injection(
                model, tokenizer, context_ids, k_tensor, device
            )
            if pred == row["label"]:
                correct += 1

            if (i + 1) % 200 == 0:
                log_msg("INFO", f"  MedQA {mode_str} 进度: {i+1}/{total}, acc={correct/(i+1):.4f}")

    acc = correct / total if total > 0 else 0.0
    log_msg("INFO", f"MedQA {mode_str} 完成 | acc={acc:.4f} ({correct}/{total})")
    return {"acc": acc, "correct": correct, "total": total}


# ═══════════════════════════════════════════════════════════════════
# 多 GPU 并行化核心
# ═══════════════════════════════════════════════════════════════════


def _eval_worker(
    rank: int,
    world_size: int,
    task_spec: Dict[str, Any],
    tmp_dir: str,
) -> None:
    """mp.spawn worker — 在 rank 对应的 GPU 上评测数据分片。

    top-level 函数（mp.spawn 要求 picklable）。

    Args:
        rank: GPU 序号（0-indexed，自动映射到 CUDA_VISIBLE_DEVICES 中的设备）
        world_size: 总 GPU 数
        task_spec: 任务规格字典，字段见 _run_parallel 文档
        tmp_dir: 临时目录，shard 结果写入 tmp_dir/shard_{rank}.json
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Phase 1: 加载 tokenizer + model
    model_path = task_spec["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if task_spec["model_type"] == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        model = model.to(device).eval()
    else:
        from model_compat import create_model
        model_cfg = task_spec["model_cfg"]
        model = create_model(
            model_path=model_path,
            injection_method=model_cfg["method"],
            injection_layers=model_cfg["layers"],
            encoder_depth=model_cfg.get("encoder_depth", 6),
            knowledge_adapter=model_cfg.get("knowledge_adapter", False),
            device=str(device),
        )
        model.load_injection_weights(task_spec["injection_weights"])
        model = model.to(device).eval()

    # Phase 2: 数据分片
    dataset = task_spec["dataset"]
    total = len(dataset)
    per_shard = total // world_size
    start = rank * per_shard
    end = start + per_shard if rank < world_size - 1 else total
    shard_ds = dataset.select(range(start, end))

    log_msg("INFO", f"[GPU {rank}] 分片 [{start}:{end}]（共 {end - start} 条）")

    # Phase 3: 根据 dataset_name 调用对应 eval 函数
    dataset_name = task_spec["dataset_name"]
    knowledge_map = task_spec.get("knowledge_map")
    knowledge_length = task_spec.get("knowledge_length", 64)

    if dataset_name == "medqa":
        if task_spec["model_type"] == "baseline":
            result = _eval_medqa_baseline(model, tokenizer, shard_ds, device)
        else:
            result = _eval_medqa_injection(
                model, tokenizer, shard_ds, device,
                knowledge_map=knowledge_map,
                knowledge_length=knowledge_length,
                use_knowledge=task_spec.get("use_knowledge", True),
            )
    elif dataset_name == "arc":
        result = eval_arc(
            model, tokenizer, shard_ds, device,
            knowledge_map=knowledge_map,
            knowledge_length=knowledge_length,
            is_injection=task_spec.get("is_injection", False),
            limit=len(shard_ds),  # 已分片，全量跑
        )
    elif dataset_name == "mmlu":
        result = eval_mmlu(
            model, tokenizer, shard_ds, device,
            knowledge_map=knowledge_map,
            knowledge_length=knowledge_length,
            is_injection=task_spec.get("is_injection", False),
            limit=len(shard_ds),  # 已分片，全量跑
        )
    else:
        raise ValueError(f"未知数据集: {dataset_name}")

    # Phase 4: 写入 shard 结果
    shard_path = Path(tmp_dir) / f"shard_{rank}.json"
    shard_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    log_msg("INFO", f"[GPU {rank}] 完成，结果写入 {shard_path}")

    # Phase 5: 释放模型
    del model
    torch.cuda.empty_cache()


def _merge_shard_results(
    tmp_dir: str,
    num_shards: int,
    dataset_name: str,
) -> Dict[str, Any]:
    """合并多个 shard 的评测结果。

    Args:
        tmp_dir: 临时目录
        num_shards: shard 数量
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        合并后的结果字典
    """
    shards = []
    for i in range(num_shards):
        shard_path = Path(tmp_dir) / f"shard_{i}.json"
        assert shard_path.exists(), f"缺少 shard 文件: {shard_path}"
        shards.append(json.loads(shard_path.read_text(encoding="utf-8")))

    if dataset_name in ("medqa", "mmlu", "arc"):
        total = sum(s["total"] for s in shards)
        correct = sum(s["correct"] for s in shards)
        acc = correct / total if total > 0 else 0.0
        result: Dict[str, Any] = {"acc": acc, "correct": correct, "total": total}
        if dataset_name == "arc":
            result["skipped"] = sum(s.get("skipped", 0) for s in shards)
        return result
    else:
        raise ValueError(f"未知数据集: {dataset_name}")


def _run_parallel(
    task_spec: Dict[str, Any],
    num_gpus: int,
) -> Dict[str, Any]:
    """任务并行分发。单 GPU 退化为直接调用。

    task_spec 字段:
        model_type: "baseline" | "injection"
        model_path: str
        model_cfg: dict          # injection 专用
        injection_weights: str   # injection 专用
        dataset_name: "medqa" | "arc" | "mmlu"
        dataset: HF Dataset
        knowledge_map: dict | None
        knowledge_length: int
        use_knowledge: bool      # medqa injection 专用
        is_injection: bool
        limit: int | None        # arc/mmlu 专用

    Args:
        task_spec: 任务规格字典
        num_gpus: 可用 GPU 数

    Returns:
        评测结果字典
    """
    dataset_name = task_spec["dataset_name"]

    # ─── 如果有 limit，先截取数据集 ──────────────────────────────
    dataset = task_spec["dataset"]
    limit = task_spec.get("limit")
    if limit and limit < len(dataset):
        if hasattr(dataset, "select"):
            dataset = dataset.select(range(limit))
        task_spec = {**task_spec, "dataset": dataset}

    if num_gpus <= 1:
        # ─── 单 GPU: 直接在当前进程调用 eval ──────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = task_spec["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if task_spec["model_type"] == "baseline":
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            model = model.to(device).eval()
        else:
            from model_compat import create_model
            model_cfg = task_spec["model_cfg"]
            model = create_model(
                model_path=model_path,
                injection_method=model_cfg["method"],
                injection_layers=model_cfg["layers"],
                encoder_depth=model_cfg.get("encoder_depth", 6),
                knowledge_adapter=model_cfg.get("knowledge_adapter", False),
                device=str(device),
            )
            model.load_injection_weights(task_spec["injection_weights"])
            model = model.to(device).eval()

        if dataset_name == "medqa":
            if task_spec["model_type"] == "baseline":
                result = _eval_medqa_baseline(model, tokenizer, dataset, device)
            else:
                result = _eval_medqa_injection(
                    model, tokenizer, dataset, device,
                    knowledge_map=task_spec.get("knowledge_map"),
                    knowledge_length=task_spec.get("knowledge_length", 64),
                    use_knowledge=task_spec.get("use_knowledge", True),
                )
        elif dataset_name == "arc":
            result = eval_arc(
                model, tokenizer, dataset, device,
                knowledge_map=task_spec.get("knowledge_map"),
                knowledge_length=task_spec.get("knowledge_length", 64),
                is_injection=task_spec.get("is_injection", False),
                limit=len(dataset),
            )
        elif dataset_name == "mmlu":
            result = eval_mmlu(
                model, tokenizer, dataset, device,
                knowledge_map=task_spec.get("knowledge_map"),
                knowledge_length=task_spec.get("knowledge_length", 64),
                is_injection=task_spec.get("is_injection", False),
                limit=len(dataset),
            )
        else:
            raise ValueError(f"未知数据集: {dataset_name}")

        del model
        torch.cuda.empty_cache()
        return result

    # ─── 多 GPU: mp.spawn ──────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="e2_shard_")
    log_msg("INFO", f"多 GPU 并行 | num_gpus={num_gpus}, dataset={dataset_name}, tmp={tmp_dir}")

    mp.spawn(
        _eval_worker,
        args=(num_gpus, task_spec, tmp_dir),
        nprocs=num_gpus,
        join=True,
    )

    result = _merge_shard_results(tmp_dir, num_gpus, dataset_name)
    log_msg("INFO", f"并行合并完成 | {dataset_name}: {result}")
    return result


# ═══════════════════════════════════════════════════════════════════
# 主编排入口
# ═══════════════════════════════════════════════════════════════════


def run_e2_all(
    injection_weights: str,
    config: Dict[str, Any],
    tag: str = "phase1",
) -> Dict[str, Any]:
    """E2 跨域评测完整入口。

    在 MedQA / ARC-Challenge / MMLU 上运行三组对比:
      A) Baseline（原始 Qwen3-0.6B）
      B) Fusion + 知识（ModifiedQwen + injection_weights）
      C) Fusion + 空知识（ModifiedQwen + 全 pad）

    每个评测步骤通过 _run_parallel 自动利用多 GPU 数据并行。
    单 GPU 时退化为串行执行（与旧版行为一致）。

    Args:
        injection_weights: 注入权重路径
        config: 完整配置字典
        tag: 结果文件标签（默认 "phase1"）

    Returns:
        完整结果字典
    """
    num_gpus = torch.cuda.device_count()
    log_msg("INFO", f"E2 跨域评测 | 检测到 {num_gpus} 块 GPU")

    model_path = config["paths"]["model_dir"]
    model_cfg = config["model"]["injection"]
    eval_cfg = config["evaluation"]
    knowledge_length = eval_cfg["medqa"].get("knowledge_length", 64)

    # ─── 加载数据集 ──────────────────────────────────────────────
    log_msg("INFO", "加载数据集...")

    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    log_msg("INFO", f"MedQA test: {len(medqa_ds)} 题")

    arc_split = eval_cfg.get("arc", {}).get("split", "test")
    arc_limit = eval_cfg.get("arc", {}).get("limit")
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=arc_split)
    log_msg("INFO", f"ARC-Challenge {arc_split}: {len(arc_ds)} 题 (limit={arc_limit})")

    mmlu_split = eval_cfg.get("mmlu", {}).get("split", "test")
    mmlu_limit = eval_cfg.get("mmlu", {}).get("limit")
    mmlu_ds = load_dataset("cais/mmlu", "all", split=mmlu_split)
    log_msg("INFO", f"MMLU {mmlu_split}: {len(mmlu_ds)} 题 (limit={mmlu_limit})")

    # ─── 加载知识映射 ─────────────────────────────────────────────
    medqa_km_path = eval_cfg["medqa"]["knowledge_map"]
    arc_km_path = eval_cfg.get("arc", {}).get("knowledge_map", "data/arc_knowledge.jsonl")
    mmlu_km_path = eval_cfg.get("mmlu", {}).get("knowledge_map", "data/mmlu_knowledge.jsonl")

    assert Path(medqa_km_path).exists(), f"MedQA 知识映射不存在: {medqa_km_path}"
    assert Path(arc_km_path).exists(), f"ARC 知识映射不存在: {arc_km_path}（请先运行 build-arc-knowledge）"
    assert Path(mmlu_km_path).exists(), f"MMLU 知识映射不存在: {mmlu_km_path}（请先运行 build-mmlu-knowledge）"

    medqa_km = MedQAKnowledgeBuilder.load(medqa_km_path)
    arc_km = MedQAKnowledgeBuilder.load(arc_km_path)
    mmlu_km = MedQAKnowledgeBuilder.load(mmlu_km_path)

    results: Dict[str, Any] = {
        "weights": injection_weights,
        "tag": tag,
    }

    # ─── 公共 task_spec 基础字段 ──────────────────────────────────
    base_spec = {
        "model_path": model_path,
        "knowledge_length": knowledge_length,
    }
    injection_spec = {
        **base_spec,
        "model_type": "injection",
        "model_cfg": model_cfg,
        "injection_weights": injection_weights,
        "is_injection": True,
    }
    baseline_spec = {
        **base_spec,
        "model_type": "baseline",
        "is_injection": False,
    }

    # ══════════════════════════════════════════════════════════════
    # Phase A: Baseline 评测（原始 Qwen3-0.6B）
    # ══════════════════════════════════════════════════════════════
    log_msg("INFO", "=" * 64)
    log_msg("INFO", "Phase A: Baseline 评测（原始 Qwen3-0.6B）")
    log_msg("INFO", "=" * 64)

    # MedQA baseline
    medqa_baseline = _run_parallel({
        **baseline_spec,
        "dataset_name": "medqa",
        "dataset": medqa_ds,
        "knowledge_map": None,
    }, num_gpus)
    results["medqa"] = {"baseline": medqa_baseline}

    # ARC baseline
    arc_baseline = _run_parallel({
        **baseline_spec,
        "dataset_name": "arc",
        "dataset": arc_ds,
        "knowledge_map": None,
        "limit": arc_limit,
    }, num_gpus)
    results["arc"] = {"baseline": arc_baseline}

    # MMLU baseline
    mmlu_baseline = _run_parallel({
        **baseline_spec,
        "dataset_name": "mmlu",
        "dataset": mmlu_ds,
        "knowledge_map": None,
        "limit": mmlu_limit,
    }, num_gpus)
    results["mmlu"] = {"baseline": mmlu_baseline}

    # ══════════════════════════════════════════════════════════════
    # Phase B: Fusion 评测（ModifiedQwen + injection_weights）
    # ══════════════════════════════════════════════════════════════
    log_msg("INFO", "=" * 64)
    log_msg("INFO", f"Phase B: Fusion 评测（权重: {injection_weights}）")
    log_msg("INFO", "=" * 64)

    # MedQA Fusion + 知识
    medqa_fusion_k = _run_parallel({
        **injection_spec,
        "dataset_name": "medqa",
        "dataset": medqa_ds,
        "knowledge_map": medqa_km,
        "use_knowledge": True,
    }, num_gpus)
    results["medqa"]["fusion_knowledge"] = medqa_fusion_k

    # MedQA Fusion + 空知识
    medqa_fusion_empty = _run_parallel({
        **injection_spec,
        "dataset_name": "medqa",
        "dataset": medqa_ds,
        "knowledge_map": medqa_km,
        "use_knowledge": False,
    }, num_gpus)
    results["medqa"]["fusion_empty"] = medqa_fusion_empty

    # ARC Fusion + 知识
    arc_fusion_k = _run_parallel({
        **injection_spec,
        "dataset_name": "arc",
        "dataset": arc_ds,
        "knowledge_map": arc_km,
        "limit": arc_limit,
    }, num_gpus)
    results["arc"]["fusion_knowledge"] = arc_fusion_k

    # ARC Fusion + 空知识
    arc_fusion_empty = _run_parallel({
        **injection_spec,
        "dataset_name": "arc",
        "dataset": arc_ds,
        "knowledge_map": None,
        "limit": arc_limit,
    }, num_gpus)
    results["arc"]["fusion_empty"] = arc_fusion_empty

    # MMLU Fusion + 知识
    mmlu_fusion_k = _run_parallel({
        **injection_spec,
        "dataset_name": "mmlu",
        "dataset": mmlu_ds,
        "knowledge_map": mmlu_km,
        "limit": mmlu_limit,
    }, num_gpus)
    results["mmlu"]["fusion_knowledge"] = mmlu_fusion_k

    # MMLU Fusion + 空知识
    mmlu_fusion_empty = _run_parallel({
        **injection_spec,
        "dataset_name": "mmlu",
        "dataset": mmlu_ds,
        "knowledge_map": None,
        "limit": mmlu_limit,
    }, num_gpus)
    results["mmlu"]["fusion_empty"] = mmlu_fusion_empty

    # ══════════════════════════════════════════════════════════════
    # Phase C: 汇总并保存
    # ══════════════════════════════════════════════════════════════
    log_msg("INFO", "=" * 64)
    log_msg("INFO", "Phase C: 汇总结果")
    log_msg("INFO", "=" * 64)

    # 计算 Δacc（三个数据集格式统一）
    for ds_name in ["medqa", "arc", "mmlu"]:
        baseline_acc = results[ds_name]["baseline"]["acc"]
        fusion_k_acc = results[ds_name]["fusion_knowledge"]["acc"]
        fusion_e_acc = results[ds_name]["fusion_empty"]["acc"]
        results[ds_name]["delta_acc"] = fusion_k_acc - baseline_acc
        results[ds_name]["delta_acc_empty"] = fusion_e_acc - baseline_acc

    # 打印结构化报告
    sep = "=" * 72
    log_msg("INFO", f"\n{sep}")
    log_msg("INFO", f"E2 跨域评测结果 | 权重: {injection_weights} (tag={tag})")
    log_msg("INFO", sep)
    log_msg("INFO", f"{'数据集':<12} {'Baseline':>10} {'Fusion+知识':>12} {'Fusion+空':>10} {'Δacc':>8}")
    log_msg("INFO", "-" * 72)

    for ds_name, ds_label in [("medqa", "MedQA"), ("arc", "ARC"), ("mmlu", "MMLU")]:
        log_msg(
            "INFO",
            f"{ds_label:<12} "
            f"{results[ds_name]['baseline']['acc']:>10.4f} "
            f"{results[ds_name]['fusion_knowledge']['acc']:>12.4f} "
            f"{results[ds_name]['fusion_empty']['acc']:>10.4f} "
            f"{results[ds_name]['delta_acc']:>+8.4f}",
        )
    log_msg("INFO", sep)

    # 保存结果
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"e2_cross_domain_{tag}.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_msg("INFO", f"结果保存: {output_path}")

    return results
