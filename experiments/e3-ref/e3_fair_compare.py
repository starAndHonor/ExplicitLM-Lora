"""E3：层注入 vs RAG 公平对比。

在 MedQA / ARC-Challenge / MMLU 三个数据集上运行 G0-G4 五组完整对比矩阵：
  G0  Baseline        — 原始 Qwen3-0.6B，无知识
  G1  RAG-compressed  — decode(knowledge_ids) → 文本前缀拼接
  G2  Fusion-Phase1   — knowledge_ids → 层注入（Phase 1 权重）
  G3  Fusion-Phase2   — knowledge_ids → 层注入（Phase 2 权重）
  G4  RAG-original    — question+correct_answer 原文前缀拼接（不截断）

核心公平性保证：G1 与 G2 使用完全相同的 knowledge_ids，仅融合方式不同。
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from arc_eval import _build_arc_prompt, answer_key_to_index
from compare_eval import _build_question_prompt, _score_choices
from counterfactual_eval import _score_choices_injection
from medqa_knowledge_builder import MedQAKnowledgeBuilder
from mmlu_eval import _build_mmlu_prompt
from logger_system import log_msg


# ═══════════════════════════════════════════════════════════════════
# 数据集适配层
# ═══════════════════════════════════════════════════════════════════


def _get_prompt_builder(dataset_name: str) -> Callable:
    """返回对应数据集的 prompt 构建函数。

    Args:
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        prompt 构建函数 (row) -> str
    """
    if dataset_name == "medqa":
        return _build_question_prompt
    elif dataset_name == "arc":
        return _build_arc_prompt
    elif dataset_name == "mmlu":
        return _build_mmlu_prompt
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def _get_label(row: Dict[str, Any], dataset_name: str) -> int:
    """返回对应数据集的正确答案索引。

    Args:
        row: 数据集中的一条记录
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        正确答案索引（0-3）
    """
    if dataset_name == "medqa":
        return row["label"]
    elif dataset_name == "arc":
        return answer_key_to_index(row["answerKey"])
    elif dataset_name == "mmlu":
        return row["answer"]
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def _get_question_key(row: Dict[str, Any], dataset_name: str) -> str:
    """返回知识映射的 key（与构建知识映射时一致）。

    Args:
        row: 数据集中的一条记录
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        knowledge_map 的 key
    """
    if dataset_name == "medqa":
        return row["sent1"][:200].strip()
    elif dataset_name == "arc":
        return row["question"][:200].strip()
    elif dataset_name == "mmlu":
        return row["question"][:200].strip()
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def _get_original_text(row: Dict[str, Any], dataset_name: str) -> str:
    """返回 question + correct_answer 的原始完整文本（G4 用，不截断）。

    Args:
        row: 数据集中的一条记录
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        原始文本字符串
    """
    if dataset_name == "medqa":
        label = row["label"]
        return f"{row['sent1']} {row[f'ending{label}']}"
    elif dataset_name == "arc":
        answer_idx = answer_key_to_index(row["answerKey"])
        correct_answer = row["choices"]["text"][answer_idx]
        return f"{row['question']} {correct_answer}"
    elif dataset_name == "mmlu":
        correct_answer = row["choices"][row["answer"]]
        return f"{row['question']} {correct_answer}"
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def _should_skip(row: Dict[str, Any], dataset_name: str) -> bool:
    """判断是否应跳过该题（仅 ARC 跳过非 4 选项题）。

    Args:
        row: 数据集中的一条记录
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        True 表示应跳过
    """
    if dataset_name == "arc":
        return len(row["choices"]["text"]) != 4
    return False


# ═══════════════════════════════════════════════════════════════════
# 各组评测函数
# ═══════════════════════════════════════════════════════════════════


def _eval_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    device: torch.device,
    dataset_name: str,
) -> Dict[str, Any]:
    """G0: Baseline 评测（原始 Qwen3-0.6B，无知识）。

    Args:
        model: AutoModelForCausalLM
        tokenizer: tokenizer
        ds: HuggingFace Dataset
        device: 计算设备
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        {"acc": float, "correct": int, "total": int, "skipped": int}
    """
    build_prompt = _get_prompt_builder(dataset_name)
    total_rows = len(ds)
    correct = 0
    total = 0
    skipped = 0

    log_msg("INFO", f"G0 Baseline | {dataset_name} | 总行数={total_rows}")

    with torch.no_grad():
        for i, row in enumerate(ds):
            if _should_skip(row, dataset_name):
                skipped += 1
                continue

            label = _get_label(row, dataset_name)
            context = build_prompt(row)
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            pred = _score_choices(model, tokenizer, context_ids, device)

            if pred == label:
                correct += 1
            total += 1

            if (i + 1) % 500 == 0:
                log_msg("INFO", f"  G0 {dataset_name} 进度: {i+1}/{total_rows}, acc={correct/total:.4f}")

    acc = correct / total if total > 0 else 0.0
    log_msg("INFO", f"G0 {dataset_name} 完成 | acc={acc:.4f} ({correct}/{total}), skipped={skipped}")
    return {"acc": acc, "correct": correct, "total": total, "skipped": skipped}


def _eval_rag_compressed(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    knowledge_map: Dict[str, List[int]],
    device: torch.device,
    dataset_name: str,
) -> Dict[str, Any]:
    """G1: RAG-compressed 评测。

    decode(knowledge_ids) → 文本前缀拼接到 prompt。
    与 G2 使用完全相同的 knowledge_ids，仅融合方式不同。

    Args:
        model: AutoModelForCausalLM（原始 Qwen3）
        tokenizer: tokenizer
        ds: HuggingFace Dataset
        knowledge_map: {key: knowledge_ids} 压缩知识映射
        device: 计算设备
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        {"acc": float, "correct": int, "total": int, "skipped": int}
    """
    build_prompt = _get_prompt_builder(dataset_name)
    pad_id = tokenizer.pad_token_id
    total_rows = len(ds)
    correct = 0
    total = 0
    skipped = 0
    missing_knowledge = 0

    log_msg("INFO", f"G1 RAG-compressed | {dataset_name} | 总行数={total_rows}")

    with torch.no_grad():
        for i, row in enumerate(ds):
            if _should_skip(row, dataset_name):
                skipped += 1
                continue

            key = _get_question_key(row, dataset_name)
            label = _get_label(row, dataset_name)
            question_prompt = build_prompt(row)

            k_ids = knowledge_map.get(key)
            if k_ids is not None:
                # Phase 1: decode knowledge_ids → 文本（过滤 pad token）
                clean_ids = [t for t in k_ids if t != pad_id]
                compressed_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
                context = f"Context: {compressed_text}\n\n{question_prompt}"
            else:
                # 无知识 fallback
                missing_knowledge += 1
                context = question_prompt

            context_ids = tokenizer.encode(context, add_special_tokens=False)
            pred = _score_choices(model, tokenizer, context_ids, device)

            if pred == label:
                correct += 1
            total += 1

            if (i + 1) % 500 == 0:
                log_msg("INFO", f"  G1 {dataset_name} 进度: {i+1}/{total_rows}, acc={correct/total:.4f}")

    acc = correct / total if total > 0 else 0.0
    log_msg(
        "INFO",
        f"G1 {dataset_name} 完成 | acc={acc:.4f} ({correct}/{total}), "
        f"skipped={skipped}, missing_knowledge={missing_knowledge}",
    )
    return {"acc": acc, "correct": correct, "total": total, "skipped": skipped}


def _eval_fusion(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    knowledge_map: Dict[str, List[int]],
    device: torch.device,
    dataset_name: str,
    knowledge_length: int = 64,
    group_name: str = "G2",
) -> Dict[str, Any]:
    """G2/G3: Fusion 层注入评测。

    knowledge_ids → Cross-Attention 层注入。

    Args:
        model: ModifiedQwen（已加载注入权重）
        tokenizer: tokenizer
        ds: HuggingFace Dataset
        knowledge_map: {key: knowledge_ids} 压缩知识映射
        device: 计算设备
        dataset_name: "medqa" | "arc" | "mmlu"
        knowledge_length: 知识 token 长度（默认 64）
        group_name: 组名（用于日志，"G2" 或 "G3"）

    Returns:
        {"acc": float, "correct": int, "total": int, "skipped": int}
    """
    build_prompt = _get_prompt_builder(dataset_name)
    pad_id = tokenizer.pad_token_id
    pad_knowledge = [pad_id] * knowledge_length
    total_rows = len(ds)
    correct = 0
    total = 0
    skipped = 0

    log_msg("INFO", f"{group_name} Fusion | {dataset_name} | 总行数={total_rows}")

    with torch.no_grad():
        for i, row in enumerate(ds):
            if _should_skip(row, dataset_name):
                skipped += 1
                continue

            key = _get_question_key(row, dataset_name)
            label = _get_label(row, dataset_name)
            context = build_prompt(row)
            context_ids = tokenizer.encode(context, add_special_tokens=False)

            k_ids = knowledge_map.get(key, pad_knowledge)
            k_tensor = torch.tensor([k_ids], dtype=torch.long, device=device)
            pred = _score_choices_injection(model, tokenizer, context_ids, k_tensor, device)

            if pred == label:
                correct += 1
            total += 1

            if (i + 1) % 500 == 0:
                log_msg("INFO", f"  {group_name} {dataset_name} 进度: {i+1}/{total_rows}, acc={correct/total:.4f}")

    acc = correct / total if total > 0 else 0.0
    log_msg("INFO", f"{group_name} {dataset_name} 完成 | acc={acc:.4f} ({correct}/{total}), skipped={skipped}")
    return {"acc": acc, "correct": correct, "total": total, "skipped": skipped}


def _eval_rag_original(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    device: torch.device,
    dataset_name: str,
) -> Dict[str, Any]:
    """G4: RAG-original 评测（不截断）。

    question + correct_answer 原始完整文本 → 文本前缀拼接。
    与旧 compare_eval G3 不同：不做 256-token 截断。

    Args:
        model: AutoModelForCausalLM（原始 Qwen3）
        tokenizer: tokenizer
        ds: HuggingFace Dataset
        device: 计算设备
        dataset_name: "medqa" | "arc" | "mmlu"

    Returns:
        {"acc": float, "correct": int, "total": int, "skipped": int}
    """
    build_prompt = _get_prompt_builder(dataset_name)
    total_rows = len(ds)
    correct = 0
    total = 0
    skipped = 0

    log_msg("INFO", f"G4 RAG-original | {dataset_name} | 总行数={total_rows}")

    with torch.no_grad():
        for i, row in enumerate(ds):
            if _should_skip(row, dataset_name):
                skipped += 1
                continue

            label = _get_label(row, dataset_name)
            original_text = _get_original_text(row, dataset_name)
            question_prompt = build_prompt(row)

            # 不截断：直接使用完整原文
            context = f"Context: {original_text}\n\n{question_prompt}"
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            pred = _score_choices(model, tokenizer, context_ids, device)

            if pred == label:
                correct += 1
            total += 1

            if (i + 1) % 500 == 0:
                log_msg("INFO", f"  G4 {dataset_name} 进度: {i+1}/{total_rows}, acc={correct/total:.4f}")

    acc = correct / total if total > 0 else 0.0
    log_msg("INFO", f"G4 {dataset_name} 完成 | acc={acc:.4f} ({correct}/{total}), skipped={skipped}")
    return {"acc": acc, "correct": correct, "total": total, "skipped": skipped}


# ═══════════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════════


def _load_base_model(
    model_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """加载原始 Qwen3-0.6B（G0/G1/G4 共用）。

    Args:
        model_path: 模型路径
        device: 目标设备

    Returns:
        (model, tokenizer) 元组
    """
    log_msg("INFO", f"加载原始 Qwen3: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    log_msg("INFO", "原始 Qwen3 加载完成")
    return model, tokenizer


def _load_injection_model(
    model_path: str,
    model_cfg: Dict[str, Any],
    injection_weights: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """加载 ModifiedQwen + 注入权重（G2/G3 共用）。

    Args:
        model_path: 基础模型路径
        model_cfg: 注入配置字典
        injection_weights: 注入权重路径
        device: 目标设备

    Returns:
        (model, tokenizer) 元组
    """
    from model_compat import create_model

    log_msg("INFO", f"加载 ModifiedQwen + 权重: {injection_weights}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = create_model(
        model_path=model_path,
        injection_method=model_cfg["method"],
        injection_layers=model_cfg["layers"],
        encoder_depth=model_cfg.get("encoder_depth", 6),
        device=str(device),
    )
    model.load_injection_weights(injection_weights)
    model = model.to(device).eval()
    log_msg("INFO", f"ModifiedQwen 加载完成 | 权重: {injection_weights}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════
# 多 GPU 并行化
# ═══════════════════════════════════════════════════════════════════


def _e3_worker(
    rank: int,
    world_size: int,
    task_spec: Dict[str, Any],
    tmp_dir: str,
) -> None:
    """mp.spawn worker — 在 rank 对应的 GPU 上评测数据分片。

    top-level 函数（mp.spawn 要求 picklable）。

    Args:
        rank: GPU 序号
        world_size: 总 GPU 数
        task_spec: 任务规格字典
        tmp_dir: shard 结果写入目录
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Phase 1: 加载模型
    model_path = task_spec["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if task_spec["model_type"] == "baseline":
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
    else:
        from model_compat import create_model
        model_cfg = task_spec["model_cfg"]
        model = create_model(
            model_path=model_path,
            injection_method=model_cfg["method"],
            injection_layers=model_cfg["layers"],
            encoder_depth=model_cfg.get("encoder_depth", 6),
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

    dataset_name = task_spec["dataset_name"]
    eval_mode = task_spec["eval_mode"]
    knowledge_map = task_spec.get("knowledge_map")
    knowledge_length = task_spec.get("knowledge_length", 64)
    group_name = task_spec.get("group_name", eval_mode)

    log_msg("INFO", f"[GPU {rank}] {group_name} {dataset_name} 分片 [{start}:{end}]（共 {end - start} 条）")

    # Phase 3: 根据 eval_mode 调用对应 eval 函数
    if eval_mode == "G0":
        result = _eval_baseline(model, tokenizer, shard_ds, device, dataset_name)
    elif eval_mode == "G1":
        result = _eval_rag_compressed(model, tokenizer, shard_ds, knowledge_map, device, dataset_name)
    elif eval_mode in ("G2", "G3"):
        result = _eval_fusion(
            model, tokenizer, shard_ds, knowledge_map, device, dataset_name,
            knowledge_length=knowledge_length, group_name=group_name,
        )
    elif eval_mode == "G4":
        result = _eval_rag_original(model, tokenizer, shard_ds, device, dataset_name)
    else:
        raise ValueError(f"未知 eval_mode: {eval_mode}")

    # Phase 4: 写入 shard 结果
    shard_path = Path(tmp_dir) / f"shard_{rank}.json"
    shard_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    log_msg("INFO", f"[GPU {rank}] {group_name} {dataset_name} 完成")

    del model
    torch.cuda.empty_cache()


def _merge_e3_shards(tmp_dir: str, num_shards: int) -> Dict[str, Any]:
    """合并多个 shard 的评测结果。

    Args:
        tmp_dir: 临时目录
        num_shards: shard 数量

    Returns:
        合并后的结果字典
    """
    shards = []
    for i in range(num_shards):
        shard_path = Path(tmp_dir) / f"shard_{i}.json"
        assert shard_path.exists(), f"缺少 shard 文件: {shard_path}"
        shards.append(json.loads(shard_path.read_text(encoding="utf-8")))

    total = sum(s["total"] for s in shards)
    correct = sum(s["correct"] for s in shards)
    skipped = sum(s.get("skipped", 0) for s in shards)
    acc = correct / total if total > 0 else 0.0
    return {"acc": acc, "correct": correct, "total": total, "skipped": skipped}


def _run_e3_parallel(
    task_spec: Dict[str, Any],
    num_gpus: int,
) -> Dict[str, Any]:
    """E3 任务并行分发。单 GPU 退化为直接调用。

    task_spec 必需字段:
        eval_mode: "G0" | "G1" | "G2" | "G3" | "G4"
        model_type: "baseline" | "injection"
        model_path: str
        dataset_name: "medqa" | "arc" | "mmlu"
        dataset: HF Dataset
    可选字段:
        model_cfg, injection_weights: injection 专用
        knowledge_map: G1/G2/G3 专用
        knowledge_length: int
        group_name: str

    Args:
        task_spec: 任务规格字典
        num_gpus: 可用 GPU 数

    Returns:
        评测结果字典
    """
    eval_mode = task_spec["eval_mode"]
    dataset_name = task_spec["dataset_name"]
    group_name = task_spec.get("group_name", eval_mode)

    if num_gpus <= 1:
        # ─── 单 GPU: 直接在当前进程调用 ─────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = task_spec["model_path"]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if task_spec["model_type"] == "baseline":
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
            ).to(device).eval()
        else:
            from model_compat import create_model
            model_cfg = task_spec["model_cfg"]
            model = create_model(
                model_path=model_path,
                injection_method=model_cfg["method"],
                injection_layers=model_cfg["layers"],
                encoder_depth=model_cfg.get("encoder_depth", 6),
                device=str(device),
            )
            model.load_injection_weights(task_spec["injection_weights"])
            model = model.to(device).eval()

        knowledge_map = task_spec.get("knowledge_map")
        knowledge_length = task_spec.get("knowledge_length", 64)
        dataset = task_spec["dataset"]

        if eval_mode == "G0":
            result = _eval_baseline(model, tokenizer, dataset, device, dataset_name)
        elif eval_mode == "G1":
            result = _eval_rag_compressed(model, tokenizer, dataset, knowledge_map, device, dataset_name)
        elif eval_mode in ("G2", "G3"):
            result = _eval_fusion(
                model, tokenizer, dataset, knowledge_map, device, dataset_name,
                knowledge_length=knowledge_length, group_name=group_name,
            )
        elif eval_mode == "G4":
            result = _eval_rag_original(model, tokenizer, dataset, device, dataset_name)
        else:
            raise ValueError(f"未知 eval_mode: {eval_mode}")

        del model
        torch.cuda.empty_cache()
        return result

    # ─── 多 GPU: mp.spawn ──────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix=f"e3_{group_name}_{dataset_name}_")
    log_msg("INFO", f"多 GPU 并行 | {group_name} {dataset_name} | num_gpus={num_gpus}")

    mp.spawn(
        _e3_worker,
        args=(num_gpus, task_spec, tmp_dir),
        nprocs=num_gpus,
        join=True,
    )

    result = _merge_e3_shards(tmp_dir, num_gpus)
    log_msg("INFO", f"并行合并完成 | {group_name} {dataset_name}: acc={result['acc']:.4f}")
    return result


def _print_e3_report(results: Dict[str, Any]) -> None:
    """打印 E3 结构化报告。

    Args:
        results: 完整结果字典
    """
    sep = "=" * 80
    log_msg("INFO", f"\n{sep}")
    log_msg("INFO", "E3 公平对比结果（层注入 vs RAG）")
    log_msg("INFO", sep)

    # Phase 1: 各数据集结果表
    groups = ["G0_baseline", "G1_rag_compressed", "G2_fusion_phase1", "G3_fusion_phase2", "G4_rag_original"]
    group_labels = ["G0 Baseline", "G1 RAG-compressed", "G2 Fusion-Phase1", "G3 Fusion-Phase2", "G4 RAG-original"]

    header = f"{'组别':<22}"
    for ds_name in ["medqa", "arc", "mmlu"]:
        header += f" {ds_name.upper():>10}"
    log_msg("INFO", header)
    log_msg("INFO", "-" * 80)

    for gkey, glabel in zip(groups, group_labels):
        line = f"{glabel:<22}"
        for ds_name in ["medqa", "arc", "mmlu"]:
            acc = results[ds_name][gkey]["acc"]
            line += f" {acc:>9.2%}"
        log_msg("INFO", line)

    log_msg("INFO", "-" * 80)

    # Phase 2: 核心对比（G2 vs G1, G3 vs G1）
    log_msg("INFO", "\n核心对比（Fusion vs RAG-compressed，使用相同 knowledge_ids）:")
    for ds_name in ["medqa", "arc", "mmlu"]:
        g1_acc = results[ds_name]["G1_rag_compressed"]["acc"]
        g2_acc = results[ds_name]["G2_fusion_phase1"]["acc"]
        g3_acc = results[ds_name]["G3_fusion_phase2"]["acc"]
        g4_acc = results[ds_name]["G4_rag_original"]["acc"]
        g0_acc = results[ds_name]["G0_baseline"]["acc"]

        delta_g2_g1 = g2_acc - g1_acc
        delta_g3_g1 = g3_acc - g1_acc

        # 知识利用效率 = Δacc(Fusion) / Δacc(G4)
        delta_g4 = g4_acc - g0_acc
        eff_g2 = (g2_acc - g0_acc) / delta_g4 if delta_g4 > 0 else 0.0
        eff_g3 = (g3_acc - g0_acc) / delta_g4 if delta_g4 > 0 else 0.0

        log_msg("INFO", f"  {ds_name.upper()}: G2-G1={delta_g2_g1:+.2%}, G3-G1={delta_g3_g1:+.2%}, "
                         f"效率 G2={eff_g2:.1%}, G3={eff_g3:.1%}")

    log_msg("INFO", sep)


# ═══════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════


def run_e3_all(
    phase1_weights: str,
    phase2_weights: str,
    config: Dict[str, Any],
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """E3 完整入口：5 组 × 3 数据集公平对比。

    支持多 GPU 数据并行（自动检测可用 GPU 数）。
    单 GPU 时自动退化为串行执行。

    执行顺序（最小化模型切换）：
      Phase A: 原始 Qwen3 → G0, G1, G4 (×3 数据集)
      Phase B: ModifiedQwen + Phase1 → G2 (×3 数据集)
      Phase C: ModifiedQwen + Phase2 → G3 (×3 数据集)
      Phase D: 汇总报告 + 保存

    Args:
        config: 完整配置字典
        phase1_weights: Phase 1 注入权重路径
        phase2_weights: Phase 2 注入权重路径

    Returns:
        完整结果字典
    """
    num_gpus = torch.cuda.device_count()
    model_path = config["paths"]["model_dir"]
    model_cfg = config["model"]["injection"]
    eval_cfg = config["evaluation"]
    knowledge_length = eval_cfg["medqa"].get("knowledge_length", 64)

    log_msg("INFO", "=" * 80)
    log_msg("INFO", "E3：层注入 vs RAG 公平对比")
    log_msg("INFO", f"检测到 {num_gpus} 块 GPU（{'多卡并行' if num_gpus > 1 else '单卡串行'}）")
    log_msg("INFO", f"Phase 1 权重: {phase1_weights}")
    log_msg("INFO", f"Phase 2 权重: {phase2_weights}")
    log_msg("INFO", "=" * 80)

    # ─── 加载数据集 ──────────────────────────────────────────────
    log_msg("INFO", "加载数据集...")

    medqa_ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    log_msg("INFO", f"MedQA test: {len(medqa_ds)} 题")

    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    log_msg("INFO", f"ARC-Challenge test: {len(arc_ds)} 题")

    mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
    log_msg("INFO", f"MMLU test: {len(mmlu_ds)} 题")

    all_datasets = {
        "medqa": medqa_ds,
        "arc": arc_ds,
        "mmlu": mmlu_ds,
    }

    # ─── 加载知识映射 ─────────────────────────────────────────────
    medqa_km_path = eval_cfg["medqa"]["knowledge_map"]
    arc_km_path = eval_cfg.get("arc", {}).get("knowledge_map", "data/arc_knowledge.jsonl")
    mmlu_km_path = eval_cfg.get("mmlu", {}).get("knowledge_map", "data/mmlu_knowledge.jsonl")

    assert Path(medqa_km_path).exists(), f"MedQA 知识映射不存在: {medqa_km_path}"
    assert Path(arc_km_path).exists(), f"ARC 知识映射不存在: {arc_km_path}"
    assert Path(mmlu_km_path).exists(), f"MMLU 知识映射不存在: {mmlu_km_path}"

    knowledge_maps = {
        "medqa": MedQAKnowledgeBuilder.load(medqa_km_path),
        "arc": MedQAKnowledgeBuilder.load(arc_km_path),
        "mmlu": MedQAKnowledgeBuilder.load(mmlu_km_path),
    }

    log_msg("INFO", f"知识映射加载完成 | MedQA: {len(knowledge_maps['medqa'])} 条, "
                     f"ARC: {len(knowledge_maps['arc'])} 条, MMLU: {len(knowledge_maps['mmlu'])} 条")

    # 初始化结果字典
    results: Dict[str, Any] = {
        "phase1_weights": phase1_weights,
        "phase2_weights": phase2_weights,
        "num_gpus": num_gpus,
    }
    for ds_name in ["medqa", "arc", "mmlu"]:
        results[ds_name] = {}

    # ─── 公共 task_spec 基础字段 ──────────────────────────────────
    base_spec = {
        "model_path": model_path,
        "knowledge_length": knowledge_length,
    }
    baseline_spec = {**base_spec, "model_type": "baseline"}
    injection_spec_p1 = {
        **base_spec,
        "model_type": "injection",
        "model_cfg": model_cfg,
        "injection_weights": phase1_weights,
    }
    injection_spec_p2 = {
        **base_spec,
        "model_type": "injection",
        "model_cfg": model_cfg,
        "injection_weights": phase2_weights,
    }

    # ══════════════════════════════════════════════════════════════
    # Phase A: 原始 Qwen3-0.6B → G0, G1, G4
    # ══════════════════════════════════════════════════════════════
    log_msg("INFO", "\n" + "=" * 80)
    log_msg("INFO", "Phase A: 原始 Qwen3-0.6B（G0, G1, G4）")
    log_msg("INFO", "=" * 80)

    for ds_name in ["medqa", "arc", "mmlu"]:
        ds = all_datasets[ds_name]
        km = knowledge_maps[ds_name]

        # G0: Baseline
        results[ds_name]["G0_baseline"] = _run_e3_parallel({
            **baseline_spec,
            "eval_mode": "G0",
            "group_name": "G0",
            "dataset_name": ds_name,
            "dataset": ds,
        }, num_gpus)

        # G1: RAG-compressed
        results[ds_name]["G1_rag_compressed"] = _run_e3_parallel({
            **baseline_spec,
            "eval_mode": "G1",
            "group_name": "G1",
            "dataset_name": ds_name,
            "dataset": ds,
            "knowledge_map": km,
        }, num_gpus)

        # G4: RAG-original（不截断）
        results[ds_name]["G4_rag_original"] = _run_e3_parallel({
            **baseline_spec,
            "eval_mode": "G4",
            "group_name": "G4",
            "dataset_name": ds_name,
            "dataset": ds,
        }, num_gpus)

    # ══════════════════════════════════════════════════════════════
    # Phase B: ModifiedQwen + Phase 1 → G2
    # ══════════════════════════════════════════════════════════════
    log_msg("INFO", "\n" + "=" * 80)
    log_msg("INFO", f"Phase B: Fusion-Phase1（G2, 权重: {phase1_weights}）")
    log_msg("INFO", "=" * 80)

    for ds_name in ["medqa", "arc", "mmlu"]:
        ds = all_datasets[ds_name]
        km = knowledge_maps[ds_name]

        results[ds_name]["G2_fusion_phase1"] = _run_e3_parallel({
            **injection_spec_p1,
            "eval_mode": "G2",
            "group_name": "G2",
            "dataset_name": ds_name,
            "dataset": ds,
            "knowledge_map": km,
        }, num_gpus)

    # ══════════════════════════════════════════════════════════════
    # Phase C: ModifiedQwen + Phase 2 → G3
    # ══════════════════════════════════════════════════════════════
    log_msg("INFO", "\n" + "=" * 80)
    log_msg("INFO", f"Phase C: Fusion-Phase2（G3, 权重: {phase2_weights}）")
    log_msg("INFO", "=" * 80)

    for ds_name in ["medqa", "arc", "mmlu"]:
        ds = all_datasets[ds_name]
        km = knowledge_maps[ds_name]

        results[ds_name]["G3_fusion_phase2"] = _run_e3_parallel({
            **injection_spec_p2,
            "eval_mode": "G3",
            "group_name": "G3",
            "dataset_name": ds_name,
            "dataset": ds,
            "knowledge_map": km,
        }, num_gpus)

    # ══════════════════════════════════════════════════════════════
    # Phase D: 汇总报告 + 保存
    # ══════════════════════════════════════════════════════════════
    log_msg("INFO", "\n" + "=" * 80)
    log_msg("INFO", "Phase D: 汇总与保存")
    log_msg("INFO", "=" * 80)

    # 计算 summary 指标
    results["summary"] = {}
    for ds_name in ["medqa", "arc", "mmlu"]:
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

    # 打印报告
    _print_e3_report(results)

    # 保存结果
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_name = "e3_fair_compare.json" if not tag else f"e3_fair_compare_{tag}.json"
    output_path = results_dir / output_name
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_msg("INFO", f"结果保存: {output_path}")

    return results
