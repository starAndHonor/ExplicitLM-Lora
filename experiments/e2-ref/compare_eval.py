"""四组对比评测模块。

与 evaluate_medqa_inline 使用完全相同的 loglikelihood 方法，
确保与 Phase 2（71.1%）的结果可直接对比。

四组：
  1. 原始 Qwen3-0.6B，无任何外部知识
  2. RAG + compressed 知识（decode knowledge_ids 后文本前缀拼接）
  3. RAG + 原始知识（question + correct_answer，完美检索）
  4. Prefix token 拼接（compressed knowledge_ids 直接拼在 question 前）
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from logger_system import log_msg


def _build_question_prompt(row: Dict[str, Any]) -> str:
    """构造标准 MedQA question prompt。

    与 evaluate_medqa_inline 的格式完全一致，确保对比公平。

    Args:
        row: MedQA 数据集中的一条记录

    Returns:
        格式化后的 question prompt 字符串
    """
    options = {
        "A": row["ending0"],
        "B": row["ending1"],
        "C": row["ending2"],
        "D": row["ending3"],
    }
    answers_str = "".join(f"{k}. {v}\n" for k, v in options.items())
    return f"Question: {row['sent1']}\n{answers_str}Answer:"


def _score_choices(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context_ids: List[int],
    device: torch.device,
) -> int:
    """用 loglikelihood 方法在4个选项中选最优。

    与 evaluate_medqa_inline 的逻辑完全一致：
    对每个选项做完整 forward，取 continuation 位置 log-prob 之和，
    选最高分的选项作为预测。

    Args:
        model: 原始 Qwen3 AutoModelForCausalLM
        tokenizer: 对应的 tokenizer
        context_ids: 已编码的 context token id 列表
        device: 计算设备

    Returns:
        预测选项索引（0=A, 1=B, 2=C, 3=D）
    """
    scores = []
    for choice in [" A", " B", " C", " D"]:
        cont_ids = tokenizer.encode(choice, add_special_tokens=False)
        input_ids = context_ids + cont_ids

        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(input_ids=input_t)
        logits = outputs.logits  # [1, L, V]

        # 取 continuation 对应位置的 log-prob
        cont_start = len(context_ids) - 1
        cont_end = len(input_ids) - 1
        cont_logits = logits[0, cont_start:cont_end, :]
        cont_tokens = torch.tensor(cont_ids, dtype=torch.long, device=device)
        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_ll = log_probs.gather(1, cont_tokens.unsqueeze(-1)).squeeze(-1)
        scores.append(token_ll.sum().item())

    return scores.index(max(scores))


def eval_group1_baseline(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """组1：原始 Qwen3-0.6B，无任何外部知识。

    Args:
        model: 原始 Qwen3 AutoModelForCausalLM
        tokenizer: 对应的 tokenizer
        ds: MedQA test dataset
        device: 计算设备

    Returns:
        {"group", "method", "acc", "correct", "total"}
    """
    log_msg("INFO", "组1 (原始 Qwen3-0.6B) 评测开始...")
    correct = 0
    total = len(ds)

    for i, row in enumerate(ds):
        context = _build_question_prompt(row)
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        pred = _score_choices(model, tokenizer, context_ids, device)
        if pred == row["label"]:
            correct += 1
        if (i + 1) % 200 == 0:
            log_msg("INFO", f"  组1 进度: {i+1}/{total}, acc={correct/(i+1):.4f}")

    acc = correct / total
    log_msg("INFO", f"组1 完成 | acc={acc:.4f} ({correct}/{total})")
    return {"group": 1, "method": "原始 Qwen3-0.6B", "acc": acc, "correct": correct, "total": total}


def eval_group2_rag_compressed(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    knowledge_map: Dict[str, List[int]],
    device: torch.device,
) -> Dict[str, Any]:
    """组2：RAG + compressed 知识（decode knowledge_ids 后文本前缀拼接）。

    将 knowledge_ids decode 回文本，以 "Context: ...\n\n{question_prompt}" 格式输入。
    模拟 RAG 使用 LLMLingua 压缩后知识的场景。

    Args:
        model: 原始 Qwen3 AutoModelForCausalLM
        tokenizer: 对应的 tokenizer
        ds: MedQA test dataset
        knowledge_map: {key: knowledge_ids} 映射
        device: 计算设备

    Returns:
        {"group", "method", "acc", "correct", "total"}
    """
    log_msg("INFO", "组2 (RAG + compressed 知识) 评测开始...")
    pad_id = tokenizer.pad_token_id
    correct = 0
    total = len(ds)

    for i, row in enumerate(ds):
        key = row["sent1"][:200].strip()
        k_ids = knowledge_map.get(key)
        question_prompt = _build_question_prompt(row)

        if k_ids is not None:
            clean_ids = [t for t in k_ids if t != pad_id]
            compressed_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
            context = f"Context: {compressed_text}\n\n{question_prompt}"
        else:
            # fallback：无知识，等价于组1
            context = question_prompt

        context_ids = tokenizer.encode(context, add_special_tokens=False)
        pred = _score_choices(model, tokenizer, context_ids, device)
        if pred == row["label"]:
            correct += 1
        if (i + 1) % 200 == 0:
            log_msg("INFO", f"  组2 进度: {i+1}/{total}, acc={correct/(i+1):.4f}")

    acc = correct / total
    log_msg("INFO", f"组2 完成 | acc={acc:.4f} ({correct}/{total})")
    return {"group": 2, "method": "RAG + compressed 知识", "acc": acc, "correct": correct, "total": total}


def eval_group3_rag_original(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    device: torch.device,
    max_knowledge_tokens: int = 256,
) -> Dict[str, Any]:
    """组3：RAG + 原始知识（question + correct_answer，完美检索）。

    直接用 sent1 + correct_answer 作为 context，截断至 max_knowledge_tokens。
    模拟传统 RAG 完美检索到正确知识的场景。

    Args:
        model: 原始 Qwen3 AutoModelForCausalLM
        tokenizer: 对应的 tokenizer
        ds: MedQA test dataset
        device: 计算设备
        max_knowledge_tokens: 原始知识截断长度（默认 256，与数据处理一致）

    Returns:
        {"group", "method", "acc", "correct", "total"}
    """
    log_msg("INFO", "组3 (RAG + 原始知识) 评测开始...")
    correct = 0
    total = len(ds)

    for i, row in enumerate(ds):
        label = row["label"]
        original_text = f"{row['sent1']} {row[f'ending{label}']}"

        # 截断到 max_knowledge_tokens
        orig_ids = tokenizer.encode(original_text, add_special_tokens=False)
        orig_ids = orig_ids[:max_knowledge_tokens]
        original_text_trunc = tokenizer.decode(orig_ids)

        question_prompt = _build_question_prompt(row)
        context = f"Context: {original_text_trunc}\n\n{question_prompt}"
        context_ids = tokenizer.encode(context, add_special_tokens=False)

        pred = _score_choices(model, tokenizer, context_ids, device)
        if pred == row["label"]:
            correct += 1
        if (i + 1) % 200 == 0:
            log_msg("INFO", f"  组3 进度: {i+1}/{total}, acc={correct/(i+1):.4f}")

    acc = correct / total
    log_msg("INFO", f"组3 完成 | acc={acc:.4f} ({correct}/{total})")
    return {"group": 3, "method": "RAG + 原始知识（完美检索）", "acc": acc, "correct": correct, "total": total}


def eval_group4_prefix_token(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    knowledge_map: Dict[str, List[int]],
    device: torch.device,
) -> Dict[str, Any]:
    """组4：compressed knowledge_ids 直接 token 前缀拼接。

    不 decode，直接将 knowledge_ids（过滤 pad）拼接到 question prompt 的
    token ids 前面。测试模型在无专门训练时能否利用 LLMLingua token 作为 prefix。

    Args:
        model: 原始 Qwen3 AutoModelForCausalLM
        tokenizer: 对应的 tokenizer
        ds: MedQA test dataset
        knowledge_map: {key: knowledge_ids} 映射
        device: 计算设备

    Returns:
        {"group", "method", "acc", "correct", "total"}
    """
    log_msg("INFO", "组4 (Prefix token 拼接) 评测开始...")
    pad_id = tokenizer.pad_token_id
    correct = 0
    total = len(ds)

    for i, row in enumerate(ds):
        key = row["sent1"][:200].strip()
        k_ids = knowledge_map.get(key)
        question_prompt = _build_question_prompt(row)
        question_ids = tokenizer.encode(question_prompt, add_special_tokens=False)

        if k_ids is not None:
            clean_ids = [t for t in k_ids if t != pad_id]
            context_ids = clean_ids + question_ids
        else:
            context_ids = question_ids

        pred = _score_choices(model, tokenizer, context_ids, device)
        if pred == row["label"]:
            correct += 1
        if (i + 1) % 200 == 0:
            log_msg("INFO", f"  组4 进度: {i+1}/{total}, acc={correct/(i+1):.4f}")

    acc = correct / total
    log_msg("INFO", f"组4 完成 | acc={acc:.4f} ({correct}/{total})")
    return {"group": 4, "method": "Prefix token 拼接（compressed）", "acc": acc, "correct": correct, "total": total}


def _print_results_table(results: List[Dict[str, Any]]) -> None:
    """打印格式化对比表格。

    Args:
        results: 四组评测结果列表
    """
    sep = "=" * 64
    log_msg("INFO", f"\n{sep}")
    log_msg("INFO", "MedQA 四组对比评测结果")
    log_msg("INFO", sep)
    log_msg("INFO", f"{'方法':<40} {'Acc':>7}  {'Correct':>7}  {'Total':>5}")
    log_msg("INFO", "-" * 64)
    for r in results:
        log_msg(
            "INFO",
            f"组{r['group']}: {r['method']:<36} {r['acc']:>6.2%}  {r['correct']:>7}  {r['total']:>5}",
        )
    log_msg("INFO", "-" * 64)
    log_msg("INFO", f"{'[参考] Explicit-LoRA Phase2 Epoch1':<40} {'71.10%':>7}  {'905':>7}  {'1273':>5}")
    log_msg("INFO", sep)


def _load_shared_resources(model_path: str):
    """加载共享资源（模型、tokenizer、数据集）。

    Args:
        model_path: Qwen3-0.6B 本地路径

    Returns:
        (model, tokenizer, ds, device) 元组
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg("INFO", f"使用设备: {device}")

    log_msg("INFO", f"加载 Qwen3-0.6B: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    log_msg("INFO", "模型加载完成")

    log_msg("INFO", "加载 MedQA test split...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    log_msg("INFO", f"共 {len(ds)} 题")
    return model, tokenizer, ds, device


def _save_result(result: Dict[str, Any], results_dir: str, filename: str) -> None:
    """保存单组评测结果到 JSON 文件。

    Args:
        result: 单组评测结果
        results_dir: 结果保存目录
        filename: 输出文件名
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / filename
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_msg("INFO", f"结果已保存: {output_path}")


def run_group1(model_path: str, results_dir: str = "results") -> Dict[str, Any]:
    """独立运行组1：原始 Qwen3-0.6B（无任何外部知识）。

    Args:
        model_path: Qwen3-0.6B 本地路径
        results_dir: 结果保存目录

    Returns:
        {"group", "method", "acc", "correct", "total"}
    """
    model, tokenizer, ds, device = _load_shared_resources(model_path)
    result = eval_group1_baseline(model, tokenizer, ds, device)
    _save_result(result, results_dir, "compare_group1_baseline.json")
    log_msg("INFO", f"组1 结果: acc={result['acc']:.4f} ({result['correct']}/{result['total']})")
    return result


def run_group2(
    model_path: str,
    knowledge_map_path: str,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """独立运行组2：RAG + compressed 知识。

    Args:
        model_path: Qwen3-0.6B 本地路径
        knowledge_map_path: 压缩知识映射路径
        results_dir: 结果保存目录

    Returns:
        {"group", "method", "acc", "correct", "total"}
    """
    from medqa_knowledge_builder import MedQAKnowledgeBuilder

    model, tokenizer, ds, device = _load_shared_resources(model_path)
    knowledge_map = MedQAKnowledgeBuilder.load(knowledge_map_path)
    result = eval_group2_rag_compressed(model, tokenizer, ds, knowledge_map, device)
    _save_result(result, results_dir, "compare_group2_rag_compressed.json")
    log_msg("INFO", f"组2 结果: acc={result['acc']:.4f} ({result['correct']}/{result['total']})")
    return result


def run_group3(model_path: str, results_dir: str = "results") -> Dict[str, Any]:
    """独立运行组3：RAG + 原始知识（完美检索）。

    Args:
        model_path: Qwen3-0.6B 本地路径
        results_dir: 结果保存目录

    Returns:
        {"group", "method", "acc", "correct", "total"}
    """
    model, tokenizer, ds, device = _load_shared_resources(model_path)
    result = eval_group3_rag_original(model, tokenizer, ds, device)
    _save_result(result, results_dir, "compare_group3_rag_original.json")
    log_msg("INFO", f"组3 结果: acc={result['acc']:.4f} ({result['correct']}/{result['total']})")
    return result


def run_group4(
    model_path: str,
    knowledge_map_path: str,
    results_dir: str = "results",
) -> Dict[str, Any]:
    """独立运行组4：compressed token 前缀拼接。

    Args:
        model_path: Qwen3-0.6B 本地路径
        knowledge_map_path: 压缩知识映射路径
        results_dir: 结果保存目录

    Returns:
        {"group", "method", "acc", "correct", "total"}
    """
    from medqa_knowledge_builder import MedQAKnowledgeBuilder

    model, tokenizer, ds, device = _load_shared_resources(model_path)
    knowledge_map = MedQAKnowledgeBuilder.load(knowledge_map_path)
    result = eval_group4_prefix_token(model, tokenizer, ds, knowledge_map, device)
    _save_result(result, results_dir, "compare_group4_prefix_token.json")
    log_msg("INFO", f"组4 结果: acc={result['acc']:.4f} ({result['correct']}/{result['total']})")
    return result


def run_all_groups(
    model_path: str,
    knowledge_map_path: str,
    results_dir: str = "results",
) -> List[Dict[str, Any]]:
    """依次运行四组评测，输出对比表格并保存结果。

    Args:
        model_path: Qwen3-0.6B 本地路径
        knowledge_map_path: data/medqa_knowledge.jsonl 路径（test split）
        results_dir: 结果保存目录

    Returns:
        四组结果列表，每项为 {"group", "method", "acc", "correct", "total"}
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg("INFO", f"使用设备: {device}")

    # Phase 1: 加载模型和 tokenizer
    log_msg("INFO", f"加载 Qwen3-0.6B: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()
    log_msg("INFO", "模型加载完成")

    # Phase 2: 加载知识映射和数据集
    from medqa_knowledge_builder import MedQAKnowledgeBuilder
    knowledge_map = MedQAKnowledgeBuilder.load(knowledge_map_path)

    log_msg("INFO", "加载 MedQA test split...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    log_msg("INFO", f"共 {len(ds)} 题")

    # Phase 3: 依次运行四组
    results = []
    results.append(eval_group1_baseline(model, tokenizer, ds, device))
    results.append(eval_group2_rag_compressed(model, tokenizer, ds, knowledge_map, device))
    results.append(eval_group3_rag_original(model, tokenizer, ds, device))
    results.append(eval_group4_prefix_token(model, tokenizer, ds, knowledge_map, device))

    # Phase 4: 输出和保存
    _print_results_table(results)

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(results_dir) / "compare_eval.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log_msg("INFO", f"结果已保存: {output_path}")

    return results
