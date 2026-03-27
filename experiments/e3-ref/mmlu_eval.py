"""E2 MMLU 评测模块。

构造 MMLU 知识映射（Oracle: question + correct_answer → LLMLingua 压缩 → 64 tokens），
并提供 loglikelihood 多选题评测（与 MedQA 评测方式一致）。

知识来源: compress(question + choices[answer])，Oracle 设置
Key 格式: question[:200].strip()
评测方式: loglikelihood（对 " A"/" B"/" C"/" D" 各做 forward，取最高 log-prob）
"""

from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from compare_eval import _score_choices
from counterfactual_eval import _score_choices_injection
from logger_system import log_msg


def build_mmlu_knowledge(
    output_path: str,
    config: Dict[str, Any],
    split: str = "test",
    limit: Optional[int] = None,
) -> Dict[str, List[int]]:
    """构造 MMLU 知识映射（Oracle 设置）。

    source_text = question + " " + choices[answer]
    → LLMLingua-2 压缩 → tokenize → knowledge_ids[64]

    Args:
        output_path: 输出 JSONL 路径
        config: 完整配置字典
        split: MMLU split（默认 test）
        limit: 限制题数（None 表示全量）

    Returns:
        {key: knowledge_ids} 知识映射
    """
    from medqa_knowledge_builder import MedQAKnowledgeBuilder
    from data_builder.compressor import KnowledgeCompressor

    knowledge_length = config["evaluation"]["medqa"].get("knowledge_length", 64)
    compression_gpu = config["evaluation"]["medqa"].get("compression_gpu", 0)

    tokenizer = AutoTokenizer.from_pretrained(
        config["paths"]["model_dir"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    compressor = KnowledgeCompressor(
        model_name=config["paths"]["llmlingua_model_dir"],
        compression_rate=0.25,
        gpu_id=compression_gpu,
    )

    ds = load_dataset("cais/mmlu", "all", split=split)
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    total = len(ds)
    failed = 0
    knowledge_map: Dict[str, List[int]] = {}

    log_msg("INFO", f"MMLU 知识构建开始 | split={split}, 总数={total}")

    for i, row in enumerate(ds):
        question = row["question"]
        answer_text = row["choices"][row["answer"]]
        source_text = f"{question} {answer_text}"
        key = question[:200].strip()

        # LLMLingua-2 压缩
        compressed = compressor.compress_text(source_text)
        if compressed is None:
            failed += 1
            compressed = source_text[:100]

        # tokenize + pad/truncate
        tokens = tokenizer.encode(compressed, add_special_tokens=False)
        tokens = tokens[:knowledge_length]
        if len(tokens) < knowledge_length:
            tokens = tokens + [tokenizer.pad_token_id] * (
                knowledge_length - len(tokens)
            )

        knowledge_map[key] = tokens

        if (i + 1) % 500 == 0:
            log_msg("INFO", f"MMLU 知识构建进度: {i+1}/{total}")

    log_msg(
        "INFO",
        f"MMLU 知识构建完成 | 成功: {len(knowledge_map)}, 压缩失败: {failed}",
    )

    MedQAKnowledgeBuilder.save(knowledge_map, output_path)
    return knowledge_map


def _build_mmlu_prompt(row: Dict[str, Any]) -> str:
    """构造 MMLU question prompt（4 选 1 格式，与 MedQA 对齐）。

    Args:
        row: MMLU 一条记录（含 question, choices）

    Returns:
        格式化后的 prompt 字符串
    """
    choices = row["choices"]
    labels = ["A", "B", "C", "D"]
    answers_str = "".join(f"{l}. {c}\n" for l, c in zip(labels, choices))
    return f"Question: {row['question']}\n{answers_str}Answer:"


def eval_mmlu(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    device: torch.device,
    knowledge_map: Optional[Dict[str, List[int]]] = None,
    knowledge_length: int = 64,
    is_injection: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """MMLU loglikelihood 评测。

    复用 compare_eval._score_choices（baseline）
    和 counterfactual_eval._score_choices_injection（injection）。

    Args:
        model: AutoModelForCausalLM（baseline）或 ModifiedQwen（Fusion）
        tokenizer: tokenizer
        ds: MMLU HuggingFace Dataset
        device: 计算设备
        knowledge_map: 知识映射（None 时 injection 模式用全 pad）
        knowledge_length: 知识 token 长度
        is_injection: True=ModifiedQwen, False=AutoModelForCausalLM
        limit: 评测题数上限（None 表示全量）

    Returns:
        {"acc": float, "correct": int, "total": int}
    """
    pad_id = tokenizer.pad_token_id
    pad_knowledge = [pad_id] * knowledge_length

    actual_ds = ds
    if limit and limit < len(ds):
        if hasattr(ds, "select"):
            actual_ds = ds.select(range(limit))
        else:
            actual_ds = ds[:limit]

    total = len(actual_ds)
    correct = 0

    model.eval()
    log_msg(
        "INFO",
        f"MMLU 评测开始 | 总数={total}, "
        f"is_injection={is_injection}, has_knowledge={knowledge_map is not None}",
    )

    with torch.no_grad():
        for i, row in enumerate(actual_ds):
            question = row["question"]
            key = question[:200].strip()
            label = row["answer"]

            context = _build_mmlu_prompt(row)
            context_ids = tokenizer.encode(context, add_special_tokens=False)

            if is_injection:
                k_ids = (
                    knowledge_map.get(key, pad_knowledge)
                    if knowledge_map is not None
                    else pad_knowledge
                )
                k_tensor = torch.tensor([k_ids], dtype=torch.long, device=device)
                pred = _score_choices_injection(
                    model, tokenizer, context_ids, k_tensor, device
                )
            else:
                pred = _score_choices(model, tokenizer, context_ids, device)

            if pred == label:
                correct += 1

            if (i + 1) % 500 == 0:
                log_msg(
                    "INFO",
                    f"  MMLU 进度: {i+1}/{total}, acc={correct/(i+1):.4f}",
                )

    acc = correct / total if total > 0 else 0.0
    log_msg("INFO", f"MMLU 评测完成 | acc={acc:.4f} ({correct}/{total})")

    return {"acc": acc, "correct": correct, "total": total}
