"""E2 ARC-Challenge 评测模块。

构造 ARC 知识映射（Oracle: question + correct_answer → LLMLingua 压缩 → 64 tokens），
并提供 loglikelihood 多选题评测（与 MedQA/MMLU 评测方式一致）。

数据集: allenai/ai2_arc, ARC-Challenge config
知识来源: compress(question + correct_answer)，Oracle 设置
Key 格式: question[:200].strip()
评测方式: loglikelihood（对 " A"/" B"/" C"/" D" 各做 forward，取最高 log-prob）
仅保留恰好 4 选项的题（跳过 3/5 选项题），与 MedQA/MMLU 统一。
"""

from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from compare_eval import _score_choices
from counterfactual_eval import _score_choices_injection
from logger_system import log_msg


# answerKey → 选项索引映射（ARC 数据集中 answerKey 可能是字母或数字）
_ANSWER_KEY_TO_INDEX: Dict[str, int] = {
    "A": 0, "B": 1, "C": 2, "D": 3,
    "1": 0, "2": 1, "3": 2, "4": 3,
}


def answer_key_to_index(key: str) -> int:
    """将 ARC answerKey 转换为选项索引。

    Args:
        key: 答案标签（"A"/"B"/"C"/"D" 或 "1"/"2"/"3"/"4"）

    Returns:
        选项索引（0-3）

    Raises:
        ValueError: 不支持的 answerKey
    """
    if key not in _ANSWER_KEY_TO_INDEX:
        raise ValueError(f"不支持的 answerKey: {key!r}")
    return _ANSWER_KEY_TO_INDEX[key]


def build_arc_knowledge(
    output_path: str,
    config: Dict[str, Any],
    split: str = "test",
    limit: Optional[int] = None,
) -> Dict[str, List[int]]:
    """构造 ARC-Challenge 知识映射（Oracle 设置）。

    source_text = question + " " + correct_answer
    → LLMLingua-2 压缩 → tokenize → knowledge_ids[64]

    仅处理恰好 4 选项的题，跳过其余。

    Args:
        output_path: 输出 JSONL 路径
        config: 完整配置字典
        split: ARC split（默认 test）
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

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    if limit and limit < len(ds):
        ds = ds.select(range(limit))

    total = len(ds)
    skipped = 0
    failed = 0
    knowledge_map: Dict[str, List[int]] = {}

    log_msg("INFO", f"ARC 知识构建开始 | split={split}, 总数={total}")

    for i, row in enumerate(ds):
        # Phase 1: 过滤非 4 选项题
        choices_text = row["choices"]["text"]
        if len(choices_text) != 4:
            skipped += 1
            continue

        question = row["question"]
        answer_idx = answer_key_to_index(row["answerKey"])
        correct_answer = choices_text[answer_idx]
        source_text = f"{question} {correct_answer}"
        key = question[:200].strip()

        # Phase 2: LLMLingua-2 压缩
        compressed = compressor.compress_text(source_text)
        if compressed is None:
            failed += 1
            compressed = source_text[:100]

        # Phase 3: tokenize + pad/truncate
        tokens = tokenizer.encode(compressed, add_special_tokens=False)
        tokens = tokens[:knowledge_length]
        if len(tokens) < knowledge_length:
            tokens = tokens + [tokenizer.pad_token_id] * (
                knowledge_length - len(tokens)
            )

        knowledge_map[key] = tokens

        if (i + 1) % 200 == 0:
            log_msg("INFO", f"ARC 知识构建进度: {i+1}/{total}")

    log_msg(
        "INFO",
        f"ARC 知识构建完成 | "
        f"成功: {len(knowledge_map)}, 跳过(非4选项): {skipped}, 压缩失败: {failed}",
    )

    MedQAKnowledgeBuilder.save(knowledge_map, output_path)
    return knowledge_map


def _build_arc_prompt(row: Dict[str, Any]) -> str:
    """构造 ARC question prompt（4 选 1 格式，与 MedQA/MMLU 对齐）。

    Args:
        row: ARC 一条记录（含 question, choices）

    Returns:
        格式化后的 prompt 字符串
    """
    choices_text = row["choices"]["text"]
    labels = ["A", "B", "C", "D"]
    answers_str = "".join(f"{l}. {c}\n" for l, c in zip(labels, choices_text))
    return f"Question: {row['question']}\n{answers_str}Answer:"


def eval_arc(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    device: torch.device,
    knowledge_map: Optional[Dict[str, List[int]]] = None,
    knowledge_length: int = 64,
    is_injection: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """ARC-Challenge loglikelihood 评测。

    复用 compare_eval._score_choices（baseline）
    和 counterfactual_eval._score_choices_injection（injection）。
    仅评测恰好 4 选项的题。

    Args:
        model: AutoModelForCausalLM（baseline）或 ModifiedQwen（Fusion）
        tokenizer: tokenizer
        ds: ARC HuggingFace Dataset
        device: 计算设备
        knowledge_map: 知识映射（None 时 injection 模式用全 pad）
        knowledge_length: 知识 token 长度
        is_injection: True=ModifiedQwen, False=AutoModelForCausalLM
        limit: 评测题数上限（None 表示全量）

    Returns:
        {"acc": float, "correct": int, "total": int, "skipped": int}
    """
    pad_id = tokenizer.pad_token_id
    pad_knowledge = [pad_id] * knowledge_length

    actual_ds = ds
    if limit and limit < len(ds):
        if hasattr(ds, "select"):
            actual_ds = ds.select(range(limit))
        else:
            actual_ds = ds[:limit]

    total_rows = len(actual_ds)
    correct = 0
    total = 0
    skipped = 0

    model.eval()
    log_msg(
        "INFO",
        f"ARC 评测开始 | 总行数={total_rows}, "
        f"is_injection={is_injection}, has_knowledge={knowledge_map is not None}",
    )

    with torch.no_grad():
        for i, row in enumerate(actual_ds):
            # 过滤非 4 选项题
            if len(row["choices"]["text"]) != 4:
                skipped += 1
                continue

            question = row["question"]
            key = question[:200].strip()
            label = answer_key_to_index(row["answerKey"])

            context = _build_arc_prompt(row)
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
            total += 1

            if (i + 1) % 200 == 0:
                log_msg(
                    "INFO",
                    f"  ARC 进度: {i+1}/{total_rows}, acc={correct/total:.4f}",
                )

    acc = correct / total if total > 0 else 0.0
    log_msg(
        "INFO",
        f"ARC 评测完成 | acc={acc:.4f} ({correct}/{total}), skipped={skipped}",
    )

    return {"acc": acc, "correct": correct, "total": total, "skipped": skipped}
