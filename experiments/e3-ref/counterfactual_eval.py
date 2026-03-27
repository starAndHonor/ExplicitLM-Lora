"""E1 Sanity Check：反事实知识评测模块。

通过对比正确知识 / 反事实知识 / 无知识三组评测，
验证 Fusion 模块是否真正在使用注入的知识。

指标: Knowledge Sensitivity (KS) = acc_correct - acc_counterfactual
- KS > 15%: 模块有效利用知识
- KS < 5%:  模块基本忽略知识

反事实知识构造: wrong_answer = ending[(label+1)%4]，确定性复现。
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from logger_system import log_msg


def build_counterfactual_knowledge(
    split: str,
    output_path: str,
    config: Dict[str, Any],
) -> Dict[str, List[int]]:
    """构造反事实知识映射。

    对每道题取 wrong_answer = ending[(label+1)%4]（确定性复现），
    compress(question + wrong_answer) → knowledge_ids[64]。
    Key 格式与正确知识映射完全一致（sent1[:200]）。

    Args:
        split: 数据集 split（默认 test）
        output_path: 输出 JSONL 路径
        config: 完整配置字典

    Returns:
        {key: knowledge_ids} 反事实知识映射
    """
    from medqa_knowledge_builder import MedQAKnowledgeBuilder

    eval_cfg = config["evaluation"]["medqa"]

    builder = MedQAKnowledgeBuilder(
        tokenizer_path=config["paths"]["model_dir"],
        compressor_model=config["paths"]["llmlingua_model_dir"],
        knowledge_length=eval_cfg.get("knowledge_length", 64),
        gpu_id=eval_cfg.get("compression_gpu", 4),
    )

    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split=split)
    knowledge_length = eval_cfg.get("knowledge_length", 64)
    total = len(ds)
    failed = 0

    log_msg("INFO", f"开始构建反事实知识 | split={split}, 总数={total}")

    knowledge_map: Dict[str, List[int]] = {}

    for i, row in enumerate(ds):
        # Phase 1: 取反事实答案（确定性选择，避免随机性）
        question = row["sent1"]
        label = row["label"]
        wrong_idx = (label + 1) % 4
        wrong_answer = row[f"ending{wrong_idx}"]
        source_text = f"{question} {wrong_answer}"

        # Phase 2: LLMLingua-2 压缩
        compressed = builder.compressor.compress_text(source_text)
        if compressed is None:
            failed += 1
            compressed = question[:100]  # fallback

        # Phase 3: tokenize + pad/truncate 到 knowledge_length
        tokens = builder.tokenizer.encode(compressed, add_special_tokens=False)
        tokens = tokens[:knowledge_length]
        if len(tokens) < knowledge_length:
            tokens = tokens + [builder.tokenizer.pad_token_id] * (
                knowledge_length - len(tokens)
            )

        key = question[:200].strip()
        knowledge_map[key] = tokens

        if (i + 1) % 100 == 0:
            log_msg("INFO", f"反事实知识构建进度: {i+1}/{total}")

    log_msg("INFO", f"反事实知识构建完成 | 成功: {total - failed}, 失败: {failed}")

    MedQAKnowledgeBuilder.save(knowledge_map, output_path)
    return knowledge_map


def _score_choices_injection(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    context_ids: List[int],
    knowledge_ids: torch.LongTensor,
    device: torch.device,
) -> int:
    """对注入模型做 loglikelihood 多选题评分。

    完全复用 compare_eval._score_choices() 的算法逻辑，
    唯一区别: 调用 model(input_ids, knowledge_ids) 带知识注入。

    Args:
        model: ModifiedQwen（已在 device 上，eval 模式）
        tokenizer: 对应的 tokenizer
        context_ids: 已编码的 context token id 列表
        knowledge_ids: shape [1, knowledge_length]，已在 device 上
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
            logits = model(input_t, knowledge_ids)  # [1, L, V]

        # 取 continuation 对应位置的 log-prob
        cont_start = len(context_ids) - 1
        cont_end = len(input_ids) - 1
        cont_logits = logits[0, cont_start:cont_end, :]
        cont_tokens = torch.tensor(cont_ids, dtype=torch.long, device=device)
        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_ll = log_probs.gather(1, cont_tokens.unsqueeze(-1)).squeeze(-1)
        scores.append(token_ll.sum().item())

    return scores.index(max(scores))


def _build_question_prompt(row: Dict[str, Any]) -> str:
    """构造标准 MedQA question prompt（与 compare_eval 格式一致）。

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


def eval_e1_sanity_check(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    ds: Any,
    correct_km: Dict[str, List[int]],
    cf_km: Dict[str, List[int]],
    device: torch.device,
    knowledge_length: int = 64,
) -> Dict[str, Any]:
    """在 MedQA test split 上运行三组对比评测（E1 Sanity Check）。

    组 A: 注入正确知识 → acc_correct
    组 B: 注入反事实知识 → acc_counterfactual
    组 C: 注入全 pad 知识（无知识）→ acc_no_knowledge

    KS = acc_correct - acc_counterfactual

    Args:
        model: ModifiedQwen（已加载权重，已在 device 上，eval 模式）
        tokenizer: 对应的 tokenizer
        ds: HuggingFace Dataset（MedQA test split）
        correct_km: 正确知识映射 {key: knowledge_ids}
        cf_km: 反事实知识映射 {key: knowledge_ids}
        device: 计算设备
        knowledge_length: 知识 token 长度（默认 64）

    Returns:
        包含三组 acc 和 KS 指标的字典
    """
    pad_id = tokenizer.pad_token_id
    total = len(ds)

    # 预构建全 pad 知识（无知识组）
    no_knowledge = [pad_id] * knowledge_length

    correct_a = 0  # 正确知识组
    correct_cf = 0  # 反事实知识组
    correct_nk = 0  # 无知识组

    model.eval()
    log_msg("INFO", f"E1 Sanity Check 开始 | 总题数={total}, knowledge_length={knowledge_length}")

    with torch.no_grad():
        for i, row in enumerate(ds):
            key = row["sent1"][:200].strip()
            label = row["label"]

            context = _build_question_prompt(row)
            context_ids = tokenizer.encode(context, add_special_tokens=False)

            # 获取各组知识 IDs
            k_ids_correct = correct_km.get(key, no_knowledge)
            k_ids_cf = cf_km.get(key, no_knowledge)

            # 转换为 tensor（shape [1, knowledge_length]）
            k_correct = torch.tensor([k_ids_correct], dtype=torch.long, device=device)
            k_cf = torch.tensor([k_ids_cf], dtype=torch.long, device=device)
            k_nk = torch.tensor([no_knowledge], dtype=torch.long, device=device)

            # 组 A: 正确知识
            pred_correct = _score_choices_injection(
                model, tokenizer, context_ids, k_correct, device
            )
            if pred_correct == label:
                correct_a += 1

            # 组 B: 反事实知识
            pred_cf = _score_choices_injection(
                model, tokenizer, context_ids, k_cf, device
            )
            if pred_cf == label:
                correct_cf += 1

            # 组 C: 无知识（全 pad）
            pred_nk = _score_choices_injection(
                model, tokenizer, context_ids, k_nk, device
            )
            if pred_nk == label:
                correct_nk += 1

            if (i + 1) % 200 == 0:
                log_msg(
                    "INFO",
                    f"  进度: {i+1}/{total} | "
                    f"acc_correct={correct_a/(i+1):.4f}, "
                    f"acc_cf={correct_cf/(i+1):.4f}, "
                    f"acc_nk={correct_nk/(i+1):.4f}",
                )

    acc_correct = correct_a / total
    acc_cf = correct_cf / total
    acc_nk = correct_nk / total
    ks = acc_correct - acc_cf

    return {
        "acc_correct": acc_correct,
        "acc_counterfactual": acc_cf,
        "acc_no_knowledge": acc_nk,
        "knowledge_sensitivity": ks,
        "correct_correct": correct_a,
        "correct_counterfactual": correct_cf,
        "correct_no_knowledge": correct_nk,
        "total": total,
    }


def run_e1_sanity_check(
    injection_weights: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """E1 Sanity Check 完整入口函数。

    自动确保反事实知识映射存在，加载模型和数据，运行三组对比，
    打印结构化报告并保存结果到 results/ 目录。

    Args:
        injection_weights: 注入权重路径（如 checkpoints/phase2_best）
        config: 完整配置字典

    Returns:
        eval_e1_sanity_check 的结果字典
    """
    from model_compat import create_model
    from medqa_knowledge_builder import MedQAKnowledgeBuilder

    eval_cfg = config["evaluation"]["medqa"]
    model_cfg = config["model"]["injection"]

    # Phase 1: 确保反事实知识映射存在
    cf_path = eval_cfg.get(
        "knowledge_map_counterfactual",
        "data/medqa_knowledge_counterfactual.jsonl",
    )
    if not Path(cf_path).exists():
        log_msg("INFO", f"反事实知识映射不存在，自动构建: {cf_path}")
        build_counterfactual_knowledge("test", cf_path, config)
    else:
        log_msg("INFO", f"反事实知识映射已存在: {cf_path}")

    # Phase 2: 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg("INFO", f"使用设备: {device}")

    log_msg("INFO", "创建 ModifiedQwen 模型...")
    model = create_model(
        model_path=config["paths"]["model_dir"],
        injection_method=model_cfg["method"],
        injection_layers=model_cfg["layers"],
        encoder_depth=model_cfg.get("encoder_depth", 6),
        knowledge_adapter=model_cfg.get("knowledge_adapter", False),
        device=str(device),
    )
    model.load_injection_weights(injection_weights)
    log_msg("INFO", f"已加载注入权重: {injection_weights}")
    model = model.to(device).eval()

    # Phase 3: 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config["paths"]["model_dir"], trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Phase 4: 加载数据集和知识映射
    log_msg("INFO", "加载 MedQA test split...")
    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    log_msg("INFO", f"共 {len(ds)} 题")

    correct_km_path = eval_cfg["knowledge_map"]
    log_msg("INFO", f"加载正确知识映射: {correct_km_path}")
    correct_km = MedQAKnowledgeBuilder.load(correct_km_path)

    log_msg("INFO", f"加载反事实知识映射: {cf_path}")
    cf_km = MedQAKnowledgeBuilder.load(cf_path)

    # Phase 5: 运行三组对比评测
    knowledge_length = eval_cfg.get("knowledge_length", 64)
    results = eval_e1_sanity_check(
        model=model,
        tokenizer=tokenizer,
        ds=ds,
        correct_km=correct_km,
        cf_km=cf_km,
        device=device,
        knowledge_length=knowledge_length,
    )

    # Phase 6: 打印结构化报告
    sep = "=" * 64
    log_msg("INFO", f"\n{sep}")
    log_msg("INFO", "E1 Sanity Check 完成")
    log_msg("INFO", sep)
    log_msg("INFO", f"acc_correct_knowledge:        {results['acc_correct']:.4f}  ({results['correct_correct']}/{results['total']})")
    log_msg("INFO", f"acc_counterfactual_knowledge: {results['acc_counterfactual']:.4f}  ({results['correct_counterfactual']}/{results['total']})")
    log_msg("INFO", f"acc_no_knowledge:             {results['acc_no_knowledge']:.4f}  ({results['correct_no_knowledge']}/{results['total']})")
    log_msg("INFO", f"knowledge_sensitivity (KS):  {results['knowledge_sensitivity']:+.4f}")
    if results["knowledge_sensitivity"] > 0.15:
        log_msg("INFO", "结论: KS > 15% → Fusion 模块有效利用知识 ✓")
    elif results["knowledge_sensitivity"] < 0.05:
        log_msg("INFO", "结论: KS < 5% → Fusion 模块基本忽略知识 ✗")
    else:
        log_msg("INFO", "结论: 5% <= KS <= 15% → Fusion 模块部分利用知识 △")
    log_msg("INFO", sep)

    # Phase 7: 保存结果
    tag = Path(injection_weights).name
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"e1_sanity_check_{tag}.json"
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_msg("INFO", f"结果保存: {output_path}")

    return results
