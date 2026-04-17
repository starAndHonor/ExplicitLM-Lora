"""E9: 顺序写入 + 闭卷探测 Benchmark。

流程：
  Step 1: 从三个数据集（MedQA / ARC / MMLU）各随机抽取 N=100 条，
          使用 row["original_text"]（question + 正确答案）作为知识源。
  Step 2: 对每条 original_text 使用 LLMLingua 在线压缩为 fusion token IDs（单视图）。
  Step 3: 将新知识条目逐一 overlay 到 dense index（每次替换一个随机位置，
          严格按时序一条一条写入，模拟真实场景）。
  Step 4 / 5: 写入完成后，从 N 条中随机选 M=10 条做闭卷探测：
              dense retriever 检索 → Phase3 injection 模型回答。
  评测：比较"写入前"与"写入后"的检索命中率与 QA 准确率。
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from config import Config
from experiments.e1.counterfactual_eval import KnowledgeCompressor
from experiments.e2.common import PROJECT_ROOT, build_injection_model, setup_logging
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices_injection
from experiments.e3.data_loading import load_arc_rows, load_medqa_rows, load_mmlu_rows
from experiments.e8.common import build_fusion_encoder_and_tokenizer, encode_fusion_texts
from models.qwen_wrapper import KnowledgeEncoder
from retrieval.dense_index import DenseKnowledgeIndex

logger = logging.getLogger(__name__)

DATASETS = ("medqa", "arc", "mmlu")



# ── 压缩工具（完全对齐 chat_overlay_answer.py）────────────────────────────

def _compress(
    compressor: Optional[KnowledgeCompressor],
    tokenizer: AutoTokenizer,
    text: str,
    fusion_length: int,
    rate: float,
    backend: str,
) -> Tuple[str, List[int]]:
    """返回 (compressed_text, fusion_token_ids[fusion_length])。"""
    if backend == "mock_tokenize":
        ids = tokenizer.encode(text, add_special_tokens=False)[:fusion_length]
    else:
        assert compressor is not None
        compressed = compressor.compress_text(text, target_token=fusion_length)
        if not compressed:
            compressed = text[:200]
        ids = tokenizer.encode(compressed, add_special_tokens=False)[:fusion_length]
        text = compressed

    if len(ids) < fusion_length:
        ids += [tokenizer.pad_token_id] * (fusion_length - len(ids))

    decoded = tokenizer.decode(
        [x for x in ids if x != tokenizer.pad_token_id], skip_special_tokens=True
    ).strip()
    return decoded, ids


# ── 数据加载 ──────────────────────────────────────────────────────────────

def _load_rows(dataset: str) -> List[Dict[str, Any]]:
    if dataset == "medqa":
        return load_medqa_rows(limit=None)
    if dataset == "arc":
        return load_arc_rows(limit=None)
    if dataset == "mmlu":
        return load_mmlu_rows(limit=None)
    raise ValueError(f"unsupported dataset: {dataset}")


def _sample_rows(dataset: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """从 dataset 中随机采样 n 条（保证有 original_text 字段）。"""
    rows = _load_rows(dataset)
    eligible = [r for r in rows if r.get("original_text", "").strip()]
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(eligible), generator=generator).tolist()[:n]
    return [eligible[i] for i in idx]


# ── 检索 & 推理 ───────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_query(
    text: str,
    encoder: KnowledgeEncoder,
    tokenizer: AutoTokenizer,
    fusion_length: int,
    device: torch.device,
) -> torch.Tensor:
    """将 query 文本原始截断后编码为 embedding [1, hidden_dim]（CPU）。

    双视图方案：检索侧统一使用原始 token 截断（前 fusion_length 个 token），
    不做 LLMLingua 压缩，与写入侧 r0(original_text[:64]) 对称对齐。
    """
    enc = tokenizer(
        [text],
        max_length=fusion_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
        return_tensors="pt",
    )
    emb = encoder.encode_mean(
        enc["input_ids"].to(device),
        enc["attention_mask"].long().to(device),
    )
    return emb.detach().cpu()


def _probe_single(
    row: Dict[str, Any],
    index: DenseKnowledgeIndex,
    encoder: KnowledgeEncoder,
    enc_tokenizer: AutoTokenizer,
    phase3_model: torch.nn.Module,
    phase3_tokenizer: AutoTokenizer,
    fusion_length: int,
    device: torch.device,
    query_mode: str,
) -> Dict[str, Any]:
    """对单条 row 做闭卷探测（检索 + 注入 + QA），直接操作 in-memory index。"""
    # Step 4: 检索（原始截断，与写入侧 r0(original_text[:64]) 对称）
    if query_mode == "question_only":
        query = row["question"]
    else:
        query = build_multiple_choice_prompt(row["question"], row["choices"])

    q_emb = _encode_query(query, encoder, enc_tokenizer, fusion_length, device)
    search = index.search(q_emb, top_k=1)

    idx = int(search.indices[0][0].item())
    valid = bool(search.valid_mask[0][0].item())
    retrieved_key = index.keys[idx] if valid and idx >= 0 else ""
    retrieval_hit = retrieved_key == str(row["key"])

    # Step 5: 注入 + 回答
    knowledge_ids = search.fusion_ids[0, 0].unsqueeze(0).to(device)  # [1, fusion_length]

    prompt = build_multiple_choice_prompt(row["question"], row["choices"])
    context_ids = phase3_tokenizer.encode(prompt, add_special_tokens=False)

    with torch.no_grad():
        pred = score_choices_injection(
            phase3_model, phase3_tokenizer, context_ids, knowledge_ids, device
        )

    return {
        "key": str(row["key"]),
        "question": row["question"][:120],
        "label": row["label"],
        "pred": pred,
        "retrieved_key": retrieved_key,
        "retrieval_hit": retrieval_hit,
        "qa_correct": pred == row["label"],
    }


# ── 单数据集实验 ──────────────────────────────────────────────────────────

def _run_dataset(
    dataset: str,
    cfg: Config,
    base_index_path: Path,
    encoder: KnowledgeEncoder,
    enc_tokenizer: AutoTokenizer,
    phase3_model: torch.nn.Module,
    phase3_tokenizer: AutoTokenizer,
    device: torch.device,
    n_writes: int,
    n_probes: int,
    seed: int,
    query_mode: str,
    compressor: Optional[KnowledgeCompressor],
    compression_rate: float,
    compression_backend: str,
) -> Dict[str, Any]:
    fusion_length = cfg.model.fusion_length
    logger.info("[E9][%s] 加载 %d 条写入数据 seed=%d", dataset, n_writes, seed)
    write_rows = _sample_rows(dataset, n_writes, seed)

    # 选出 M 条探测行（从写入集中随机选）
    probe_rng = random.Random(seed + 99)
    probe_rows = probe_rng.sample(write_rows, min(n_probes, len(write_rows)))

    # 加载 base index 副本（in-memory，不修改磁盘）
    current_index = DenseKnowledgeIndex.load(base_index_path)
    logger.info("[E9][%s] base index loaded | size=%d", dataset, len(current_index))

    # ── 写入前评测（baseline：base index 无任务知识）──
    logger.info("[E9][%s] 写入前探测 %d 条 ...", dataset, len(probe_rows))
    before_results = [
        _probe_single(row, current_index, encoder, enc_tokenizer,
                      phase3_model, phase3_tokenizer, fusion_length, device, query_mode)
        for row in probe_rows
    ]
    before_retrieval = sum(r["retrieval_hit"] for r in before_results) / max(len(before_results), 1)
    before_qa = sum(r["qa_correct"] for r in before_results) / max(len(before_results), 1)
    logger.info("[E9][%s] before | retrieval_hit=%.3f qa_acc=%.3f", dataset, before_retrieval, before_qa)

    # ── 顺序逐条写入 ──
    write_log: List[Dict[str, Any]] = []
    rng_replace = random.Random(seed + 1)

    for i, row in enumerate(write_rows):
        original_text = row["original_text"]
        t0 = time.time()

        # Step 2a: 检索 embedding — original_text 原始截断（前 fusion_length 个 token）
        # 双视图：embedding 与 query 均使用原始截断，确保检索侧对齐
        emb = encode_fusion_texts(
            cfg=cfg,
            encoder=encoder,
            tokenizer=enc_tokenizer,
            texts=[original_text],
            device=device,
        )  # [1, hidden_dim]

        # Step 2b: 注入 fusion_ids — LLMLingua 压缩（信息密度高，供 cross-attention 注入）
        compressed_text, fusion_ids_list = _compress(
            compressor=compressor,
            tokenizer=enc_tokenizer,
            text=original_text,
            fusion_length=fusion_length,
            rate=compression_rate,
            backend=compression_backend,
        )

        fusion_ids_tensor = torch.tensor([fusion_ids_list], dtype=torch.long)  # [1, fusion_length]

        # Step 3: 替换 index 中的一个随机位置（one-by-one）
        replace_pos = rng_replace.randrange(len(current_index))
        replaced_key = current_index.keys[replace_pos]
        current_index.delete_by_keys([replaced_key])
        current_index.add_entries(
            embeddings=emb,
            fusion_ids=fusion_ids_tensor,
            keys=[str(row["key"])],
            texts=[compressed_text],
            replace_existing=True,
        )

        latency_ms = (time.time() - t0) * 1000.0

        # ── 写一条查一条诊断（仅前 3 条）──
        if i < 3:
            # 用 original_text 自查（与写入 embedding 完全一致，理论上必须命中）
            self_emb = encode_fusion_texts(cfg=cfg, encoder=encoder, tokenizer=enc_tokenizer,
                                           texts=[original_text], device=device)
            self_search = current_index.search(self_emb.cpu(), top_k=1)
            self_idx = int(self_search.indices[0][0].item())
            self_valid = bool(self_search.valid_mask[0][0].item())
            self_key = current_index.keys[self_idx] if self_valid and self_idx >= 0 else ""
            self_hit = self_key == str(row["key"])
            # 用 raw question 查（双视图检索侧）
            q_emb = _encode_query(row["question"], encoder, enc_tokenizer, fusion_length, device)
            q_search = current_index.search(q_emb, top_k=1)
            q_idx = int(q_search.indices[0][0].item())
            q_valid = bool(q_search.valid_mask[0][0].item())
            q_key = current_index.keys[q_idx] if q_valid and q_idx >= 0 else ""
            q_hit = q_key == str(row["key"])
            logger.info(
                "[E9][%s] diag step=%d | self_hit=%s(valid=%s) q_hit=%s(valid=%s) | "
                "original='%s...' question='%s...'",
                dataset, i + 1, self_hit, self_valid, q_hit, q_valid,
                original_text[:40], row["question"][:40],
            )

        write_log.append({
            "step": i + 1,
            "key": str(row["key"]),
            "replaced_key": replaced_key,
            "latency_ms": latency_ms,
        })

        if (i + 1) % 20 == 0:
            logger.info("[E9][%s] 已写入 %d / %d | 最近耗时 %.1f ms", dataset, i + 1, n_writes, latency_ms)

    logger.info("[E9][%s] 全部 %d 条写入完成", dataset, len(write_rows))

    # ── 写入后评测（probe 同一批 10 条）──
    logger.info("[E9][%s] 写入后探测 %d 条 ...", dataset, len(probe_rows))
    after_results = [
        _probe_single(row, current_index, encoder, enc_tokenizer,
                      phase3_model, phase3_tokenizer, fusion_length, device, query_mode)
        for row in probe_rows
    ]
    after_retrieval = sum(r["retrieval_hit"] for r in after_results) / max(len(after_results), 1)
    after_qa = sum(r["qa_correct"] for r in after_results) / max(len(after_results), 1)
    logger.info(
        "[E9][%s] after  | retrieval_hit=%.3f qa_acc=%.3f | delta_retrieval=+%.3f delta_qa=+%.3f",
        dataset,
        after_retrieval,
        after_qa,
        after_retrieval - before_retrieval,
        after_qa - before_qa,
    )

    return {
        "dataset": dataset,
        "n_writes": len(write_rows),
        "n_probes": len(probe_rows),
        "metrics": {
            "before_retrieval_top1": before_retrieval,
            "before_qa_acc": before_qa,
            "after_retrieval_top1": after_retrieval,
            "after_qa_acc": after_qa,
            "delta_retrieval": after_retrieval - before_retrieval,
            "delta_qa": after_qa - before_qa,
            "mean_write_latency_ms": sum(w["latency_ms"] for w in write_log) / max(len(write_log), 1),
            "total_write_time_ms": sum(w["latency_ms"] for w in write_log),
        },
        "before_probe_results": before_results,
        "after_probe_results": after_results,
        "write_log": write_log,
    }


# ── 主入口 ────────────────────────────────────────────────────────────────

def run_e9(
    cfg: Config,
    base_index_path: str,
    phase3_weights: str,
    device: str = "cuda:0",
    n_writes: int = 100,
    n_probes: int = 10,
    seed: int = 0,
    query_mode: str = "question_only",
    compression_backend: str = "llmlingua",
    compression_rate: float = 0.25,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """E9 主流程：对三个数据集各顺序写入 N 条，探测 M 条。"""
    setup_logging()
    logger.info("================ E9 Sequential Write + Closed-book Probe ================")
    device_obj = torch.device(device)

    base_index_resolved = Path(base_index_path) if Path(base_index_path).is_absolute() \
        else (PROJECT_ROOT / base_index_path).resolve()
    phase3_resolved = Path(phase3_weights) if Path(phase3_weights).is_absolute() \
        else (PROJECT_ROOT / phase3_weights).resolve()

    logger.info("base_index=%s | phase3=%s | fusion_length=%d | backend=%s",
                base_index_resolved, phase3_resolved, cfg.model.fusion_length, compression_backend)

    # ── 初始化 encoder & phase3 model（共享，跨数据集复用）──
    encoder, enc_tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device_obj)
    phase3_model, phase3_tokenizer = build_injection_model(
        cfg, str(phase3_resolved), device=str(device_obj), log_prefix="E9"
    )

    # ── 初始化 LLMLingua compressor（仅 llmlingua 模式）──
    compressor: Optional[KnowledgeCompressor] = None
    if compression_backend != "mock_tokenize":
        gpu_id = device_obj.index if device_obj.type == "cuda" else None
        logger.info("[E9] 初始化 LLMLingua compressor model=%s rate=%.2f ...",
                    cfg.paths.llmlingua_model_dir, compression_rate)
        compressor = KnowledgeCompressor(
            model_name=cfg.paths.llmlingua_model_dir,
            compression_rate=compression_rate,
            gpu_id=gpu_id,
        )

    # ── 逐数据集运行 ──
    dataset_results: Dict[str, Any] = {}
    for ds in DATASETS:
        logger.info("===== [E9] 数据集: %s =====", ds)
        dataset_results[ds] = _run_dataset(
            dataset=ds,
            cfg=cfg,
            base_index_path=base_index_resolved,
            encoder=encoder,
            enc_tokenizer=enc_tokenizer,
            phase3_model=phase3_model,
            phase3_tokenizer=phase3_tokenizer,
            device=device_obj,
            n_writes=n_writes,
            n_probes=n_probes,
            seed=seed,
            query_mode=query_mode,
            compressor=compressor,
            compression_rate=compression_rate,
            compression_backend=compression_backend,
        )

    # ── 汇总 ──
    overall_before_retrieval = sum(
        r["metrics"]["before_retrieval_top1"] for r in dataset_results.values()
    ) / len(DATASETS)
    overall_after_retrieval = sum(
        r["metrics"]["after_retrieval_top1"] for r in dataset_results.values()
    ) / len(DATASETS)
    overall_before_qa = sum(
        r["metrics"]["before_qa_acc"] for r in dataset_results.values()
    ) / len(DATASETS)
    overall_after_qa = sum(
        r["metrics"]["after_qa_acc"] for r in dataset_results.values()
    ) / len(DATASETS)

    output: Dict[str, Any] = {
        "experiment": "e9",
        "n_writes_per_dataset": n_writes,
        "n_probes_per_dataset": n_probes,
        "seed": seed,
        "fusion_length": cfg.model.fusion_length,
        "query_mode": query_mode,
        "compression_backend": compression_backend,
        "compression_rate": compression_rate,
        "phase3_weights": str(phase3_resolved),
        "base_index": str(base_index_resolved),
        "overall_metrics": {
            "before_retrieval_top1": overall_before_retrieval,
            "after_retrieval_top1": overall_after_retrieval,
            "before_qa_acc": overall_before_qa,
            "after_qa_acc": overall_after_qa,
            "delta_retrieval": overall_after_retrieval - overall_before_retrieval,
            "delta_qa": overall_after_qa - overall_before_qa,
        },
        "datasets": dataset_results,
    }

    logger.info(
        "E9 Overall | before_retrieval=%.3f after_retrieval=%.3f | before_qa=%.3f after_qa=%.3f",
        overall_before_retrieval, overall_after_retrieval, overall_before_qa, overall_after_qa,
    )

    if output_path is not None:
        out = Path(output_path) if Path(output_path).is_absolute() \
            else (PROJECT_ROOT / output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("结果已保存至 %s", out)

    return output
