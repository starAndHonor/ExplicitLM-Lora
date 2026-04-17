#!/usr/bin/env python
from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e1.counterfactual_eval import KnowledgeCompressor
from experiments.e2.common import build_injection_model, get_model_path
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices_injection
from experiments.e8.common import (
    build_fusion_encoder_and_tokenizer,
    encode_fusion_texts,
    resolve_path,
)
from retrieval.dense_index import DenseKnowledgeIndex

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-sample pipeline: dataset original knowledge -> LLMLingua compression -> one-entry overlay -> retrieval + fusion answer"
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/default.yaml"))
    parser.add_argument("--override", nargs="?", action="append", help="config overrides")
    parser.add_argument("--dataset", choices=["medqa", "arc", "mmlu"], default="medqa")
    parser.add_argument("--base-index", required=True, help="Base dense index path used before single-entry overlay")
    parser.add_argument("--phase3-ckpt", required=True, help="Phase3 checkpoint used for fusion answering")
    parser.add_argument("--question", required=True)
    parser.add_argument("--option-a", required=True)
    parser.add_argument("--option-b", required=True)
    parser.add_argument("--option-c", required=True)
    parser.add_argument("--option-d", required=True)
    parser.add_argument("--source-key", default="", help="Knowledge key to fetch original knowledge; defaults to question[:200].strip()")
    parser.add_argument("--source-question", default="", help="Use another source question to fetch dataset-provided original knowledge")
    parser.add_argument("--source-original-text", default="", help="Directly provide original uncompressed knowledge text (no answer), bypassing dataset lookup")
    parser.add_argument("--no-answer-in-knowledge", action="store_true",
                        help="Strip correct answer from dataset original_text before compression (不带答案版本). "
                             "dataset original_text = question + correct_answer; this flag uses question-only as knowledge source.")
    parser.add_argument("--query-mode", choices=["question_only", "question_choices"], default="question_only")
    parser.add_argument("--anchor-source", choices=["original_text", "compressed_decode"], default="original_text")
    parser.add_argument(
        "--compression-backend",
        choices=["llmlingua", "mock_tokenize"],
        default="llmlingua",
        help="Knowledge compression backend; use mock_tokenize to mimic online compression when LLMLingua is unavailable",
    )
    parser.add_argument("--compression-rate", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--artifacts-dir",
        default=str(PROJECT_ROOT / "results" / "single_overlay"),
        help="Directory to save one-row constructed data artifacts and optional overlay index",
    )
    parser.add_argument("--save-overlay-index", action="store_true", help="Persist the overlaid dense index to artifacts_dir")
    parser.add_argument("--json", action="store_true", help="Print final result as JSON")
    return parser.parse_args()


def _parse_overrides(overrides: List[str] | None) -> Dict[str, Any]:
    if overrides is None:
        return {}
    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"invalid override (missing '='): {item}")
        key, value = item.split("=", 1)
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
            continue
        try:
            result[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            result[key] = float(value)
            continue
        except ValueError:
            pass
        result[key] = value
    return result


def _dataset_knowledge_path(dataset: str, fusion_length: int) -> Path:
    if fusion_length == 64:
        return resolve_path(f"data/{dataset}_knowledge.jsonl")
    return resolve_path(f"data/{dataset}_knowledge_k{fusion_length}.jsonl")


def _load_original_knowledge_map(path: Path) -> Dict[str, str]:
    """从 k-specific knowledge.jsonl 读取 text/original_text 字段作为原文映射。"""
    result: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("original_text") or ""
            result[str(obj["key"])] = str(text).strip()
    return result


def _build_retrieval_query(args: argparse.Namespace) -> str:
    if args.query_mode == "question_only":
        return args.question
    return build_multiple_choice_prompt(
        args.question,
        [args.option_a, args.option_b, args.option_c, args.option_d],
    )


def _build_compressor(
    cfg: Any,
    compression_rate: float,
    device: torch.device,
    backend: str,
) -> Optional[KnowledgeCompressor]:
    """初始化 LLMLingua compressor（mock_tokenize 模式返回 None）。"""
    if backend == "mock_tokenize":
        return None
    gpu_id = device.index if device.type == "cuda" else None
    return KnowledgeCompressor(
        model_name=cfg.paths.llmlingua_model_dir,
        compression_rate=compression_rate,
        gpu_id=gpu_id,
    )


def _compress_text_to_ids(
    tokenizer: AutoTokenizer,
    source_text: str,
    compressor: Optional[KnowledgeCompressor],
    backend: str,
    fusion_length: int,
) -> tuple[str, List[int]]:
    """压缩源文本为 fusion token IDs（使用已初始化的 compressor）。

    参数：
        fusion_length: 目标 token 长度（target_token）。
    """
    if backend == "mock_tokenize":
        token_ids = tokenizer.encode(source_text, add_special_tokens=False)
        token_ids = token_ids[:fusion_length]
        if len(token_ids) < fusion_length:
            token_ids.extend([tokenizer.pad_token_id] * (fusion_length - len(token_ids)))
        compressed_text = tokenizer.decode(
            [x for x in token_ids if x != tokenizer.pad_token_id],
            skip_special_tokens=True,
        ).strip()
        if not compressed_text:
            compressed_text = source_text[:200].strip()
        return compressed_text, token_ids

    assert compressor is not None
    compressed_text = compressor.compress_text(source_text, target_token=fusion_length)
    if compressed_text is None or not compressed_text.strip():
        raise RuntimeError("LLMLingua compression returned empty result")
    token_ids = tokenizer.encode(compressed_text, add_special_tokens=False)
    token_ids = token_ids[:fusion_length]
    if len(token_ids) < fusion_length:
        token_ids.extend([tokenizer.pad_token_id] * (fusion_length - len(token_ids)))
    return compressed_text, token_ids


def _compress_to_fusion_ids(
    cfg: Any,
    tokenizer: AutoTokenizer,
    source_text: str,
    compression_rate: float,
    device: torch.device,
    backend: str,
    retrieval_fusion_length: int,
) -> tuple[str, List[int]]:
    """压缩源文本为 fusion token IDs（保留旧接口，内部新建 compressor）。"""
    compressor = _build_compressor(cfg, compression_rate, device, backend)
    return _compress_text_to_ids(tokenizer, source_text, compressor, backend, retrieval_fusion_length)


def _choice_label(index: int) -> str:
    return ["A", "B", "C", "D"][index]


def _resolve_source_knowledge(
    dataset: str,
    *,
    fusion_length: int,
    target_question: str,
    source_key: str,
    source_question: str,
    source_original_text: str,
    no_answer: bool = False,
) -> tuple[str, str]:
    if source_original_text.strip():
        return "__direct_source_text__", source_original_text.strip()

    resolved_key = source_key.strip()
    if not resolved_key and source_question.strip():
        resolved_key = source_question[:200].strip()
    if not resolved_key:
        resolved_key = target_question[:200].strip()

    original_map = _load_original_knowledge_map(_dataset_knowledge_path(dataset, fusion_length))
    if resolved_key in original_map:
        full_text = original_map[resolved_key]
        if no_answer:
            # original_text = question_scenario + " " + correct_answer
            # 不带答案版本：使用 target_question 作为知识来源（只含场景，无答案）
            question_only = target_question.strip()
            logger.info("--no-answer-in-knowledge: 使用 question_only 作为知识（去掉 correct_answer）")
            return resolved_key, question_only
        return resolved_key, full_text

    suggestions = difflib.get_close_matches(resolved_key, list(original_map.keys()), n=3, cutoff=0.3)
    suggestion_text = f" Close matches: {suggestions}" if suggestions else ""
    raise KeyError(
        f"source key not found in {dataset} original knowledge map: {resolved_key!r}."
        f" Try --source-question, --source-key, or --source-original-text.{suggestion_text}"
    )


def main() -> None:
    _setup_logging()
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    device = torch.device(args.device)

    source_key, original_knowledge = _resolve_source_knowledge(
        args.dataset,
        fusion_length=cfg.model.fusion_length,
        target_question=args.question,
        source_key=args.source_key,
        source_question=args.source_question,
        source_original_text=args.source_original_text,
        no_answer=args.no_answer_in_knowledge,
    )

    model_path = get_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 加载 dense index ──
    dense_index = DenseKnowledgeIndex.load(resolve_path(args.base_index))
    fusion_length = cfg.model.fusion_length  # 检索与注入统一使用 k64

    logger.info(
        "Loaded base index | path=%s | total=%d | active=%d | fusion_length=%d",
        resolve_path(args.base_index),
        len(dense_index),
        dense_index.num_active,
        fusion_length,
    )

    # ── 单视图压缩（k64），检索与注入共用同一份 fusion_ids ──
    # compressor 初始化一次，后续 query 压缩复用
    compressor = _build_compressor(cfg, args.compression_rate, device, args.compression_backend)
    compressed_text, fusion_ids = _compress_text_to_ids(
        tokenizer=tokenizer,
        source_text=original_knowledge,
        compressor=compressor,
        backend=args.compression_backend,
        fusion_length=fusion_length,
    )

    anchor_text = original_knowledge
    if args.anchor_source == "compressed_decode":
        anchor_text = tokenizer.decode(
            [x for x in fusion_ids if x != tokenizer.pad_token_id],
            skip_special_tokens=True,
        ).strip()
        if not anchor_text:
            raise RuntimeError("decoded compressed knowledge is empty; cannot use as anchor text")

    artifacts_dir = resolve_path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    anchor_jsonl = artifacts_dir / f"{args.dataset}_single_anchor.jsonl"
    fusion_jsonl = artifacts_dir / f"{args.dataset}_single_fusion.jsonl"
    anchor_jsonl.write_text(
        json.dumps({"key": source_key, "original_text": anchor_text}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    fusion_jsonl.write_text(
        json.dumps(
            {"key": source_key, "compressed_text": compressed_text, "knowledge_ids": fusion_ids},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # 单视图：用 compressed_text 在 fusion_encoder_depth 下编码
    encoder, encoder_tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device)
    embedding = encode_fusion_texts(
        cfg=cfg,
        encoder=encoder,
        tokenizer=encoder_tokenizer,
        texts=[compressed_text],
        device=device,
    )

    fusion_ids_for_index = torch.tensor([fusion_ids], dtype=torch.long)

    rng = random.Random(args.seed)
    replace_pos = rng.randrange(len(dense_index))
    replaced_key = dense_index.keys[replace_pos]
    deleted = dense_index.delete_by_keys([replaced_key])
    dense_index.add_entries(
        embeddings=embedding,
        fusion_ids=fusion_ids_for_index,
        keys=[source_key],
        texts=[anchor_text],
        replace_existing=True,
    )
    dense_index.compact()

    overlay_index_path = ""
    if args.save_overlay_index:
        overlay_index = artifacts_dir / f"{args.dataset}_single_overlay.pt"
        dense_index.save(overlay_index)
        overlay_index_path = str(overlay_index)

    # query 与写入路径对称：也经过 LLMLingua 压缩（target_token=fusion_length）后再编码
    retrieval_query_raw = _build_retrieval_query(args)
    compressed_query, _ = _compress_text_to_ids(
        tokenizer=tokenizer,
        source_text=retrieval_query_raw,
        compressor=compressor,
        backend=args.compression_backend,
        fusion_length=fusion_length,
    )
    logger.info("Retrieval query (compressed): %s", compressed_query[:120])
    encoded_query = encoder_tokenizer(
        [compressed_query],
        max_length=cfg.model.fusion_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
        return_tensors="pt",
    )
    with torch.no_grad():
        query_emb = encoder.encode_mean(
            encoded_query["input_ids"].to(device),
            encoded_query["attention_mask"].long().to(device),
        )
    search_out = dense_index.search(query_emb.detach().cpu(), top_k=args.top_k)

    retrieved_rows: List[Dict[str, Any]] = []
    for idx, score, valid in zip(
        search_out.indices[0].tolist(),
        search_out.scores[0].tolist(),
        search_out.valid_mask[0].tolist(),
    ):
        if idx < 0 or not valid:
            continue
        retrieved_rows.append(
            {
                "rank": len(retrieved_rows) + 1,
                "key": dense_index.keys[idx],
                "text": dense_index.texts[idx],
                "score": float(score),
                "fusion_text": tokenizer.decode(
                    [x for x in dense_index.fusion_ids[idx].tolist() if x != tokenizer.pad_token_id],
                    skip_special_tokens=True,
                ).strip(),
            }
        )

    # 注入使用检索结果（rank 1），与存入 index 的 fusion_ids 完全一致（k64）
    top1_knowledge_ids = search_out.fusion_ids[0, 0].unsqueeze(0).to(device)  # [1, K_f]
    logger.info("Model injection using retrieved fusion_ids (k64, %d tokens)", fusion_length)

    model, answer_tokenizer = build_injection_model(
        cfg,
        fusion_ckpt=args.phase3_ckpt,
        device=str(device),
        log_prefix="SingleOverlay",
    )
    prompt = build_multiple_choice_prompt(
        args.question,
        [args.option_a, args.option_b, args.option_c, args.option_d],
    )
    context_ids = answer_tokenizer.encode(prompt, add_special_tokens=False)
    pred_idx = score_choices_injection(model, answer_tokenizer, context_ids, top1_knowledge_ids, device)

    result = {
        "dataset": args.dataset,
        "source_key": source_key,
        "query_mode": args.query_mode,
        "anchor_source": args.anchor_source,
        "compression_rate": args.compression_rate,
        "compression_backend": args.compression_backend,
        "base_index": str(resolve_path(args.base_index)),
        "phase3_ckpt": str(resolve_path(args.phase3_ckpt)),
        "artifacts": {
            "anchor_jsonl": str(anchor_jsonl),
            "fusion_jsonl": str(fusion_jsonl),
            "overlay_index": overlay_index_path,
        },
        "overlay": {
            "replaced_key": replaced_key,
            "deleted_from_base": deleted,
            "new_key": source_key,
        },
        "source_knowledge": {
            "original_text": original_knowledge,
            "compressed_text": compressed_text,
            "fusion_ids": fusion_ids,
            "fusion_length": fusion_length,
        },
        "question": {
            "question": args.question,
            "choices": [args.option_a, args.option_b, args.option_c, args.option_d],
            "retrieval_query": retrieval_query_raw,
        },
        "retrieval": {
            "top_k": args.top_k,
            "results": retrieved_rows,
        },
        "answer": {
            "pred_index": pred_idx,
            "pred_label": _choice_label(pred_idx),
            "pred_choice": [args.option_a, args.option_b, args.option_c, args.option_d][pred_idx],
        },
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("[SingleFlow] Source key:", source_key)
        print("[SingleFlow] Original knowledge:")
        print(original_knowledge)
        print("\n[SingleFlow] Compressed knowledge (k%d):" % fusion_length)
        print(compressed_text)
        print("\n[SingleFlow] Retrieved top results:")
        for item in retrieved_rows:
            print(item)
        print("\n[SingleFlow] Predicted answer:")
        print(result["answer"])
        print("\n[SingleFlow] Artifacts:")
        print(result["artifacts"])


if __name__ == "__main__":
    main()
