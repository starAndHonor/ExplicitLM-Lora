from __future__ import annotations

import json
import time
from typing import Any, Dict

import torch

from config import Config
from experiments.e8.common import (
    build_fusion_encoder_and_tokenizer,
    build_phase3_injection_model,
    encode_fusion_texts,
    evaluate_edit_rows,
    get_compressed_text,
    init_logging,
    load_medqa_knowledge_entries,
    resolve_path,
    save_temp_index,
    select_edit_rows,
)
from retrieval.dense_index import DenseKnowledgeIndex
from training.dense_retriever import DenseRetriever


def run_e8a_upsert(
    cfg: Config,
    full_index_path: str,
    phase3_weights: str,
    device: str = "cuda:0",
    n_edits: int = 100,
    seed: int = 0,
    query_mode: str = "question_only",
    output_path: str | None = None,
) -> Dict[str, Any]:
    init_logging()
    device_obj = torch.device(device)

    full_index_path_resolved = resolve_path(full_index_path)
    phase3_weights_resolved = resolve_path(phase3_weights)

    knowledge_entries = load_medqa_knowledge_entries(cfg.model.fusion_length)
    edit_rows = select_edit_rows(limit=n_edits, seed=seed, knowledge_entries=knowledge_entries)
    edit_keys = [row["key"] for row in edit_rows]

    base_missing_index = DenseKnowledgeIndex.load(full_index_path_resolved)
    deleted = base_missing_index.delete_by_keys(edit_keys)
    base_missing_index.compact()
    base_missing_index_path = save_temp_index(base_missing_index, prefix="e8a_base_missing")

    encoder, encoding_tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device_obj)
    phase3_model, phase3_tokenizer = build_phase3_injection_model(cfg, phase3_weights_resolved, device=device_obj)

    base_missing_retriever = DenseRetriever(
        cfg=cfg,
        index_path=str(base_missing_index_path),
        device=device_obj,
        tokenizer=phase3_tokenizer,
    )
    before_eval = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        base_missing_retriever,
        edit_rows,
        device=device_obj,
        query_mode=query_mode,
    )
    pre_write_acc = before_eval["qa_acc"]
    retrieval_top1_before = before_eval["retrieval_top1"]

    edit_texts = [get_compressed_text(knowledge_entries[key], encoding_tokenizer) for key in edit_keys]
    edit_embeddings = encode_fusion_texts(
        cfg=cfg,
        encoder=encoder,
        tokenizer=encoding_tokenizer,
        texts=edit_texts,
        device=device_obj,
    )
    fusion_ids = torch.tensor(
        [
            knowledge_entries[key].knowledge_ids[: cfg.model.fusion_length]
            for key in edit_keys
        ],
        dtype=torch.long,
    )
    if fusion_ids.shape[1] < cfg.model.fusion_length:
        raise ValueError("knowledge_ids shorter than fusion_length are not expected in MedQA knowledge map")

    updated_index = DenseKnowledgeIndex.load(base_missing_index_path)
    write_t0 = time.time()
    updated_index.add_entries(
        embeddings=edit_embeddings,
        fusion_ids=fusion_ids,
        keys=edit_keys,
        texts=edit_texts,
        replace_existing=True,
    )
    mean_write_latency_ms = (time.time() - write_t0) * 1000.0 / max(len(edit_keys), 1)
    updated_index_path = save_temp_index(updated_index, prefix="e8a_updated")

    updated_retriever = DenseRetriever(
        cfg=cfg,
        index_path=str(updated_index_path),
        device=device_obj,
        tokenizer=phase3_tokenizer,
    )
    after_eval = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        updated_retriever,
        edit_rows,
        device=device_obj,
        query_mode=query_mode,
    )
    post_write_acc = after_eval["qa_acc"]
    retrieval_top1_after = after_eval["retrieval_top1"]

    write_success_rate = (
        (post_write_acc - pre_write_acc) / max(1.0 - pre_write_acc, 1e-8)
        if post_write_acc >= pre_write_acc
        else 0.0
    )

    result: Dict[str, Any] = {
        "experiment": "e8a",
        "dataset": "medqa",
        "query_mode": query_mode,
        "n_edits": len(edit_rows),
        "seed": seed,
        "phase3_weights": str(phase3_weights_resolved),
        "full_index": str(full_index_path_resolved),
        "base_missing_index": str(base_missing_index_path),
        "updated_index": str(updated_index_path),
        "deleted_from_base": deleted,
        "metrics": {
            "pre_write_acc": pre_write_acc,
            "post_write_acc": post_write_acc,
            "write_success_rate": write_success_rate,
            "mean_write_latency_ms": mean_write_latency_ms,
            "retrieval_top1_before": retrieval_top1_before,
            "retrieval_top1_after": retrieval_top1_after,
        },
    }

    if output_path is not None:
        out_path = resolve_path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
