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


def run_e8b_delete_rollback(
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

    original_text_path = resolve_path("data/medqa_knowledge_original_text.jsonl")
    knowledge_map_path = resolve_path(cfg.eval.medqa_knowledge_map)
    full_index_path_resolved = resolve_path(full_index_path)
    phase3_weights_resolved = resolve_path(phase3_weights)

    knowledge_entries = load_medqa_knowledge_entries(original_text_path, knowledge_map_path)
    edit_rows = select_edit_rows(limit=n_edits, seed=seed, knowledge_entries=knowledge_entries)
    edit_keys = [row["key"] for row in edit_rows]

    phase3_model, phase3_tokenizer = build_phase3_injection_model(cfg, phase3_weights_resolved, device=device_obj)

    full_retriever = DenseRetriever(
        cfg=cfg,
        index_path=str(full_index_path_resolved),
        device=device_obj,
        tokenizer=phase3_tokenizer,
    )
    before_eval = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        full_retriever,
        edit_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    deleted_index = DenseKnowledgeIndex.load(full_index_path_resolved)
    delete_t0 = time.time()
    deleted = deleted_index.delete_by_keys(edit_keys)
    deleted_index.compact()
    mean_delete_latency_ms = (time.time() - delete_t0) * 1000.0 / max(len(edit_keys), 1)
    deleted_index_path = save_temp_index(deleted_index, prefix="e8b_deleted")

    deleted_retriever = DenseRetriever(
        cfg=cfg,
        index_path=str(deleted_index_path),
        device=device_obj,
        tokenizer=phase3_tokenizer,
    )
    deleted_eval = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        deleted_retriever,
        edit_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    encoder, encoding_tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device_obj)
    restored_index = DenseKnowledgeIndex.load(deleted_index_path)
    edit_texts = [get_compressed_text(knowledge_entries[key], encoding_tokenizer) for key in edit_keys]
    edit_embeddings = encode_fusion_texts(
        cfg=cfg,
        encoder=encoder,
        tokenizer=encoding_tokenizer,
        texts=edit_texts,
        device=device_obj,
    )
    fusion_ids = torch.tensor(
        [knowledge_entries[key].knowledge_ids[: cfg.model.fusion_length] for key in edit_keys],
        dtype=torch.long,
    )
    if fusion_ids.shape[1] < cfg.model.fusion_length:
        raise ValueError("knowledge_ids shorter than fusion_length are not expected in MedQA knowledge map")

    rollback_t0 = time.time()
    restored_index.add_entries(
        embeddings=edit_embeddings,
        fusion_ids=fusion_ids,
        keys=edit_keys,
        texts=edit_texts,
        replace_existing=True,
    )
    mean_rollback_latency_ms = (time.time() - rollback_t0) * 1000.0 / max(len(edit_keys), 1)
    restored_index_path = save_temp_index(restored_index, prefix="e8b_restored")

    restored_retriever = DenseRetriever(
        cfg=cfg,
        index_path=str(restored_index_path),
        device=device_obj,
        tokenizer=phase3_tokenizer,
    )
    restored_eval = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        restored_retriever,
        edit_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    before_qa_flags = before_eval["qa_correct_flags"]
    deleted_qa_flags = deleted_eval["qa_correct_flags"]
    restored_qa_flags = restored_eval["qa_correct_flags"]
    before_retrieval_flags = before_eval["retrieval_exact_flags"]
    deleted_retrieval_flags = deleted_eval["retrieval_exact_flags"]
    restored_retrieval_flags = restored_eval["retrieval_exact_flags"]

    delete_success_count = sum(
        1 for before_ok, after_ok in zip(before_qa_flags, deleted_qa_flags) if before_ok and not after_ok
    )
    rollback_fidelity_count = sum(
        1
        for before_ok, restored_ok in zip(before_qa_flags, restored_qa_flags)
        if before_ok == restored_ok
    )
    retrieval_drop_after_delete = sum(
        1
        for before_ok, after_ok in zip(before_retrieval_flags, deleted_retrieval_flags)
        if before_ok and not after_ok
    ) / max(len(edit_rows), 1)
    qa_drop_after_delete = sum(
        1 for before_ok, after_ok in zip(before_qa_flags, deleted_qa_flags) if before_ok and not after_ok
    ) / max(len(edit_rows), 1)
    qa_recovery_after_rollback = sum(
        1 for before_ok, restored_ok in zip(before_qa_flags, restored_qa_flags) if before_ok and restored_ok
    ) / max(len(edit_rows), 1)
    retrieval_recovery_after_rollback = sum(
        1
        for before_ok, restored_ok in zip(before_retrieval_flags, restored_retrieval_flags)
        if before_ok and restored_ok
    ) / max(len(edit_rows), 1)

    result: Dict[str, Any] = {
        "experiment": "e8b",
        "dataset": "medqa",
        "query_mode": query_mode,
        "n_cases": len(edit_rows),
        "seed": seed,
        "phase3_weights": str(phase3_weights_resolved),
        "full_index": str(full_index_path_resolved),
        "deleted_index": str(deleted_index_path),
        "restored_index": str(restored_index_path),
        "deleted_from_full": deleted,
        "metrics": {
            "pre_delete_acc": before_eval["qa_acc"],
            "post_delete_acc": deleted_eval["qa_acc"],
            "post_rollback_acc": restored_eval["qa_acc"],
            "retrieval_top1_before": before_eval["retrieval_top1"],
            "retrieval_top1_after_delete": deleted_eval["retrieval_top1"],
            "retrieval_top1_after_rollback": restored_eval["retrieval_top1"],
            "delete_success_rate": delete_success_count / max(len(edit_rows), 1),
            "rollback_fidelity": rollback_fidelity_count / max(len(edit_rows), 1),
            "retrieval_drop_after_delete": retrieval_drop_after_delete,
            "qa_drop_after_delete": qa_drop_after_delete,
            "qa_recovery_after_rollback": qa_recovery_after_rollback,
            "retrieval_recovery_after_rollback": retrieval_recovery_after_rollback,
            "mean_delete_latency_ms": mean_delete_latency_ms,
            "mean_rollback_latency_ms": mean_rollback_latency_ms,
        },
    }

    if output_path is not None:
        out_path = resolve_path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
