from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
    select_disjoint_row_groups,
)
from retrieval.dense_index import DenseKnowledgeIndex
from training.dense_retriever import DenseRetriever


def _rows_to_keys(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(row["key"]) for row in rows]


def _prepare_embeddings_and_fusion(
    cfg: Config,
    encoder,
    tokenizer,
    knowledge_entries,
    keys: Sequence[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    texts = [get_compressed_text(knowledge_entries[key], tokenizer) for key in keys]
    embeddings = encode_fusion_texts(
        cfg=cfg,
        encoder=encoder,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
    )
    fusion_ids = torch.tensor(
        [knowledge_entries[key].knowledge_ids[: cfg.model.fusion_length] for key in keys],
        dtype=torch.long,
    )
    if fusion_ids.shape[1] < cfg.model.fusion_length:
        raise ValueError("knowledge_ids shorter than fusion_length are not expected in MedQA knowledge map")
    return embeddings, fusion_ids, texts


def _split_batches(keys: Sequence[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [list(keys[i : i + batch_size]) for i in range(0, len(keys), batch_size)]


def _make_retriever(cfg: Config, index_path: str, device: torch.device, tokenizer) -> DenseRetriever:
    return DenseRetriever(cfg=cfg, index_path=index_path, device=device, tokenizer=tokenizer)


def _save_index_to_path(index: DenseKnowledgeIndex, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    index.save(path)
    return path


def run_e8d_batch_ingest(
    cfg: Config,
    full_index_path: str,
    phase3_weights: str,
    device: str = "cuda:0",
    n_edits: int = 100,
    seed: int = 0,
    query_mode: str = "question_only",
    locality_samples: int = 200,
    output_path: str | None = None,
) -> Dict[str, Any]:
    init_logging()
    device_obj = torch.device(device)

    knowledge_entries = load_medqa_knowledge_entries(
        resolve_path("data/medqa_knowledge_original_text.jsonl"),
        resolve_path(cfg.eval.medqa_knowledge_map),
    )
    ingest_rows, old_rows = select_disjoint_row_groups(
        [n_edits, locality_samples],
        seed=seed,
        knowledge_entries=knowledge_entries,
    )
    ingest_keys = _rows_to_keys(ingest_rows)

    full_index_path_resolved = resolve_path(full_index_path)
    phase3_weights_resolved = resolve_path(phase3_weights)

    phase3_model, phase3_tokenizer = build_phase3_injection_model(cfg, phase3_weights_resolved, device=device_obj)
    encoder, encoding_tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device_obj)

    base_missing_index = DenseKnowledgeIndex.load(full_index_path_resolved)
    deleted = base_missing_index.delete_by_keys(ingest_keys)
    base_missing_index.compact()
    base_missing_index_path = save_temp_index(base_missing_index, prefix="e8d_a_base_missing")

    base_retriever = _make_retriever(cfg, str(base_missing_index_path), device_obj, phase3_tokenizer)
    old_before = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        base_retriever,
        old_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )
    ingest_before = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        base_retriever,
        ingest_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    embeddings, fusion_ids, texts = _prepare_embeddings_and_fusion(
        cfg, encoder, encoding_tokenizer, knowledge_entries, ingest_keys, device_obj
    )
    updated_index = DenseKnowledgeIndex.load(base_missing_index_path)
    t0 = time.time()
    updated_index.add_entries(
        embeddings=embeddings,
        fusion_ids=fusion_ids,
        keys=ingest_keys,
        texts=texts,
        replace_existing=True,
    )
    bulk_ingest_latency_ms = (time.time() - t0) * 1000.0
    updated_index_path = save_temp_index(updated_index, prefix="e8d_a_updated")

    updated_retriever = _make_retriever(cfg, str(updated_index_path), device_obj, phase3_tokenizer)
    old_after = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        updated_retriever,
        old_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )
    ingest_after = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        updated_retriever,
        ingest_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    result: Dict[str, Any] = {
        "experiment": "e8d_a",
        "dataset": "medqa",
        "query_mode": query_mode,
        "n_ingested": len(ingest_rows),
        "locality_samples": len(old_rows),
        "seed": seed,
        "phase3_weights": str(phase3_weights_resolved),
        "full_index": str(full_index_path_resolved),
        "base_missing_index": str(base_missing_index_path),
        "updated_index": str(updated_index_path),
        "deleted_from_base": deleted,
        "metrics": {
            "old_qa_acc_before": old_before["qa_acc"],
            "old_qa_acc_after": old_after["qa_acc"],
            "old_retrieval_top1_before": old_before["retrieval_top1"],
            "old_retrieval_top1_after": old_after["retrieval_top1"],
            "old_qa_retention": (
                old_after["qa_acc"] / old_before["qa_acc"] if old_before["qa_acc"] > 0 else 0.0
            ),
            "old_retrieval_retention": (
                old_after["retrieval_top1"] / old_before["retrieval_top1"]
                if old_before["retrieval_top1"] > 0
                else 0.0
            ),
            "ingest_qa_acc_before": ingest_before["qa_acc"],
            "ingest_qa_acc_after": ingest_after["qa_acc"],
            "ingest_retrieval_top1_before": ingest_before["retrieval_top1"],
            "ingest_retrieval_top1_after": ingest_after["retrieval_top1"],
            "bulk_ingest_latency_ms": bulk_ingest_latency_ms,
            "mean_ingest_latency_ms": bulk_ingest_latency_ms / max(len(ingest_rows), 1),
            "ingest_throughput_docs_per_sec": len(ingest_rows) / max(bulk_ingest_latency_ms / 1000.0, 1e-8),
        },
    }

    if output_path is not None:
        out_path = resolve_path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


def run_e8d_incremental_add_delete(
    cfg: Config,
    full_index_path: str,
    phase3_weights: str,
    device: str = "cuda:0",
    n_edits: int = 100,
    update_batch_size: int = 10,
    seed: int = 0,
    query_mode: str = "question_only",
    locality_samples: int = 200,
    output_path: str | None = None,
) -> Dict[str, Any]:
    init_logging()
    device_obj = torch.device(device)

    knowledge_entries = load_medqa_knowledge_entries(
        resolve_path("data/medqa_knowledge_original_text.jsonl"),
        resolve_path(cfg.eval.medqa_knowledge_map),
    )
    add_rows, delete_rows, old_rows = select_disjoint_row_groups(
        [n_edits, n_edits, locality_samples],
        seed=seed,
        knowledge_entries=knowledge_entries,
    )
    add_keys = _rows_to_keys(add_rows)
    delete_keys = _rows_to_keys(delete_rows)

    full_index_path_resolved = resolve_path(full_index_path)
    phase3_weights_resolved = resolve_path(phase3_weights)

    phase3_model, phase3_tokenizer = build_phase3_injection_model(cfg, phase3_weights_resolved, device=device_obj)
    encoder, encoding_tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device_obj)
    run_tmp_dir = Path(tempfile.mkdtemp(prefix="e8_e8d_b_", dir=str(resolve_path("results/e8/tmp"))))

    current_index = DenseKnowledgeIndex.load(full_index_path_resolved)
    current_index.delete_by_keys(add_keys)
    current_index.compact()

    add_embeddings, add_fusion_ids, add_texts = _prepare_embeddings_and_fusion(
        cfg, encoder, encoding_tokenizer, knowledge_entries, add_keys, device_obj
    )
    add_embedding_map = {key: add_embeddings[i : i + 1] for i, key in enumerate(add_keys)}
    add_fusion_map = {key: add_fusion_ids[i : i + 1] for i, key in enumerate(add_keys)}
    add_text_map = {key: add_texts[i] for i, key in enumerate(add_keys)}

    base_index_path = _save_index_to_path(current_index, run_tmp_dir / "base.pt")
    base_retriever = _make_retriever(cfg, str(base_index_path), device_obj, phase3_tokenizer)
    old_baseline = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        base_retriever,
        old_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )
    add_before = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        base_retriever,
        add_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    add_batches = _split_batches(add_keys, update_batch_size)
    delete_batches = _split_batches(delete_keys, update_batch_size)
    add_stage: List[Dict[str, Any]] = []
    delete_stage: List[Dict[str, Any]] = []
    add_latencies_ms: List[float] = []
    delete_latencies_ms: List[float] = []
    add_eval_index_path = run_tmp_dir / "add_eval.pt"
    post_add_index_path = run_tmp_dir / "post_add.pt"
    delete_eval_index_path = run_tmp_dir / "delete_eval.pt"
    final_index_path = run_tmp_dir / "final.pt"

    for batch_idx, batch_keys in enumerate(add_batches, start=1):
        t0 = time.time()
        current_index.add_entries(
            embeddings=torch.cat([add_embedding_map[key] for key in batch_keys], dim=0),
            fusion_ids=torch.cat([add_fusion_map[key] for key in batch_keys], dim=0),
            keys=batch_keys,
            texts=[add_text_map[key] for key in batch_keys],
            replace_existing=True,
        )
        latency_ms = (time.time() - t0) * 1000.0
        add_latencies_ms.append(latency_ms)

        _save_index_to_path(current_index, add_eval_index_path)
        retriever = _make_retriever(cfg, str(add_eval_index_path), device_obj, phase3_tokenizer)
        current_add_rows = add_rows[: batch_idx * update_batch_size]
        add_eval = evaluate_edit_rows(
            phase3_model,
            phase3_tokenizer,
            retriever,
            current_add_rows,
            knowledge_entries,
            device=device_obj,
            query_mode=query_mode,
            fusion_length=cfg.model.fusion_length,
        )
        old_eval = evaluate_edit_rows(
            phase3_model,
            phase3_tokenizer,
            retriever,
            old_rows,
            knowledge_entries,
            device=device_obj,
            query_mode=query_mode,
            fusion_length=cfg.model.fusion_length,
        )
        add_stage.append(
            {
                "batch_index": batch_idx,
                "batch_size": len(batch_keys),
                "added_so_far": len(current_add_rows),
                "latency_ms": latency_ms,
                "add_qa_acc": add_eval["qa_acc"],
                "add_retrieval_top1": add_eval["retrieval_top1"],
                "old_qa_acc": old_eval["qa_acc"],
                "old_retrieval_top1": old_eval["retrieval_top1"],
                "index_path": str(add_eval_index_path),
                "index_size": len(current_index),
                "active_entries": current_index.num_active,
            }
        )

    _save_index_to_path(current_index, post_add_index_path)
    post_add_retriever = _make_retriever(cfg, str(post_add_index_path), device_obj, phase3_tokenizer)
    delete_before = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        post_add_retriever,
        delete_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    deleted_so_far: List[Dict[str, Any]] = []
    for batch_idx, batch_keys in enumerate(delete_batches, start=1):
        t0 = time.time()
        current_index.delete_by_keys(batch_keys)
        latency_ms = (time.time() - t0) * 1000.0
        delete_latencies_ms.append(latency_ms)

        _save_index_to_path(current_index, delete_eval_index_path)
        retriever = _make_retriever(cfg, str(delete_eval_index_path), device_obj, phase3_tokenizer)
        deleted_so_far.extend([row for row in delete_rows if row["key"] in set(batch_keys)])
        delete_eval = evaluate_edit_rows(
            phase3_model,
            phase3_tokenizer,
            retriever,
            deleted_so_far,
            knowledge_entries,
            device=device_obj,
            query_mode=query_mode,
            fusion_length=cfg.model.fusion_length,
        )
        old_eval = evaluate_edit_rows(
            phase3_model,
            phase3_tokenizer,
            retriever,
            old_rows,
            knowledge_entries,
            device=device_obj,
            query_mode=query_mode,
            fusion_length=cfg.model.fusion_length,
        )
        delete_stage.append(
            {
                "batch_index": batch_idx,
                "batch_size": len(batch_keys),
                "deleted_so_far": len(deleted_so_far),
                "latency_ms": latency_ms,
                "delete_qa_acc": delete_eval["qa_acc"],
                "delete_retrieval_top1": delete_eval["retrieval_top1"],
                "old_qa_acc": old_eval["qa_acc"],
                "old_retrieval_top1": old_eval["retrieval_top1"],
                "index_path": str(delete_eval_index_path),
                "index_size": len(current_index),
                "active_entries": current_index.num_active,
                "tombstone_ratio": 1.0 - (current_index.num_active / max(len(current_index), 1)),
            }
        )

    _save_index_to_path(current_index, final_index_path)
    final_retriever = _make_retriever(cfg, str(final_index_path), device_obj, phase3_tokenizer)
    add_after = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        final_retriever,
        add_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )
    delete_after = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        final_retriever,
        delete_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )
    old_after = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        final_retriever,
        old_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    result: Dict[str, Any] = {
        "experiment": "e8d_b",
        "dataset": "medqa",
        "query_mode": query_mode,
        "n_add": len(add_rows),
        "n_delete": len(delete_rows),
        "locality_samples": len(old_rows),
        "update_batch_size": update_batch_size,
        "seed": seed,
        "phase3_weights": str(phase3_weights_resolved),
        "full_index": str(full_index_path_resolved),
        "base_index": str(base_index_path),
        "post_add_index": str(post_add_index_path),
        "final_index": str(final_index_path),
        "metrics": {
            "add_qa_acc_before": add_before["qa_acc"],
            "add_qa_acc_after": add_after["qa_acc"],
            "add_retrieval_top1_before": add_before["retrieval_top1"],
            "add_retrieval_top1_after": add_after["retrieval_top1"],
            "delete_qa_acc_before": delete_before["qa_acc"],
            "delete_qa_acc_after": delete_after["qa_acc"],
            "delete_retrieval_top1_before": delete_before["retrieval_top1"],
            "delete_retrieval_top1_after": delete_after["retrieval_top1"],
            "old_qa_acc_before": old_baseline["qa_acc"],
            "old_qa_acc_after": old_after["qa_acc"],
            "old_retrieval_top1_before": old_baseline["retrieval_top1"],
            "old_retrieval_top1_after": old_after["retrieval_top1"],
            "old_qa_retention_after_updates": (
                old_after["qa_acc"] / old_baseline["qa_acc"] if old_baseline["qa_acc"] > 0 else 0.0
            ),
            "old_retrieval_retention_after_updates": (
                old_after["retrieval_top1"] / old_baseline["retrieval_top1"]
                if old_baseline["retrieval_top1"] > 0
                else 0.0
            ),
            "mean_add_latency_ms": sum(add_latencies_ms) / max(len(add_latencies_ms), 1),
            "mean_delete_latency_ms": sum(delete_latencies_ms) / max(len(delete_latencies_ms), 1),
            "tombstone_ratio_final": 1.0 - (current_index.num_active / max(len(current_index), 1)),
        },
        "add_stage": add_stage,
        "delete_stage": delete_stage,
    }

    if output_path is not None:
        out_path = resolve_path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
