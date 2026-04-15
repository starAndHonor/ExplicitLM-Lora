from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

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
    select_locality_rows,
)
from retrieval.dense_index import DenseKnowledgeIndex
from training.dense_retriever import DenseRetriever


@dataclass(frozen=True)
class _EditOp:
    op: str
    key: str


def _parse_steps(steps: Sequence[int]) -> List[int]:
    parsed = sorted({int(s) for s in steps if int(s) > 0})
    if not parsed:
        raise ValueError("steps must contain at least one positive integer")
    return parsed


def _measure_retrieval_latency_ms(
    retriever: DenseRetriever,
    rows: Sequence[Dict[str, Any]],
    query_mode: str,
) -> float:
    if not rows:
        return 0.0
    from experiments.e8.common import build_retrieval_query

    started = time.time()
    for row in rows:
        retriever.retrieve_from_texts([build_retrieval_query(row, query_mode)])
    return (time.time() - started) * 1000.0 / len(rows)


def _build_operation_stream(edit_keys: Sequence[str], total_steps: int) -> List[_EditOp]:
    ops: List[_EditOp] = []
    for key in edit_keys:
        ops.extend(
            [
                _EditOp("upsert", key),
                _EditOp("delete", key),
                _EditOp("rollback", key),
            ]
        )
        if len(ops) >= total_steps:
            break
    return ops[:total_steps]


def _build_row_map(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["key"]): row for row in rows}


def _subset_rows(row_map: Dict[str, Dict[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    return [row_map[key] for key in keys if key in row_map]


def run_e8c_sequential_edits(
    cfg: Config,
    full_index_path: str,
    phase3_weights: str,
    device: str = "cuda:0",
    steps: Sequence[int] = (1, 10, 100, 1000),
    seed: int = 0,
    query_mode: str = "question_only",
    locality_samples: int = 200,
    output_path: str | None = None,
) -> Dict[str, Any]:
    init_logging()
    device_obj = torch.device(device)
    step_list = _parse_steps(steps)
    max_step = max(step_list)

    original_text_path = resolve_path("data/medqa_knowledge_original_text.jsonl")
    knowledge_map_path = resolve_path(cfg.eval.medqa_knowledge_map)
    full_index_path_resolved = resolve_path(full_index_path)
    phase3_weights_resolved = resolve_path(phase3_weights)

    # Each unique key contributes three possible operations:
    # upsert -> delete -> rollback.
    n_unique_keys = (max_step + 2) // 3
    knowledge_entries = load_medqa_knowledge_entries(original_text_path, knowledge_map_path)
    edit_rows = select_edit_rows(limit=n_unique_keys, seed=seed, knowledge_entries=knowledge_entries)
    if len(edit_rows) < n_unique_keys:
        raise ValueError(f"requested {n_unique_keys} editable rows, but only {len(edit_rows)} are available")
    edit_keys = [row["key"] for row in edit_rows]
    op_stream = _build_operation_stream(edit_keys, max_step)
    if len(op_stream) < max_step:
        raise ValueError(f"failed to build {max_step} sequential edits, got {len(op_stream)}")

    row_map = _build_row_map(edit_rows)
    locality_rows = select_locality_rows(
        limit=locality_samples,
        seed=seed + 1,
        exclude_keys=edit_keys,
        knowledge_entries=knowledge_entries,
    )

    phase3_model, phase3_tokenizer = build_phase3_injection_model(cfg, phase3_weights_resolved, device=device_obj)
    encoder, encoding_tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device_obj)

    full_retriever = DenseRetriever(
        cfg=cfg,
        index_path=str(full_index_path_resolved),
        device=device_obj,
        tokenizer=phase3_tokenizer,
    )
    full_eval = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        full_retriever,
        edit_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )
    full_qa_by_key = {
        row["key"]: bool(flag) for row, flag in zip(edit_rows, full_eval["qa_correct_flags"])
    }
    full_retrieval_by_key = {
        row["key"]: bool(flag) for row, flag in zip(edit_rows, full_eval["retrieval_exact_flags"])
    }
    locality_baseline = evaluate_edit_rows(
        phase3_model,
        phase3_tokenizer,
        full_retriever,
        locality_rows,
        knowledge_entries,
        device=device_obj,
        query_mode=query_mode,
        fusion_length=cfg.model.fusion_length,
    )

    # Start from a base-missing state and then edit the same index online.
    current_index = DenseKnowledgeIndex.load(full_index_path_resolved)
    current_index.delete_by_keys(edit_keys)
    current_index.compact()

    edit_texts = [get_compressed_text(knowledge_entries[key], encoding_tokenizer) for key in edit_keys]
    edit_embeddings = encode_fusion_texts(
        cfg=cfg,
        encoder=encoder,
        tokenizer=encoding_tokenizer,
        texts=edit_texts,
        device=device_obj,
    )
    text_by_key = {key: text for key, text in zip(edit_keys, edit_texts)}
    embedding_by_key = {
        key: edit_embeddings[idx : idx + 1] for idx, key in enumerate(edit_keys)
    }
    fusion_by_key = {
        key: torch.tensor(
            [knowledge_entries[key].knowledge_ids[: cfg.model.fusion_length]],
            dtype=torch.long,
        )
        for key in edit_keys
    }
    for key, fusion in fusion_by_key.items():
        if fusion.shape[1] < cfg.model.fusion_length:
            raise ValueError(f"knowledge_ids shorter than fusion_length for key={key}")

    state_by_key = {key: "missing" for key in edit_keys}
    rolled_back_keys: set[str] = set()
    op_cursor = 0
    op_history: List[Dict[str, Any]] = []
    write_latencies_ms: List[float] = []
    delete_latencies_ms: List[float] = []

    series: Dict[str, List[float | int]] = {
        "edit_success_rate": [],
        "edit_retrieval_top1": [],
        "delete_success_rate": [],
        "delete_retrieval_success_rate": [],
        "rollback_fidelity": [],
        "rollback_retrieval_fidelity": [],
        "locality_retention": [],
        "locality_qa_acc": [],
        "locality_retrieval_top1": [],
        "retrieval_latency_ms": [],
        "index_size": [],
        "active_entries": [],
        "tombstone_ratio": [],
        "present_count": [],
        "deleted_count": [],
        "rolled_back_count": [],
        "mean_edit_latency_ms": [],
        "mean_write_latency_ms": [],
        "mean_delete_latency_ms": [],
        "compact_count": [],
    }
    step_details: List[Dict[str, Any]] = []

    for step in step_list:
        while op_cursor < step:
            edit_op = op_stream[op_cursor]
            op_t0 = time.time()
            if edit_op.op in {"upsert", "rollback"}:
                current_index.add_entries(
                    embeddings=embedding_by_key[edit_op.key],
                    fusion_ids=fusion_by_key[edit_op.key],
                    keys=[edit_op.key],
                    texts=[text_by_key[edit_op.key]],
                    replace_existing=True,
                )
                elapsed_ms = (time.time() - op_t0) * 1000.0
                write_latencies_ms.append(elapsed_ms)
                state_by_key[edit_op.key] = "present"
                if edit_op.op == "rollback":
                    rolled_back_keys.add(edit_op.key)
            elif edit_op.op == "delete":
                current_index.delete_by_keys([edit_op.key])
                elapsed_ms = (time.time() - op_t0) * 1000.0
                delete_latencies_ms.append(elapsed_ms)
                state_by_key[edit_op.key] = "deleted"
            else:  # pragma: no cover
                raise ValueError(f"unsupported edit op: {edit_op.op}")

            op_history.append(
                {
                    "index": op_cursor + 1,
                    "op": edit_op.op,
                    "key": edit_op.key,
                    "state_after": state_by_key[edit_op.key],
                }
            )
            op_cursor += 1

        current_index_path = save_temp_index(current_index, prefix=f"e8c_step_{step}")
        retriever = DenseRetriever(
            cfg=cfg,
            index_path=str(current_index_path),
            device=device_obj,
            tokenizer=phase3_tokenizer,
        )

        present_keys = [key for key, state in state_by_key.items() if state == "present"]
        deleted_keys = [key for key, state in state_by_key.items() if state == "deleted"]
        rolled_back_present_keys = [key for key in present_keys if key in rolled_back_keys]

        present_rows = _subset_rows(row_map, present_keys)
        deleted_rows = _subset_rows(row_map, deleted_keys)
        rolled_back_rows = _subset_rows(row_map, rolled_back_present_keys)

        present_eval = evaluate_edit_rows(
            phase3_model,
            phase3_tokenizer,
            retriever,
            present_rows,
            knowledge_entries,
            device=device_obj,
            query_mode=query_mode,
            fusion_length=cfg.model.fusion_length,
        ) if present_rows else {
            "qa_acc": 0.0,
            "retrieval_top1": 0.0,
            "qa_correct_flags": [],
            "retrieval_exact_flags": [],
            "total": 0,
        }

        deleted_eval = evaluate_edit_rows(
            phase3_model,
            phase3_tokenizer,
            retriever,
            deleted_rows,
            knowledge_entries,
            device=device_obj,
            query_mode=query_mode,
            fusion_length=cfg.model.fusion_length,
        ) if deleted_rows else {
            "qa_acc": 0.0,
            "retrieval_top1": 0.0,
            "qa_correct_flags": [],
            "retrieval_exact_flags": [],
            "total": 0,
        }

        rolled_back_eval = evaluate_edit_rows(
            phase3_model,
            phase3_tokenizer,
            retriever,
            rolled_back_rows,
            knowledge_entries,
            device=device_obj,
            query_mode=query_mode,
            fusion_length=cfg.model.fusion_length,
        ) if rolled_back_rows else {
            "qa_acc": 0.0,
            "retrieval_top1": 0.0,
            "qa_correct_flags": [],
            "retrieval_exact_flags": [],
            "total": 0,
        }

        locality_eval = evaluate_edit_rows(
            phase3_model,
            phase3_tokenizer,
            retriever,
            locality_rows,
            knowledge_entries,
            device=device_obj,
            query_mode=query_mode,
            fusion_length=cfg.model.fusion_length,
        )
        retrieval_latency_ms = _measure_retrieval_latency_ms(retriever, locality_rows, query_mode=query_mode)

        locality_retention = (
            locality_eval["qa_acc"] / locality_baseline["qa_acc"]
            if locality_baseline["qa_acc"] > 0
            else 0.0
        )

        delete_success_rate = 0.0
        delete_retrieval_success_rate = 0.0
        if deleted_rows:
            delete_qa_successes = 0
            delete_retrieval_successes = 0
            for row, qa_flag, retrieval_flag in zip(
                deleted_rows,
                deleted_eval["qa_correct_flags"],
                deleted_eval["retrieval_exact_flags"],
            ):
                key = row["key"]
                if full_qa_by_key.get(key, False) and not qa_flag:
                    delete_qa_successes += 1
                if full_retrieval_by_key.get(key, False) and not retrieval_flag:
                    delete_retrieval_successes += 1
            delete_success_rate = delete_qa_successes / len(deleted_rows)
            delete_retrieval_success_rate = delete_retrieval_successes / len(deleted_rows)

        rollback_fidelity = 0.0
        rollback_retrieval_fidelity = 0.0
        if rolled_back_rows:
            rollback_qa_match = 0
            rollback_retrieval_match = 0
            for row, qa_flag, retrieval_flag in zip(
                rolled_back_rows,
                rolled_back_eval["qa_correct_flags"],
                rolled_back_eval["retrieval_exact_flags"],
            ):
                key = row["key"]
                if qa_flag == full_qa_by_key.get(key, False):
                    rollback_qa_match += 1
                if retrieval_flag == full_retrieval_by_key.get(key, False):
                    rollback_retrieval_match += 1
            rollback_fidelity = rollback_qa_match / len(rolled_back_rows)
            rollback_retrieval_fidelity = rollback_retrieval_match / len(rolled_back_rows)

        mean_write_latency_ms = (
            sum(write_latencies_ms) / len(write_latencies_ms) if write_latencies_ms else 0.0
        )
        mean_delete_latency_ms = (
            sum(delete_latencies_ms) / len(delete_latencies_ms) if delete_latencies_ms else 0.0
        )
        all_latencies = write_latencies_ms + delete_latencies_ms
        mean_edit_latency_ms = sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

        series["edit_success_rate"].append(present_eval["qa_acc"])
        series["edit_retrieval_top1"].append(present_eval["retrieval_top1"])
        series["delete_success_rate"].append(delete_success_rate)
        series["delete_retrieval_success_rate"].append(delete_retrieval_success_rate)
        series["rollback_fidelity"].append(rollback_fidelity)
        series["rollback_retrieval_fidelity"].append(rollback_retrieval_fidelity)
        series["locality_retention"].append(locality_retention)
        series["locality_qa_acc"].append(locality_eval["qa_acc"])
        series["locality_retrieval_top1"].append(locality_eval["retrieval_top1"])
        series["retrieval_latency_ms"].append(retrieval_latency_ms)
        series["index_size"].append(len(current_index))
        series["active_entries"].append(current_index.num_active)
        series["tombstone_ratio"].append(1.0 - (current_index.num_active / max(len(current_index), 1)))
        series["present_count"].append(len(present_rows))
        series["deleted_count"].append(len(deleted_rows))
        series["rolled_back_count"].append(len(rolled_back_rows))
        series["mean_edit_latency_ms"].append(mean_edit_latency_ms)
        series["mean_write_latency_ms"].append(mean_write_latency_ms)
        series["mean_delete_latency_ms"].append(mean_delete_latency_ms)
        series["compact_count"].append(0)

        step_details.append(
            {
                "step": step,
                "index_path": str(current_index_path),
                "ops_applied": op_cursor,
                "last_operation": op_history[-1] if op_history else None,
                "present_count": len(present_rows),
                "deleted_count": len(deleted_rows),
                "rolled_back_count": len(rolled_back_rows),
                "present_qa_acc": present_eval["qa_acc"],
                "present_retrieval_top1": present_eval["retrieval_top1"],
                "delete_success_rate": delete_success_rate,
                "delete_retrieval_success_rate": delete_retrieval_success_rate,
                "rollback_fidelity": rollback_fidelity,
                "rollback_retrieval_fidelity": rollback_retrieval_fidelity,
                "locality_qa_acc": locality_eval["qa_acc"],
                "locality_retrieval_top1": locality_eval["retrieval_top1"],
                "locality_retention": locality_retention,
                "retrieval_latency_ms": retrieval_latency_ms,
                "index_size": len(current_index),
                "active_entries": current_index.num_active,
                "tombstone_ratio": 1.0 - (current_index.num_active / max(len(current_index), 1)),
                "mean_edit_latency_ms": mean_edit_latency_ms,
                "mean_write_latency_ms": mean_write_latency_ms,
                "mean_delete_latency_ms": mean_delete_latency_ms,
            }
        )

    result: Dict[str, Any] = {
        "experiment": "e8c",
        "dataset": "medqa",
        "query_mode": query_mode,
        "steps": step_list,
        "seed": seed,
        "locality_samples": len(locality_rows),
        "phase3_weights": str(phase3_weights_resolved),
        "full_index": str(full_index_path_resolved),
        "operation_pattern": "online_upsert_delete_rollback",
        "operations": [op.__dict__ for op in op_stream],
        "series": series,
        "baseline": {
            "full_edit_qa_acc": full_eval["qa_acc"],
            "full_edit_retrieval_top1": full_eval["retrieval_top1"],
            "locality_qa_acc": locality_baseline["qa_acc"],
            "locality_retrieval_top1": locality_baseline["retrieval_top1"],
        },
        "step_details": step_details,
    }

    if output_path is not None:
        out_path = resolve_path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result
