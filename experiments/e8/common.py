from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from config import Config
from experiments.e2.common import (
    PROJECT_ROOT,
    build_injection_model,
    get_model_path,
    setup_logging,
)
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices_injection
from experiments.e3.data_loading import load_medqa_rows
from models.qwen_wrapper import KnowledgeEncoder, load_base_model
from retrieval.dense_index import DenseKnowledgeIndex
from training.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


@dataclass
class MedQAKnowledgeEntry:
    key: str
    original_text: str
    knowledge_ids: List[int]


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_medqa_knowledge_entries(
    original_text_path: str | Path,
    knowledge_map_path: str | Path,
) -> Dict[str, MedQAKnowledgeEntry]:
    original_path = resolve_path(str(original_text_path))
    knowledge_path = resolve_path(str(knowledge_map_path))

    original_map: Dict[str, str] = {}
    with original_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            original_map[str(obj["key"])] = str(obj["original_text"])

    entries: Dict[str, MedQAKnowledgeEntry] = {}
    with knowledge_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj["key"])
            if key not in original_map:
                continue
            entries[key] = MedQAKnowledgeEntry(
                key=key,
                original_text=original_map[key],
                knowledge_ids=[int(x) for x in obj["knowledge_ids"]],
            )
    return entries


def select_edit_rows(
    limit: int,
    seed: int,
    knowledge_entries: Dict[str, MedQAKnowledgeEntry],
) -> List[Dict[str, Any]]:
    rows = load_medqa_rows(limit=None)
    eligible = [row for row in rows if row["key"] in knowledge_entries]
    if limit <= 0 or limit >= len(eligible):
        return eligible
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(eligible), generator=generator).tolist()[:limit]
    return [eligible[i] for i in indices]


def select_locality_rows(
    limit: int,
    seed: int,
    exclude_keys: Sequence[str],
    knowledge_entries: Dict[str, MedQAKnowledgeEntry],
) -> List[Dict[str, Any]]:
    rows = load_medqa_rows(limit=None)
    exclude = set(exclude_keys)
    eligible = [row for row in rows if row["key"] in knowledge_entries and row["key"] not in exclude]
    if limit <= 0 or limit >= len(eligible):
        return eligible
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(eligible), generator=generator).tolist()[:limit]
    return [eligible[i] for i in indices]


def select_disjoint_row_groups(
    group_sizes: Sequence[int],
    seed: int,
    knowledge_entries: Dict[str, MedQAKnowledgeEntry],
) -> List[List[Dict[str, Any]]]:
    rows = load_medqa_rows(limit=None)
    eligible = [row for row in rows if row["key"] in knowledge_entries]
    total_needed = sum(max(0, int(size)) for size in group_sizes)
    if total_needed > len(eligible):
        raise ValueError(
            f"requested {total_needed} disjoint rows, but only {len(eligible)} eligible MedQA rows are available"
        )

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(len(eligible), generator=generator).tolist()
    cursor = 0
    groups: List[List[Dict[str, Any]]] = []
    for size in group_sizes:
        size = max(0, int(size))
        group = [eligible[i] for i in shuffled_indices[cursor : cursor + size]]
        groups.append(group)
        cursor += size
    return groups


def build_retrieval_query(row: Dict[str, Any], query_mode: str) -> str:
    if query_mode == "question_only":
        return row["question"]
    if query_mode == "question_choices":
        return build_multiple_choice_prompt(row["question"], row["choices"])
    raise ValueError(f"unsupported query_mode: {query_mode}")


def _pad_knowledge_ids(ids: Sequence[int], fusion_length: int, pad_token_id: int) -> List[int]:
    trimmed = list(ids[:fusion_length])
    if len(trimmed) < fusion_length:
        trimmed.extend([pad_token_id] * (fusion_length - len(trimmed)))
    return trimmed


def build_knowledge_encoder_and_tokenizer(
    cfg: Config,
    device: torch.device | str,
) -> tuple[KnowledgeEncoder, AutoTokenizer]:
    device_obj = torch.device(device)
    model_path = get_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = load_base_model(model_path, bf16=cfg.train.bf16 and device_obj.type != "cpu")
    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.retrieval_encoder_depth,
        hidden_dim=cfg.model.hidden_dim,
        mode=cfg.model.knowledge_encoder_mode,
    )
    encoder.requires_grad_(False)
    encoder = encoder.to(device_obj).eval()
    return encoder, tokenizer


@torch.no_grad()
def encode_anchor_texts(
    cfg: Config,
    encoder: KnowledgeEncoder,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: torch.device | str,
) -> torch.Tensor:
    device_obj = torch.device(device)
    encoded = tokenizer(
        list(texts),
        max_length=cfg.model.anchor_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
        return_tensors="pt",
    )
    return encoder.encode_mean(
        encoded["input_ids"].to(device_obj),
        encoded["attention_mask"].long().to(device_obj),
    ).detach().cpu()


def get_compressed_text(
    entry: "MedQAKnowledgeEntry",
    tokenizer: AutoTokenizer,
) -> str:
    """从 knowledge_ids 解码压缩文本（过滤 pad token）。"""
    valid = [x for x in entry.knowledge_ids if x != tokenizer.pad_token_id]
    return tokenizer.decode(valid, skip_special_tokens=True).strip()


def build_fusion_encoder_and_tokenizer(
    cfg: Config,
    device: torch.device | str,
) -> tuple[KnowledgeEncoder, AutoTokenizer]:
    """检索侧编码器：retrieval_encoder_depth（r0=0，纯词嵌入）+ fusion_length。"""
    device_obj = torch.device(device)
    model_path = get_model_path(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = load_base_model(model_path, bf16=cfg.train.bf16 and device_obj.type != "cpu")
    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.retrieval_encoder_depth,   # r0=0
        hidden_dim=cfg.model.hidden_dim,
        mode=cfg.model.knowledge_encoder_mode,
    )
    encoder.requires_grad_(False)
    encoder = encoder.to(device_obj).eval()
    return encoder, tokenizer


@torch.no_grad()
def encode_fusion_texts(
    cfg: Config,
    encoder: KnowledgeEncoder,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: torch.device | str,
) -> torch.Tensor:
    """单视图：对压缩文本在 fusion_length 下编码为 embedding。"""
    device_obj = torch.device(device)
    encoded = tokenizer(
        list(texts),
        max_length=cfg.model.fusion_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=False,
        return_tensors="pt",
    )
    return encoder.encode_mean(
        encoded["input_ids"].to(device_obj),
        encoded["attention_mask"].long().to(device_obj),
    ).detach().cpu()


def save_temp_index(index: DenseKnowledgeIndex, prefix: str) -> Path:
    tmp_root = PROJECT_ROOT / "results/e8/tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"e8_{prefix}_", dir=str(tmp_root)))
    out_path = tmp_dir / f"{prefix}.pt"
    index.save(out_path)
    return out_path


def _load_medqa_anchor_rows(anchor_variant: str) -> List[Dict[str, str]]:
    if anchor_variant == "original_text":
        path = resolve_path("data/medqa_knowledge_original_text.jsonl")
        rows: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(
                    {
                        "key": str(obj["key"]),
                        "text": str(obj["original_text"]).strip(),
                    }
                )
        return rows

    if anchor_variant == "k256":
        path = resolve_path("data/medqa_knowledge_k256.jsonl")
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(
                    {
                        "key": str(obj["key"]),
                        "knowledge_ids": [int(x) for x in obj["knowledge_ids"]],
                    }
                )
        return rows

    raise ValueError(f"unsupported MedQA anchor variant: {anchor_variant}")


def build_medqa_overlay_index(
    cfg: Config,
    base_index_path: str | Path,
    device: torch.device | str,
    *,
    anchor_variant: str = "original_text",
    overlay_seed: int = 42,
) -> Tuple[Path, int]:
    """Build a temporary MedQA overlay index on top of a FineWeb base index."""

    base_index_resolved = resolve_path(str(base_index_path))
    dense_index = DenseKnowledgeIndex.load(base_index_resolved)
    target_size = len(dense_index)
    logger.info(
        "[E8] Building MedQA overlay index | base=%s | total=%d | active=%d | variant=%s | seed=%d",
        base_index_resolved,
        len(dense_index),
        dense_index.num_active,
        anchor_variant,
        overlay_seed,
    )

    knowledge_entries = load_medqa_knowledge_entries(
        original_text_path=resolve_path("data/medqa_knowledge_original_text.jsonl"),
        knowledge_map_path=resolve_path(cfg.eval.medqa_knowledge_map),
    )
    anchor_rows = _load_medqa_anchor_rows(anchor_variant)
    filtered_rows = [row for row in anchor_rows if row["key"] in knowledge_entries]
    if not filtered_rows:
        raise ValueError(f"no MedQA rows loaded for anchor variant {anchor_variant}")

    encoder, tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device)
    texts: List[str] = []
    keys: List[str] = []
    fusion_ids_rows: List[List[int]] = []
    for row in filtered_rows:
        key = row["key"]
        keys.append(key)
        # 单视图：embedding 来自 compressed knowledge_ids（与注入同源）
        compressed = get_compressed_text(knowledge_entries[key], tokenizer)
        texts.append(compressed)
        fusion_ids_rows.append(knowledge_entries[key].knowledge_ids)

    if len(keys) > target_size:
        raise ValueError(f"overlay docs {len(keys)} exceed index size {target_size}")

    embeddings = encode_fusion_texts(
        cfg=cfg,
        encoder=encoder,
        tokenizer=tokenizer,
        texts=texts,
        device=device,
    )
    pad_token_id = tokenizer.pad_token_id
    fusion_ids = torch.full((len(keys), cfg.model.fusion_length), pad_token_id, dtype=torch.long)
    for idx, token_ids in enumerate(fusion_ids_rows):
        cur = list(token_ids[: cfg.model.fusion_length])
        if len(cur) < cfg.model.fusion_length:
            cur.extend([pad_token_id] * (cfg.model.fusion_length - len(cur)))
        fusion_ids[idx] = torch.tensor(cur, dtype=torch.long)

    generator = torch.Generator().manual_seed(overlay_seed)
    replace_positions = torch.randperm(target_size, generator=generator).tolist()[: len(keys)]
    old_keys = [dense_index.keys[pos] for pos in replace_positions]
    deleted = dense_index.delete_by_keys(old_keys)
    dense_index.add_entries(
        embeddings=embeddings,
        fusion_ids=fusion_ids,
        keys=keys,
        texts=texts,
        replace_existing=True,
    )
    dense_index.compact()
    if len(dense_index) != target_size:
        raise RuntimeError(f"overlay size mismatch: expected {target_size}, got {len(dense_index)}")

    out_path = save_temp_index(dense_index, prefix=f"e8_medqa_overlay_{anchor_variant}")
    logger.info(
        "[E8] MedQA overlay index ready | output=%s | replaced=%d | deleted=%d",
        out_path,
        len(old_keys),
        deleted,
    )
    return out_path, deleted


def prepare_medqa_full_index(
    cfg: Config,
    *,
    memory_setting: str,
    full_index_path: str | None,
    base_index_path: str | None,
    device: torch.device | str,
    anchor_variant: str = "original_text",
    overlay_seed: int = 42,
) -> Dict[str, Any]:
    if memory_setting == "controlled":
        if not full_index_path:
            raise ValueError("--full-index is required when memory_setting=controlled")
        resolved = resolve_path(full_index_path)
        return {
            "memory_setting": memory_setting,
            "full_index_path": str(resolved),
            "source_index_path": str(resolved),
            "anchor_variant": anchor_variant,
            "overlay_seed": overlay_seed,
            "overlay_deleted": 0,
        }

    if memory_setting == "overlay_1m":
        if not base_index_path:
            raise ValueError("--base-index is required when memory_setting=overlay_1m")
        overlay_path, deleted = build_medqa_overlay_index(
            cfg=cfg,
            base_index_path=base_index_path,
            device=device,
            anchor_variant=anchor_variant,
            overlay_seed=overlay_seed,
        )
        return {
            "memory_setting": memory_setting,
            "full_index_path": str(overlay_path),
            "source_index_path": str(resolve_path(base_index_path)),
            "anchor_variant": anchor_variant,
            "overlay_seed": overlay_seed,
            "overlay_deleted": deleted,
        }

    raise ValueError(f"unsupported memory setting: {memory_setting}")


def compute_retrieval_top1_exact(
    retriever: DenseRetriever,
    rows: Sequence[Dict[str, Any]],
    knowledge_entries: Dict[str, MedQAKnowledgeEntry],
    query_mode: str,
    pad_token_id: int,
    fusion_length: int,
) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        query = build_retrieval_query(row, query_mode)
        pred = retriever.retrieve_from_texts([query])[0].detach().cpu().tolist()
        gold = _pad_knowledge_ids(
            knowledge_entries[row["key"]].knowledge_ids,
            fusion_length=fusion_length,
            pad_token_id=pad_token_id,
        )
        if pred == gold:
            correct += 1
    return correct / len(rows)


def compute_qa_accuracy(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    retriever: DenseRetriever,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    query_mode: str,
) -> float:
    if not rows:
        return 0.0
    correct = 0
    with torch.no_grad():
        for row in rows:
            retrieval_query = build_retrieval_query(row, query_mode)
            knowledge_ids = retriever.retrieve_from_texts([retrieval_query])
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pred = score_choices_injection(model, tokenizer, context_ids, knowledge_ids, device)
            if pred == row["label"]:
                correct += 1
    return correct / len(rows)


def evaluate_edit_rows(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    retriever: DenseRetriever,
    rows: Sequence[Dict[str, Any]],
    knowledge_entries: Dict[str, MedQAKnowledgeEntry],
    device: torch.device,
    query_mode: str,
    fusion_length: int,
) -> Dict[str, Any]:
    qa_correct_flags: List[bool] = []
    retrieval_exact_flags: List[bool] = []
    pad_token_id = tokenizer.pad_token_id

    with torch.no_grad():
        for row in rows:
            retrieval_query = build_retrieval_query(row, query_mode)
            knowledge_ids = retriever.retrieve_from_texts([retrieval_query])
            pred_knowledge = knowledge_ids[0].detach().cpu().tolist()
            gold_knowledge = _pad_knowledge_ids(
                knowledge_entries[row["key"]].knowledge_ids,
                fusion_length=fusion_length,
                pad_token_id=pad_token_id,
            )
            retrieval_exact_flags.append(pred_knowledge == gold_knowledge)

            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pred = score_choices_injection(model, tokenizer, context_ids, knowledge_ids, device)
            qa_correct_flags.append(pred == row["label"])

    total = len(rows)
    qa_acc = sum(qa_correct_flags) / total if total else 0.0
    retrieval_top1 = sum(retrieval_exact_flags) / total if total else 0.0
    return {
        "qa_acc": qa_acc,
        "retrieval_top1": retrieval_top1,
        "qa_correct_flags": qa_correct_flags,
        "retrieval_exact_flags": retrieval_exact_flags,
        "total": total,
    }


def build_phase3_injection_model(
    cfg: Config,
    phase3_weights: str | Path,
    device: torch.device | str,
):
    return build_injection_model(cfg, str(resolve_path(str(phase3_weights))), device=str(device), log_prefix="E8")


def init_logging() -> None:
    setup_logging()
    logger.info("================ E8 Editable Memory Benchmark ================")
