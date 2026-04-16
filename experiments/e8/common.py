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
    fusion_length: int,
) -> Dict[str, MedQAKnowledgeEntry]:
    """从 k-specific knowledge jsonl 加载 MedQA knowledge 条目（single-view）。"""
    path = _medqa_knowledge_path(fusion_length)
    entries: Dict[str, MedQAKnowledgeEntry] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = str(obj["key"])
            knowledge_ids = [int(x) for x in obj["knowledge_ids"]]
            original_text = str(obj.get("text") or obj.get("original_text") or "")
            entries[key] = MedQAKnowledgeEntry(
                key=key,
                original_text=original_text,
                knowledge_ids=knowledge_ids,
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


def _medqa_knowledge_path(fusion_length: int) -> Path:
    """根据 fusion_length 返回对应的预处理知识文件路径。"""
    if fusion_length == 64:
        return resolve_path("data/medqa_knowledge.jsonl")
    return resolve_path(f"data/medqa_knowledge_k{fusion_length}.jsonl")


def _load_medqa_knowledge_rows(fusion_length: int) -> List[Dict[str, object]]:
    """加载对应 k-size 的 MedQA knowledge 文件（含 knowledge_ids）。"""
    path = _medqa_knowledge_path(fusion_length)
    rows: List[Dict[str, object]] = []
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


def build_medqa_overlay_index(
    cfg: Config,
    base_index_path: str | Path,
    device: torch.device | str,
    *,
    overlay_seed: int = 42,
) -> Tuple[Path, int]:
    """构建临时 MedQA overlay 索引（单视图，fusion_length 由 cfg.model.fusion_length 控制）。"""
    base_index_resolved = resolve_path(str(base_index_path))
    dense_index = DenseKnowledgeIndex.load(base_index_resolved)
    target_size = len(dense_index)
    logger.info(
        "[E8] Building MedQA overlay index | base=%s | total=%d | fusion_length=%d | knowledge_path=%s | seed=%d",
        base_index_resolved,
        len(dense_index),
        cfg.model.fusion_length,
        _medqa_knowledge_path(cfg.model.fusion_length),
        overlay_seed,
    )

    knowledge_rows = _load_medqa_knowledge_rows(cfg.model.fusion_length)
    if not knowledge_rows:
        raise ValueError(f"no MedQA knowledge rows loaded for fusion_length={cfg.model.fusion_length}")

    encoder, tokenizer = build_fusion_encoder_and_tokenizer(cfg, device=device)
    pad_token_id = tokenizer.pad_token_id

    keys: List[str] = []
    texts: List[str] = []
    fusion_ids_list: List[List[int]] = []
    for row in knowledge_rows:
        keys.append(str(row["key"]))
        # 解码 knowledge_ids 作为 embedding 输入文本
        valid_ids = [x for x in row["knowledge_ids"] if x != pad_token_id]
        texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True).strip())
        fusion_ids_list.append(list(row["knowledge_ids"]))

    if len(keys) > target_size:
        raise ValueError(f"overlay docs {len(keys)} exceed index size {target_size}")

    embeddings = encode_fusion_texts(cfg=cfg, encoder=encoder, tokenizer=tokenizer, texts=texts, device=device)
    fusion_ids = torch.full((len(keys), cfg.model.fusion_length), pad_token_id, dtype=torch.long)
    for idx, ids in enumerate(fusion_ids_list):
        cur = ids[: cfg.model.fusion_length]
        fusion_ids[idx, : len(cur)] = torch.tensor(cur, dtype=torch.long)

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

    out_path = save_temp_index(dense_index, prefix="e8_medqa_overlay")
    logger.info("[E8] MedQA overlay index ready | output=%s | replaced=%d | deleted=%d", out_path, len(old_keys), deleted)
    return out_path, deleted


def prepare_medqa_full_index(
    cfg: Config,
    *,
    memory_setting: str,
    full_index_path: str | None,
    base_index_path: str | None,
    device: torch.device | str,
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
            "fusion_length": cfg.model.fusion_length,
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
            overlay_seed=overlay_seed,
        )
        return {
            "memory_setting": memory_setting,
            "full_index_path": str(overlay_path),
            "source_index_path": str(resolve_path(base_index_path)),
            "fusion_length": cfg.model.fusion_length,
            "overlay_seed": overlay_seed,
            "overlay_deleted": deleted,
        }

    raise ValueError(f"unsupported memory setting: {memory_setting}")


def compute_retrieval_top1_exact(
    retriever: DenseRetriever,
    rows: Sequence[Dict[str, Any]],
    query_mode: str,
) -> float:
    """按 key 匹配计算 top-1 检索正确率（单视图模式）。"""
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        query = build_retrieval_query(row, query_mode)
        search = retriever.search_from_texts([query], top_k=1)
        idx = int(search.indices[0][0].item())
        valid = bool(search.valid_mask[0][0].item())
        pred_key = retriever.index.keys[idx] if valid and idx >= 0 else ""
        if pred_key == str(row["key"]):
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
    device: torch.device,
    query_mode: str,
) -> Dict[str, Any]:
    """评测编辑行的 QA 准确率和 top-1 检索正确率（key 匹配，单视图模式）。"""
    qa_correct_flags: List[bool] = []
    retrieval_exact_flags: List[bool] = []

    with torch.no_grad():
        for row in rows:
            retrieval_query = build_retrieval_query(row, query_mode)
            # key 匹配检索正确率
            search = retriever.search_from_texts([retrieval_query], top_k=1)
            idx = int(search.indices[0][0].item())
            valid = bool(search.valid_mask[0][0].item())
            pred_key = retriever.index.keys[idx] if valid and idx >= 0 else ""
            retrieval_exact_flags.append(pred_key == str(row["key"]))

            # QA 准确率：取 top-1 fusion_ids 注入
            knowledge_ids = retriever.retrieve_from_texts([retrieval_query])
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
