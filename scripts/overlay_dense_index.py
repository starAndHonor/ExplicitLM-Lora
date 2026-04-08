#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from models.qwen_wrapper import KnowledgeEncoder, load_base_model
from retrieval.dense_index import DenseKnowledgeIndex

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay task knowledge onto an existing dense index while keeping total size fixed")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", default="", help="Single-source jsonl/parquet/txt")
    parser.add_argument("--anchor-input", default="", help="Dual-source anchor input")
    parser.add_argument("--fusion-input", default="", help="Dual-source fusion input")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tokenize-batch-size", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_rows(path: Path, limit: int) -> List[Dict[str, object]]:
    suffix = path.suffix.lower()
    rows: List[Dict[str, object]] = []
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                if key is None:
                    key = str(idx)
                text = obj.get("text")
                if text is None:
                    text = obj.get("original_text")
                knowledge_ids = obj.get("knowledge_ids")
                compressed = obj.get("compressed_text")
                if compressed is None and text is not None:
                    compressed = text
                rows.append(
                    {
                        "key": str(key),
                        "text": None if text is None else str(text).strip(),
                        "compressed_text": None if compressed is None else str(compressed).strip(),
                        "knowledge_ids": None if knowledge_ids is None else [int(x) for x in knowledge_ids],
                    }
                )
                if limit > 0 and len(rows) >= limit:
                    break
        return rows
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        if limit > 0:
            records = records[:limit]
        for idx, record in enumerate(records):
            key = record.get("key")
            if key is None:
                key = str(idx)
            text = record.get("text")
            if text is None:
                text = record.get("original_text")
            knowledge_ids = record.get("knowledge_ids")
            compressed = record.get("compressed_text")
            if compressed is None and text is not None:
                compressed = text
            rows.append(
                {
                    "key": str(key),
                    "text": None if text is None else str(text).strip(),
                    "compressed_text": None if compressed is None else str(compressed).strip(),
                    "knowledge_ids": None if knowledge_ids is None else [int(x) for x in knowledge_ids],
                }
            )
        return rows
    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                rows.append({"key": str(idx), "text": text, "compressed_text": text, "knowledge_ids": None})
                if limit > 0 and len(rows) >= limit:
                    break
        return rows
    raise ValueError(f"unsupported input suffix: {suffix}")


def _resolve_text(
    row: Dict[str, object],
    tokenizer: AutoTokenizer,
    *,
    fallback_to_key: bool,
) -> str:
    text = row.get("text")
    if text is not None:
        text_str = str(text).strip()
        if text_str:
            return text_str

    knowledge_ids = row.get("knowledge_ids")
    if knowledge_ids is not None:
        decoded = tokenizer.decode([int(x) for x in knowledge_ids], skip_special_tokens=True).strip()
        if decoded:
            return decoded

    compressed = row.get("compressed_text")
    if compressed is not None:
        compressed_str = str(compressed).strip()
        if compressed_str:
            return compressed_str

    if fallback_to_key:
        key = row.get("key")
        if key is not None:
            return str(key).strip()

    raise ValueError("row missing decodable text/original_text/knowledge_ids")


def _materialize_rows(
    rows: Sequence[Dict[str, object]],
    tokenizer: AutoTokenizer,
    *,
    fallback_to_key: bool,
) -> List[Dict[str, str]]:
    materialized: List[Dict[str, str]] = []
    for idx, row in enumerate(rows):
        try:
            text = _resolve_text(row, tokenizer, fallback_to_key=fallback_to_key)
        except Exception as exc:
            raise ValueError(f"row {idx} missing usable anchor text") from exc
        compressed = row.get("compressed_text")
        compressed_text = text if compressed is None else str(compressed).strip() or text
        materialized.append(
            {
                "key": str(row["key"]),
                "text": text,
                "compressed_text": compressed_text,
            }
        )
    return materialized


def _load_fusion_map(path: Path, limit: int) -> Dict[str, List[int]]:
    suffix = path.suffix.lower()
    mapping: Dict[str, List[int]] = {}
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                knowledge_ids = obj.get("knowledge_ids")
                if key is None or knowledge_ids is None:
                    raise ValueError(f"row {idx} missing key/knowledge_ids")
                mapping[str(key)] = [int(x) for x in knowledge_ids]
                if limit > 0 and len(mapping) >= limit:
                    break
        return mapping
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        if limit > 0:
            records = records[:limit]
        for idx, record in enumerate(records):
            key = record.get("key")
            knowledge_ids = record.get("knowledge_ids")
            if key is None or knowledge_ids is None:
                raise ValueError(f"row {idx} missing key/knowledge_ids")
            mapping[str(key)] = [int(x) for x in knowledge_ids]
        return mapping
    raise ValueError(f"unsupported fusion input suffix: {suffix}")


def _tokenize_texts(texts: Sequence[str], tokenizer: AutoTokenizer, max_length: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(texts)
    ids = torch.zeros((n, max_length), dtype=torch.long)
    mask = torch.zeros((n, max_length), dtype=torch.long)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        enc = tokenizer(
            list(texts[start:end]),
            max_length=max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )
        ids[start:end] = enc["input_ids"]
        mask[start:end] = enc["attention_mask"].long()
    return ids, mask


def _encode_embeddings(encoder: KnowledgeEncoder, ids: torch.Tensor, mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, ids.shape[0], batch_size):
            end = min(start + batch_size, ids.shape[0])
            emb = encoder.encode_mean(ids[start:end].to(encoder.device), mask[start:end].to(encoder.device))
            outs.append(emb.cpu().float())
    return torch.cat(outs, dim=0) if outs else torch.empty((0, encoder.hidden_dim), dtype=torch.float32)


def main() -> None:
    _setup_logging()
    args = _parse_args()
    cfg = load_config(args.config)

    dense_index = DenseKnowledgeIndex.load(args.index)
    target_size = len(dense_index)
    logger.info("Loaded dense index | total=%d | active=%d", len(dense_index), dense_index.num_active)

    model_path = os.environ.get("MODEL_PATH", cfg.paths.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.anchor_input or args.fusion_input:
        if not args.anchor_input or not args.fusion_input:
            raise ValueError("dual-source mode requires both --anchor-input and --fusion-input")
        anchor_rows_raw = _load_rows(Path(args.anchor_input), args.limit)
        fusion_map = _load_fusion_map(Path(args.fusion_input), args.limit)
        anchor_rows = _materialize_rows(anchor_rows_raw, tokenizer, fallback_to_key=False)
        keys: List[str] = []
        texts: List[str] = []
        fusion_ids_rows: List[List[int]] = []
        for row in anchor_rows:
            key = row["key"]
            if key not in fusion_map:
                continue
            keys.append(key)
            texts.append(row["text"])
            fusion_ids_rows.append(fusion_map[key])
        fusion_ids = torch.full((len(keys), cfg.model.fusion_length), tokenizer.pad_token_id, dtype=torch.long)
        for idx, token_ids in enumerate(fusion_ids_rows):
            cur = list(token_ids[: cfg.model.fusion_length])
            if len(cur) < cfg.model.fusion_length:
                cur.extend([tokenizer.pad_token_id] * (cfg.model.fusion_length - len(cur)))
            fusion_ids[idx] = torch.tensor(cur, dtype=torch.long)
    else:
        if not args.input:
            raise ValueError("either --input or dual-source inputs are required")
        rows_raw = _load_rows(Path(args.input), args.limit)
        rows = _materialize_rows(rows_raw, tokenizer, fallback_to_key=True)
        keys = [row["key"] for row in rows]
        texts = [row["text"] for row in rows]
        compressed = [row["compressed_text"] for row in rows]
        fusion_ids, _ = _tokenize_texts(compressed, tokenizer, cfg.model.fusion_length, args.tokenize_batch_size)

    anchor_ids, anchor_mask = _tokenize_texts(texts, tokenizer, cfg.model.anchor_length, args.tokenize_batch_size)
    base_model = load_base_model(model_path, bf16=cfg.train.bf16 and args.device != "cpu")
    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.retrieval_encoder_depth,
        hidden_dim=cfg.model.hidden_dim,
        mode=cfg.model.knowledge_encoder_mode,
    )
    encoder.requires_grad_(False)
    encoder = encoder.to(args.device).eval()
    embeddings = _encode_embeddings(encoder, anchor_ids, anchor_mask, args.batch_size)

    n_new = len(keys)
    if n_new > target_size:
        raise ValueError(f"overlay docs {n_new} exceed index size {target_size}")

    rng = random.Random(args.seed)
    positions = list(range(target_size))
    rng.shuffle(positions)
    replace_positions = positions[:n_new]

    logger.info(
        "Overlaying %d new docs onto dense index of size %d (seed=%d)",
        n_new,
        target_size,
        args.seed,
    )

    old_keys = [dense_index.keys[pos] for pos in replace_positions]
    dense_index.delete_by_keys(old_keys)
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

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dense_index.save(output)
    logger.info(
        "Overlay dense index saved | output=%s | total=%d | active=%d | replaced=%d",
        output,
        len(dense_index),
        dense_index.num_active,
        len(old_keys),
    )


if __name__ == "__main__":
    main()
