#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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


class ParquetEpochSampler:
    """与 Phase1 一致：按文件粒度打乱后顺序读取直到凑满 n_samples。"""

    def __init__(self, parquet_dir: str, n_samples: int) -> None:
        self.n_samples = n_samples
        files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"no parquet files found under {parquet_dir}")
        self.files = files

    def sample_epoch_data(self, seed: int) -> List[Dict[str, str]]:
        rng = random.Random(seed)
        shuffled_files = list(self.files)
        rng.shuffle(shuffled_files)

        rows: List[Dict[str, str]] = []
        for fpath in shuffled_files:
            if len(rows) >= self.n_samples:
                break
            df = pd.read_parquet(fpath, columns=["text", "compressed_text", "uuid", "original_uuid"])
            records = df.to_dict(orient="records")
            for idx, record in enumerate(records):
                key = record.get("uuid")
                if key is None:
                    key = record.get("original_uuid")
                if key is None:
                    key = f"{Path(fpath).stem}:{idx}"
                rows.append(
                    {
                        "key": str(key),
                        "text": str(record["text"]).strip(),
                        "compressed_text": str(record["compressed_text"]).strip(),
                    }
                )
                if len(rows) >= self.n_samples:
                    break
        return rows[: self.n_samples]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 1M dense index from FineWeb compressed parquet data")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--override", nargs="?", action="append", help="config overrides")
    parser.add_argument("--parquet-dir", default=str(PROJECT_ROOT / "data/compressed/v2"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample-size", type=int, default=1048576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tokenize-batch-size", type=int, default=10000)
    parser.add_argument("--index-type", choices=["flat", "hnsw"], default="flat")
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    return parser.parse_args()


def _parse_overrides(overrides: list[str] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    flat: list[str] = []
    for item in overrides:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    result: dict[str, Any] = {}
    for item in flat:
        if "=" not in item:
            raise ValueError(f"invalid override (missing '='): {item}")
        key, value = item.split("=", 1)
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


def _tokenize_texts(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = len(texts)
    input_ids = torch.zeros((n, max_length), dtype=torch.long)
    attention_mask = torch.zeros((n, max_length), dtype=torch.long)
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
        input_ids[start:end] = enc["input_ids"]
        attention_mask[start:end] = enc["attention_mask"].long()
    return input_ids, attention_mask


def _encode_embeddings(
    encoder: KnowledgeEncoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, input_ids.shape[0], batch_size):
            end = min(start + batch_size, input_ids.shape[0])
            emb = encoder.encode_mean(
                input_ids[start:end].to(encoder.device),
                attention_mask[start:end].to(encoder.device),
            )
            outputs.append(emb.cpu().float())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, encoder.hidden_dim), dtype=torch.float32)


def main() -> None:
    _setup_logging()
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))

    sampler = ParquetEpochSampler(args.parquet_dir, args.sample_size)
    rows = sampler.sample_epoch_data(seed=args.seed)
    logger.info(
        "Sampled FineWeb rows | parquet_dir=%s | seed=%d | requested=%d | actual=%d",
        args.parquet_dir,
        args.seed,
        args.sample_size,
        len(rows),
    )

    model_path = os.environ.get("MODEL_PATH", cfg.paths.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    keys = [row["key"] for row in rows]
    texts = [row["text"] for row in rows]
    compressed = [row["compressed_text"] for row in rows]
    # 单视图：fusion_ids 同时用于 embedding 和注入
    fusion_ids, fusion_mask = _tokenize_texts(
        compressed, tokenizer, max_length=cfg.model.fusion_length, batch_size=args.tokenize_batch_size
    )

    base_model = load_base_model(model_path, bf16=cfg.train.bf16 and args.device != "cpu")
    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.retrieval_encoder_depth,   # r0：纯词嵌入，0 层
        hidden_dim=cfg.model.hidden_dim,
        mode=cfg.model.knowledge_encoder_mode,
    )
    encoder.requires_grad_(False)
    encoder = encoder.to(args.device).eval()

    logger.info("Encoding FineWeb dense embeddings (fusion view) | count=%d | batch_size=%d", len(texts), args.batch_size)
    embeddings = _encode_embeddings(encoder, fusion_ids, fusion_mask, batch_size=args.batch_size)

    index = DenseKnowledgeIndex(
        dim=cfg.model.hidden_dim,
        fusion_length=cfg.model.fusion_length,
        index_type=args.index_type,
        normalize=True,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_ef_search=args.hnsw_ef_search,
    )
    index.add_entries(embeddings=embeddings, fusion_ids=fusion_ids, keys=keys, texts=texts)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    index.save(output)
    logger.info(
        "FineWeb dense index built | output=%s | total=%d | active=%d | seed=%d",
        output,
        len(index),
        index.num_active,
        args.seed,
    )


if __name__ == "__main__":
    main()
