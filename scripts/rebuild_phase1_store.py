#!/usr/bin/env python
"""
重建 Phase 1 知识库与索引结构，不执行 Router 监督训练。

用途：
    当需要替换 knowledge source 时，基于新的知识文本重建：
      - anchor_bank
      - fusion_bank
      - PCA / row_centroids / col_centroids
      - inverted_index / cluster_offsets / cluster_counts
      - router.pkm.row_keys / router.pkm.col_keys

输入格式：
    1. 单文件模式（兼容）
       JSONL / Parquet / TXT
       每条记录至少需要：
         - text
       可选：
         - compressed_text
       若无 compressed_text，则退化为使用 text 直接写入 fusion_bank

    2. 双文件模式（推荐）
       --anchor-input: 原始知识文本（text / original_text）
       --fusion-input: 压缩知识（key -> knowledge_ids）
       脚本会按 key 对齐，生成：
         - anchor_bank: 原文 token 化
         - fusion_bank: 直接使用压缩 knowledge_ids

注意：
    - 本脚本不会更新 router 的可训练参数，只会同步新的 PKM keys。
    - 输入条目数若不足 cfg.router.knowledge_num，会循环复制已有条目直到充足。
    - 若输入条目数更多，则只取前 N 条。
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from models.qwen_wrapper import KnowledgeEncoder, load_base_model
from router.memory_bank import DualKnowledgeStore
from router.model import MemoryRouter
from training.phase1_router import tokenize_parquet_batch

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Phase1 store from a new knowledge source")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--input",
        default="",
        help="Single-source mode only: knowledge source file (.jsonl / .parquet / .txt)",
    )
    parser.add_argument("--anchor-input", default="", help="Anchor source file: original text jsonl/parquet/txt")
    parser.add_argument("--fusion-input", default="", help="Fusion source file: compressed knowledge jsonl/parquet")
    parser.add_argument(
        "--phase1-ckpt",
        default="checkpoints/phase1_best",
        help="Existing Phase1 checkpoint dir (used to load router.pt before updating PKM keys)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output checkpoint dir, e.g. checkpoints/phase1_medical_rebuilt",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for encoder and recluster",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Recluster encode chunk size; default uses cfg.data.phase1_recluster_chunk_size",
    )
    parser.add_argument(
        "--tokenize-batch-size",
        type=int,
        default=None,
        help="Tokenize batch size; default uses cfg.data.phase1_tokenize_batch_size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional hard cap before matching knowledge_num; -1 means use all rows",
    )
    parser.add_argument(
        "--overlay-on-existing",
        action="store_true",
        help="Use existing phase1 store as the base and randomly overwrite part of it with new knowledge",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when overlaying new knowledge onto the existing store",
    )
    return parser.parse_args()


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


def _normalize_record(record: Dict[str, object], idx: int) -> Dict[str, str]:
    text = record.get("text")
    if text is None:
        text = record.get("original_text")
    if text is None:
        raise ValueError(f"row {idx} missing 'text'/'original_text'")

    compressed = record.get("compressed_text")
    if compressed is None:
        compressed = text

    return {
        "text": str(text).strip(),
        "compressed_text": str(compressed).strip(),
    }


def _load_rows(path: Path, limit: int) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    rows: List[Dict[str, str]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                rows.append(_normalize_record(obj, idx))
                if limit > 0 and len(rows) >= limit:
                    break
        return rows

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        if limit > 0:
            records = records[:limit]
        return [_normalize_record(record, idx) for idx, record in enumerate(records)]

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                rows.append({"text": text, "compressed_text": text})
                if limit > 0 and len(rows) >= limit:
                    break
        return rows

    raise ValueError(f"unsupported input suffix: {suffix}")


def _load_anchor_rows(path: Path, limit: int) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    rows: List[Dict[str, str]] = []

    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get("text")
                if text is None:
                    text = obj.get("original_text")
                if text is None:
                    raise ValueError(f"row {idx} missing 'text'/'original_text' in {path}")
                key = obj.get("key")
                if key is None:
                    key = str(idx)
                rows.append({"key": str(key), "text": str(text).strip()})
                if limit > 0 and len(rows) >= limit:
                    break
        return rows

    if suffix == ".parquet":
        df = pd.read_parquet(path)
        records = df.to_dict(orient="records")
        if limit > 0:
            records = records[:limit]
        for idx, record in enumerate(records):
            text = record.get("text")
            if text is None:
                text = record.get("original_text")
            if text is None:
                raise ValueError(f"row {idx} missing 'text'/'original_text' in {path}")
            key = record.get("key")
            if key is None:
                key = str(idx)
            rows.append({"key": str(key), "text": str(text).strip()})
        return rows

    if suffix == ".txt":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                text = line.strip()
                if not text:
                    continue
                rows.append({"key": str(idx), "text": text})
                if limit > 0 and len(rows) >= limit:
                    break
        return rows

    raise ValueError(f"unsupported anchor input suffix: {suffix}")


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
                    raise ValueError(f"row {idx} missing 'key'/'knowledge_ids' in {path}")
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
                raise ValueError(f"row {idx} missing 'key'/'knowledge_ids' in {path}")
            mapping[str(key)] = [int(x) for x in knowledge_ids]
        return mapping

    raise ValueError(f"unsupported fusion input suffix: {suffix}")


def _repeat_rows(rows: List[Dict[str, str]], target_size: int) -> List[Dict[str, str]]:
    if len(rows) >= target_size:
        return rows[:target_size]

    if not rows:
        raise ValueError("cannot repeat empty rows")

    original = list(rows)
    repeat_count = target_size - len(rows)
    logger.warning(
        "Knowledge rows not enough; repeating existing rows to add %d entries and reach knowledge_num=%d",
        repeat_count,
        target_size,
    )
    i = 0
    rows = list(rows)
    while len(rows) < target_size:
        src = original[i % len(original)]
        rows.append(
            {
                "text": src["text"],
                "compressed_text": src["compressed_text"],
            }
        )
        i += 1
    return rows


def _repeat_anchor_and_fusion(
    anchor_rows: List[Dict[str, str]],
    fusion_ids_rows: List[List[int]],
    target_size: int,
) -> Tuple[List[Dict[str, str]], List[List[int]]]:
    if len(anchor_rows) >= target_size:
        return anchor_rows[:target_size], fusion_ids_rows[:target_size]

    if not anchor_rows or not fusion_ids_rows:
        raise ValueError("cannot repeat empty anchor/fusion rows")

    original_anchor = list(anchor_rows)
    original_fusion = [list(x) for x in fusion_ids_rows]
    repeat_count = target_size - len(anchor_rows)
    logger.warning(
        "Knowledge rows not enough; repeating existing aligned rows to add %d entries and reach knowledge_num=%d",
        repeat_count,
        target_size,
    )
    anchor_rows = list(anchor_rows)
    fusion_ids_rows = [list(x) for x in fusion_ids_rows]
    i = 0
    while len(anchor_rows) < target_size:
        src_anchor = original_anchor[i % len(original_anchor)]
        src_fusion = original_fusion[i % len(original_fusion)]
        anchor_rows.append(
            {
                "key": src_anchor.get("key", f"__repeat_{i}"),
                "text": src_anchor["text"],
                "compressed_text": src_anchor["compressed_text"],
            }
        )
        fusion_ids_rows.append(list(src_fusion))
        i += 1
    return anchor_rows, fusion_ids_rows


def _overlay_new_on_existing(
    new_anchor_ids: torch.Tensor,
    new_fusion_ids: torch.Tensor,
    phase1_ckpt: Path,
    cfg,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    store_path = phase1_ckpt / "store.pt"
    if not store_path.exists():
        raise FileNotFoundError(f"existing store checkpoint not found: {store_path}")

    base_store = DualKnowledgeStore(
        cfg.router,
        cfg.model.fusion_length,
        cfg.model.anchor_length,
        device="cpu",
    )
    base_store.load_state(str(store_path))

    base_anchor = base_store.anchor_bank.data.clone()
    base_fusion = base_store.fusion_bank.data.clone()
    knowledge_num = cfg.router.knowledge_num
    n_new = int(new_anchor_ids.shape[0])
    if n_new > knowledge_num:
        raise ValueError(
            f"new knowledge count {n_new} exceeds knowledge_num={knowledge_num}; "
            "cannot guarantee all new knowledge is included in overlay mode"
        )

    rng = random.Random(seed)
    positions = list(range(knowledge_num))
    rng.shuffle(positions)
    replace_positions = positions[:n_new]

    logger.info(
        "Overlaying %d new knowledge rows onto existing store of size %d (seed=%d)",
        n_new,
        knowledge_num,
        seed,
    )
    for src_idx, dst_idx in enumerate(replace_positions):
        base_anchor[dst_idx] = new_anchor_ids[src_idx]
        base_fusion[dst_idx] = new_fusion_ids[src_idx]

    return base_anchor, base_fusion


def main() -> None:
    _setup_logging()
    args = _parse_args()

    cfg = load_config(args.config)
    phase1_ckpt = Path(args.phase1_ckpt)
    router_path = phase1_ckpt / "router.pt"
    if not router_path.exists():
        raise FileNotFoundError(f"router checkpoint not found: {router_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    knowledge_num = cfg.router.knowledge_num
    chunk_size = args.chunk_size or cfg.data.phase1_recluster_chunk_size
    tokenize_batch_size = args.tokenize_batch_size or cfg.data.phase1_tokenize_batch_size

    model_path = cfg.paths.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.anchor_input or args.fusion_input:
        if not args.anchor_input or not args.fusion_input:
            raise ValueError("dual-source mode requires both --anchor-input and --fusion-input")

        anchor_path = Path(args.anchor_input)
        fusion_path = Path(args.fusion_input)
        if not anchor_path.exists():
            raise FileNotFoundError(f"anchor source not found: {anchor_path}")
        if not fusion_path.exists():
            raise FileNotFoundError(f"fusion source not found: {fusion_path}")

        logger.info("Loading anchor rows from %s", anchor_path)
        anchor_raw = _load_anchor_rows(anchor_path, args.limit)
        logger.info("Loading fusion map from %s", fusion_path)
        fusion_map = _load_fusion_map(fusion_path, args.limit)

        anchor_rows: List[Dict[str, str]] = []
        fusion_ids_rows: List[List[int]] = []
        for row in anchor_raw:
            key = row["key"]
            if key not in fusion_map:
                continue
            anchor_rows.append(
                {
                    "text": row["text"],
                    "compressed_text": row["text"],
                }
            )
            fusion_ids_rows.append(fusion_map[key])

        logger.info(
            "Aligned anchor/fusion rows | anchors=%d | matched=%d",
            len(anchor_raw),
            len(anchor_rows),
        )
        if args.overlay_on_existing and len(anchor_rows) > knowledge_num:
            raise ValueError(
                f"overlay mode requires new knowledge count <= knowledge_num, got {len(anchor_rows)} > {knowledge_num}"
            )
        if not args.overlay_on_existing:
            if len(anchor_rows) > knowledge_num:
                anchor_rows = anchor_rows[:knowledge_num]
                fusion_ids_rows = fusion_ids_rows[:knowledge_num]
            else:
                anchor_rows, fusion_ids_rows = _repeat_anchor_and_fusion(
                    anchor_rows,
                    fusion_ids_rows,
                    knowledge_num,
                )

        logger.info(
            "Tokenizing anchor rows | count=%d | anchor_length=%d",
            len(anchor_rows),
            cfg.model.anchor_length,
        )
        anchor_ids, _ = tokenize_parquet_batch(
            anchor_rows,
            tokenizer,
            anchor_length=cfg.model.anchor_length,
            fusion_length=cfg.model.fusion_length,
            batch_size=tokenize_batch_size,
        )
        fusion_ids = torch.full(
            (knowledge_num, cfg.model.fusion_length),
            fill_value=tokenizer.pad_token_id,
            dtype=torch.long,
        )
        for i, token_ids in enumerate(fusion_ids_rows):
            ids = list(token_ids[: cfg.model.fusion_length])
            if len(ids) < cfg.model.fusion_length:
                ids.extend([tokenizer.pad_token_id] * (cfg.model.fusion_length - len(ids)))
            fusion_ids[i] = torch.tensor(ids, dtype=torch.long)
        if args.overlay_on_existing:
            n_new = len(anchor_rows)
            anchor_ids = anchor_ids[:n_new]
            fusion_ids = fusion_ids[:n_new]
            anchor_ids, fusion_ids = _overlay_new_on_existing(
                anchor_ids,
                fusion_ids,
                phase1_ckpt,
                cfg,
                args.seed,
            )
            source_desc = (
                f"overlay_on_existing(anchor={anchor_path},fusion={fusion_path},seed={args.seed})"
            )
        else:
            source_desc = f"anchor={anchor_path},fusion={fusion_path}"
    else:
        source_path = Path(args.input)
        if not source_path.exists():
            raise FileNotFoundError(f"knowledge source not found: {source_path}")
        logger.info("Loading knowledge rows from %s", source_path)
        rows = _load_rows(source_path, args.limit)
        if args.overlay_on_existing and len(rows) > knowledge_num:
            raise ValueError(
                f"overlay mode requires new knowledge count <= knowledge_num, got {len(rows)} > {knowledge_num}"
            )
        if not args.overlay_on_existing:
            if len(rows) > knowledge_num:
                logger.info("Input has %d rows; only first %d will be used", len(rows), knowledge_num)
                rows = rows[:knowledge_num]
            else:
                rows = _repeat_rows(rows, knowledge_num)

        logger.info(
            "Tokenizing rows | count=%d | anchor_length=%d | fusion_length=%d",
            len(rows),
            cfg.model.anchor_length,
            cfg.model.fusion_length,
        )
        anchor_ids, fusion_ids = tokenize_parquet_batch(
            rows,
            tokenizer,
            anchor_length=cfg.model.anchor_length,
            fusion_length=cfg.model.fusion_length,
            batch_size=tokenize_batch_size,
        )
        if args.overlay_on_existing:
            anchor_ids, fusion_ids = _overlay_new_on_existing(
                anchor_ids,
                fusion_ids,
                phase1_ckpt,
                cfg,
                args.seed,
            )
            source_desc = f"overlay_on_existing(source={source_path},seed={args.seed})"
        else:
            source_desc = str(source_path)

    logger.info("Loading base model / knowledge encoder on %s", args.device)
    base_model = load_base_model(model_path, bf16=cfg.train.bf16 and args.device != "cpu")
    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.encoder_depth,
        hidden_dim=cfg.model.hidden_dim,
        mode=cfg.model.knowledge_encoder_mode,
    )
    encoder.requires_grad_(False)
    encoder = encoder.to(args.device).eval()

    logger.info("Building new DualKnowledgeStore")
    store = DualKnowledgeStore(
        cfg.router,
        cfg.model.fusion_length,
        cfg.model.anchor_length,
        device="cpu",
    )
    store.anchor_bank.update_all(anchor_ids)
    store.fusion_bank.update_all(fusion_ids)
    store.valid_mask[:knowledge_num] = True
    store.next_free = knowledge_num

    logger.info("Running compact_and_recluster | chunk_size=%d", chunk_size)
    store.compact_and_recluster(encoder, chunk_size=chunk_size)

    logger.info("Loading existing router and refreshing PKM keys")
    router = MemoryRouter(cfg.router, encoder)
    router_state = torch.load(router_path, map_location="cpu", weights_only=True)
    router.load_state_dict(router_state, strict=True)
    assert store.row_centroids is not None and store.col_centroids is not None
    router.pkm.update_keys(store.row_centroids, store.col_centroids)

    torch.save(router.state_dict(), output_dir / "router.pt")
    store.save_state(str(output_dir / "store.pt"))
    meta_lines = [
        f"source={source_desc}",
        f"rows={knowledge_num}",
        f"anchor_length={cfg.model.anchor_length}",
        f"fusion_length={cfg.model.fusion_length}",
        f"chunk_size={chunk_size}",
        "note=rebuild_phase1_store_without_router_training",
    ]
    (output_dir / "meta.txt").write_text("\n".join(meta_lines), encoding="utf-8")

    logger.info("Rebuild done | output_dir=%s", output_dir)


if __name__ == "__main__":
    main()
