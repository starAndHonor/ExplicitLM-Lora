from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import Config
from experiments.e2.common import (
    PROJECT_ROOT,
    build_baseline_model,
    build_injection_model,
    setup_logging,
)
from experiments.e2.scoring import build_multiple_choice_prompt, score_choices_injection
from experiments.e3.evaluator import eval_baseline
from training.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


class E5KnowledgeRetriever:
    """E5 dense retriever that returns fusion knowledge ids for Phase3 injection."""

    def __init__(
        self,
        cfg: Config,
        anchor_path: str,
        fusion_path: str,
        device: torch.device,
        e5_model_name: str,
        pad_token_id: int,
        hf_tokenizer: Optional[AutoTokenizer] = None,
        hf_model: Optional[AutoModel] = None,
    ) -> None:
        self.cfg = cfg
        self.anchor_path = Path(anchor_path)
        self.fusion_path = Path(fusion_path)
        self.device = device
        self.e5_model_name = e5_model_name
        self.pad_token_id = int(pad_token_id)
        self.fusion_length = int(cfg.model.fusion_length)

        if not self.anchor_path.exists():
            raise FileNotFoundError(f"E5 anchor file not found: {self.anchor_path}")
        if not self.fusion_path.exists():
            raise FileNotFoundError(f"E5 fusion file not found: {self.fusion_path}")

        self._tokenizer = hf_tokenizer or AutoTokenizer.from_pretrained(self.e5_model_name, use_fast=True)
        self._model = hf_model or AutoModel.from_pretrained(self.e5_model_name)
        self._model = self._model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad = False

        fusion_map = self._load_fusion_map(self.fusion_path)
        keys: List[str] = []
        texts: List[str] = []
        fusion_rows: List[List[int]] = []
        for item in self._load_anchor_rows(self.anchor_path):
            key = item["key"]
            if key not in fusion_map:
                continue
            keys.append(key)
            texts.append(item["text"])
            fusion_rows.append(fusion_map[key])
        if not keys:
            raise RuntimeError(
                f"E5 retriever got empty aligned set from anchor={self.anchor_path} and fusion={self.fusion_path}"
            )

        self.keys = keys
        self.fusion_ids = self._build_fusion_tensor(fusion_rows).to(self.device)
        self.doc_embeddings = self._encode_texts(
            [f"passage: {text}" for text in texts], batch_size=32, max_length=256
        ).to(self.device)
        logger.info(
            "[E5Retriever] built | model=%s | docs=%d | emb_shape=%s | anchor=%s | fusion=%s",
            self.e5_model_name,
            len(self.keys),
            tuple(self.doc_embeddings.shape),
            self.anchor_path,
            self.fusion_path,
        )

    def _load_anchor_rows(self, path: Path) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = str(obj.get("key", idx))
                text = obj.get("text")
                if text is None:
                    text = obj.get("original_text")
                if text is None:
                    continue
                text_s = str(text).strip()
                if not text_s:
                    continue
                rows.append({"key": key, "text": text_s})
        return rows

    def _load_fusion_map(self, path: Path) -> Dict[str, List[int]]:
        mapping: Dict[str, List[int]] = {}
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                ids = obj.get("knowledge_ids")
                if key is None or ids is None:
                    continue
                mapping[str(key)] = [int(x) for x in ids]
        if not mapping:
            raise RuntimeError(f"E5 retriever fusion map is empty: {path}")
        return mapping

    def _build_fusion_tensor(self, rows: List[List[int]]) -> torch.Tensor:
        out = torch.full(
            (len(rows), self.fusion_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        for i, ids in enumerate(rows):
            cur = ids[: self.fusion_length]
            if cur:
                out[i, : len(cur)] = torch.tensor(cur, dtype=torch.long)
        return out

    @torch.no_grad()
    def _encode_texts(self, texts: List[str], batch_size: int, max_length: int) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            hidden = self._model(**encoded, return_dict=True).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = F.normalize(pooled, p=2, dim=-1)
            outputs.append(pooled)
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def retrieve_from_texts(self, texts: List[str]) -> torch.Tensor:
        query_emb = self._encode_texts(
            [f"query: {text}" for text in texts],
            batch_size=min(32, max(1, len(texts))),
            max_length=256,
        )
        scores = torch.matmul(query_emb, self.doc_embeddings.T)
        best_idx = torch.argmax(scores, dim=-1)
        return self.fusion_ids[best_idx]


def _log_section(title: str) -> None:
    logger.info("%s %s %s", "=" * 16, title, "=" * 16)


def _resolve_output(cfg: Config, output_path: Optional[str]) -> Path:
    if output_path is not None:
        return Path(output_path)
    return PROJECT_ROOT / cfg.paths.results_dir / "e7" / "e7_benchmark_compare.json"


def _load_datasets(max_samples: int) -> Dict[str, List[Dict[str, Any]]]:
    from experiments.e3.data_loading import load_arc_rows, load_medqa_rows, load_mmlu_rows

    limit = None if max_samples < 0 else max_samples
    return {
        "medqa": load_medqa_rows(limit=limit),
        "arc": load_arc_rows(limit=limit),
        "mmlu": load_mmlu_rows(limit=limit),
    }


def _build_dense_query(row: Dict[str, Any], query_mode: str) -> str:
    if query_mode == "question_only":
        return row["question"]
    if query_mode == "question_choices":
        return build_multiple_choice_prompt(row["question"], row["choices"])
    raise ValueError(f"unsupported query_mode: {query_mode}")


def eval_dense_fusion(
    model: torch.nn.Module,
    tokenizer: Any,
    retriever: Any,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    group_name: str,
    query_mode: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    from tqdm.auto import tqdm

    total = len(rows)
    correct = 0
    iterator = tqdm(
        rows,
        total=total,
        desc=f"📌 {dataset_name.upper()} / {group_name}",
        leave=True,
        disable=not show_progress,
    )
    logger.info("📌 %s | %s | start | samples=%d", dataset_name.upper(), group_name, total)
    with torch.no_grad():
        for i, row in enumerate(iterator, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            retrieval_query = _build_dense_query(row, query_mode)
            knowledge_ids = retriever.retrieve_from_texts([retrieval_query])
            context_ids = tokenizer.encode(prompt, add_special_tokens=False)
            pred = score_choices_injection(model, tokenizer, context_ids, knowledge_ids, device)
            if pred == row["label"]:
                correct += 1
            iterator.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}")
    acc = correct / total if total else 0.0
    logger.info("✅ %s | %s | done | acc=%.4f | correct=%d/%d", dataset_name.upper(), group_name, acc, correct, total)
    return {"acc": acc, "correct": correct, "total": total}


def eval_dense_rag(
    model: torch.nn.Module,
    tokenizer: Any,
    retriever: DenseRetriever,
    rows: List[Dict[str, Any]],
    device: torch.device,
    dataset_name: str,
    query_mode: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    from tqdm.auto import tqdm

    total = len(rows)
    correct = 0
    iterator = tqdm(
        rows,
        total=total,
        desc=f"📌 {dataset_name.upper()} / RAG",
        leave=True,
        disable=not show_progress,
    )
    logger.info("📌 %s | Dense RAG | start | samples=%d", dataset_name.upper(), total)
    with torch.no_grad():
        for i, row in enumerate(iterator, start=1):
            prompt = build_multiple_choice_prompt(row["question"], row["choices"])
            retrieval_query = _build_dense_query(row, query_mode)
            knowledge_ids = retriever.retrieve_from_texts([retrieval_query])
            pad_id = tokenizer.pad_token_id
            clean_ids = [t for t in knowledge_ids[0].tolist() if t != pad_id]
            compressed_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
            context = f"Context: {compressed_text}\n\n{prompt}"
            context_ids = tokenizer.encode(context, add_special_tokens=False)
            from experiments.e2.scoring import score_choices

            pred = score_choices(model, tokenizer, context_ids, device)
            if pred == row["label"]:
                correct += 1
            iterator.set_postfix(acc=f"{correct / i:.4f}", correct=f"{correct}/{i}")
    acc = correct / total if total else 0.0
    logger.info("✅ %s | Dense RAG | done | acc=%.4f | correct=%d/%d", dataset_name.upper(), acc, correct, total)
    return {"acc": acc, "correct": correct, "total": total}


def _print_report(results: Dict[str, Any]) -> None:
    _log_section("📊 E7 SUMMARY")
    logger.info("%-24s %10s %10s %10s", "group", "MEDQA", "ARC", "MMLU")
    logger.info("%s", "-" * 64)
    summary_rows = [
        ("B0_qwen3_base", "B0 Qwen3-0.6B"),
        ("TF_dense_p3_infer", "TF Dense->P3"),
        ("RAG_dense", "Dense RAG"),
    ]
    if "TF_e5_p3_infer" in results["medqa"]:
        summary_rows.insert(2, ("TF_e5_p3_infer", "TF E5->P3"))
    for key, label in summary_rows:
        logger.info(
            "%-24s %10.2f%% %10.2f%% %10.2f%%",
            label,
            results["medqa"][key]["acc"] * 100,
            results["arc"][key]["acc"] * 100,
            results["mmlu"][key]["acc"] * 100,
        )
    logger.info("%s", "-" * 64)
    logger.info("%-24s %10s %10s %10s", "delta", "MEDQA", "ARC", "MMLU")
    logger.info("%s", "-" * 64)
    delta_rows = [
        ("TF_minus_B0", "TF - B0"),
        ("RAG_minus_B0", "RAG - B0"),
    ]
    if "E5_minus_B0" in results["summary"]["medqa"]:
        delta_rows.insert(1, ("E5_minus_B0", "E5 - B0"))
    for key, label in delta_rows:
        logger.info(
            "%-24s %10.2f%% %10.2f%% %10.2f%%",
            label,
            results["summary"]["medqa"][key] * 100,
            results["summary"]["arc"][key] * 100,
            results["summary"]["mmlu"][key] * 100,
        )


def run_e7_all(
    cfg: Config,
    dense_indices: Dict[str, str],
    training_free_weights: str,
    device: str = "cuda:0",
    max_samples: int = -1,
    output_path: Optional[str] = None,
    query_mode: str = "question_only",
    e5_model_name: Optional[str] = None,
    e5_sources: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Any]:
    setup_logging()
    started = time.time()
    device_obj = torch.device(device)

    _log_section("🌍 E7 BENCHMARK COMPARE")
    logger.info(
        "E7 start | dense_indices=%s | tf=%s | device=%s | max_samples=%d | query_mode=%s",
        dense_indices,
        training_free_weights,
        device,
        max_samples,
        query_mode,
    )

    datasets = _load_datasets(max_samples=max_samples)
    logger.info(
        "Datasets loaded | MedQA=%d | ARC=%d | MMLU=%d",
        len(datasets["medqa"]),
        len(datasets["arc"]),
        len(datasets["mmlu"]),
    )

    results: Dict[str, Any] = {
        "dense_indices": dense_indices,
        "training_free_weights": training_free_weights,
        "device": device,
        "max_samples": max_samples,
        "query_mode": query_mode,
        "e5_model_name": e5_model_name,
        "e5_sources": e5_sources,
        "medqa": {},
        "arc": {},
        "mmlu": {},
    }

    _log_section("📌 B0")
    baseline_model, baseline_tokenizer = build_baseline_model(cfg, device=device)
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            rows = datasets[ds_name]
            results[ds_name]["B0_qwen3_base"] = eval_baseline(
                baseline_model,
                baseline_tokenizer,
                rows,
                device_obj,
                ds_name,
            )
    finally:
        del baseline_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    _log_section("📌 TF")
    training_free_model, training_free_tokenizer = build_injection_model(
        cfg,
        training_free_weights,
        device=device,
        log_prefix="E7LoadTF",
    )
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            retriever = DenseRetriever(cfg=cfg, index_path=dense_indices[ds_name], device=device_obj, tokenizer=training_free_tokenizer)
            results[ds_name]["TF_dense_p3_infer"] = eval_dense_fusion(
                training_free_model,
                training_free_tokenizer,
                retriever,
                datasets[ds_name],
                device_obj,
                ds_name,
                "TF",
                query_mode=query_mode,
            )
            del retriever
            if torch.cuda.is_available() and device_obj.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del training_free_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    if e5_sources is not None and e5_model_name is not None:
        _log_section("📌 TF-E5")
        e5_tokenizer = AutoTokenizer.from_pretrained(e5_model_name, use_fast=True)
        e5_model = AutoModel.from_pretrained(e5_model_name).to(device_obj).eval()
        for p in e5_model.parameters():
            p.requires_grad = False
        training_free_model, training_free_tokenizer = build_injection_model(
            cfg,
            training_free_weights,
            device=device,
            log_prefix="E7LoadTFE5",
        )
        try:
            for ds_name in ("medqa", "arc", "mmlu"):
                src = e5_sources[ds_name]
                retriever = E5KnowledgeRetriever(
                    cfg=cfg,
                    anchor_path=src["anchor"],
                    fusion_path=src["fusion"],
                    device=device_obj,
                    e5_model_name=e5_model_name,
                    pad_token_id=training_free_tokenizer.pad_token_id,
                    hf_tokenizer=e5_tokenizer,
                    hf_model=e5_model,
                )
                results[ds_name]["TF_e5_p3_infer"] = eval_dense_fusion(
                    training_free_model,
                    training_free_tokenizer,
                    retriever,
                    datasets[ds_name],
                    device_obj,
                    ds_name,
                    "TF-E5",
                    query_mode=query_mode,
                )
                del retriever
                if torch.cuda.is_available() and device_obj.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            del training_free_model
            del e5_model
            if torch.cuda.is_available() and device_obj.type == "cuda":
                torch.cuda.empty_cache()

    _log_section("📌 RAG")
    rag_model, rag_tokenizer = build_baseline_model(cfg, device=device)
    try:
        for ds_name in ("medqa", "arc", "mmlu"):
            retriever = DenseRetriever(cfg=cfg, index_path=dense_indices[ds_name], device=device_obj, tokenizer=rag_tokenizer)
            results[ds_name]["RAG_dense"] = eval_dense_rag(
                rag_model,
                rag_tokenizer,
                retriever,
                datasets[ds_name],
                device_obj,
                ds_name,
                query_mode=query_mode,
            )
            del retriever
            if torch.cuda.is_available() and device_obj.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del rag_model
        if torch.cuda.is_available() and device_obj.type == "cuda":
            torch.cuda.empty_cache()

    results["summary"] = {}
    for ds_name in ("medqa", "arc", "mmlu"):
        b0 = results[ds_name]["B0_qwen3_base"]["acc"]
        tf = results[ds_name]["TF_dense_p3_infer"]["acc"]
        rag = results[ds_name]["RAG_dense"]["acc"]
        candidates = [
            ("B0_qwen3_base", b0),
            ("TF_dense_p3_infer", tf),
            ("RAG_dense", rag),
        ]
        summary_item: Dict[str, Any] = {
            "TF_minus_B0": tf - b0,
            "RAG_minus_B0": rag - b0,
        }
        if "TF_e5_p3_infer" in results[ds_name]:
            e5 = results[ds_name]["TF_e5_p3_infer"]["acc"]
            summary_item["E5_minus_B0"] = e5 - b0
            candidates.append(("TF_e5_p3_infer", e5))
        summary_item["best_acc"] = max(x[1] for x in candidates)
        summary_item["best_group"] = max(candidates, key=lambda x: x[1])[0]
        results["summary"][ds_name] = summary_item

    results["elapsed_sec"] = time.time() - started
    _print_report(results)

    output_file = _resolve_output(cfg, output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("E7 finished | elapsed_sec=%.2f | output=%s", results["elapsed_sec"], output_file)
    return results
