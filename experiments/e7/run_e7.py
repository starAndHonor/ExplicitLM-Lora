from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e7.comparison import run_e7_all


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E7 dense benchmark comparison")
    parser.add_argument("--config", default="config/default.yaml", help="config file path")
    parser.add_argument("--dense-index-medqa", required=True, help="Dense retriever index for MedQA")
    parser.add_argument("--dense-index-arc", required=True, help="Dense retriever index for ARC")
    parser.add_argument("--dense-index-mmlu", required=True, help="Dense retriever index for MMLU")
    parser.add_argument(
        "--training-free-weights",
        required=True,
        help="Fusion checkpoint directory used with dense retrieval",
    )
    parser.add_argument("--device", default="cuda:0", help="device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-samples", type=int, default=-1, help="max samples per benchmark")
    parser.add_argument("--output", default=None, help="output json path")
    parser.add_argument(
        "--query-mode",
        choices=["question_only", "question_choices"],
        default="question_only",
        help="Dense retrieval query formulation",
    )
    parser.add_argument(
        "--enable-e5-phase3",
        action="store_true",
        help="Enable extra group: E5 retriever -> Phase3 injection model",
    )
    parser.add_argument(
        "--e5-model-name",
        default="intfloat/e5-base-v2",
        help="HF model name for E5 retriever encoder",
    )
    parser.add_argument("--e5-anchor-medqa", default="", help="E5 anchor JSONL for MedQA")
    parser.add_argument("--e5-fusion-medqa", default="", help="E5 fusion JSONL for MedQA")
    parser.add_argument("--e5-anchor-arc", default="", help="E5 anchor JSONL for ARC")
    parser.add_argument("--e5-fusion-arc", default="", help="E5 fusion JSONL for ARC")
    parser.add_argument("--e5-anchor-mmlu", default="", help="E5 anchor JSONL for MMLU")
    parser.add_argument("--e5-fusion-mmlu", default="", help="E5 fusion JSONL for MMLU")
    parser.add_argument("--override", nargs="?", action="append", help="config overrides")
    return parser.parse_args()


def _parse_overrides(overrides: Optional[Iterable[str | List[str]]]) -> dict[str, Any]:
    if overrides is None:
        return {}
    flat: List[str] = []
    for item in overrides:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    result: dict[str, Any] = {}
    for item in flat:
        if "=" not in item:
            raise ValueError(f"invalid override: {item}")
        key, value = item.split("=", 1)
        if value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
            continue
        try:
            result[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            result[key] = float(value)
            continue
        except ValueError:
            pass
        result[key] = value
    return result


def _resolve(path_str: str) -> str:
    path = Path(path_str)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def _default_e5_sources() -> dict[str, dict[str, str]]:
    return {
        "medqa": {
            "anchor": str((PROJECT_ROOT / "data/medqa_knowledge_original_text.jsonl").resolve()),
            "fusion": str((PROJECT_ROOT / "data/medqa_knowledge.jsonl").resolve()),
        },
        "arc": {
            "anchor": str((PROJECT_ROOT / "data/arc_knowledge_original_text.jsonl").resolve()),
            "fusion": str((PROJECT_ROOT / "data/arc_knowledge.jsonl").resolve()),
        },
        "mmlu": {
            "anchor": str((PROJECT_ROOT / "data/mmlu_knowledge_original_text.jsonl").resolve()),
            "fusion": str((PROJECT_ROOT / "data/mmlu_knowledge.jsonl").resolve()),
        },
    }


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    e5_sources = None
    if args.enable_e5_phase3:
        defaults = _default_e5_sources()
        e5_sources = {
            "medqa": {
                "anchor": _resolve(args.e5_anchor_medqa) if args.e5_anchor_medqa else defaults["medqa"]["anchor"],
                "fusion": _resolve(args.e5_fusion_medqa) if args.e5_fusion_medqa else defaults["medqa"]["fusion"],
            },
            "arc": {
                "anchor": _resolve(args.e5_anchor_arc) if args.e5_anchor_arc else defaults["arc"]["anchor"],
                "fusion": _resolve(args.e5_fusion_arc) if args.e5_fusion_arc else defaults["arc"]["fusion"],
            },
            "mmlu": {
                "anchor": _resolve(args.e5_anchor_mmlu) if args.e5_anchor_mmlu else defaults["mmlu"]["anchor"],
                "fusion": _resolve(args.e5_fusion_mmlu) if args.e5_fusion_mmlu else defaults["mmlu"]["fusion"],
            },
        }
    run_e7_all(
        cfg=cfg,
        dense_indices={
            "medqa": _resolve(args.dense_index_medqa),
            "arc": _resolve(args.dense_index_arc),
            "mmlu": _resolve(args.dense_index_mmlu),
        },
        training_free_weights=_resolve(args.training_free_weights),
        device=args.device,
        max_samples=args.max_samples,
        output_path=args.output,
        query_mode=args.query_mode,
        e5_model_name=args.e5_model_name if args.enable_e5_phase3 else None,
        e5_sources=e5_sources,
    )


if __name__ == "__main__":
    main()
