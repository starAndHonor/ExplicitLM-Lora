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


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
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
    )


if __name__ == "__main__":
    main()
