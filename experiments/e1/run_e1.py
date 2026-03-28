from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e1.counterfactual_eval import run_e1_sanity_check
from experiments.e2.common import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E1 sanity check experiment")
    parser.add_argument("--config", default="config/default.yaml", help="config file path")
    parser.add_argument(
        "--fusion-ckpt",
        default="checkpoints/phase2_best",
        help="fusion checkpoint directory",
    )
    parser.add_argument("--output", default=None, help="output json path")
    parser.add_argument("--max-samples", type=int, default=-1, help="max MedQA samples, -1 means full split")
    parser.add_argument(
        "--counterfactual-map",
        default="data/medqa_knowledge_counterfactual.jsonl",
        help="counterfactual knowledge map path; auto-build if missing",
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


def main() -> None:
    setup_logging()
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    fusion_ckpt = (
        str(Path(args.fusion_ckpt).resolve())
        if Path(args.fusion_ckpt).is_absolute()
        else str((PROJECT_ROOT / args.fusion_ckpt).resolve())
    )
    run_e1_sanity_check(
        cfg=cfg,
        fusion_ckpt=fusion_ckpt,
        output_path=args.output,
        max_samples=args.max_samples,
        counterfactual_map_path=args.counterfactual_map,
    )


if __name__ == "__main__":
    main()

