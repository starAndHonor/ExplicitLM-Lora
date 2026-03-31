from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e2.cross_domain_runner import run_e2_all


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E2 cross-domain experiment")
    parser.add_argument("--config", default="config/default.yaml", help="config file path")
    parser.add_argument("--phase2-weights", default="checkpoints/phase2_best", help="Phase 2 checkpoint directory")
    parser.add_argument("--phase3-weights", default="checkpoints/phase3_best", help="Phase 3 checkpoint directory")
    parser.add_argument("--phase2-ckpt", dest="phase2_weights", help=argparse.SUPPRESS)
    parser.add_argument("--phase3-ckpt", dest="phase3_weights", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help="device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="max samples per dataset, -1 means full split",
    )
    parser.add_argument("--output", default=None, help="output json path")
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
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    phase2_weights = (
        str(Path(args.phase2_weights).resolve())
        if Path(args.phase2_weights).is_absolute()
        else str((Path(__file__).resolve().parents[2] / args.phase2_weights).resolve())
    )
    phase3_weights = (
        str(Path(args.phase3_weights).resolve())
        if Path(args.phase3_weights).is_absolute()
        else str((Path(__file__).resolve().parents[2] / args.phase3_weights).resolve())
    )
    run_e2_all(
        cfg=cfg,
        phase2_ckpt=phase2_weights,
        phase3_ckpt=phase3_weights,
        device=args.device,
        max_samples=args.max_samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
