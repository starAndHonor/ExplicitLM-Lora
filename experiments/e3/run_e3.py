from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e3.fair_compare import run_e3_all


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E3 fair compare")
    parser.add_argument("--config", default="config/default.yaml", help="config file path")
    parser.add_argument("--phase1-weights", required=True, help="phase1 checkpoint directory")
    parser.add_argument("--phase2-weights", required=True, help="phase2 checkpoint directory")
    parser.add_argument("--k", type=int, default=64, help="knowledge token budget: 32/64/128/256")
    parser.add_argument("--device", default="cuda:0", help="device, e.g. cuda:0 or cpu")
    parser.add_argument("--max-samples", type=int, default=-1, help="max samples per dataset")
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


def _resolve(path_str: str) -> str:
    path = Path(path_str)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    run_e3_all(
        cfg=cfg,
        phase1_weights=_resolve(args.phase1_weights),
        phase2_weights=_resolve(args.phase2_weights),
        k=args.k,
        device=args.device,
        max_samples=args.max_samples,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
