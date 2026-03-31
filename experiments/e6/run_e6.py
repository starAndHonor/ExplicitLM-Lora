from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e6.efficiency import run_e6_all


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E6 inference efficiency benchmark")
    parser.add_argument("--config", default="config/default.yaml", help="config file path")
    parser.add_argument("--phase3-weights", required=True, help="Phase 3 checkpoint directory")
    parser.add_argument("--phase2-weights", dest="phase3_weights", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help="single device, e.g. cuda:0 or cpu")
    parser.add_argument("--n-warmup", type=int, default=10, help="warmup samples")
    parser.add_argument("--n-measure", type=int, default=200, help="measured samples")
    parser.add_argument("--output", default=None, help="output json path")
    parser.add_argument("--e3-result", default=None, help="optional E3 result json path")
    parser.add_argument("--e5-result", default=None, help="optional E5 result json path")
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


def _resolve(path_str: Optional[str]) -> Optional[str]:
    if path_str is None:
        return None
    path = Path(path_str)
    return str(path if path.is_absolute() else (PROJECT_ROOT / path).resolve())


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    run_e6_all(
        cfg=cfg,
        phase2_weights=_resolve(args.phase3_weights) or args.phase3_weights,
        device=args.device,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
        output_path=args.output,
        e3_result_path=_resolve(args.e3_result),
        e5_result_path=_resolve(args.e5_result),
    )


if __name__ == "__main__":
    main()
