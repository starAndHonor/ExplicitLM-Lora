from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e8.bulk_update import (
    run_e8d_batch_ingest,
    run_e8d_incremental_add_delete,
)
from experiments.e8.common import prepare_medqa_full_index
from experiments.e8.delete_rollback import run_e8b_delete_rollback
from experiments.e8.sequential_edits import run_e8c_sequential_edits
from experiments.e8.upsert import run_e8a_upsert


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E8 editable memory benchmark")
    parser.add_argument("--config", default="config/default.yaml", help="config file path")
    parser.add_argument("--experiment", choices=["e8a", "e8b", "e8c", "e8d_a", "e8d_b"], default="e8a")
    parser.add_argument(
        "--memory-setting",
        choices=["controlled", "overlay_1m"],
        default="controlled",
        help="controlled uses a prebuilt task index; overlay_1m builds a temporary MedQA overlay on top of a FineWeb 1M base",
    )
    parser.add_argument("--full-index", default=None, help="full dense index path (required for controlled)")
    parser.add_argument("--base-index", default=None, help="FineWeb base dense index path (required for overlay_1m)")
    parser.add_argument(
        "--anchor-variant",
        choices=["original_text", "k256"],
        default="original_text",
        help="anchor representation used when memory_setting=overlay_1m",
    )
    parser.add_argument("--overlay-seed", type=int, default=42, help="random overlay replacement seed")
    parser.add_argument("--phase3-weights", required=True, help="phase3 checkpoint directory")
    parser.add_argument("--device", default="cuda:0", help="device")
    parser.add_argument("--n-edits", type=int, default=100, help="number of edited knowledge entries")
    parser.add_argument("--seed", type=int, default=0, help="random seed for edit set selection")
    parser.add_argument(
        "--steps",
        default="1,2,3,10,11,12,100,101,102",
        help="comma-separated edit budgets used by e8c",
    )
    parser.add_argument(
        "--locality-samples",
        type=int,
        default=200,
        help="number of locality rows evaluated in e8c/e8d",
    )
    parser.add_argument(
        "--update-batch-size",
        type=int,
        default=10,
        help="incremental add/delete batch size used by e8d_b",
    )
    parser.add_argument(
        "--query-mode",
        choices=["question_only", "question_choices"],
        default="question_only",
        help="query formulation used during evaluation",
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
    step_values = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    prepared = prepare_medqa_full_index(
        cfg=cfg,
        memory_setting=args.memory_setting,
        full_index_path=args.full_index,
        base_index_path=args.base_index,
        device=args.device,
        anchor_variant=args.anchor_variant,
        overlay_seed=args.overlay_seed,
    )
    full_index_path = prepared["full_index_path"]

    if args.experiment == "e8a":
        result = run_e8a_upsert(
            cfg=cfg,
            full_index_path=full_index_path,
            phase3_weights=args.phase3_weights,
            device=args.device,
            n_edits=args.n_edits,
            seed=args.seed,
            query_mode=args.query_mode,
            output_path=args.output,
        )
    elif args.experiment == "e8b":
        result = run_e8b_delete_rollback(
            cfg=cfg,
            full_index_path=full_index_path,
            phase3_weights=args.phase3_weights,
            device=args.device,
            n_edits=args.n_edits,
            seed=args.seed,
            query_mode=args.query_mode,
            output_path=args.output,
        )
    elif args.experiment == "e8c":
        result = run_e8c_sequential_edits(
            cfg=cfg,
            full_index_path=full_index_path,
            phase3_weights=args.phase3_weights,
            device=args.device,
            steps=step_values,
            seed=args.seed,
            query_mode=args.query_mode,
            locality_samples=args.locality_samples,
            output_path=args.output,
        )
    elif args.experiment == "e8d_a":
        result = run_e8d_batch_ingest(
            cfg=cfg,
            full_index_path=full_index_path,
            phase3_weights=args.phase3_weights,
            device=args.device,
            n_edits=args.n_edits,
            seed=args.seed,
            query_mode=args.query_mode,
            locality_samples=args.locality_samples,
            output_path=args.output,
        )
    elif args.experiment == "e8d_b":
        result = run_e8d_incremental_add_delete(
            cfg=cfg,
            full_index_path=full_index_path,
            phase3_weights=args.phase3_weights,
            device=args.device,
            n_edits=args.n_edits,
            update_batch_size=args.update_batch_size,
            seed=args.seed,
            query_mode=args.query_mode,
            locality_samples=args.locality_samples,
            output_path=args.output,
        )
    else:  # pragma: no cover
        raise ValueError(f"unsupported experiment: {args.experiment}")

    result["memory"] = prepared
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
