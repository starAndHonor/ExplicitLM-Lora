from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

from config import Config, load_config  # noqa: E402
from e3_fair_compare import run_e3_all  # noqa: E402


def _cfg_to_reference_dict(cfg: Config) -> Dict[str, Any]:
    return {
        "paths": {
            "model_dir": cfg.paths.model_dir,
            "llmlingua_model_dir": cfg.paths.llmlingua_model_dir,
            "results_dir": str(PROJECT_ROOT / cfg.paths.results_dir),
        },
        "model": {
            "injection": {
                "method": cfg.model.injection_method,
                "layers": cfg.model.injection_layers,
                "encoder_depth": cfg.model.encoder_depth,
                "knowledge_adapter": False,
            }
        },
        "evaluation": {
            "medqa": {
                "knowledge_map": str(PROJECT_ROOT / cfg.eval.medqa_knowledge_map),
                "knowledge_length": cfg.model.fusion_length,
            },
            "arc": {
                "knowledge_map": str(PROJECT_ROOT / cfg.eval.arc_knowledge_map),
                "split": "test",
                "limit": None,
            },
            "mmlu": {
                "knowledge_map": str(PROJECT_ROOT / cfg.eval.mmlu_knowledge_map),
                "split": "test",
                "limit": None,
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reference-style E3 fair compare on current checkpoints")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--phase1-weights", default="checkpoints/phase1_best")
    parser.add_argument("--phase2-weights", default="checkpoints/phase2_best")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    phase1_weights = (
        Path(args.phase1_weights)
        if Path(args.phase1_weights).is_absolute()
        else (PROJECT_ROOT / args.phase1_weights)
    ).resolve()
    phase2_weights = (
        Path(args.phase2_weights)
        if Path(args.phase2_weights).is_absolute()
        else (PROJECT_ROOT / args.phase2_weights)
    ).resolve()
    tag = args.tag
    run_e3_all(
        str(phase1_weights),
        str(phase2_weights),
        _cfg_to_reference_dict(cfg),
        tag=tag,
    )


if __name__ == "__main__":
    main()
