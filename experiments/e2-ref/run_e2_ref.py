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
from cross_domain_runner import run_e2_all  # noqa: E402


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
    parser = argparse.ArgumentParser(description="Run reference-style E2 evaluation on current checkpoints")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--fusion-ckpt", default="checkpoints/phase2_best")
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    fusion_ckpt = (
        Path(args.fusion_ckpt)
        if Path(args.fusion_ckpt).is_absolute()
        else (PROJECT_ROOT / args.fusion_ckpt)
    ).resolve()
    tag = args.tag or fusion_ckpt.name
    run_e2_all(str(fusion_ckpt), _cfg_to_reference_dict(cfg), tag=tag)


if __name__ == "__main__":
    main()
