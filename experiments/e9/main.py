"""E9 CLI 入口。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from experiments.e9.run_e9 import run_e9


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E9: Sequential Write + Closed-book Probe Benchmark")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--base-index", required=True,
                        help="FineWeb 1M base dense index 路径（不含任务知识）")
    parser.add_argument("--phase3-weights", required=True, help="Phase3 checkpoint 目录")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-writes", type=int, default=100, help="每数据集顺序写入条数")
    parser.add_argument("--n-probes", type=int, default=10, help="每数据集闭卷探测条数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--query-mode",
        choices=["question_only", "question_choices"],
        default="question_only",
    )
    parser.add_argument(
        "--compression-backend",
        choices=["llmlingua", "mock_tokenize"],
        default="llmlingua",
        help="压缩后端：llmlingua（LLMLingua-2）或 mock_tokenize（直接截断，用于调试）",
    )
    parser.add_argument("--compression-rate", type=float, default=0.25, help="LLMLingua 压缩率")
    parser.add_argument("--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--override", nargs="?", action="append")
    return parser.parse_args()


def _parse_overrides(overrides: Optional[Iterable[str | List[str]]]) -> Dict[str, Any]:
    if overrides is None:
        return {}
    flat: List[str] = []
    for item in overrides:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    result: Dict[str, Any] = {}
    for item in flat:
        if "=" not in item:
            raise ValueError(f"invalid override: {item}")
        k, v = item.split("=", 1)
        if v.lower() in {"true", "false"}:
            result[k] = v.lower() == "true"
            continue
        try:
            result[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            result[k] = float(v)
            continue
        except ValueError:
            pass
        result[k] = v
    return result


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config, cli_overrides=_parse_overrides(args.override))
    result = run_e9(
        cfg=cfg,
        base_index_path=args.base_index,
        phase3_weights=args.phase3_weights,
        device=args.device,
        n_writes=args.n_writes,
        n_probes=args.n_probes,
        seed=args.seed,
        query_mode=args.query_mode,
        compression_backend=args.compression_backend,
        compression_rate=args.compression_rate,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
