#!/usr/bin/env python
"""
DenseRetriever -> Phase3-FusionInference

这是 run_phase1_phase3_infer.py 的 dense 版便捷入口：
    - 默认使用 dense_retriever
    - 其余参数完全透传
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "scripts" / "run_phase1_phase3_infer.py"


def main() -> None:
    sys.argv = [
        str(TARGET),
        "--knowledge-source",
        "dense_retriever",
        *sys.argv[1:],
    ]
    runpy.run_path(str(TARGET), run_name="__main__")


if __name__ == "__main__":
    main()
