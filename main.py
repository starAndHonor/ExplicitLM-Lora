"""
Explicit-LoRA 生产 CLI 入口

子命令：
    build-knowledge  Phase 0: 知识构建（LLMLingua 压缩 + 双 Bank 写入）
    train            Phase 1-3: 训练管线
    eval             评测入口
    answer           端到端 QA

用法：
    conda run -n ExplicitLLM python main.py build-knowledge --config config/default.yaml
    conda run -n ExplicitLLM python main.py train --phase 1 --config config/default.yaml
    conda run -n ExplicitLLM python main.py eval --config config/default.yaml
    conda run -n ExplicitLLM python main.py answer --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# 路径设置
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402


# ─────────────────────────────────────────────
# CLI 参数解析
# ─────────────────────────────────────────────


def _parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """
    解析 key=value 列表为嵌套 dict。

    参数：
        overrides: 形如 ["router.dim=512", "train.bf16=true"] 的列表

    返回：
        扁平 dict，key 为点路径，value 为字符串
    """
    result: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            print(f"[ERROR] 无效的 override 格式（缺少 '='）: {item}")
            sys.exit(1)
        key, value = item.split("=", 1)
        # 尝试转换类型
        if value.lower() in ("true", "false"):
            result[key] = value.lower() == "true"
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


# ─────────────────────────────────────────────
# 子命令实现（占位）
# ─────────────────────────────────────────────


def _cmd_build_knowledge(cfg: Any, args: argparse.Namespace) -> None:
    """Phase 0: 知识构建（LLMLingua 压缩 → Fusion Bank + Anchor Bank）。"""
    print("[WARN] build-knowledge: 前置模块 §1.5 (clustering) 未实现，无法执行。")


def _cmd_train(cfg: Any, args: argparse.Namespace) -> None:
    """Phase 1-3: 训练管线。"""
    print(
        f"[WARN] train --phase {args.phase}: 前置模块 §1.10 (pipeline) 未实现，无法执行。"
    )


def _cmd_eval(cfg: Any, args: argparse.Namespace) -> None:
    """评测入口。"""
    print("[WARN] eval: 前置模块 §1.10 (pipeline) 未实现，无法执行。")


def _cmd_answer(cfg: Any, args: argparse.Namespace) -> None:
    """端到端 QA。"""
    print("[WARN] answer: 前置模块 §1.10 (pipeline) 未实现，无法执行。")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────


def main() -> None:
    """Explicit-LoRA CLI 入口函数。"""
    parser = argparse.ArgumentParser(description="Explicit-LoRA CLI")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--device", default="cpu", help="设备（cpu / cuda:0 等）")
    parser.add_argument("--override", nargs="*", help="key=value 配置覆盖")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 生产命令
    subparsers.add_parser("build-knowledge", help="Phase 0: 知识构建")
    train_parser = subparsers.add_parser("train", help="Phase 1-3: 训练管线")
    train_parser.add_argument(
        "--phase", type=int, required=True, choices=[0, 1, 2, 3], help="训练阶段"
    )
    subparsers.add_parser("eval", help="评测入口")
    subparsers.add_parser("answer", help="端到端 QA")

    args = parser.parse_args()
    cli_overrides = _parse_overrides(args.override or [])
    cfg = load_config(args.config, cli_overrides=cli_overrides)

    dispatch = {
        "build-knowledge": _cmd_build_knowledge,
        "train": _cmd_train,
        "eval": _cmd_eval,
        "answer": _cmd_answer,
    }
    dispatch[args.command](cfg, args)


if __name__ == "__main__":
    main()
