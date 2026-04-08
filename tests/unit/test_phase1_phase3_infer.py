from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_phase1_phase3_infer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_phase1_phase3_infer_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_retrieval_query_question_only() -> None:
    module = _load_module()
    args = SimpleNamespace(
        query_mode="question_only",
        question="What is the diagnosis?",
        option_a="A",
        option_b="B",
        option_c="C",
        option_d="D",
    )

    assert module._build_retrieval_query(args) == "What is the diagnosis?"


def test_build_retrieval_query_question_choices() -> None:
    module = _load_module()
    args = SimpleNamespace(
        query_mode="question_choices",
        question="What is the diagnosis?",
        option_a="Alpha",
        option_b="Beta",
        option_c="Gamma",
        option_d="Delta",
    )

    assert module._build_retrieval_query(args) == (
        "Question: What is the diagnosis?\n"
        "A. Alpha\n"
        "B. Beta\n"
        "C. Gamma\n"
        "D. Delta\n"
        "Answer:"
    )
