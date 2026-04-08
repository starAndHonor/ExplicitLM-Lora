from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "test_scheme2_medqa.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("test_scheme2_medqa_integration", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "knowledge:" + ",".join(str(x) for x in ids)

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class _FakeRetriever:
    def __init__(self, *args, **kwargs):
        self.tokenizer = _FakeTokenizer()

    def retrieve_from_texts(self, texts):
        assert len(texts) == 1
        return torch.tensor([[201, 202, 0, 0]], dtype=torch.long)


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def test_scheme2_medqa_script_prints_expected_summary(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_module()

    rows = [
        {
            "question": "Question 1",
            "choices": ["A1", "B1", "C1", "D1"],
            "label": 1,
        },
        {
            "question": "Question 2",
            "choices": ["A2", "B2", "C2", "D2"],
            "label": 2,
        },
    ]

    monkeypatch.setattr(module, "load_config", lambda path: SimpleNamespace(train=SimpleNamespace(bf16=False)))
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, "load_medqa_rows", lambda limit: rows[:limit])
    monkeypatch.setattr(module, "Phase1Retriever", _FakeRetriever)
    monkeypatch.setattr(module, "_build_modified_qwen_phase3", lambda cfg, ckpt: (_FakeModel(), _FakeTokenizer()))
    monkeypatch.setattr(module, "score_choices_injection", lambda *args, **kwargs: 1)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--limit",
            "2",
            "--phase1-ckpt",
            "dummy_phase1",
            "--phase3-ckpt",
            "dummy_phase3",
        ],
    )

    module.main()
    out = capsys.readouterr().out
    summary = json.loads(out[out.rfind("{"):])

    assert summary["n"] == 2
    assert summary["acc"] == 0.5
