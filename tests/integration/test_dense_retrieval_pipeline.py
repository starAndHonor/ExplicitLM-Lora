from __future__ import annotations

import importlib.util
import json
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_phase1_phase3_infer.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_phase1_phase3_infer_integration", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "decoded:" + ",".join(str(x) for x in ids)

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _FakeRetriever:
    def __init__(self, *args, **kwargs):
        self.tokenizer = _FakeTokenizer()

    def retrieve_from_texts(self, texts):
        assert len(texts) == 1
        return torch.tensor([[101, 102, 0, 0]], dtype=torch.long)


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, knowledge_ids, attention_mask, labels=None):
        logits = torch.zeros((1, input_ids.shape[1], 256), dtype=torch.float32)
        logits[0, -1, 7] = 5.0
        logits[0, -1, 9] = 4.0
        return SimpleNamespace(logits=logits)


def test_run_phase1_phase3_infer_supports_dense_retriever(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_module()

    monkeypatch.setattr(module, "load_config", lambda path: SimpleNamespace(train=SimpleNamespace(bf16=False)))
    monkeypatch.setattr(module, "DenseRetriever", _FakeRetriever)
    monkeypatch.setattr(module, "_build_modified_qwen_phase3", lambda cfg, ckpt: (_FakeModel(), _FakeTokenizer()))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--knowledge-source",
            "dense_retriever",
            "--dense-index",
            "dummy.pt",
            "--phase3-ckpt",
            "phase3_dummy",
            "--device",
            "cpu",
            "--query-mode",
            "question_only",
            "--question",
            "What is the diagnosis?",
            "--option-a",
            "Alpha",
            "--option-b",
            "Beta",
            "--option-c",
            "Gamma",
            "--option-d",
            "Delta",
            "--json",
        ],
    )

    module.main()
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert payload["knowledge_source"] == "dense_retriever"
    assert payload["query_mode"] == "question_only"
    assert payload["retrieval_query"] == "What is the diagnosis?"
    assert payload["knowledge_text"] == "decoded:101,102"
    assert payload["pred_token"] == "decoded:7"

