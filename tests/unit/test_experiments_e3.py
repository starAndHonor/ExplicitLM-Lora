from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch

from experiments.e3 import data_loading as current_data_loading
from experiments.e3 import evaluator as current_evaluator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
E3_REF_DIR = PROJECT_ROOT / "experiments" / "e3-ref"


def _load_e3_ref_module(name: str):
    if str(E3_REF_DIR) not in sys.path:
        sys.path.insert(0, str(E3_REF_DIR))
    module_name = f"e3_ref_{name}"
    spec = importlib.util.spec_from_file_location(module_name, E3_REF_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def e3_ref():
    return {
        "e3_fair_compare": _load_e3_ref_module("e3_fair_compare"),
    }


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 2
        self._next_text_id = 1000
        self._text_to_id: Dict[str, int] = {}
        self._id_to_text: Dict[int, str] = {}
        self._knowledge_decode = {
            (11, 12): "compressed med knowledge",
            (21, 22): "compressed arc knowledge",
            (31, 32): "compressed mmlu knowledge",
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if text not in self._text_to_id:
            token_id = self._next_text_id
            self._next_text_id += 1
            self._text_to_id[text] = token_id
            self._id_to_text[token_id] = text
        return [self._text_to_id[text]]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._knowledge_decode.get(tuple(token_ids), "decoded knowledge")

    def text_from_ids(self, token_ids: List[int]) -> str:
        return self._id_to_text[token_ids[0]]


def _predict_from_context(tokenizer: FakeTokenizer, context_ids: List[int]) -> int:
    text = tokenizer.text_from_ids(context_ids)
    if "A patient has chest pain." in text:
        return 0
    if "What force pulls objects toward Earth?" in text:
        return 1
    if "What is the powerhouse of the cell?" in text:
        return 1
    raise AssertionError(f"unexpected context: {text}")


def _predict_from_injection(
    tokenizer: FakeTokenizer,
    context_ids: List[int],
    knowledge_ids: torch.LongTensor,
) -> int:
    return _predict_from_context(tokenizer, context_ids)


def _raw_rows() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "medqa": [
            {
                "sent1": "A patient has chest pain.",
                "ending0": "Myocardial infarction",
                "ending1": "Pulmonary embolism",
                "ending2": "Pneumothorax",
                "ending3": "Aortic dissection",
                "label": 0,
            }
        ],
        "arc": [
            {
                "question": "What force pulls objects toward Earth?",
                "choices": {
                    "text": ["friction", "gravity", "magnetism", "electricity"],
                    "label": ["A", "B", "C", "D"],
                },
                "answerKey": "B",
            },
            {
                "question": "Which state of matter has no fixed shape?",
                "choices": {
                    "text": ["solid", "liquid", "gas"],
                    "label": ["A", "B", "C"],
                },
                "answerKey": "C",
            },
        ],
        "mmlu": [
            {
                "question": "What is the powerhouse of the cell?",
                "choices": ["Nucleus", "Mitochondria", "Ribosome", "Golgi apparatus"],
                "answer": 1,
                "subject": "biology",
            }
        ],
    }


def _current_medqa_disk_rows() -> List[Dict[str, Any]]:
    return [
        {
            "question": "A patient has chest pain.",
            "options": {
                "A": "Myocardial infarction",
                "B": "Pulmonary embolism",
                "C": "Pneumothorax",
                "D": "Aortic dissection",
            },
            "answer_idx": "A",
        }
    ]


def _knowledge_maps() -> Dict[str, Dict[str, List[int]]]:
    return {
        "medqa": {"A patient has chest pain.": [11, 12, 0, 0]},
        "arc": {"What force pulls objects toward Earth?": [21, 22, 0, 0]},
        "mmlu": {"What is the powerhouse of the cell?": [31, 32, 0, 0]},
    }


def test_data_loading_matches_reference_helpers(monkeypatch: pytest.MonkeyPatch, e3_ref) -> None:
    ref = e3_ref["e3_fair_compare"]
    raw = _raw_rows()

    monkeypatch.setattr(current_data_loading, "load_from_disk", lambda path: {"test": _current_medqa_disk_rows()})

    def _fake_load_dataset(name: str, *args: Any, **kwargs: Any):
        if name == "allenai/ai2_arc":
            return raw["arc"]
        if name == "cais/mmlu":
            return raw["mmlu"]
        raise AssertionError(name)

    monkeypatch.setattr(current_data_loading, "load_dataset", _fake_load_dataset)

    medqa_rows = current_data_loading.load_medqa_rows()
    arc_rows = current_data_loading.load_arc_rows()
    mmlu_rows = current_data_loading.load_mmlu_rows()

    assert medqa_rows == [
        {
            "question": raw["medqa"][0]["sent1"],
            "choices": [
                raw["medqa"][0]["ending0"],
                raw["medqa"][0]["ending1"],
                raw["medqa"][0]["ending2"],
                raw["medqa"][0]["ending3"],
            ],
            "label": ref._get_label(raw["medqa"][0], "medqa"),
            "key": ref._get_question_key(raw["medqa"][0], "medqa"),
            "original_text": ref._get_original_text(raw["medqa"][0], "medqa"),
        }
    ]
    assert arc_rows == [
        {
            "question": raw["arc"][0]["question"],
            "choices": raw["arc"][0]["choices"]["text"],
            "label": ref._get_label(raw["arc"][0], "arc"),
            "key": ref._get_question_key(raw["arc"][0], "arc"),
            "original_text": ref._get_original_text(raw["arc"][0], "arc"),
        }
    ]
    assert mmlu_rows == [
        {
            "question": raw["mmlu"][0]["question"],
            "choices": raw["mmlu"][0]["choices"],
            "label": ref._get_label(raw["mmlu"][0], "mmlu"),
            "key": ref._get_question_key(raw["mmlu"][0], "mmlu"),
            "original_text": ref._get_original_text(raw["mmlu"][0], "mmlu"),
        }
    ]


@pytest.mark.parametrize("dataset_name", ["medqa", "arc", "mmlu"])
def test_group_results_match_reference_on_small_samples(
    monkeypatch: pytest.MonkeyPatch,
    e3_ref,
    dataset_name: str,
) -> None:
    ref = e3_ref["e3_fair_compare"]
    tokenizer = FakeTokenizer()
    model = object()
    device = torch.device("cpu")
    raw = _raw_rows()[dataset_name]
    km = _knowledge_maps()[dataset_name]

    if dataset_name == "medqa":
        current_rows = [
            {
                "question": raw[0]["sent1"],
                "choices": [raw[0][f"ending{i}"] for i in range(4)],
                "label": raw[0]["label"],
                "key": raw[0]["sent1"][:200].strip(),
                "original_text": ref._get_original_text(raw[0], "medqa"),
            }
        ]
    elif dataset_name == "arc":
        current_rows = [
            {
                "question": raw[0]["question"],
                "choices": list(raw[0]["choices"]["text"]),
                "label": 1,
                "key": raw[0]["question"][:200].strip(),
                "original_text": ref._get_original_text(raw[0], "arc"),
            }
        ]
    else:
        current_rows = [
            {
                "question": raw[0]["question"],
                "choices": list(raw[0]["choices"]),
                "label": raw[0]["answer"],
                "key": raw[0]["question"][:200].strip(),
                "original_text": ref._get_original_text(raw[0], "mmlu"),
            }
        ]

    monkeypatch.setattr(
        current_evaluator,
        "score_choices",
        lambda _model, tok, context_ids, _device: _predict_from_context(tok, context_ids),
    )
    monkeypatch.setattr(
        current_evaluator,
        "score_choices_injection",
        lambda _model, tok, context_ids, knowledge_ids, _device: _predict_from_injection(tok, context_ids, knowledge_ids),
    )
    monkeypatch.setattr(
        ref,
        "_score_choices",
        lambda _model, tok, context_ids, _device: _predict_from_context(tok, context_ids),
    )
    monkeypatch.setattr(
        ref,
        "_score_choices_injection",
        lambda _model, tok, context_ids, knowledge_ids, _device: _predict_from_injection(tok, context_ids, knowledge_ids),
    )

    current_g0 = current_evaluator.eval_baseline(model, tokenizer, current_rows, device, dataset_name, show_progress=False)
    ref_g0 = ref._eval_baseline(model, tokenizer, raw, device, dataset_name)
    assert current_g0["acc"] == ref_g0["acc"]
    assert current_g0["correct"] == ref_g0["correct"]
    assert current_g0["total"] == ref_g0["total"]

    current_g1 = current_evaluator.eval_rag_compressed(
        model,
        tokenizer,
        current_rows,
        device,
        dataset_name,
        knowledge_map=km,
        show_progress=False,
    )
    ref_g1 = ref._eval_rag_compressed(model, tokenizer, raw, km, device, dataset_name)
    assert current_g1["acc"] == ref_g1["acc"]
    assert current_g1["correct"] == ref_g1["correct"]
    assert current_g1["total"] == ref_g1["total"]

    current_g2 = current_evaluator.eval_fusion(
        model,
        tokenizer,
        current_rows,
        device,
        dataset_name,
        knowledge_map=km,
        group_name="G2",
        knowledge_length=4,
        show_progress=False,
    )
    ref_g2 = ref._eval_fusion(
        model,
        tokenizer,
        raw,
        km,
        device,
        dataset_name,
        knowledge_length=4,
        group_name="G2",
    )
    assert current_g2["acc"] == ref_g2["acc"]
    assert current_g2["correct"] == ref_g2["correct"]
    assert current_g2["total"] == ref_g2["total"]

    current_g3 = current_evaluator.eval_fusion(
        model,
        tokenizer,
        current_rows,
        device,
        dataset_name,
        knowledge_map=km,
        group_name="G3",
        knowledge_length=4,
        show_progress=False,
    )
    ref_g3 = ref._eval_fusion(
        model,
        tokenizer,
        raw,
        km,
        device,
        dataset_name,
        knowledge_length=4,
        group_name="G3",
    )
    assert current_g3["acc"] == ref_g3["acc"]
    assert current_g3["correct"] == ref_g3["correct"]
    assert current_g3["total"] == ref_g3["total"]

    current_g4 = current_evaluator.eval_rag_original(
        model,
        tokenizer,
        current_rows,
        device,
        dataset_name,
        show_progress=False,
    )
    ref_g4 = ref._eval_rag_original(model, tokenizer, raw, device, dataset_name)
    assert current_g4["acc"] == ref_g4["acc"]
    assert current_g4["correct"] == ref_g4["correct"]
    assert current_g4["total"] == ref_g4["total"]

    if dataset_name == "arc":
        assert ref_g0["skipped"] == 1
        assert ref_g1["skipped"] == 1
        assert ref_g2["skipped"] == 1
        assert ref_g3["skipped"] == 1
        assert ref_g4["skipped"] == 1
