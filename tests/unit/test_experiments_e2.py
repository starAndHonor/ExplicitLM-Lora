from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent.parent
REFERENCE_ROOT = PROJECT_ROOT / "Reference" / "Explicit-Lora-fusion"
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config  # noqa: E402
from experiments.e2 import arc_eval as local_arc_eval  # noqa: E402
from experiments.e2 import common as local_common  # noqa: E402
from experiments.e2 import medqa_eval as local_medqa_eval  # noqa: E402
from experiments.e2 import mmlu_eval as local_mmlu_eval  # noqa: E402
from experiments.e2.scoring import build_multiple_choice_prompt  # noqa: E402
from models import (  # noqa: E402
    AttentionInjection,
    ConcatProjection,
    GatedInjection,
    KnowledgeEncoder,
    ModifiedQwen,
    load_base_model,
)


def _import_reference_module(module_name: str):
    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    return importlib.import_module(f"evaluation.{module_name}")


def _load_reference_qwen_wrapper_class():
    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    ref_path = REFERENCE_ROOT / "models" / "qwen_wrapper.py"
    spec = importlib.util.spec_from_file_location("reference_qwen_wrapper_compare", ref_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.QwenWrapper


def test_build_multiple_choice_prompt_matches_reference_helpers() -> None:
    ref_compare_eval = _import_reference_module("compare_eval")
    ref_arc_eval = _import_reference_module("arc_eval")
    ref_mmlu_eval = _import_reference_module("mmlu_eval")

    medqa_row = {
        "sent1": "What?",
        "ending0": "a1",
        "ending1": "a2",
        "ending2": "a3",
        "ending3": "a4",
    }
    arc_row = {
        "question": "What?",
        "choices": {"text": ["a1", "a2", "a3", "a4"], "label": ["A", "B", "C", "D"]},
    }
    mmlu_row = {
        "question": "What?",
        "choices": ["a1", "a2", "a3", "a4"],
    }

    expected = "Question: What?\nA. a1\nB. a2\nC. a3\nD. a4\nAnswer:"
    assert build_multiple_choice_prompt("What?", ["a1", "a2", "a3", "a4"]) == expected
    assert ref_compare_eval._build_question_prompt(medqa_row) == expected
    assert ref_arc_eval._build_arc_prompt(arc_row) == expected
    assert ref_mmlu_eval._build_mmlu_prompt(mmlu_row) == expected


def test_arc_answer_mapping_covers_letters_and_numbers() -> None:
    assert local_arc_eval.ANSWER_KEY_TO_INDEX["A"] == 0
    assert local_arc_eval.ANSWER_KEY_TO_INDEX["4"] == 3


def test_medqa_examples_match_reference_field_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_rows = [
        {
            "sent1": "Question 1",
            "ending0": "A1",
            "ending1": "B1",
            "ending2": "C1",
            "ending3": "D1",
            "label": 2,
        }
    ]
    monkeypatch.setattr(local_medqa_eval, "load_dataset", lambda *args, **kwargs: raw_rows)

    rows = local_medqa_eval.load_medqa_examples(limit=1)

    assert rows == [
        {
            "key": "Question 1",
            "question": "Question 1",
            "choices": ["A1", "B1", "C1", "D1"],
            "label": 2,
        }
    ]


def test_arc_example_loading_matches_reference_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    ref_arc_eval = _import_reference_module("arc_eval")
    raw_rows = [
        {
            "question": "Question 1",
            "choices": {"text": ["A1", "B1", "C1", "D1"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "question": "Question 2",
            "choices": {"text": ["A2", "B2", "C2"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
        {
            "question": "Question 3",
            "choices": {"text": ["A3", "B3", "C3", "D3"], "label": ["A", "B", "C", "D"]},
            "answerKey": "4",
        },
    ]
    monkeypatch.setattr(local_arc_eval, "load_dataset", lambda *args, **kwargs: raw_rows)

    rows = local_arc_eval.load_arc_examples()

    assert rows == [
        {"key": "Question 1", "question": "Question 1", "choices": ["A1", "B1", "C1", "D1"], "label": 0},
        {"key": "Question 3", "question": "Question 3", "choices": ["A3", "B3", "C3", "D3"], "label": 3},
    ]
    assert len(raw_rows) - len(rows) == 1
    assert ref_arc_eval.answer_key_to_index("4") == rows[1]["label"]


def test_mmlu_examples_match_reference_field_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_rows = [
        {"question": "Question 1", "choices": ["A1", "B1", "C1", "D1"], "answer": 1},
        {"question": "Question 2", "choices": ["A2", "B2", "C2"], "answer": 0},
    ]
    monkeypatch.setattr(local_mmlu_eval, "load_dataset", lambda *args, **kwargs: raw_rows)

    rows = local_mmlu_eval.load_mmlu_examples()

    assert rows == [
        {"key": "Question 1", "question": "Question 1", "choices": ["A1", "B1", "C1", "D1"], "label": 1}
    ]


def test_medqa_baseline_eval_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    ref_runner = _import_reference_module("cross_domain_runner")
    raw_rows = [
        {
            "sent1": "Question 1",
            "ending0": "A1",
            "ending1": "B1",
            "ending2": "C1",
            "ending3": "D1",
            "label": 0,
        },
        {
            "sent1": "Question 2",
            "ending0": "A2",
            "ending1": "B2",
            "ending2": "C2",
            "ending3": "D2",
            "label": 2,
        },
    ]
    local_rows = [
        {"key": row["sent1"], "question": row["sent1"], "choices": [row["ending0"], row["ending1"], row["ending2"], row["ending3"]], "label": row["label"]}
        for row in raw_rows
    ]

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: list(range(len(text.splitlines()) + 1))

    monkeypatch.setattr(local_medqa_eval, "score_choices", lambda *args, **kwargs: 0)
    monkeypatch.setattr(ref_runner, "_score_choices", lambda *args, **kwargs: 0)

    local_result = local_medqa_eval.eval_medqa_baseline(mock_model, mock_tokenizer, local_rows, torch.device("cpu"), show_progress=False)
    ref_result = ref_runner._eval_medqa_baseline(mock_model, mock_tokenizer, raw_rows, torch.device("cpu"))

    assert local_result == ref_result


def test_arc_eval_matches_reference_on_same_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    ref_arc_eval = _import_reference_module("arc_eval")
    raw_rows = [
        {
            "question": "Question 1",
            "choices": {"text": ["A1", "B1", "C1", "D1"], "label": ["A", "B", "C", "D"]},
            "answerKey": "A",
        },
        {
            "question": "Question 2",
            "choices": {"text": ["A2", "B2", "C2", "D2"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "question": "Skip me",
            "choices": {"text": ["A3", "B3", "C3"], "label": ["A", "B", "C"]},
            "answerKey": "B",
        },
    ]
    local_rows = [
        {"key": "Question 1", "question": "Question 1", "choices": ["A1", "B1", "C1", "D1"], "label": 0},
        {"key": "Question 2", "question": "Question 2", "choices": ["A2", "B2", "C2", "D2"], "label": 2},
    ]

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [1, 2, 3]

    local_preds = iter([0, 1])
    ref_preds = iter([0, 1])
    monkeypatch.setattr(local_arc_eval, "score_choices", lambda *args, **kwargs: next(local_preds))
    monkeypatch.setattr(ref_arc_eval, "_score_choices", lambda *args, **kwargs: next(ref_preds))

    local_result = local_arc_eval.eval_arc(
        mock_model,
        mock_tokenizer,
        local_rows,
        torch.device("cpu"),
        is_injection=False,
        show_progress=False,
    )
    ref_result = ref_arc_eval.eval_arc(
        mock_model,
        mock_tokenizer,
        raw_rows,
        torch.device("cpu"),
        is_injection=False,
        limit=3,
    )

    assert local_result["acc"] == ref_result["acc"]
    assert local_result["correct"] == ref_result["correct"]
    assert local_result["total"] == ref_result["total"]
    assert ref_result["skipped"] == len(raw_rows) - len(local_rows)


def test_mmlu_eval_matches_reference_on_same_predictions(monkeypatch: pytest.MonkeyPatch) -> None:
    ref_mmlu_eval = _import_reference_module("mmlu_eval")
    raw_rows = [
        {"question": "Question 1", "choices": ["A1", "B1", "C1", "D1"], "answer": 0},
        {"question": "Question 2", "choices": ["A2", "B2", "C2", "D2"], "answer": 2},
    ]
    local_rows = [
        {"key": "Question 1", "question": "Question 1", "choices": ["A1", "B1", "C1", "D1"], "label": 0},
        {"key": "Question 2", "question": "Question 2", "choices": ["A2", "B2", "C2", "D2"], "label": 2},
    ]

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [1, 2, 3]

    local_preds = iter([0, 1])
    ref_preds = iter([0, 1])
    monkeypatch.setattr(local_mmlu_eval, "score_choices", lambda *args, **kwargs: next(local_preds))
    monkeypatch.setattr(ref_mmlu_eval, "_score_choices", lambda *args, **kwargs: next(ref_preds))

    local_result = local_mmlu_eval.eval_mmlu(
        mock_model,
        mock_tokenizer,
        local_rows,
        torch.device("cpu"),
        is_injection=False,
        show_progress=False,
    )
    ref_result = ref_mmlu_eval.eval_mmlu(
        mock_model,
        mock_tokenizer,
        raw_rows,
        torch.device("cpu"),
        is_injection=False,
        limit=2,
    )

    assert local_result == ref_result


def test_build_injection_model_matches_reference_checkpoint_loading_semantics_with_real_phase2_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "phase2_best"
    assert ckpt_dir.exists()
    expected_injection_state = torch.load(
        ckpt_dir / "injection_modules.pt",
        map_location="cpu",
        weights_only=True,
    )

    class DummyInjection(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim

    class DummyLoader:
        def __init__(self) -> None:
            self.loaded = None

        def load_state_dict(self, state_dict):
            self.loaded = state_dict

    class DummyModel:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.injection_modules = DummyLoader()

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

    class DummyTokenizer:
        pad_token_id = 7

    cfg = SimpleNamespace(
        train=SimpleNamespace(bf16=True),
        model=SimpleNamespace(
            encoder_depth=6,
            hidden_dim=16,
            injection_method="attention",
            injection_layers=[6, 12, 18, 24],
        ),
        paths=SimpleNamespace(model_dir="dummy-model"),
    )

    loaded_paths: list[str] = []
    captured_model: DummyModel | None = None

    monkeypatch.setattr(local_common, "load_base_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(local_common, "load_tokenizer", lambda *args, **kwargs: DummyTokenizer())
    monkeypatch.setattr(local_common, "KnowledgeEncoder", lambda *args, **kwargs: object())

    def _make_dummy_model(**kwargs):
        nonlocal captured_model
        captured_model = DummyModel(**kwargs)
        return captured_model

    monkeypatch.setattr(local_common, "ModifiedQwen", _make_dummy_model)
    monkeypatch.setattr(local_common, "AttentionInjection", DummyInjection)
    monkeypatch.setattr(local_common, "ConcatProjection", DummyInjection)
    monkeypatch.setattr(local_common, "GatedInjection", DummyInjection)

    original_torch_load = local_common.torch.load

    def _recording_torch_load(path, *args, **kwargs):
        loaded_paths.append(Path(path).name)
        return original_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(local_common.torch, "load", _recording_torch_load)

    model, tokenizer = local_common.build_injection_model(cfg, str(ckpt_dir), device="cpu")

    assert isinstance(model, DummyModel)
    assert isinstance(tokenizer, DummyTokenizer)
    assert loaded_paths == ["injection_modules.pt"]
    assert captured_model is not None
    assert captured_model.injection_modules.loaded is not None
    assert captured_model.injection_modules.loaded.keys() == expected_injection_state.keys()


def test_real_model_instance_matches_reference_style_injection_weights_phase2_best() -> None:
    model_path = PROJECT_ROOT / "Qwen3-0.6B"
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "phase2_best"
    if not model_path.exists():
        pytest.skip(f"missing base model path: {model_path}")
    if not ckpt_dir.exists():
        pytest.skip(f"missing checkpoint path: {ckpt_dir}")

    cfg = load_config(str(PROJECT_ROOT / "config" / "default.yaml"))
    device = "cpu"

    current_model, _ = local_common.build_injection_model(cfg, str(ckpt_dir), device=device)
    current_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in current_model.injection_modules.state_dict().items()
    }
    del current_model

    base_model = load_base_model(str(model_path), bf16=cfg.train.bf16)
    tokenizer = local_common.load_tokenizer(str(model_path))
    encoder = KnowledgeEncoder(
        base_model=base_model,
        encoder_depth=cfg.model.encoder_depth,
        hidden_dim=cfg.model.hidden_dim,
    )
    injection_method = cfg.model.injection_method.lower()
    if injection_method == "attention":
        factory = AttentionInjection
    elif injection_method == "concat":
        factory = ConcatProjection
    elif injection_method == "gated":
        factory = GatedInjection
    else:
        raise ValueError(f"unsupported injection_method: {cfg.model.injection_method}")

    reference_model = ModifiedQwen(
        base_model=base_model,
        knowledge_encoder=encoder,
        injection_modules=nn.ModuleList(
            [factory(cfg.model.hidden_dim) for _ in cfg.model.injection_layers]
        ),
        injection_layers=cfg.model.injection_layers,
        pad_token_id=tokenizer.pad_token_id,
    )
    reference_model.injection_modules.load_state_dict(
        torch.load(ckpt_dir / "injection_modules.pt", map_location="cpu", weights_only=True)
    )
    reference_state = {
        name: tensor.detach().cpu()
        for name, tensor in reference_model.injection_modules.state_dict().items()
    }

    assert current_state.keys() == reference_state.keys()
    for name in current_state:
        assert current_state[name].shape == reference_state[name].shape, name
        assert torch.equal(current_state[name], reference_state[name]), name


def test_knowledge_encoder_differs_from_reference_qwen_wrapper_encode_knowledge() -> None:
    model_path = PROJECT_ROOT / "Qwen3-0.6B"
    if not model_path.exists():
        pytest.skip(f"missing base model path: {model_path}")

    ids = torch.tensor(
        [
            [101, 205, 309, 401, 0, 0, 0, 0],
            [11, 22, 33, 44, 55, 66, 77, 88],
        ],
        dtype=torch.long,
    )
    mask = (ids != 0).long()

    base_model = load_base_model(str(model_path), bf16=True)
    encoder = KnowledgeEncoder(base_model=base_model, encoder_depth=6, hidden_dim=1024).eval()
    with torch.no_grad():
        local_out = encoder(ids, mask)
    del encoder, base_model

    RefQwenWrapper = _load_reference_qwen_wrapper_class()
    reference_qwen = RefQwenWrapper(model_path=str(model_path), device="cpu", freeze=True)
    with torch.no_grad():
        ref_out = reference_qwen.encode_knowledge(ids, encoder_depth=6)

    diff = (local_out.float() - ref_out.float()).abs()

    assert tuple(local_out.shape) == tuple(ref_out.shape)
    assert local_out.dtype == ref_out.dtype
    assert diff.max().item() > 1.0
    assert not torch.allclose(local_out.float(), ref_out.float(), atol=1e-2, rtol=1e-2)
