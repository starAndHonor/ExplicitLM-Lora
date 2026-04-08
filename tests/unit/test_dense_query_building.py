from __future__ import annotations

from experiments.e7.comparison import _build_dense_query


def test_build_dense_query_question_only() -> None:
    row = {
        "question": "Which law explains this motion?",
        "choices": ["First", "Second", "Third", "None"],
    }

    assert _build_dense_query(row, "question_only") == "Which law explains this motion?"


def test_build_dense_query_question_choices_contains_full_prompt() -> None:
    row = {
        "question": "Which law explains this motion?",
        "choices": ["First", "Second", "Third", "None"],
    }

    query = _build_dense_query(row, "question_choices")
    assert query.startswith("Question: Which law explains this motion?")
    assert "A. First" in query
    assert "D. None" in query
    assert query.endswith("Answer:")
