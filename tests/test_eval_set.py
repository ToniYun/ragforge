"""Tests for eval set loading and validation (scripts/eval_set.py).

Strict validation matters: a malformed eval set produces numbers that
look authoritative and mean nothing.
"""

import json

import pytest

from scripts import eval_set as eval_set_module
from scripts.eval_set import EvalQuestion, EvalSetError, parse_questions


def entry(**overrides):
    base = {"query": "what is the policy", "kind": "paraphrase", "relevant_chunk_ids": ["abc"]}
    base.update(overrides)
    return base


def test_parses_a_well_formed_entry():
    questions = parse_questions([entry(note="a note")])

    assert len(questions) == 1
    assert questions[0].query == "what is the policy"
    assert questions[0].kind == "paraphrase"
    assert questions[0].relevant_chunk_ids == ["abc"]
    assert questions[0].note == "a note"


def test_relevant_chunk_ids_default_to_empty():
    questions = parse_questions([{"query": "q", "kind": "exact"}])

    assert questions[0].relevant_chunk_ids == []


def test_rejects_a_non_list_payload():
    with pytest.raises(EvalSetError):
        parse_questions({"query": "q"})


def test_rejects_an_empty_eval_set():
    with pytest.raises(EvalSetError):
        parse_questions([])


def test_rejects_a_blank_query():
    with pytest.raises(EvalSetError):
        parse_questions([entry(query="   ")])


def test_rejects_an_unknown_kind():
    with pytest.raises(EvalSetError):
        parse_questions([entry(kind="tricky")])


def test_rejects_non_string_chunk_ids():
    with pytest.raises(EvalSetError):
        parse_questions([entry(relevant_chunk_ids=[123])])


def test_rejects_duplicate_chunk_ids():
    # A duplicate would silently inflate the denominator in recall@k.
    with pytest.raises(EvalSetError):
        parse_questions([entry(relevant_chunk_ids=["a", "a"])])


def test_rejects_unanswerable_questions_that_have_relevant_chunks():
    with pytest.raises(EvalSetError):
        parse_questions([entry(kind="unanswerable", relevant_chunk_ids=["a"])])


def test_error_message_identifies_the_offending_entry():
    with pytest.raises(EvalSetError, match="entry 1"):
        parse_questions([entry(), entry(kind="nonsense")])


# --- labeling state ---------------------------------------------------------

def test_question_with_chunk_ids_is_labeled():
    assert EvalQuestion(query="q", kind="exact", relevant_chunk_ids=["a"]).is_labeled


def test_question_without_chunk_ids_is_not_labeled():
    assert not EvalQuestion(query="q", kind="exact").is_labeled


def test_unanswerable_question_is_labeled_with_no_chunk_ids():
    # Having zero relevant chunks IS the label for these.
    assert EvalQuestion(query="q", kind="unanswerable").is_labeled


def test_unlabeled_returns_only_pending_questions():
    questions = [
        EvalQuestion(query="done", kind="exact", relevant_chunk_ids=["a"]),
        EvalQuestion(query="pending", kind="exact"),
        EvalQuestion(query="abstain", kind="unanswerable"),
    ]

    pending = eval_set_module.unlabeled(questions)

    assert [q.query for q in pending] == ["pending"]


# --- round trip -------------------------------------------------------------

def test_save_then_load_preserves_questions(tmp_path):
    path = tmp_path / "eval.json"
    original = [EvalQuestion(query="q", kind="multihop", relevant_chunk_ids=["a", "b"], note="n")]

    eval_set_module.save(original, path)
    loaded = eval_set_module.load(path)

    assert loaded == original


def test_load_reports_a_missing_file_clearly(tmp_path):
    with pytest.raises(EvalSetError, match="not found"):
        eval_set_module.load(tmp_path / "nope.json")


def test_load_reports_invalid_json_clearly(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(EvalSetError, match="not valid JSON"):
        eval_set_module.load(path)


# --- the committed eval set -------------------------------------------------

def test_shipped_eval_set_is_valid():
    questions = eval_set_module.load()

    assert len(questions) >= 15


def test_shipped_eval_set_covers_every_kind():
    # Losing a bucket means losing the ability to see a regression in it.
    kinds = {q.kind for q in eval_set_module.load()}

    assert kinds == eval_set_module.KINDS
