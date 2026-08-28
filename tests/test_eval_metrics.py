"""Tests for the Step 0 scoring harness (scripts/eval_metrics.py).

These pass today -- the harness is the measuring instrument, so it gets
verified before it is trusted to judge anything else.
"""

import pytest

from scripts import eval_metrics as metrics


# --- dedupe -----------------------------------------------------------------

def test_dedupe_preserves_first_occurrence_order():
    assert metrics.dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_dedupe_handles_empty():
    assert metrics.dedupe([]) == []


# --- recall_at_k ------------------------------------------------------------

def test_recall_at_k_counts_relevant_chunks_inside_the_cutoff():
    retrieved = ["a", "b", "c", "d"]
    assert metrics.recall_at_k(retrieved, ["a", "c"], k=4) == 1.0


def test_recall_at_k_ignores_relevant_chunks_beyond_the_cutoff():
    retrieved = ["x", "y", "a", "b"]
    assert metrics.recall_at_k(retrieved, ["a", "b"], k=2) == 0.0


def test_recall_at_k_is_partial_when_some_relevant_chunks_are_found():
    retrieved = ["a", "x", "y"]
    assert metrics.recall_at_k(retrieved, ["a", "b"], k=3) == 0.5


def test_recall_at_k_is_none_when_nothing_is_relevant():
    # Unanswerable questions are not scorable by recall.
    assert metrics.recall_at_k(["a", "b"], [], k=5) is None


def test_recall_at_k_does_not_double_count_duplicate_retrievals():
    # Fusion can surface the same chunk twice; that must not inflate recall
    # or push a genuine result out of the cutoff.
    retrieved = ["a", "a", "b"]
    assert metrics.recall_at_k(retrieved, ["a", "b"], k=2) == 1.0


def test_recall_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError):
        metrics.recall_at_k(["a"], ["a"], k=0)


# --- reciprocal_rank --------------------------------------------------------

def test_reciprocal_rank_uses_the_first_relevant_position():
    assert metrics.reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


def test_reciprocal_rank_is_one_when_the_top_hit_is_relevant():
    assert metrics.reciprocal_rank(["a", "x"], ["a"]) == 1.0


def test_reciprocal_rank_is_zero_when_no_relevant_chunk_appears():
    assert metrics.reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_reciprocal_rank_respects_the_cutoff():
    # Relevant chunk sits at position 3, cutoff is 2.
    assert metrics.reciprocal_rank(["x", "y", "a"], ["a"], k=2) == 0.0


def test_reciprocal_rank_is_none_when_nothing_is_relevant():
    assert metrics.reciprocal_rank(["a"], []) is None


def test_reciprocal_rank_ignores_duplicates_when_computing_position():
    assert metrics.reciprocal_rank(["x", "x", "a"], ["a"]) == 0.5


# --- hit_rate ---------------------------------------------------------------

def test_hit_rate_is_one_when_anything_relevant_is_retrieved():
    assert metrics.hit_rate(["x", "a"], ["a"], k=5) == 1.0


def test_hit_rate_is_zero_on_a_complete_miss():
    assert metrics.hit_rate(["x", "y"], ["a"], k=5) == 0.0


def test_hit_rate_is_none_when_nothing_is_relevant():
    assert metrics.hit_rate(["x"], [], k=5) is None


# --- aggregation ------------------------------------------------------------

def test_mean_skips_none_rather_than_treating_it_as_zero():
    # A None must not drag the average down; it is "not scorable", not "0".
    assert metrics.mean([1.0, None, 0.0]) == 0.5


def test_mean_is_none_when_nothing_is_scorable():
    assert metrics.mean([None, None]) is None


def test_mean_is_none_for_an_empty_sequence():
    assert metrics.mean([]) is None


def test_p50_returns_the_median_latency():
    assert metrics.p50([10.0, 20.0, 90.0]) == 20.0


def test_p50_is_none_for_no_samples():
    assert metrics.p50([]) is None
