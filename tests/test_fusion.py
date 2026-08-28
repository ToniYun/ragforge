"""SPEC for Step 4: merging the two arms with Reciprocal Rank Fusion.

Build in ``app/services/retrieval_service.py``::

    RRF_K = 60

    @dataclass
    class Candidate:
        chunk: Document_Chunks
        score: float
        sources: set[str] = field(default_factory=set)

    def reciprocal_rank_fusion(
        rankings: dict[str, list],   # source name -> ranked results, best first
        k: int = RRF_K,
        top_k: int = 20,
    ) -> list[Candidate]: ...

    def hybrid_search(query, db, top_k=20, document_id=None) -> list[Candidate]: ...

The values of ``rankings`` are duck-typed on ``.chunk`` -- SearchResult and
KeywordResult both qualify, and fusion never reads their scores.

    score(chunk) = sum over lists of  1 / (k + rank_in_that_list)

with rank starting at 1. This file is pure Python: no database, and no mocks
beyond the two search functions. Make these pass first -- the logic bugs
here are subtle and very cheap to catch.
"""

import uuid
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from app.models import Document_Chunks
from tests.factories import make_chunk, make_chunks

try:
    from app.services.retrieval_service import (
        RRF_K,
        Candidate,
        hybrid_search,
        reciprocal_rank_fusion,
    )
except ImportError:  # pragma: no cover - the spec is not implemented yet
    RRF_K = Candidate = hybrid_search = reciprocal_rank_fusion = None

pytestmark = pytest.mark.skipif(
    reciprocal_rank_fusion is None,
    reason="Step 4 not implemented: add Candidate, RRF_K, reciprocal_rank_fusion() "
    "and hybrid_search() to app/services/retrieval_service.py",
)


@dataclass
class Ranked:
    """Stand-in for SearchResult / KeywordResult. Only ``.chunk`` is read."""

    chunk: Document_Chunks
    score: float = 0.0


def rank(chunks) -> list[Ranked]:
    return [Ranked(chunk=c) for c in chunks]


def rrf(*ranks: int, k: int = 60) -> float:
    """Expected fused score for a chunk appearing at these 1-based ranks."""
    return sum(1 / (k + r) for r in ranks)


# --- the constant -----------------------------------------------------------


def test_rrf_k_defaults_to_sixty():
    """60 is the constant from the original RRF paper. It damps the head of
    each list so a single number-one hit cannot dominate the merge."""
    assert RRF_K == 60


# --- core scoring -----------------------------------------------------------


def test_chunk_in_both_lists_outranks_a_chunk_topping_only_one():
    """The central property. Agreement between two independent retrievers is
    stronger evidence than one confident vote."""
    shared, vector_only = make_chunk("in both"), make_chunk("vector only")

    fused = reciprocal_rank_fusion(
        {
            "vector": rank([vector_only, shared]),  # shared is #2 here
            "keyword": rank([shared]),  # and #1 here
        }
    )

    assert fused[0].chunk is shared


def test_fused_score_matches_the_rrf_formula():
    shared = make_chunk("shared")

    fused = reciprocal_rank_fusion({"vector": rank([shared]), "keyword": rank([shared])})

    assert fused[0].score == pytest.approx(rrf(1, 1))


def test_single_list_score_matches_the_rrf_formula():
    only = make_chunk("only")

    fused = reciprocal_rank_fusion({"vector": rank([make_chunk("a"), only])})

    assert fused[0].chunk.content == "a"
    assert fused[1].score == pytest.approx(rrf(2))


def test_two_mediocre_ranks_beat_one_top_rank():
    """rrf(3, 3) is about 0.0317 and rrf(1) about 0.0164 -- appearing
    mid-list in both arms beats topping a single arm."""
    both, single = make_chunk("both"), make_chunk("single")
    filler = make_chunks(2, "filler")

    fused = reciprocal_rank_fusion(
        {
            "vector": rank([single, filler[0], both]),
            "keyword": rank([filler[1], filler[0], both]),
        }
    )

    assert fused[0].chunk is both


def test_fusion_uses_ranks_not_scores():
    """Identical orderings must fuse identically regardless of raw score
    magnitude. This is exactly why cosine distance and ts_rank_cd never need
    normalising onto a shared scale."""
    chunks = make_chunks(3)

    small = {"a": [Ranked(c, 0.001 * i) for i, c in enumerate(chunks)]}
    huge = {"a": [Ranked(c, 10_000.0 * i) for i, c in enumerate(chunks)]}

    fused_small = reciprocal_rank_fusion(small)
    fused_huge = reciprocal_rank_fusion(huge)

    assert [c.chunk.id for c in fused_small] == [c.chunk.id for c in fused_huge]
    assert [c.score for c in fused_small] == pytest.approx([c.score for c in fused_huge])


def test_larger_k_compresses_the_gap_between_ranks():
    """As k grows, rank position matters less and fusion approaches a plain
    vote count. Worth feeling before you tune anything."""
    chunks = make_chunks(2)

    tight = reciprocal_rank_fusion({"a": rank(chunks)}, k=1)
    loose = reciprocal_rank_fusion({"a": rank(chunks)}, k=1000)

    assert tight[0].score - tight[1].score > loose[0].score - loose[1].score


# --- deduplication and provenance -------------------------------------------


def test_a_chunk_appearing_in_both_lists_yields_one_candidate():
    shared = make_chunk("shared")

    fused = reciprocal_rank_fusion({"vector": rank([shared]), "keyword": rank([shared])})

    assert len(fused) == 1


def test_candidate_records_every_source_that_found_it():
    shared = make_chunk("shared")

    fused = reciprocal_rank_fusion({"vector": rank([shared]), "keyword": rank([shared])})

    assert fused[0].sources == {"vector", "keyword"}


def test_candidate_records_a_single_source_when_only_one_arm_found_it():
    only = make_chunk("only")

    fused = reciprocal_rank_fusion({"vector": rank([only]), "keyword": []})

    assert fused[0].sources == {"vector"}


def test_deduplication_is_by_chunk_id_not_object_identity():
    """The arms are separate queries, so SQLAlchemy can hand back distinct
    Python objects for the same row."""
    chunk = make_chunk("shared")
    twin = make_chunk("shared", chunk_id=chunk.id)

    fused = reciprocal_rank_fusion({"vector": rank([chunk]), "keyword": rank([twin])})

    assert len(fused) == 1
    assert fused[0].sources == {"vector", "keyword"}


# --- shape and edges --------------------------------------------------------


def test_results_are_sorted_by_descending_fused_score():
    chunks = make_chunks(4)

    fused = reciprocal_rank_fusion({"vector": rank(chunks)})

    assert [c.score for c in fused] == sorted((c.score for c in fused), reverse=True)


def test_top_k_truncates_the_fused_list():
    fused = reciprocal_rank_fusion({"vector": rank(make_chunks(10))}, top_k=3)

    assert len(fused) == 3


def test_empty_rankings_produce_no_candidates():
    assert reciprocal_rank_fusion({}) == []


def test_all_empty_lists_produce_no_candidates():
    assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []


def test_one_empty_arm_does_not_break_fusion():
    """Keyword search legitimately returns nothing when no lexeme matches."""
    chunks = make_chunks(2)

    fused = reciprocal_rank_fusion({"vector": rank(chunks), "keyword": []})

    assert len(fused) == 2


def test_fusion_returns_candidates_not_input_objects():
    fused = reciprocal_rank_fusion({"vector": rank(make_chunks(1))})

    assert isinstance(fused[0], Candidate)


def test_fusion_is_deterministic_for_equal_scores():
    """Ties must not reorder between runs, or eval numbers drift for reasons
    that have nothing to do with your changes."""
    chunks = make_chunks(5)
    rankings = {"vector": rank(chunks), "keyword": rank(list(reversed(chunks)))}

    first = [c.chunk.id for c in reciprocal_rank_fusion(rankings)]
    second = [c.chunk.id for c in reciprocal_rank_fusion(rankings)]

    assert first == second


def test_fusion_does_not_mutate_its_inputs():
    chunks = make_chunks(3)
    rankings = {"vector": rank(chunks)}
    before = list(rankings["vector"])

    reciprocal_rank_fusion(rankings)

    assert rankings["vector"] == before


# --- hybrid_search wiring ---------------------------------------------------


def test_hybrid_search_queries_both_arms_and_fuses():
    db = MagicMock()
    vector_hit, keyword_hit = make_chunk("vector"), make_chunk("keyword")

    with patch(
        "app.services.retrieval_service.search_chunks", return_value=rank([vector_hit])
    ) as mock_vector, patch(
        "app.services.retrieval_service.keyword_search", return_value=rank([keyword_hit])
    ) as mock_keyword:
        results = hybrid_search("vacation policy", db, top_k=20)

    mock_vector.assert_called_once()
    mock_keyword.assert_called_once()
    assert {c.chunk.id for c in results} == {vector_hit.id, keyword_hit.id}


def test_hybrid_search_passes_the_document_filter_to_both_arms():
    db = MagicMock()
    doc_id = uuid.uuid4()

    with patch(
        "app.services.retrieval_service.search_chunks", return_value=[]
    ) as mock_vector, patch(
        "app.services.retrieval_service.keyword_search", return_value=[]
    ) as mock_keyword:
        hybrid_search("q", db, document_id=doc_id)

    assert mock_vector.call_args.kwargs["document_id"] == doc_id
    assert mock_keyword.call_args.kwargs["document_id"] == doc_id


def test_hybrid_search_labels_sources_as_vector_and_keyword():
    db = MagicMock()
    shared = make_chunk("shared")

    with patch(
        "app.services.retrieval_service.search_chunks", return_value=rank([shared])
    ), patch(
        "app.services.retrieval_service.keyword_search", return_value=rank([shared])
    ):
        results = hybrid_search("q", db)

    assert results[0].sources == {"vector", "keyword"}
