"""SPEC for Step 6: the evidence pool.

Build in ``app/services/research_service.py``::

    class EvidencePool:
        def __init__(self) -> None: ...
        def add(self, query: str, candidates: list[Candidate]) -> None: ...
        def candidates(self) -> list[Candidate]: ...
        trail: list[str]        # every query issued, in order

    # Single-shot hybrid search + rerank. No loop yet:
    def retrieve_and_rerank(query, db, top_k=5, retrieve_k=20, document_id=None) -> list[Candidate]: ...

This is the step that makes Step 8 possible. Separating "accumulate
evidence" from "run a search" is what lets iteration 2 build on iteration 1
instead of replacing it -- and it is why citation numbers can stay stable
across a multi-iteration run.

Step 6 is a refactor: after it, your eval numbers must be *identical* to
Step 5. If they move, the refactor changed behaviour.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_chunk, make_chunks

try:
    from app.services.research_service import EvidencePool, retrieve_and_rerank
    from app.services.retrieval_service import Candidate
except ImportError:  # pragma: no cover - the spec is not implemented yet
    EvidencePool = retrieve_and_rerank = Candidate = None

pytestmark = pytest.mark.skipif(
    EvidencePool is None,
    reason="Step 6 not implemented: add EvidencePool and retrieve_and_rerank() "
    "to app/services/research_service.py",
)


def candidate(content="chunk", score=1.0, sources=None, chunk=None):
    return Candidate(
        chunk=chunk or make_chunk(content),
        score=score,
        sources=sources or {"vector"},
    )


# --- accumulation -----------------------------------------------------------


def test_new_pool_is_empty():
    pool = EvidencePool()

    assert pool.candidates() == []
    assert pool.trail == []


def test_add_collects_candidates():
    pool = EvidencePool()

    pool.add("q1", [candidate("a"), candidate("b")])

    assert len(pool.candidates()) == 2


def test_candidates_accumulate_across_multiple_adds():
    """The defining behaviour: a second search widens the pool, it does not
    replace it."""
    pool = EvidencePool()

    pool.add("q1", [candidate("a")])
    pool.add("q2", [candidate("b")])

    assert {c.chunk.content for c in pool.candidates()} == {"a", "b"}


def test_adding_an_empty_result_set_is_harmless():
    pool = EvidencePool()

    pool.add("q1", [candidate("a")])
    pool.add("q2", [])

    assert len(pool.candidates()) == 1


# --- deduplication ----------------------------------------------------------


def test_the_same_chunk_from_two_queries_is_stored_once():
    """Two follow-up queries very often surface the same chunk. Duplicates
    would waste prompt budget and corrupt citation numbering."""
    pool = EvidencePool()
    chunk = make_chunk("shared")

    pool.add("q1", [candidate(chunk=chunk, score=1.0)])
    pool.add("q2", [candidate(chunk=chunk, score=2.0)])

    assert len(pool.candidates()) == 1


def test_duplicate_chunk_keeps_the_higher_score():
    pool = EvidencePool()
    chunk = make_chunk("shared")

    pool.add("q1", [candidate(chunk=chunk, score=1.0)])
    pool.add("q2", [candidate(chunk=chunk, score=9.0)])

    assert pool.candidates()[0].score == pytest.approx(9.0)


def test_duplicate_chunk_does_not_downgrade_an_existing_higher_score():
    pool = EvidencePool()
    chunk = make_chunk("shared")

    pool.add("q1", [candidate(chunk=chunk, score=9.0)])
    pool.add("q2", [candidate(chunk=chunk, score=1.0)])

    assert pool.candidates()[0].score == pytest.approx(9.0)


def test_deduplication_is_by_chunk_id_not_object_identity():
    pool = EvidencePool()
    chunk = make_chunk("shared")
    twin = make_chunk("shared", chunk_id=chunk.id)

    pool.add("q1", [candidate(chunk=chunk)])
    pool.add("q2", [candidate(chunk=twin)])

    assert len(pool.candidates()) == 1


# --- the trail --------------------------------------------------------------


def test_trail_records_every_query_in_order():
    """The trail is what you show the user and what you read when debugging
    a bad answer."""
    pool = EvidencePool()

    pool.add("original question", [])
    pool.add("follow up one", [])
    pool.add("follow up two", [])

    assert pool.trail == ["original question", "follow up one", "follow up two"]


def test_trail_records_queries_that_returned_nothing():
    """A search that found nothing is a fact about the corpus worth keeping."""
    pool = EvidencePool()

    pool.add("fruitless query", [])

    assert pool.trail == ["fruitless query"]


# --- isolation --------------------------------------------------------------


def test_candidates_returns_a_copy():
    """Callers rerank and truncate this list; that must not empty the pool."""
    pool = EvidencePool()
    pool.add("q", [candidate("a"), candidate("b")])

    returned = pool.candidates()
    returned.clear()

    assert len(pool.candidates()) == 2


def test_two_pools_do_not_share_state():
    first, second = EvidencePool(), EvidencePool()

    first.add("q", [candidate("a")])

    assert second.candidates() == []
    assert second.trail == []


# --- retrieve_and_rerank ----------------------------------------------------


def test_retrieve_and_rerank_runs_hybrid_search_then_rerank():
    db = MagicMock()
    found = [candidate(c) for c in ("a", "b", "c")]

    with patch(
        "app.services.research_service.hybrid_search", return_value=found
    ) as mock_search, patch(
        "app.services.research_service.rerank", return_value=found[:2]
    ) as mock_rerank:
        results = retrieve_and_rerank("q", db, top_k=2)

    mock_search.assert_called_once()
    mock_rerank.assert_called_once()
    assert len(results) == 2


def test_retrieve_and_rerank_retrieves_wider_than_it_returns():
    """Handing the reranker only five candidates wastes it. Retrieval should
    ask for materially more than top_k."""
    db = MagicMock()

    with patch(
        "app.services.research_service.hybrid_search", return_value=[]
    ) as mock_search, patch("app.services.research_service.rerank", return_value=[]):
        retrieve_and_rerank("q", db, top_k=5)

    requested = mock_search.call_args.kwargs.get("top_k")
    assert requested is not None and requested > 5


def test_retrieve_and_rerank_reranks_against_the_original_query():
    db = MagicMock()

    with patch("app.services.research_service.hybrid_search", return_value=[candidate()]), \
         patch("app.services.research_service.rerank", return_value=[]) as mock_rerank:
        retrieve_and_rerank("how many vacation days", db)

    assert mock_rerank.call_args[0][0] == "how many vacation days"


def test_retrieve_and_rerank_passes_the_document_filter_through():
    db = MagicMock()
    doc_id = uuid.uuid4()

    with patch(
        "app.services.research_service.hybrid_search", return_value=[]
    ) as mock_search, patch("app.services.research_service.rerank", return_value=[]):
        retrieve_and_rerank("q", db, document_id=doc_id)

    assert mock_search.call_args.kwargs["document_id"] == doc_id


def test_retrieve_and_rerank_returns_empty_when_nothing_is_found():
    db = MagicMock()

    with patch("app.services.research_service.hybrid_search", return_value=[]), \
         patch("app.services.research_service.rerank", return_value=[]):
        assert retrieve_and_rerank("q", db) == []
