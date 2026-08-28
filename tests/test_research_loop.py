"""SPEC for Step 9: the research loop.

Build in ``app/services/research_service.py``::

    @dataclass
    class ResearchResult:
        answer: str
        citations: list[Citation]
        search_trail: list[str]
        iterations: int
        sufficient: bool

    def research(
        query: str,
        db: Session,
        top_k: int = 5,
        max_iterations: int = 3,
        document_id: uuid.UUID | None = None,
    ) -> ResearchResult: ...

    # Retrieval-only variant so scripts/evaluate.py can score the loop:
    def research_retrieval(query, db, top_k=5, **kwargs) -> list[Candidate]: ...

This is the file to read before you write the loop. These tests are the
spec, and the six invariants below are the ones people get wrong:

1. It always terminates.
2. It never issues the same query twice.
3. The final iteration skips the assessment call (no budget left to act).
4. Reranking always targets the ORIGINAL query, never a follow-up.
5. Evidence accumulates; it is never replaced.
6. A failing assessor ends the loop rather than the request.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_chunk

try:
    from app.services.research_service import Assessment, ResearchResult, research
    from app.services.retrieval_service import Candidate
except ImportError:  # pragma: no cover - the spec is not implemented yet
    Assessment = ResearchResult = research = Candidate = None

pytestmark = pytest.mark.skipif(
    research is None,
    reason="Step 9 not implemented: add ResearchResult and research() to "
    "app/services/research_service.py",
)


def candidate(content="evidence"):
    return Candidate(chunk=make_chunk(content), score=5.0, sources={"vector"})


def sufficient():
    return Assessment(sufficient=True, missing="", next_queries=[])


def insufficient(*next_queries):
    return Assessment(
        sufficient=False,
        missing="something",
        next_queries=list(next_queries) or ["follow up"],
    )


class Harness:
    """Patches the four collaborators and records how they were called."""

    def __init__(self, assessments, search_results=None, answer="an answer [1]"):
        self.assessments = list(assessments)
        self.search_results = search_results
        self.answer = answer
        self.searched = []
        self.reranked = []
        self.assessed = 0

    def hybrid_search(self, query, db, **kwargs):
        self.searched.append(query)
        if self.search_results is not None:
            return self.search_results.pop(0) if self.search_results else []
        return [candidate(f"for {query}")]

    def rerank(self, query, candidates, top_k=5):
        self.reranked.append(query)
        return list(candidates)[:top_k]

    def assess_sufficiency(self, query, candidates):
        self.assessed += 1
        if self.assessments:
            return self.assessments.pop(0)
        return sufficient()


def run(harness, query="the question", **kwargs):
    with patch("app.services.research_service.hybrid_search", harness.hybrid_search), \
         patch("app.services.research_service.rerank", harness.rerank), \
         patch("app.services.research_service.assess_sufficiency", harness.assess_sufficiency), \
         patch(
             "app.services.research_service.generate_from_candidates",
             return_value=(harness.answer, []),
         ):
        return research(query, MagicMock(), **kwargs)


# --- invariant 1: termination -----------------------------------------------


def test_stops_after_the_first_iteration_when_evidence_is_sufficient():
    harness = Harness([sufficient()])

    result = run(harness)

    assert result.iterations == 1
    assert len(harness.searched) == 1


def test_stops_at_max_iterations_when_never_satisfied():
    """The ceiling is a hard stop, not a suggestion. Without it a stubborn
    assessor loops until the request times out."""
    harness = Harness([insufficient("a"), insufficient("b"), insufficient("c")])

    result = run(harness, max_iterations=3)

    assert result.iterations == 3


def test_max_iterations_of_one_behaves_like_single_shot_rag():
    """The sane production default: opt into the loop per request."""
    harness = Harness([insufficient("a")])

    result = run(harness, max_iterations=1)

    assert result.iterations == 1
    assert len(harness.searched) == 1


def test_stops_when_the_assessor_offers_no_follow_up_queries():
    """Insufficient but with nothing to search for is a dead end, not a
    reason to search the same thing again."""
    harness = Harness([Assessment(sufficient=False, missing="unclear", next_queries=[])])

    result = run(harness, max_iterations=3)

    assert result.iterations == 1


# --- invariant 2: no repeated searches --------------------------------------


def test_never_issues_the_same_query_twice():
    """Models love re-proposing a rephrasing of the original question."""
    harness = Harness([insufficient("the question"), insufficient("something new")])

    run(harness, query="the question", max_iterations=3)

    assert len(harness.searched) == len(set(harness.searched))


def test_repeated_queries_are_matched_case_and_whitespace_insensitively():
    harness = Harness([insufficient("  THE Question  "), sufficient()])

    run(harness, query="the question", max_iterations=3)

    assert len(harness.searched) == 1


def test_a_wholly_duplicate_follow_up_set_ends_the_loop():
    """If every proposal was already searched there is nothing left to do."""
    harness = Harness([insufficient("the question"), insufficient("the question")])

    result = run(harness, query="the question", max_iterations=3)

    assert len(harness.searched) == 1
    assert result.iterations == 1


# --- invariant 3: no wasted assessment on the last pass ---------------------


def test_skips_the_assessment_call_on_the_final_iteration():
    """There is no budget left to act on the verdict, so the call is pure
    latency."""
    harness = Harness([insufficient("second"), insufficient("third")])

    run(harness, max_iterations=3)

    assert harness.assessed == 2


def test_single_iteration_run_never_assesses():
    harness = Harness([])

    run(harness, max_iterations=1)

    assert harness.assessed == 0


# --- invariant 4: rerank against the original query -------------------------


def test_reranks_against_the_original_query_not_the_follow_ups():
    """Follow-ups exist to widen recall. Relevance is always judged against
    what the user actually asked."""
    harness = Harness([insufficient("a narrower follow up"), sufficient()])

    run(harness, query="the original question", max_iterations=3)

    assert set(harness.reranked) == {"the original question"}


# --- invariant 5: evidence accumulates --------------------------------------


def test_evidence_from_earlier_iterations_survives_into_the_final_answer():
    """Iteration 2 must widen the pool, not replace it."""
    first, second = candidate("from iteration one"), candidate("from iteration two")
    harness = Harness([insufficient("follow up"), sufficient()], search_results=[[first], [second]])

    with patch("app.services.research_service.hybrid_search", harness.hybrid_search), \
         patch("app.services.research_service.rerank", harness.rerank), \
         patch("app.services.research_service.assess_sufficiency", harness.assess_sufficiency), \
         patch(
             "app.services.research_service.generate_from_candidates",
             return_value=("answer", []),
         ) as mock_generate:
        research("q", MagicMock(), max_iterations=3)

    final_candidates = mock_generate.call_args[0][1]
    contents = {c.chunk.content for c in final_candidates}
    assert contents == {"from iteration one", "from iteration two"}


def test_search_trail_records_every_query_issued():
    harness = Harness([insufficient("follow up"), sufficient()])

    result = run(harness, query="original", max_iterations=3)

    assert result.search_trail == ["original", "follow up"]


def test_search_trail_starts_with_the_user_query():
    harness = Harness([sufficient()])

    result = run(harness, query="original")

    assert result.search_trail[0] == "original"


# --- invariant 6: a failing assessor does not fail the request --------------


def test_an_assessor_that_raises_ends_the_loop_and_still_answers():
    def exploding(query, candidates):
        raise RuntimeError("ollama fell over")

    harness = Harness([])

    with patch("app.services.research_service.hybrid_search", harness.hybrid_search), \
         patch("app.services.research_service.rerank", harness.rerank), \
         patch("app.services.research_service.assess_sufficiency", exploding), \
         patch(
             "app.services.research_service.generate_from_candidates",
             return_value=("answer", []),
         ):
        result = research("q", MagicMock(), max_iterations=3)

    assert result.answer == "answer"
    assert result.iterations == 1


# --- reported outcome -------------------------------------------------------


def test_reports_sufficient_true_when_it_stopped_satisfied():
    harness = Harness([sufficient()])

    assert run(harness).sufficient is True


def test_reports_sufficient_false_when_it_ran_out_of_budget():
    """Honest signal: the caller learns this was a best effort under a
    budget, not a confident answer."""
    harness = Harness([insufficient("a"), insufficient("b")])

    assert run(harness, max_iterations=3).sufficient is False


def test_returns_a_research_result():
    harness = Harness([sufficient()])

    assert isinstance(run(harness), ResearchResult)


# --- empty corpus -----------------------------------------------------------


def test_returns_an_honest_answer_when_nothing_is_ever_retrieved():
    harness = Harness([], search_results=[[], [], []])

    with patch("app.services.research_service.hybrid_search", harness.hybrid_search), \
         patch("app.services.research_service.rerank", harness.rerank), \
         patch(
             "app.services.research_service.assess_sufficiency",
             lambda q, c: Assessment(False, "no evidence", []),
         ), \
         patch("app.services.research_service.generate_from_candidates") as mock_generate:
        result = research("q", MagicMock(), max_iterations=3)

    assert result.citations == []
    mock_generate.assert_not_called()


def test_passes_the_document_filter_into_every_search():
    doc_id = uuid.uuid4()
    seen = []

    def recording_search(query, db, **kwargs):
        seen.append(kwargs.get("document_id"))
        return [candidate()]

    harness = Harness([insufficient("follow up"), sufficient()])

    with patch("app.services.research_service.hybrid_search", recording_search), \
         patch("app.services.research_service.rerank", harness.rerank), \
         patch("app.services.research_service.assess_sufficiency", harness.assess_sufficiency), \
         patch(
             "app.services.research_service.generate_from_candidates",
             return_value=("answer", []),
         ):
        research("q", MagicMock(), max_iterations=3, document_id=doc_id)

    assert seen == [doc_id, doc_id]
