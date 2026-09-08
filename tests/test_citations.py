"""SPEC for Step 8: citation numbering, validation and reporting.

Build in ``app/services/generation_service.py``::

    CITATION_PATTERN = re.compile(r"\\[(\\d+)\\]")

    @dataclass
    class Citation:
        number: int
        chunk_id: uuid.UUID
        document_id: uuid.UUID
        document_filename: str      # NEW: a page number alone is not locatable
        page_number: int
        content: str

    def number_citations(candidates: list[Candidate]) -> list[Citation]: ...
    def cited_numbers(answer: str) -> set[int]: ...
    def used_citations(answer: str, citations: list[Citation]) -> list[Citation]: ...
    def unknown_citations(answer: str, citations: list[Citation]) -> set[int]: ...

    def generate_from_candidates(
        query: str, candidates: list[Candidate]
    ) -> tuple[str, list[Citation]]: ...

Two ideas carry this step:

* Numbers are assigned ONCE, after the final rerank. Numbering per
  iteration would renumber everything the model saw last time round.
* A number the model emits that does not exist is a hallucination signal.
  Detect it, log it, and never hand it back as a real source.
"""

import re
import uuid
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_chunk, make_document

try:
    from app.services.generation_service import (
        Citation,
        cited_numbers,
        generate_from_candidates,
        number_citations,
        unknown_citations,
        used_citations,
    )
    from app.services.retrieval_service import Candidate
except ImportError:  # pragma: no cover - the spec is not implemented yet
    Citation = cited_numbers = generate_from_candidates = None
    number_citations = unknown_citations = used_citations = Candidate = None

pytestmark = pytest.mark.skipif(
    number_citations is None,
    reason="Step 8 not implemented: add number_citations(), cited_numbers(), "
    "used_citations(), unknown_citations() and generate_from_candidates() "
    "to app/services/generation_service.py",
)


def candidate(content="evidence", filename="handbook.pdf", page=1):
    document = make_document(filename)
    chunk = make_chunk(content, document=document, page_number=page)
    return Candidate(chunk=chunk, score=5.0, sources={"vector"})


def citations_for(*contents):
    return number_citations([candidate(c) for c in contents])


# --- numbering --------------------------------------------------------------


def test_citations_are_numbered_from_one():
    """Numbered for humans, not indexed for computers -- these end up in
    prose as [1], [2]."""
    citations = citations_for("a", "b", "c")

    assert [c.number for c in citations] == [1, 2, 3]


def test_numbering_follows_the_reranked_order():
    citations = citations_for("best", "second", "third")

    assert [c.content for c in citations] == ["best", "second", "third"]


def test_numbering_an_empty_candidate_list_yields_no_citations():
    assert number_citations([]) == []


def test_citation_carries_the_document_filename():
    """A page number without a filename cannot be located in a corpus of a
    hundred PDFs."""
    citations = number_citations([candidate("c", filename="employee-handbook.pdf")])

    assert citations[0].document_filename == "employee-handbook.pdf"


def test_citation_carries_chunk_document_and_page_identifiers():
    source = candidate("c", page=7)

    citation = number_citations([source])[0]

    assert citation.chunk_id == source.chunk.id
    assert citation.document_id == source.chunk.document_id
    assert citation.page_number == 7


def test_citation_carries_the_chunk_content():
    citations = citations_for("the exact supporting text")

    assert citations[0].content == "the exact supporting text"


def test_numbering_is_stable_for_the_same_candidate_order():
    """Stability is what lets the loop run twice without renumbering the
    evidence the model already saw."""
    candidates = [candidate("a"), candidate("b")]

    first = [(c.number, c.chunk_id) for c in number_citations(candidates)]
    second = [(c.number, c.chunk_id) for c in number_citations(candidates)]

    assert first == second


# --- parsing what the model emitted -----------------------------------------


def test_cited_numbers_finds_a_single_reference():
    assert cited_numbers("The policy allows 15 days [1].") == {1}


def test_cited_numbers_finds_several_references():
    assert cited_numbers("Both [1] and [3] agree.") == {1, 3}


def test_cited_numbers_deduplicates_repeated_references():
    assert cited_numbers("[2] says this, and [2] also says that.") == {2}


def test_cited_numbers_is_empty_for_an_uncited_answer():
    assert cited_numbers("I could not find that information.") == set()


def test_cited_numbers_ignores_bracketed_non_numbers():
    assert cited_numbers("See [appendix] and [1].") == {1}


def test_cited_numbers_handles_adjacent_references():
    assert cited_numbers("Several sources [1][2][3] concur.") == {1, 2, 3}


# --- filtering to sources actually used -------------------------------------


def test_used_citations_returns_only_the_referenced_sources():
    """Returning all retrieved chunks shows the caller the candidate pool.
    Returning the used ones shows them the evidence."""
    citations = citations_for("a", "b", "c")

    used = used_citations("Only the first matters [1].", citations)

    assert [c.number for c in used] == [1]


def test_used_citations_preserves_numbering_order():
    citations = citations_for("a", "b", "c")

    used = used_citations("Later [3] and earlier [1].", citations)

    assert [c.number for c in used] == [1, 3]


def test_used_citations_is_empty_when_the_answer_cites_nothing():
    citations = citations_for("a", "b")

    assert used_citations("No relevant information was found.", citations) == []


def test_used_citations_does_not_invent_sources():
    citations = citations_for("a")

    used = used_citations("As shown in [9].", citations)

    assert used == []


# --- hallucination detection ------------------------------------------------


def test_unknown_citations_flags_a_number_with_no_source():
    """The model invented a source. This is the alarm worth logging."""
    citations = citations_for("a", "b")

    assert unknown_citations("According to [7], yes.", citations) == {7}


def test_unknown_citations_is_empty_when_every_reference_is_real():
    citations = citations_for("a", "b")

    assert unknown_citations("Per [1] and [2].", citations) == set()


def test_unknown_citations_ignores_zero_when_no_source_is_numbered_zero():
    citations = citations_for("a")

    assert unknown_citations("See [0].", citations) == {0}


# --- generate_from_candidates -----------------------------------------------


def ollama_reply(text):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": text}
    return response


def generate(answer_text, candidates=None, query="the question"):
    with patch(
        "app.services.generation_service.httpx.post",
        return_value=ollama_reply(answer_text),
    ) as mock_post:
        answer, citations = generate_from_candidates(
            query, candidates if candidates is not None else [candidate("a"), candidate("b")]
        )
    return answer, citations, mock_post


def test_generate_returns_the_model_answer_and_the_used_citations():
    answer, citations, _ = generate("Fifteen days [1].")

    assert answer == "Fifteen days [1]."
    assert [c.number for c in citations] == [1]


def test_generate_strips_surrounding_whitespace():
    answer, _, _ = generate("  Fifteen days [1].  ")

    assert answer == "Fifteen days [1]."


def test_generate_numbers_sources_in_the_prompt():
    _, _, mock_post = generate("[1]", candidates=[candidate("First fact."), candidate("Second fact.")])

    prompt = mock_post.call_args.kwargs["json"]["prompt"]
    assert "[1] First fact." in prompt
    assert "[2] Second fact." in prompt


def test_generate_includes_the_query_in_the_prompt():
    _, _, mock_post = generate("[1]", query="how many vacation days")

    assert "how many vacation days" in mock_post.call_args.kwargs["json"]["prompt"]


def test_generate_skips_the_model_call_when_there_are_no_candidates():
    with patch("app.services.generation_service.httpx.post") as mock_post:
        answer, citations = generate_from_candidates("q", [])

    assert citations == []
    assert answer
    mock_post.assert_not_called()


def test_generate_drops_hallucinated_citation_numbers():
    answer, citations, _ = generate(
        "Real [1] and invented [8].", candidates=[candidate("a")]
    )

    assert [c.number for c in citations] == [1]


def test_generate_logs_hallucinated_citation_numbers(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        generate("Invented [8].", candidates=[candidate("a")])

    assert "8" in caplog.text


# --- the compiled pattern ---------------------------------------------------


def test_citation_pattern_matches_bracketed_integers():
    from app.services.generation_service import CITATION_PATTERN

    assert CITATION_PATTERN.findall("[1] and [22]") == ["1", "22"]


def test_citation_pattern_is_precompiled():
    from app.services.generation_service import CITATION_PATTERN

    assert isinstance(CITATION_PATTERN, re.Pattern)


# --- uuid hygiene -----------------------------------------------------------


def test_citation_identifiers_are_uuids_not_strings():
    """The response schema declares uuid.UUID; a string here only fails at
    serialisation time."""
    citation = number_citations([candidate("a")])[0]

    assert isinstance(citation.chunk_id, uuid.UUID)
    assert isinstance(citation.document_id, uuid.UUID)
