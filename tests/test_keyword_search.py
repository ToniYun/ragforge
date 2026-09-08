"""SPEC for Step 3: keyword search over Postgres full-text search.

Build in ``app/services/retrieval_service.py``::

    @dataclass
    class KeywordResult:
        chunk: Document_Chunks
        score: float          # ts_rank_cd value; HIGHER is better

    def keyword_search(
        query: str,
        db: Session,
        top_k: int = 20,
        document_id: uuid.UUID | None = None,
    ) -> list[KeywordResult]: ...

Why a separate result type instead of reusing ``SearchResult``? Because a
``distance`` (lower is better) and a lexical ``rank`` (higher is better)
are not the same quantity, and pretending otherwise is how you end up
sorting one of them backwards. Fusion in Step 4 only needs rank order, so
the two never have to share a scale.

These follow the mocked-session pattern from ``test_retrieval.py``: the
session is a MagicMock and the assertions are made against the compiled
SQL, so no database is required.
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import make_chunk

try:
    from app.services.retrieval_service import KeywordResult, keyword_search
except ImportError:  # pragma: no cover - the spec is not implemented yet
    KeywordResult = None
    keyword_search = None

pytestmark = pytest.mark.skipif(
    keyword_search is None,
    reason="Step 3 not implemented: add KeywordResult and keyword_search() "
           "to app/services/retrieval_service.py",
)


def run_keyword_search(query="vacation policy", top_k=20, document_id=None, rows=None):
    db = MagicMock()
    db.execute.return_value.all.return_value = rows or []

    results = keyword_search(query, db, top_k=top_k, document_id=document_id)

    statement = db.execute.call_args[0][0]
    return results, statement


def compiled(statement) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": True}))


# --- query construction -----------------------------------------------------

def test_keyword_search_does_not_embed_the_query():
    """The whole point is a lexical path that costs no model inference."""
    with patch("app.services.retrieval_service.embed_text") as mock_embed:
        run_keyword_search()

    mock_embed.assert_not_called()


def test_keyword_search_uses_websearch_to_tsquery():
    """``to_tsquery`` raises a syntax error on ordinary user input such as
    "what's the policy?". ``websearch_to_tsquery`` accepts free text and
    additionally honours "quoted phrases" and OR."""
    _, statement = run_keyword_search()

    assert "websearch_to_tsquery" in compiled(statement).lower()


def test_keyword_search_never_uses_bare_to_tsquery():
    sql = compiled(run_keyword_search(query="what's the policy?")[1]).lower()

    # websearch_to_tsquery / plainto_tsquery both contain "to_tsquery", so
    # check that no *bare* call appears.
    assert " to_tsquery(" not in sql and "(to_tsquery(" not in sql


def test_keyword_search_matches_with_the_tsvector_operator():
    sql = compiled(run_keyword_search()[1])

    assert "@@" in sql


def test_keyword_search_queries_the_indexed_column():
    """Match against the stored ``content_tsv``, not a recomputed
    ``to_tsvector(content)`` -- the latter cannot use the GIN index."""
    sql = compiled(run_keyword_search()[1]).lower()

    assert "content_tsv" in sql


def test_keyword_search_passes_the_query_text_through():
    sql = compiled(run_keyword_search(query="vacation policy")[1])

    assert "vacation policy" in sql


def test_keyword_search_uses_the_english_configuration():
    sql = compiled(run_keyword_search()[1]).lower()

    assert "english" in sql


# --- ordering and limits ----------------------------------------------------

def test_keyword_search_orders_by_rank_descending():
    """Higher ts_rank_cd is a better match -- the opposite of cosine distance."""
    sql = compiled(run_keyword_search()[1]).upper()

    assert "ORDER BY" in sql
    assert "DESC" in sql


def test_keyword_search_limit_matches_top_k():
    sql = compiled(run_keyword_search(top_k=17)[1])

    assert "LIMIT 17" in sql


def test_keyword_search_defaults_to_a_wide_top_k():
    """Retrieve wide, return narrow: fusion needs candidates to work with."""
    import inspect

    default = inspect.signature(keyword_search).parameters["top_k"].default
    assert default >= 20


# --- filtering --------------------------------------------------------------

def test_keyword_search_filters_by_document_id_when_given():
    doc_id = uuid.uuid4()

    _, statement = run_keyword_search(document_id=doc_id)

    assert "document_chunks.document_id" in str(statement)
    assert statement.compile().params["document_id_1"] == doc_id


def test_keyword_search_does_not_filter_by_document_id_when_omitted():
    _, statement = run_keyword_search(document_id=None)

    assert "document_chunks.document_id =" not in str(statement)


# --- result mapping ---------------------------------------------------------

def test_keyword_search_wraps_rows_into_keyword_results():
    chunk = make_chunk("Employees get 15 vacation days.")

    results, _ = run_keyword_search(rows=[(chunk, 0.42)])

    assert len(results) == 1
    assert isinstance(results[0], KeywordResult)
    assert results[0].chunk is chunk
    assert results[0].score == pytest.approx(0.42)


def test_keyword_search_returns_scores_as_floats():
    """Postgres may hand back a Decimal; downstream arithmetic wants a float."""
    from decimal import Decimal

    results, _ = run_keyword_search(rows=[(make_chunk(), Decimal("0.25"))])

    assert isinstance(results[0].score, float)


def test_keyword_search_preserves_database_ordering():
    first, second = make_chunk("best"), make_chunk("worse")

    results, _ = run_keyword_search(rows=[(first, 0.9), (second, 0.1)])

    assert [r.chunk.content for r in results] == ["best", "worse"]


def test_keyword_search_returns_empty_list_when_nothing_matches():
    results, _ = run_keyword_search(rows=[])

    assert results == []
