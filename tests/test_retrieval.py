import uuid
from unittest.mock import MagicMock, patch

from app.models import Document_Chunks
from app.services.retrieval_service import search_chunks


def make_chunk(content="hello", distance=0.1):
    chunk = Document_Chunks(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content=content,
        token_count=5,
        embedding=[0.1] * 384,
    )
    return chunk, distance


def run_search(query="what is the vacation policy", top_k=5, document_id=None, rows=None):
    db = MagicMock()
    db.execute.return_value.all.return_value = rows or []

    with patch(
        "app.services.retrieval_service.embed_text", return_value=[0.1] * 384
    ) as mock_embed:
        results = search_chunks(query, db, top_k=top_k, document_id=document_id)

    statement = db.execute.call_args[0][0]
    return results, statement, mock_embed


def test_search_embeds_the_query_text():
    _, _, mock_embed = run_search(query="how many vacation days do employees get")

    mock_embed.assert_called_once_with("how many vacation days do employees get")


def test_search_wraps_rows_into_search_results_with_similarity():
    chunk, distance = make_chunk("Employees get 15 vacation days.", distance=0.2)

    results, _, _ = run_search(rows=[(chunk, distance)])

    assert len(results) == 1
    assert results[0].chunk is chunk
    assert results[0].distance == distance
    assert results[0].similarity == 1 - distance


def test_search_returns_empty_list_when_no_rows_match():
    results, _, _ = run_search(rows=[])

    assert results == []


def test_search_limit_matches_top_k():
    _, statement, _ = run_search(top_k=9)

    sql = str(statement.compile(compile_kwargs={"literal_binds": True}))
    assert "LIMIT 9" in sql


def test_search_excludes_chunks_with_null_embedding():
    _, statement, _ = run_search()

    sql = str(statement)
    assert "document_chunks.embedding IS NOT NULL" in sql


def test_search_filters_by_document_id_when_given():
    doc_id = uuid.uuid4()

    _, statement, _ = run_search(document_id=doc_id)

    compiled = statement.compile()
    sql = str(statement)
    assert "document_chunks.document_id" in sql
    assert compiled.params["document_id_1"] == doc_id


def test_search_does_not_filter_by_document_id_when_omitted():
    _, statement, _ = run_search(document_id=None)

    sql = str(statement)
    assert "document_chunks.document_id =" not in sql


def test_search_orders_by_distance_ascending():
    _, statement, _ = run_search()

    sql = str(statement)
    assert "ORDER BY distance" in sql
