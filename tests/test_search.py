import uuid
from unittest.mock import patch

from app.models import Document_Chunks
from app.services.retrieval_service import SearchResult


def make_result(content="Employees get 15 vacation days.", distance=0.2):
    chunk = Document_Chunks(
        id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        chunk_index=0,
        page_number=1,
        content=content,
        token_count=5,
        embedding=[0.1] * 384,
    )
    return SearchResult(chunk=chunk, distance=distance)


def test_search_returns_ranked_results(client):
    result = make_result()

    with patch("app.routes.search.search_chunks", return_value=[result]) as mock_search:
        response = client.post("/search", json={"query": "vacation policy"})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == "Employees get 15 vacation days."
    assert data[0]["distance"] == 0.2
    assert data[0]["similarity"] == 0.8
    assert data[0]["chunk_id"] == str(result.chunk.id)
    assert data[0]["document_id"] == str(result.chunk.document_id)
    mock_search.assert_called_once()


def test_search_requires_query_field(client):
    response = client.post("/search", json={})

    assert response.status_code == 422


def test_search_rejects_empty_query(client):
    response = client.post("/search", json={"query": ""})

    assert response.status_code == 422


def test_search_rejects_top_k_out_of_bounds(client):
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 51}).status_code == 422


def test_search_defaults_top_k_to_five(client):
    with patch("app.routes.search.search_chunks", return_value=[]) as mock_search:
        client.post("/search", json={"query": "vacation policy"})

    _, kwargs = mock_search.call_args
    assert kwargs["top_k"] == 5


def test_search_passes_document_id_through_to_service(client):
    document_id = uuid.uuid4()

    with patch("app.routes.search.search_chunks", return_value=[]) as mock_search:
        client.post(
            "/search",
            json={"query": "vacation policy", "document_id": str(document_id)},
        )

    _, kwargs = mock_search.call_args
    assert kwargs["document_id"] == document_id


def test_search_with_no_matches_returns_empty_list(client):
    with patch("app.routes.search.search_chunks", return_value=[]):
        response = client.post("/search", json={"query": "vacation policy"})

    assert response.status_code == 200
    assert response.json() == []
