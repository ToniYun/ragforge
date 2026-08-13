import uuid
from unittest.mock import patch

from app.services.generation_service import Citation, GenerationResult


def make_generation_result(answer="15 days [1].", chunk_id=None, document_id=None):
    citation = Citation(
        number=1,
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        page_number=1,
        content="Employees get 15 vacation days.",
    )
    return GenerationResult(answer=answer, citations=[citation])


def test_generate_endpoint_returns_answer_and_citations(client):
    result = make_generation_result()

    with patch("app.routes.generation.generate_answer", return_value=result) as mock_gen:
        response = client.post("/generate", json={"query": "vacation policy"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "15 days [1]."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["number"] == 1
    assert data["citations"][0]["content"] == "Employees get 15 vacation days."
    mock_gen.assert_called_once()


def test_generate_endpoint_requires_query_field(client):
    response = client.post("/generate", json={})

    assert response.status_code == 422


def test_generate_endpoint_rejects_empty_query(client):
    response = client.post("/generate", json={"query": ""})

    assert response.status_code == 422


def test_generate_endpoint_rejects_top_k_out_of_bounds(client):
    assert client.post("/generate", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/generate", json={"query": "x", "top_k": 51}).status_code == 422


def test_generate_endpoint_passes_document_id_through(client):
    document_id = uuid.uuid4()
    result = make_generation_result()

    with patch("app.routes.generation.generate_answer", return_value=result) as mock_gen:
        client.post(
            "/generate",
            json={"query": "vacation policy", "document_id": str(document_id)},
        )

    _, kwargs = mock_gen.call_args
    assert kwargs["document_id"] == document_id


def test_generate_endpoint_with_no_citations(client):
    result = GenerationResult(answer="I couldn't find any relevant information.", citations=[])

    with patch("app.routes.generation.generate_answer", return_value=result):
        response = client.post("/generate", json={"query": "vacation policy"})

    assert response.status_code == 200
    assert response.json()["citations"] == []
