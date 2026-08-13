import uuid
from unittest.mock import MagicMock, patch

import httpx

from app.models import Document_Chunks
from app.services.generation_service import (
    GenerationError,
    build_prompt,
    generate_answer,
)
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


def mock_ollama_response(text="The vacation policy allows 15 days [1]."):
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": text}
    return response


def test_build_prompt_numbers_sources_and_includes_query():
    results = [make_result("First fact."), make_result("Second fact.")]

    prompt = build_prompt("what is the policy", results)

    assert "[1] First fact." in prompt
    assert "[2] Second fact." in prompt
    assert "what is the policy" in prompt


def test_generate_answer_returns_citations_matching_sources():
    result = make_result("Employees get 15 vacation days.", distance=0.2)

    with patch("app.services.generation_service.search_chunks", return_value=[result]):
        with patch(
            "app.services.generation_service.httpx.post",
            return_value=mock_ollama_response("15 days [1]."),
        ) as mock_post:
            generation = generate_answer("how many vacation days", db=MagicMock())

    assert generation.answer == "15 days [1]."
    assert len(generation.citations) == 1
    assert generation.citations[0].number == 1
    assert generation.citations[0].chunk_id == result.chunk.id
    assert generation.citations[0].content == "Employees get 15 vacation days."
    mock_post.assert_called_once()


def test_generate_answer_sends_configured_model_and_url():
    result = make_result()

    with patch("app.services.generation_service.search_chunks", return_value=[result]):
        with patch(
            "app.services.generation_service.httpx.post",
            return_value=mock_ollama_response(),
        ) as mock_post:
            generate_answer("test query", db=MagicMock())

    args, kwargs = mock_post.call_args
    from app.config import settings

    assert args[0] == settings.ollama_url
    assert kwargs["json"]["model"] == settings.ollama_model
    assert kwargs["json"]["stream"] is False


def test_generate_answer_skips_llm_call_when_no_results():
    with patch("app.services.generation_service.search_chunks", return_value=[]):
        with patch("app.services.generation_service.httpx.post") as mock_post:
            generation = generate_answer("anything", db=MagicMock())

    assert generation.citations == []
    assert "couldn't find" in generation.answer.lower()
    mock_post.assert_not_called()


def test_generate_answer_raises_generation_error_on_http_failure():
    result = make_result()

    with patch("app.services.generation_service.search_chunks", return_value=[result]):
        with patch(
            "app.services.generation_service.httpx.post",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            try:
                generate_answer("test query", db=MagicMock())
                assert False, "expected GenerationError"
            except GenerationError as e:
                assert "Failed to reach Ollama" in str(e)
