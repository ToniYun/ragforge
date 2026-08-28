"""SPEC for Step 7: the sufficiency assessor.

Build in ``app/services/research_service.py``::

    @dataclass
    class Assessment:
        sufficient: bool
        missing: str
        next_queries: list[str]

    def assess_sufficiency(query: str, candidates: list[Candidate]) -> Assessment: ...

This is an LLM used as a *classifier* inside a pipeline, which means three
non-negotiables: temperature 0, a constrained output format, and a
fallback for when it misbehaves anyway. A malformed assessment must end
the loop, never raise -- degrading to single-shot RAG is a fine failure
mode, a 500 is not.

Ollama's /api/generate accepts ``"format": "json"``, which constrains
decoding so you never have to scrape prose for a yes/no.
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tests.factories import make_chunk

try:
    from app.services.research_service import Assessment, assess_sufficiency
    from app.services.retrieval_service import Candidate
except ImportError:  # pragma: no cover - the spec is not implemented yet
    Assessment = assess_sufficiency = Candidate = None

pytestmark = pytest.mark.skipif(
    assess_sufficiency is None,
    reason="Step 7 not implemented: add Assessment and assess_sufficiency() "
    "to app/services/research_service.py",
)


def candidate(content="Employees get 15 vacation days."):
    return Candidate(chunk=make_chunk(content), score=5.0, sources={"vector"})


def ollama_reply(payload):
    """Ollama wraps the model output in {"response": "<text>"}."""
    body = payload if isinstance(payload, str) else json.dumps(payload)
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"response": body}
    return response


def assess(payload, candidates=None, query="how many vacation days"):
    with patch(
        "app.services.research_service.httpx.post", return_value=ollama_reply(payload)
    ) as mock_post:
        result = assess_sufficiency(query, candidates or [candidate()])
    return result, mock_post


# --- parsing ----------------------------------------------------------------


def test_parses_a_sufficient_verdict():
    result, _ = assess({"sufficient": True, "missing": "", "next_queries": []})

    assert result.sufficient is True
    assert result.missing == ""
    assert result.next_queries == []


def test_parses_an_insufficient_verdict_with_follow_ups():
    result, _ = assess(
        {
            "sufficient": False,
            "missing": "the accrual rate for new hires",
            "next_queries": ["vacation accrual rate", "new hire benefits"],
        }
    )

    assert result.sufficient is False
    assert result.missing == "the accrual rate for new hires"
    assert result.next_queries == ["vacation accrual rate", "new hire benefits"]


def test_returns_an_assessment_instance():
    result, _ = assess({"sufficient": True})

    assert isinstance(result, Assessment)


def test_missing_fields_fall_back_to_safe_defaults():
    result, _ = assess({})

    assert isinstance(result.sufficient, bool)
    assert isinstance(result.missing, str)
    assert result.next_queries == []


# --- the request ------------------------------------------------------------


def test_requests_json_formatted_output():
    """Constrained decoding beats regexing prose for a yes/no."""
    _, mock_post = assess({"sufficient": True})

    assert mock_post.call_args.kwargs["json"]["format"] == "json"


def test_uses_temperature_zero():
    """A classifier that changes its mind between identical runs makes the
    whole loop non-reproducible."""
    _, mock_post = assess({"sufficient": True})

    options = mock_post.call_args.kwargs["json"].get("options", {})
    assert options.get("temperature") == 0


def test_does_not_stream():
    _, mock_post = assess({"sufficient": True})

    assert mock_post.call_args.kwargs["json"]["stream"] is False


def test_uses_the_configured_ollama_url_and_model():
    from app.config import settings

    _, mock_post = assess({"sufficient": True})

    assert mock_post.call_args[0][0] == settings.ollama_url
    assert mock_post.call_args.kwargs["json"]["model"] == settings.ollama_model


def test_sends_a_timeout():
    """A hung Ollama call inside a three-iteration loop hangs the request
    three times over."""
    _, mock_post = assess({"sufficient": True})

    assert mock_post.call_args.kwargs.get("timeout")


def test_prompt_contains_the_question_and_the_evidence():
    with patch(
        "app.services.research_service.httpx.post",
        return_value=ollama_reply({"sufficient": True}),
    ) as mock_post:
        assess_sufficiency("what is the accrual rate", [candidate("Accrual is 1.25 days/month.")])

    prompt = mock_post.call_args.kwargs["json"]["prompt"]
    assert "what is the accrual rate" in prompt
    assert "Accrual is 1.25 days/month." in prompt


# --- the no-evidence shortcut -----------------------------------------------


def test_no_candidates_is_insufficient_without_calling_the_model():
    """Zero evidence needs no LLM to adjudicate, and skipping the call keeps
    the empty-corpus path fast."""
    with patch("app.services.research_service.httpx.post") as mock_post:
        result = assess_sufficiency("anything", [])

    assert result.sufficient is False
    mock_post.assert_not_called()


def test_no_candidates_still_returns_an_assessment():
    with patch("app.services.research_service.httpx.post"):
        result = assess_sufficiency("anything", [])

    assert isinstance(result, Assessment)
    assert isinstance(result.next_queries, list)


# --- follow-up query hygiene ------------------------------------------------


def test_next_queries_are_capped_at_two():
    """An unbounded fan-out multiplies latency by whatever the model felt
    like proposing."""
    result, _ = assess(
        {"sufficient": False, "next_queries": ["a", "b", "c", "d", "e"]}
    )

    assert len(result.next_queries) <= 2


def test_non_string_next_queries_are_discarded():
    result, _ = assess({"sufficient": False, "next_queries": ["good", 42, None]})

    assert result.next_queries == ["good"]


def test_blank_next_queries_are_discarded():
    result, _ = assess({"sufficient": False, "next_queries": ["   ", "real query"]})

    assert result.next_queries == ["real query"]


def test_a_non_list_next_queries_field_does_not_crash():
    result, _ = assess({"sufficient": False, "next_queries": "just a string"})

    assert isinstance(result.next_queries, list)


# --- failure modes ----------------------------------------------------------


def test_malformed_json_falls_back_to_sufficient():
    """Ending the loop degrades to single-shot RAG. Continuing on a verdict
    you could not read would be guessing."""
    result, _ = assess("this is not json at all")

    assert result.sufficient is True


def test_json_of_the_wrong_shape_falls_back_to_sufficient():
    result, _ = assess('["unexpected", "list"]')

    assert result.sufficient is True


def test_http_failure_falls_back_instead_of_raising():
    with patch(
        "app.services.research_service.httpx.post",
        side_effect=httpx.ConnectError("connection refused"),
    ):
        result = assess_sufficiency("q", [candidate()])

    assert result.sufficient is True
    assert result.next_queries == []


def test_http_status_error_falls_back_instead_of_raising():
    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "500", request=MagicMock(), response=MagicMock()
    )

    with patch("app.services.research_service.httpx.post", return_value=response):
        result = assess_sufficiency("q", [candidate()])

    assert result.sufficient is True


def test_a_string_true_verdict_is_coerced_to_bool():
    """Small local models happily return the string "true" instead of true."""
    result, _ = assess({"sufficient": "true", "next_queries": []})

    assert result.sufficient is True
    assert isinstance(result.sufficient, bool)


def test_a_string_false_verdict_is_not_read_as_true():
    """The trap: ``bool("false")`` is True. A naive ``bool(payload[...])``
    turns every string verdict into "sufficient" and silently disables the
    entire loop -- it would still pass the test above."""
    result, _ = assess(
        {"sufficient": "false", "missing": "the accrual rate", "next_queries": ["accrual"]}
    )

    assert result.sufficient is False
