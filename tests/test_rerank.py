"""SPEC for Step 5: cross-encoder reranking.

Build in ``app/rerank/rerank_service.py`` (with an ``__init__.py`` that
re-exports, mirroring ``app/embeddings``)::

    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    class RerankError(Exception): ...

    @lru_cache(maxsize=1)
    def _get_cross_encoder(): ...      # deferred import, same as embeddings

    def rerank(query: str, candidates: list[Candidate], top_k: int = 5) -> list[Candidate]: ...

The model is never loaded in these tests -- ``_get_cross_encoder`` is
patched throughout. A test suite that downloads 90MB of weights is a test
suite you stop running.

Note the scores are raw logits (roughly -11 to +11), not similarities, so
nothing here should assume a 0-1 range.
"""

import pytest

from tests.factories import make_chunk

try:
    from app.rerank.rerank_service import RERANK_MODEL, RerankError, rerank
    from app.services.retrieval_service import Candidate
except ImportError:  # pragma: no cover - the spec is not implemented yet
    RERANK_MODEL = RerankError = rerank = Candidate = None

pytestmark = pytest.mark.skipif(
    rerank is None,
    reason="Step 5 not implemented: add app/rerank/rerank_service.py with "
    "rerank(); it also needs Candidate from Step 4",
)


class FakeCrossEncoder:
    """Records the pairs it was asked to score and returns canned scores."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append(list(pairs))
        return self.scores[: len(pairs)]


def candidates_from(*contents) -> list:
    return [Candidate(chunk=make_chunk(c), score=0.0) for c in contents]


@pytest.fixture
def patch_encoder(monkeypatch):
    """Install a fake cross-encoder and hand it back for assertions."""

    def install(scores):
        encoder = FakeCrossEncoder(scores)
        monkeypatch.setattr(
            "app.rerank.rerank_service._get_cross_encoder", lambda: encoder
        )
        return encoder

    return install


# --- configuration ----------------------------------------------------------


def test_uses_a_cross_encoder_model():
    """A bi-encoder here would just repeat what retrieval already did. The
    point is joint attention over query and passage together."""
    assert "cross-encoder" in RERANK_MODEL


# --- ordering ---------------------------------------------------------------


def test_rerank_reorders_candidates_by_cross_encoder_score():
    """Retrieval order is discarded entirely -- that is the whole job."""
    encoder_scores = [-4.0, 8.0, 1.0]
    candidates = candidates_from("first", "second", "third")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.rerank.rerank_service._get_cross_encoder",
            lambda: FakeCrossEncoder(encoder_scores),
        )
        result = rerank("q", candidates, top_k=3)

    assert [c.chunk.content for c in result] == ["second", "third", "first"]


def test_rerank_writes_the_cross_encoder_score_onto_the_candidate(patch_encoder):
    patch_encoder([7.5])
    candidates = candidates_from("only")

    result = rerank("q", candidates, top_k=1)

    assert result[0].score == pytest.approx(7.5)


def test_rerank_scores_are_floats(patch_encoder):
    """numpy float32 leaks into JSON serialisation and comparisons."""
    import numpy as np

    patch_encoder(np.array([3.5], dtype=np.float32))
    result = rerank("q", candidates_from("only"), top_k=1)

    assert isinstance(result[0].score, float)


def test_rerank_handles_negative_logits(patch_encoder):
    """Scores are logits, not similarities. Everything can be negative and
    the ordering must still be correct."""
    patch_encoder([-9.0, -2.0])
    result = rerank("q", candidates_from("worse", "better"), top_k=2)

    assert [c.chunk.content for c in result] == ["better", "worse"]


def test_rerank_truncates_to_top_k(patch_encoder):
    patch_encoder([1.0, 2.0, 3.0, 4.0, 5.0])

    result = rerank("q", candidates_from("a", "b", "c", "d", "e"), top_k=2)

    assert len(result) == 2


def test_rerank_keeps_the_highest_scoring_candidates_when_truncating(patch_encoder):
    patch_encoder([1.0, 9.0, 2.0, 8.0])

    result = rerank("q", candidates_from("a", "b", "c", "d"), top_k=2)

    assert [c.chunk.content for c in result] == ["b", "d"]


def test_rerank_default_top_k_is_narrow():
    """Retrieve wide, rerank, return narrow -- five or so sources is what a
    generation prompt can actually use well."""
    import inspect

    assert inspect.signature(rerank).parameters["top_k"].default <= 10


# --- what gets sent to the model --------------------------------------------


def test_rerank_pairs_the_query_with_each_chunk_content(patch_encoder):
    encoder = patch_encoder([1.0, 2.0])

    rerank("how many vacation days", candidates_from("alpha", "beta"), top_k=2)

    assert encoder.calls[0] == [
        ("how many vacation days", "alpha"),
        ("how many vacation days", "beta"),
    ]


def test_rerank_scores_all_candidates_in_one_batch(patch_encoder):
    """One predict() call, not one per candidate -- batching is most of the
    speed here."""
    encoder = patch_encoder([1.0] * 20)

    rerank("q", candidates_from(*[f"c{i}" for i in range(20)]), top_k=5)

    assert len(encoder.calls) == 1
    assert len(encoder.calls[0]) == 20


# --- edges ------------------------------------------------------------------


def test_rerank_returns_empty_for_no_candidates(monkeypatch):
    """And without touching the model: an empty candidate list is the normal
    result of a query no lexeme or vector matched."""

    def explode():
        raise AssertionError("model must not be loaded for an empty list")

    monkeypatch.setattr("app.rerank.rerank_service._get_cross_encoder", explode)

    assert rerank("q", [], top_k=5) == []


def test_rerank_handles_fewer_candidates_than_top_k(patch_encoder):
    patch_encoder([1.0])

    result = rerank("q", candidates_from("only"), top_k=10)

    assert len(result) == 1


def test_rerank_returns_candidate_objects(patch_encoder):
    patch_encoder([1.0])

    result = rerank("q", candidates_from("only"), top_k=1)

    assert isinstance(result[0], Candidate)


def test_rerank_preserves_fusion_provenance(patch_encoder):
    """``sources`` survives reranking -- it is how you tell which arm found
    a chunk when an answer goes wrong."""
    patch_encoder([1.0])
    candidate = Candidate(chunk=make_chunk("c"), score=0.0, sources={"keyword"})

    result = rerank("q", [candidate], top_k=1)

    assert result[0].sources == {"keyword"}


def test_rerank_is_stable_across_repeated_calls(patch_encoder):
    patch_encoder([5.0, 5.0, 5.0])
    candidates = candidates_from("a", "b", "c")

    first = [c.chunk.id for c in rerank("q", candidates, top_k=3)]
    second = [c.chunk.id for c in rerank("q", candidates, top_k=3)]

    assert first == second


# --- model loading ----------------------------------------------------------


def test_cross_encoder_loader_is_lru_cached():
    """Same pattern as embedding_service._get_model: loading weights per
    request would dominate latency. Asserted structurally so the suite never
    downloads the model."""
    from app.rerank import rerank_service

    assert hasattr(rerank_service._get_cross_encoder, "cache_info")


def test_sentence_transformers_is_not_imported_at_module_load():
    """Deferred import, exactly like embeddings/embedding_service.py --
    otherwise every ``import app`` pays for torch."""
    import ast
    import inspect

    from app.rerank import rerank_service

    tree = ast.parse(inspect.getsource(rerank_service))
    top_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    module_names = " ".join(ast.dump(node) for node in top_level_imports)

    assert "sentence_transformers" not in module_names


def test_rerank_wraps_model_failures_in_rerank_error(monkeypatch):
    """Mirrors EmbeddingError: callers should not have to catch whatever
    torch decided to raise."""

    class Exploding:
        def predict(self, pairs, **kwargs):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        "app.rerank.rerank_service._get_cross_encoder", lambda: Exploding()
    )

    with pytest.raises(RerankError):
        rerank("q", candidates_from("a"), top_k=1)
