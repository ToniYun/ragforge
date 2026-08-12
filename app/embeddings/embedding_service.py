from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingError(Exception):
    """Raised when text cannot be embedded."""


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer  # deferred: slow import, heavy dep

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    try:
        model = _get_model()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()
    except Exception as e:
        raise EmbeddingError(f"Unable to embed {len(texts)} text(s): {e}")


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
