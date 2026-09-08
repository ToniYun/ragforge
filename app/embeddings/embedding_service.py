from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# all-MiniLM-L6-v2 truncates input at 256 wordpieces (max_seq_length in its
# sentence_bert_config.json). Anything past that is silently discarded at
# embed time, so chunking has to respect it. Two of those positions go to
# [CLS] and [SEP], leaving 254 for actual content.
MAX_SEQUENCE_TOKENS = 256
SPECIAL_TOKENS = 2
MAX_CONTENT_TOKENS = MAX_SEQUENCE_TOKENS - SPECIAL_TOKENS


class EmbeddingError(Exception):
    """Raised when text cannot be embedded."""


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer  # deferred: slow import, heavy dep

    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_tokenizer():
    return _get_model().tokenizer


def count_tokens(text: str) -> int:
    """Length in the only unit that matters: wordpieces, as this model counts them.

    Excludes [CLS]/[SEP] so it can be compared directly against
    MAX_CONTENT_TOKENS.
    """
    return len(_get_tokenizer().encode(text, add_special_tokens=False))


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
