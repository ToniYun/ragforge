from __future__ import annotations

from functools import lru_cache
from typing import Callable


def _estimate_tokens(text: str) -> int:
    """Crude fallback: roughly four characters per token.

    Only used when the embedding tokenizer cannot be loaded. It is an
    estimate, so chunks built against it can still overshoot the model's real
    limit -- see get_token_counter.
    """
    return (len(text) + 3) // 4


@lru_cache(maxsize=1)
def get_token_counter() -> Callable[[str], int]:
    """The embedding model's own wordpiece counter.

    A chunk budget is only meaningful in the unit the model truncates by.
    Counting with any other tokenizer -- tiktoken, characters, words -- gives
    a number that drifts against the real limit by an amount that varies with
    the text, so chunks pass the budget check and still get truncated at
    embed time.

    Falls back to a character estimate only if the model cannot be loaded at
    all, which keeps chunking usable offline at the cost of that guarantee.
    """
    try:
        from app.embeddings import count_tokens

        count_tokens("warm up")  # surface a load failure here, not mid-document
        return count_tokens
    except Exception:
        return _estimate_tokens
