"""The chunker must never emit a chunk the embedding model would truncate.

all-MiniLM-L6-v2 cuts input at 256 wordpieces and silently embeds only what
fits. An oversized chunk therefore produces a vector for its opening
paragraphs and nothing else, while still looking perfectly healthy in the
database -- which is exactly how half a corpus becomes unsearchable without a
single error being raised.

Most of these use a deterministic additive counter rather than the real
tokenizer, so they exercise the budget arithmetic without loading a model.
The final test does load the model and checks the whole path end to end.
"""

import pytest

from app.chunkers.recursive_chunker import RecursiveChunker
from app.chunkers.tokenizer import _estimate_tokens, get_token_counter
from app.embeddings import MAX_CONTENT_TOKENS, MAX_SEQUENCE_TOKENS, SPECIAL_TOKENS


def words(text: str) -> int:
    """Additive, exact, and trivially predictable: one token per word."""
    return len(text.split())


def word_pages(count: int, per_page: int = 500) -> list[str]:
    tokens = [f"w{i}" for i in range(count)]
    return [
        " ".join(tokens[i : i + per_page]) for i in range(0, len(tokens), per_page)
    ]


def chunk_with(chunk_size, chunk_overlap, pages, min_chunk_size=100):
    chunker = RecursiveChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
        length_fn=words,
    )
    return chunker.chunk_raw_pages(pages, "doc-1")


# --- the budget constants ---------------------------------------------------


def test_content_budget_leaves_room_for_special_tokens():
    """[CLS] and [SEP] occupy two of the model's 256 positions, so a chunk of
    exactly 256 content tokens still overflows."""
    assert MAX_CONTENT_TOKENS == MAX_SEQUENCE_TOKENS - SPECIAL_TOKENS
    assert MAX_CONTENT_TOKENS == 254


def test_chunker_defaults_to_the_model_budget():
    assert RecursiveChunker().chunk_size == MAX_CONTENT_TOKENS


# --- the budget is never exceeded -------------------------------------------


@pytest.mark.parametrize(
    "chunk_size,chunk_overlap",
    [
        (254, 75),   # the shipped configuration
        (256, 75),
        (100, 40),
        (50, 20),    # small overlap-to-size ratio: the regression case
        (50, 45),    # overlap almost as large as the chunk
        (20, 19),
        (30, 0),     # no overlap at all
    ],
)
def test_no_chunk_exceeds_the_budget(chunk_size, chunk_overlap):
    """Regression: _absorb_runt folded a short trailing group into its
    neighbour without checking the combined size. Measured overshoots before
    the fix: 50/20 produced 70, 60/10 produced 100, 20/19 produced 21."""
    chunks = chunk_with(chunk_size, chunk_overlap, word_pages(4000))

    oversized = [c for c in chunks if words(c.content) > chunk_size]
    assert not oversized, (
        f"{len(oversized)} chunk(s) over budget {chunk_size}; "
        f"largest was {max(words(c.content) for c in oversized)}"
    )


def test_absorbing_a_runt_does_not_push_the_neighbour_over_budget():
    """The trailing-runt fold merges two groups; the result must still fit.
    min_chunk_size is set high here so the fold is guaranteed to be attempted."""
    chunks = chunk_with(60, 10, word_pages(605), min_chunk_size=55)

    assert all(words(c.content) <= 60 for c in chunks)


def test_a_large_fragment_after_a_large_overlap_tail_stays_within_budget():
    """The second overshoot path, which uniform fragments cannot reach.

    _merge seeds the next group with the overlap tail, then appends the next
    fragment without re-checking. When that fragment is itself near the full
    budget -- a long sentence that _split leaves whole because it does fit --
    the chunk reaches chunk_overlap + chunk_size. This construction produced
    95 against a budget of 50 before the fix.
    """
    short = " ".join("a b c d e." for _ in range(20))   # builds a 45-token tail
    big = " ".join(f"w{i}" for i in range(50)) + "."    # one 50-token fragment
    pages = [f"{short} {big} {short}"]

    chunks = chunk_with(50, 45, pages)

    assert max(words(c.content) for c in chunks) <= 50


def test_a_single_unsplittable_fragment_is_hard_split_within_budget():
    """A table row or base64 blob with no separators still has to fit."""
    blob = "x" * 20_000  # no whitespace anywhere
    chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10, length_fn=_estimate_tokens)

    chunks = chunker.chunk_raw_pages([blob], "doc-1")

    assert chunks
    assert all(_estimate_tokens(c.content) <= 50 for c in chunks)


# --- content is preserved ---------------------------------------------------


def test_every_token_survives_chunking():
    """Budget enforcement must not silently drop text -- the failure mode we
    are fixing is text loss, so the fix must not introduce its own."""
    pages = word_pages(2000)
    expected = " ".join(pages).split()

    chunks = chunk_with(254, 75, pages)

    seen = []
    for chunk in chunks:
        for token in chunk.content.split():
            if not seen or token != seen[-1]:
                seen.append(token)

    assert set(expected).issubset(set(seen))


def test_token_count_is_recorded_in_the_same_unit_as_the_budget():
    chunks = chunk_with(100, 20, word_pages(600))

    assert all(c.token_count == words(c.content) for c in chunks)


# --- the counter itself -----------------------------------------------------


def test_default_counter_is_the_embedding_tokenizer_not_an_estimate():
    """The character estimate drifts against real wordpieces by an amount that
    varies with the text, so chunks pass the budget check and get truncated
    anyway. Only the model's own tokenizer closes that gap."""
    counter = get_token_counter()

    assert counter is not _estimate_tokens, (
        "chunking fell back to the character estimate -- the embedding "
        "tokenizer failed to load"
    )


def test_estimate_fallback_is_still_usable():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("a" * 1000) == 250


# --- end to end with the real model -----------------------------------------


def test_real_chunks_fit_the_real_model():
    """The whole point, with nothing faked: chunk realistic prose with the
    shipped defaults, then tokenize exactly as the embedding model does."""
    from app.embeddings.embedding_service import _get_tokenizer

    prose = (
        "Dense retrieval maps questions and passages into a shared vector "
        "space, where relevance is measured by inner product. Sparse methods "
        "such as BM25 instead match on overlapping terms, which makes them "
        "strong on rare tokens like FAISS or TriviaQA and weak on paraphrase. "
    )
    pages = [prose * 40, prose * 25]

    chunks = RecursiveChunker().chunk_raw_pages(pages, "doc-1")
    tokenizer = _get_tokenizer()

    assert chunks
    lengths = [
        len(tokenizer.encode(c.content, add_special_tokens=True)) for c in chunks
    ]
    assert max(lengths) <= MAX_SEQUENCE_TOKENS, (
        f"largest chunk was {max(lengths)} wordpieces against a "
        f"{MAX_SEQUENCE_TOKENS} limit"
    )
