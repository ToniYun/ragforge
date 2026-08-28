"""Pure retrieval metrics. No database, no models, no I/O.

Everything here operates on lists of chunk ids, so it can be tested in
isolation and reused for any retrieval mode.

Conventions
-----------
* ``retrieved`` is an ordered list of chunk ids, best first.
* ``relevant`` is an unordered collection of chunk ids that genuinely
  answer the question.
* A metric returns ``None`` when it is undefined for the question --
  notably when ``relevant`` is empty (an "unanswerable" eval question).
  ``None`` means "don't score this", not "scored zero"; the aggregator
  skips it rather than dragging the mean down.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from statistics import median


def dedupe(ids: Iterable[str]) -> list[str]:
    """Drop repeats while preserving order.

    Fusion can surface the same chunk from two arms; a duplicate would
    otherwise inflate recall and shift ranks.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk_id in ids:
        if chunk_id not in seen:
            seen.add(chunk_id)
            ordered.append(chunk_id)
    return ordered


def recall_at_k(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float | None:
    """Fraction of relevant chunks that appear in the top ``k``."""
    if k <= 0:
        raise ValueError("k must be positive")

    relevant_set = set(relevant)
    if not relevant_set:
        return None

    top_k = set(dedupe(retrieved)[:k])
    return len(top_k & relevant_set) / len(relevant_set)


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int | None = None,
) -> float | None:
    """1 / rank of the first relevant chunk; 0.0 if none appear in top ``k``."""
    relevant_set = set(relevant)
    if not relevant_set:
        return None

    ordered = dedupe(retrieved)
    if k is not None:
        ordered = ordered[:k]

    for position, chunk_id in enumerate(ordered, start=1):
        if chunk_id in relevant_set:
            return 1 / position
    return 0.0


def hit_rate(
    retrieved: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float | None:
    """1.0 if any relevant chunk made the top ``k``, else 0.0."""
    rr = reciprocal_rank(retrieved, relevant, k=k)
    if rr is None:
        return None
    return 1.0 if rr > 0 else 0.0


def mean(values: Iterable[float | None]) -> float | None:
    """Average, skipping ``None``. Returns ``None`` if nothing is scorable."""
    scored = [v for v in values if v is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


def p50(values: Iterable[float]) -> float | None:
    ordered = list(values)
    if not ordered:
        return None
    return median(ordered)
