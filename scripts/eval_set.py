"""Loading and validating the eval set.

The eval set is the contract for every later stage: each entry says what
was asked, which chunks actually answer it, and what kind of retrieval
problem it represents. Validation is strict on purpose -- a silently
malformed eval set produces confident, meaningless numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "eval_set.json"

# The buckets exist so you can see *where* a change helped. A reranker that
# lifts the overall average while wrecking `exact` is not an improvement.
KINDS = {
    "paraphrase",   # answer is worded differently from the question
    "exact",        # hinges on a literal token: an ID, name, number
    "multihop",     # needs facts from two or more chunks/documents
    "unanswerable", # corpus genuinely does not contain the answer
}


class EvalSetError(Exception):
    """Raised when the eval set file is missing or malformed."""


@dataclass
class EvalQuestion:
    query: str
    kind: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def is_labeled(self) -> bool:
        """Unanswerable questions are complete with zero relevant chunks."""
        if self.kind == "unanswerable":
            return True
        return bool(self.relevant_chunk_ids)


def parse_questions(payload: object) -> list[EvalQuestion]:
    if not isinstance(payload, list):
        raise EvalSetError("eval set must be a JSON list of question objects")

    questions: list[EvalQuestion] = []
    for index, raw in enumerate(payload):
        where = f"entry {index}"

        if not isinstance(raw, dict):
            raise EvalSetError(f"{where}: must be an object")

        query = raw.get("query")
        if not isinstance(query, str) or not query.strip():
            raise EvalSetError(f"{where}: 'query' must be a non-empty string")

        kind = raw.get("kind")
        if kind not in KINDS:
            raise EvalSetError(
                f"{where}: 'kind' must be one of {sorted(KINDS)}, got {kind!r}"
            )

        ids = raw.get("relevant_chunk_ids", [])
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise EvalSetError(f"{where}: 'relevant_chunk_ids' must be a list of strings")

        if kind == "unanswerable" and ids:
            raise EvalSetError(
                f"{where}: unanswerable questions must have no relevant_chunk_ids"
            )

        if len(set(ids)) != len(ids):
            raise EvalSetError(f"{where}: 'relevant_chunk_ids' contains duplicates")

        questions.append(
            EvalQuestion(
                query=query,
                kind=kind,
                relevant_chunk_ids=ids,
                note=raw.get("note", ""),
            )
        )

    if not questions:
        raise EvalSetError("eval set is empty")

    return questions


def load(path: Path | str = DEFAULT_PATH) -> list[EvalQuestion]:
    path = Path(path)
    if not path.exists():
        raise EvalSetError(f"eval set not found at {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvalSetError(f"{path} is not valid JSON: {e}")

    return parse_questions(payload)


def save(questions: list[EvalQuestion], path: Path | str = DEFAULT_PATH) -> None:
    payload = [
        {
            "query": q.query,
            "kind": q.kind,
            "relevant_chunk_ids": q.relevant_chunk_ids,
            "note": q.note,
        }
        for q in questions
    ]
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def unlabeled(questions: list[EvalQuestion]) -> list[EvalQuestion]:
    return [q for q in questions if not q.is_labeled]
