from __future__ import annotations

import uuid
from dataclasses import dataclass

import httpx
from sqlalchemy.orm import Session

from app.config import settings
from app.services.retrieval_service import SearchResult, search_chunks

PROMPT_TEMPLATE = """Answer the question using only the sources below. Cite sources inline using their number in brackets, like [1] or [2]. If the sources don't contain enough information to answer, say so plainly instead of guessing.

Sources:
{sources}

Question: {query}

Answer:"""


class GenerationError(Exception):
    """Raised when the LLM fails to produce an answer."""


@dataclass
class Citation:
    number: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    content: str


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]


def build_prompt(query: str, results: list[SearchResult]) -> str:
    sources = "\n\n".join(
        f"[{i}] {result.chunk.content}" for i, result in enumerate(results, start=1)
    )
    return PROMPT_TEMPLATE.format(sources=sources, query=query)


def generate_answer(
    query: str,
    db: Session,
    top_k: int = 5,
    document_id: uuid.UUID | None = None,
) -> GenerationResult:
    results = search_chunks(query, db, top_k=top_k, document_id=document_id)

    citations = [
        Citation(
            number=i,
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            page_number=result.chunk.page_number,
            content=result.chunk.content,
        )
        for i, result in enumerate(results, start=1)
    ]

    if not results:
        return GenerationResult(
            answer="I couldn't find any relevant information to answer that.",
            citations=[],
        )

    prompt = build_prompt(query, results)

    try:
        response = httpx.post(
            settings.ollama_url,
            json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise GenerationError(f"Failed to reach Ollama at {settings.ollama_url}: {e}")

    answer = response.json()["response"].strip()

    return GenerationResult(answer=answer, citations=citations)
