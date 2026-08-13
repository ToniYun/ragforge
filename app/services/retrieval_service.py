from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import embed_text
from app.models import Document_Chunks


@dataclass
class SearchResult:
    chunk: Document_Chunks
    distance: float

    @property
    def similarity(self) -> float:
        return 1 - self.distance


def search_chunks(
    query: str,
    db: Session,
    top_k: int = 5,
    document_id: uuid.UUID | None = None,
) -> list[SearchResult]:
    query_embedding = embed_text(query)

    statement = (
        select(
            Document_Chunks,
            Document_Chunks.embedding.cosine_distance(query_embedding).label("distance"),
        )
        .where(Document_Chunks.embedding.is_not(None))
        .order_by("distance")
        .limit(top_k)
    )

    if document_id is not None:
        statement = statement.where(Document_Chunks.document_id == document_id)

    rows = db.execute(statement).all()

    return [SearchResult(chunk=chunk, distance=distance) for chunk, distance in rows]
