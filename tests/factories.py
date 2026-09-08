"""Shared builders for retrieval tests.

Not collected by pytest (no ``test_`` prefix). These build detached ORM
objects -- no session, no database -- so retrieval logic can be tested as
plain Python.
"""

from __future__ import annotations

import uuid

from app.models import Document, Document_Chunks

EMBEDDING_DIM = 384


def make_document(filename: str = "handbook.pdf") -> Document:
    return Document(
        id=uuid.uuid4(),
        filename=filename,
        file_type="application/pdf",
        status="READY",
    )


def make_chunk(
    content: str = "Employees get 15 vacation days.",
    *,
    chunk_index: int = 0,
    page_number: int = 1,
    document: Document | None = None,
    chunk_id: uuid.UUID | None = None,
) -> Document_Chunks:
    document = document or make_document()
    chunk = Document_Chunks(
        id=chunk_id or uuid.uuid4(),
        document_id=document.id,
        chunk_index=chunk_index,
        page_number=page_number,
        content=content,
        token_count=len(content.split()),
        embedding=[0.1] * EMBEDDING_DIM,
    )
    chunk.document = document
    return chunk


def make_chunks(count: int, prefix: str = "chunk") -> list[Document_Chunks]:
    """``count`` distinct chunks sharing one document."""
    document = make_document()
    return [
        make_chunk(f"{prefix} {i}", chunk_index=i, document=document)
        for i in range(count)
    ]
