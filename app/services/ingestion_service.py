import uuid
from pathlib import Path

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.chunkers.recursive_chunker import chunk_pages
from app.embeddings import embed_texts
from app.loaders.pdf_loader import load_pdf
from app.models import Document_Chunks


def process_document(
    document_id: uuid.UUID,
    file_path: Path,
    db: Session
):
    pages = load_pdf(file_path)

    chunks = chunk_pages(pages, str(document_id))

    embeddings = embed_texts([chunk.content for chunk in chunks])

    db.execute(
        delete(Document_Chunks).where(
            Document_Chunks.document_id == document_id
        )
    )

    chunk_rows = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_row = Document_Chunks(
            document_id=document_id,
            chunk_index=chunk.chunk_index,
            page_number=chunk.page_number,
            content=chunk.content,
            token_count=chunk.token_count,
            embedding=embedding,
        )

        chunk_rows.append(chunk_row)

    db.add_all(chunk_rows)

    return chunk_rows