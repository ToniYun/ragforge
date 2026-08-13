import uuid
from pathlib import Path

from app.embeddings import EMBEDDING_DIMENSION
from app.models import Document, Document_Chunks
from app.services.ingestion_service import process_document
from tests.conftest import TestingSessionLocal


def make_text_pdf(text: str) -> bytes:
    """Build a minimal single-page PDF whose content stream is real, extractable text."""
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = f"BT /F1 12 Tf 40 750 Td 14 TL ({escaped}) Tj ET".encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >> stream\n" % len(stream) + stream + b"\nendstream",
    ]

    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for i, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj ".encode() + body + b" endobj\n"

    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n".encode()
    pdf += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        pdf += f"{off:010d} 00000 n \n".encode()
    pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode()
    return pdf


def repeated_sentences(count: int) -> str:
    return " ".join(
        f"This is sentence number {i} about employee benefits, vacation policy, "
        f"and medical coverage details for the handbook."
        for i in range(count)
    )


def upload_text_document(client, text: str, filename: str = "handbook.pdf"):
    pdf_bytes = make_text_pdf(text)
    response = client.post(
        "/documents",
        files={"file": (filename, pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    return response.json()


def get_chunks(document_id):
    if isinstance(document_id, str):
        document_id = uuid.UUID(document_id)

    db = TestingSessionLocal()
    try:
        return (
            db.query(Document_Chunks)
            .filter(Document_Chunks.document_id == document_id)
            .order_by(Document_Chunks.chunk_index)
            .all()
        )
    finally:
        db.close()


def test_processing_creates_chunks_with_correctly_sized_embeddings(client):
    data = upload_text_document(client, repeated_sentences(60))
    assert data["status"] == "READY"

    chunks = get_chunks(data["id"])

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.embedding is not None
        assert len(chunk.embedding) == EMBEDDING_DIMENSION


def test_no_null_embeddings_among_created_chunks(client):
    data = upload_text_document(client, repeated_sentences(60))

    chunks = get_chunks(data["id"])

    null_embeddings = [c for c in chunks if c.embedding is None]
    assert null_embeddings == []


def test_chunk_indexes_are_unique_and_contiguous(client):
    data = upload_text_document(client, repeated_sentences(60))

    chunks = get_chunks(data["id"])
    indexes = [c.chunk_index for c in chunks]

    assert len(indexes) == len(set(indexes)), "duplicate chunk_index values found"
    assert indexes == list(range(len(indexes)))


def test_no_duplicate_chunk_content_within_a_document(client):
    data = upload_text_document(client, repeated_sentences(60))

    chunks = get_chunks(data["id"])
    contents = [c.content for c in chunks]

    assert len(contents) == len(set(contents)), "two chunks have identical content"


def test_blank_document_produces_zero_chunks_and_no_orphaned_rows(client):
    response = client.post(
        "/documents",
        files={"file": ("blank.pdf", make_text_pdf(""), "application/pdf")},
    )
    assert response.status_code == 201
    document_id = response.json()["id"]

    chunks = get_chunks(document_id)
    assert len(chunks) == 0


def test_reprocessing_a_document_does_not_duplicate_chunks(client, tmp_path):
    db = TestingSessionLocal()
    try:
        document = Document(
            filename="handbook.pdf",
            status="UPLOADED",
            file_type="application/pdf",
            file_size=1,
            storage_path="unused",
        )
        db.add(document)
        db.commit()
        db.refresh(document)

        pdf_path: Path = tmp_path / "handbook.pdf"
        pdf_path.write_bytes(make_text_pdf(repeated_sentences(60)))

        process_document(document_id=document.id, file_path=pdf_path, db=db)
        db.commit()
        first_pass = get_chunks(document.id)
        assert len(first_pass) > 0

        process_document(document_id=document.id, file_path=pdf_path, db=db)
        db.commit()
        second_pass = get_chunks(document.id)

        assert len(second_pass) == len(first_pass), (
            "reprocessing the same document duplicated its chunks instead of replacing them"
        )
    finally:
        db.close()
