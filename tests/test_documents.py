from io import BytesIO
from uuid import uuid4

from pypdf import PdfReader, PdfWriter

def create_valid_pdf():
    buffer = BytesIO()
    
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)
    
    return buffer.getvalue()

def test_upload_pdf(client):
    pdf_bytes = create_valid_pdf()
    
    response = client.post(
        "/documents",
        files={
            "file": (
                "test.pdf",
                pdf_bytes,
                "application/pdf"
            )
        }
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["filename"] == "test.pdf"
    assert data["status"] == "READY"
    assert "id" in data
    
    
def test_reject_non_pdf(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "test.txt",
                b"Hello, World!",
                "text/plain"
            )
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Invalid file type. Only PDF files are allowed."
    
def test_missing_file(client):
    response = client.post("/documents")

    assert response.status_code == 422


def test_get_documents_empty(client):
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_after_upload(client):
    pdf_bytes = create_valid_pdf()

    client.post(
        "/documents",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
    )

    response = client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["filename"] == "test.pdf"


def test_get_document_by_id(client):
    pdf_bytes = create_valid_pdf()

    upload_response = client.post(
        "/documents",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
    )
    document_id = upload_response.json()["id"]

    response = client.get(f"/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["id"] == document_id


def test_get_document_not_found(client):
    response = client.get(f"/documents/{uuid4()}")

    assert response.status_code == 404


def test_delete_document(client):
    pdf_bytes = create_valid_pdf()

    upload_response = client.post(
        "/documents",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
    )
    document_id = upload_response.json()["id"]

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/documents/{document_id}")
    assert get_response.status_code == 404


def test_delete_document_not_found(client):
    response = client.delete(f"/documents/{uuid4()}")

    assert response.status_code == 404
    
def test_corrupt_pdf_marks_failed(client):
    response = client.post(
        "/documents",
        files={"file": ("bad.pdf", b"not a real pdf", "application/pdf")}
    )
    assert response.status_code == 201
    assert response.json()["status"] == "FAILED"


def test_upload_stores_correct_file_size(client):
    pdf_bytes = create_valid_pdf()
    response = client.post(
        "/documents",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
    )
    assert response.json()["file_size"] == len(pdf_bytes)


def test_oversized_file_rejected(client):
    big = b"%PDF-1.4\n" + b"0" * (21 * 1024 * 1024)
    response = client.post(
        "/documents",
        files={"file": ("big.pdf", big, "application/pdf")}
    )
    assert response.status_code == 413