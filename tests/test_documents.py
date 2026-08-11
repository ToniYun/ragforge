from io import BytesIO

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
    assert data["status"] == "UPLOADED"
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