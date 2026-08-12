from io import BytesIO
from uuid import uuid4

from pypdf import PdfWriter


def create_valid_pdf():
    buffer = BytesIO()

    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(buffer)

    return buffer.getvalue()


def upload_document(client):
    pdf_bytes = create_valid_pdf()

    response = client.post(
        "/documents",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")}
    )
    return response.json()["id"]


def test_create_job_for_document(client):
    document_id = upload_document(client)

    response = client.post(
        f"/documents/{document_id}/jobs",
        json={"document_id": document_id, "status": "PENDING"}
    )

    assert response.status_code == 201
    data = response.json()
    assert data["document_id"] == document_id
    assert data["status"] == "PENDING"


def test_create_job_for_missing_document(client):
    response = client.post(
        f"/documents/{uuid4()}/jobs",
        json={"document_id": str(uuid4()), "status": "PENDING"}
    )

    assert response.status_code == 404


def test_get_document_jobs(client):
    document_id = upload_document(client)

    client.post(
        f"/documents/{document_id}/jobs",
        json={"document_id": document_id, "status": "PENDING"}
    )

    response = client.get(f"/documents/{document_id}/jobs")

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["document_id"] == document_id


def test_create_job_via_jobs_endpoint(client):
    document_id = upload_document(client)

    response = client.post(
        "/jobs",
        json={"document_id": document_id, "status": "PENDING"}
    )

    assert response.status_code == 201
    assert response.json()["status"] == "PENDING"


def test_delete_job(client):
    document_id = upload_document(client)

    create_response = client.post(
        "/jobs",
        json={"document_id": document_id, "status": "PENDING"}
    )
    job_id = create_response.json()["id"]

    delete_response = client.delete(f"/jobs/{job_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/jobs/{job_id}")
    assert get_response.status_code == 404


def test_delete_job_not_found(client):
    response = client.delete(f"/jobs/{uuid4()}")

    assert response.status_code == 404
