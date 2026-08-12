import uuid
from pathlib import Path

from fastapi import Depends, File, FastAPI, HTTPException, status, APIRouter, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.loaders.pdf_loader import load_pdf, PDFLoadError
from app.models import Document, Jobs
from app.schemas import DocumentResponse, JobCreate, JobResponse, ChunkResponse
from app.services.ingestion_service import process_document

router = APIRouter(
    prefix="/documents",
    tags=["documents"],    
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 10 * 1024 * 1024 

@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def create_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only PDF files are allowed.")
    
    
    new_document = Document(
        filename=file.filename or "unnamed.pdf",
        status="UPLOADED",
        file_type=file.content_type,
        file_size=file.size,
    )
    
    db.add(new_document)
    
    db.flush()
    
    file_path = UPLOAD_DIR / f"{new_document.id}.pdf"
    new_document.storage_path = str(file_path)
    
    try:
        size_written = 0
        with file_path.open("wb") as destination:
            while chunk := file.file.read(1024 * 1024): 
                size_written += len(chunk)
                if size_written > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 10MB.")
                destination.write(chunk)
        
        
        pages = load_pdf(file_path)
        
        new_document.extracted_text = pages
        
        process_document(
            document_id=new_document.id,
            file_path=file_path,
            db=db
        )
        
        new_document.status = "READY"
        
        db.commit()
        db.refresh(new_document)
        
        return new_document
    except PDFLoadError as e:
        new_document.status = "FAILED"

        db.commit()
        db.refresh(new_document)

        return new_document
    except HTTPException:
        db.rollback()

        if file_path.exists():
            file_path.unlink()

        raise
    except Exception as e:
        db.rollback()
        
        if file_path.exists():
            file_path.unlink()
        
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred while processing the document: {e}")
    finally:
        file.file.close()
        
        
    return new_document

@router.get("", response_model=list[DocumentResponse])
def get_documents(db: Session = Depends(get_db)):
    statement = select(Document).order_by(Document.created_at.desc())
    documents = db.scalars(statement).all()
    return documents

@router.get("/{document_id}/jobs", response_model=list[JobResponse])
def get_document_jobs(document_id: uuid.UUID, db: Session = Depends(get_db)):
    statement = select(Jobs).where(Jobs.document_id == document_id).order_by(Jobs.created_at.desc())
    jobs = db.scalars(statement).all()
    return jobs

@router.post("/{document_id}/jobs", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
def create_job_for_document(document_id: uuid.UUID, job: JobCreate, db: Session = Depends(get_db)):
    # Check if the document exists
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    new_job = Jobs(
        document_id=document_id,
        status=job.status
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)
    return new_job

@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    db.delete(document)
    db.commit()
    return None

@router.get("/{document_id}/text")
def get_document_text(document_id: uuid.UUID, db: Session = Depends(get_db)):
    document = db.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    
    if not document.extracted_text:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No extracted text available for this document")
    
    return {"extracted_text": document.extracted_text}

@router.get(
    "/{document_id}/chunks",
    response_model=list[ChunkResponse],
)
def get_document_chunks(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document.chunks