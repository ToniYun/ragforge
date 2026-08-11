import uuid
from pathlib import Path

from fastapi import Depends, File, FastAPI, HTTPException, status, APIRouter, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.loaders.pdf_loader import load_pdf, PDFLoadError
from app.models import Document, Jobs
from app.schemas import DocumentResponse, JobCreate, JobResponse

router = APIRouter(
    prefix="/documents",
    tags=["documents"],    
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def create_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ["application/pdf"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only PDF files are allowed.")
    
    
    new_document = Document(
        filename=file.filename or "unnamed.pdf",
        status="UPLOADED",
        file_type=file.content_type,
        file_size=len(file.file.read()),
        storage_path=str(UPLOAD_DIR / f"{uuid.uuid4()}.pdf")
    )
    
    db.add(new_document)
    
    db.flush()
    
    file_path = UPLOAD_DIR / f"{new_document.id}.pdf"
    
    try:
        with file_path.open("wb") as destination:
            while chunk := file.file.read(1024 * 1024): 
                destination.write(chunk)
        
        pages = load_pdf(file_path)
        
        db.commit()
        db.refresh(new_document)
        
        return new_document
    except PDFLoadError as e:
        new_document.status = "FAILED"
        
        db.commit()
        db.refresh(new_document)
          
        return new_document
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