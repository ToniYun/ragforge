import uuid

from fastapi import Depends, FastAPI, HTTPException, status, APIRouter
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Document
from app.schemas import DocumentCreate, DocumentResponse

router = APIRouter(
    prefix="/documents",
    tags=["documents"],    
)

@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def create_document(document: DocumentCreate, db: Session = Depends(get_db)):
    new_document = Document(filename=document.filename)
    db.add(new_document)
    db.commit()
    db.refresh(new_document)
    return new_document

@router.get("", response_model=list[DocumentResponse])
def get_documents(db: Session = Depends(get_db)):
    statement = select(Document).order_by(Document.created_at.desc())
    documents = db.scalars(statement).all()
    return documents

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