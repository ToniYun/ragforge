from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import CitationItem, GenerateRequest, GenerateResponse
from app.services.generation_service import generate_answer

router = APIRouter(
    prefix="/generate",
    tags=["generate"],
)


@router.post("", response_model=GenerateResponse)
def generate(request: GenerateRequest, db: Session = Depends(get_db)):
    result = generate_answer(
        query=request.query,
        db=db,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    return GenerateResponse(
        answer=result.answer,
        citations=[
            CitationItem(
                number=c.number,
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                page_number=c.page_number,
                content=c.content,
            )
            for c in result.citations
        ],
    )
