from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import SearchRequest, SearchResultItem
from app.services.retrieval_service import search_chunks

router = APIRouter(
    prefix="/search",
    tags=["search"],
)


@router.post("", response_model=list[SearchResultItem])
def search(request: SearchRequest, db: Session = Depends(get_db)):
    results = search_chunks(
        query=request.query,
        db=db,
        top_k=request.top_k,
        document_id=request.document_id,
    )

    return [
        SearchResultItem(
            chunk_id=result.chunk.id,
            document_id=result.chunk.document_id,
            chunk_index=result.chunk.chunk_index,
            page_number=result.chunk.page_number,
            content=result.chunk.content,
            distance=result.distance,
            similarity=result.similarity,
        )
        for result in results
    ]
