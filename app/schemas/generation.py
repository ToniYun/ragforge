import uuid

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_id: uuid.UUID | None = None


class CitationItem(BaseModel):
    number: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    page_number: int
    content: str


class GenerateResponse(BaseModel):
    answer: str
    citations: list[CitationItem]
