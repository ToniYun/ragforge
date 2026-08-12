import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ChunkResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    page_number: int
    content: str
    token_count: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)