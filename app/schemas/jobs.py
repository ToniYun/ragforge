import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobCreate(BaseModel):
    document_id: uuid.UUID
    status: str

class JobResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    status: str
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )