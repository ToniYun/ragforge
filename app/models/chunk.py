import uuid
from datetime import datetime

from sqlalchemy import String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base  

class Document_Chunks(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False
    )
    
    chunk_index: Mapped[int] = mapped_column(
        nullable=False
    )
    
    content: Mapped[str] = mapped_column(
        String,
        nullable=False
    )
    
    page_number: Mapped[int] = mapped_column(
        nullable=True
    )
    
    token_count: Mapped[int] = mapped_column(
        nullable=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )