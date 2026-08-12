from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column,relationship

from app.database import Base
from app.embeddings import EMBEDDING_DIMENSION

if TYPE_CHECKING:
    from app.models.document import Document

class Document_Chunks(Base):
    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False
    )
    
    chunk_index: Mapped[int] = mapped_column(
        nullable=False
    )

    page_number: Mapped[int] = mapped_column(
        nullable=True
    )

    content: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSION),
        nullable=True
    )


    document: Mapped[Document] = relationship(
        back_populates="chunks"
    )
    
    token_count: Mapped[int] = mapped_column(
        nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )