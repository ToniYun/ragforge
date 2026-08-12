from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import String, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

if TYPE_CHECKING:
    from app.models.chunk import Document_Chunks

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    filename: Mapped[str] = mapped_column(
        String(255), 
        nullable=False
    )
    
    file_type: Mapped[str] = mapped_column(
        String(100),
        nullable=True
    )
    
    file_size: Mapped[int] = mapped_column(
        nullable=True
    )
    
    storage_path: Mapped[str] = mapped_column(
        String(500),
        nullable=True
    )

    extracted_text: Mapped[list | None] = mapped_column(
        JSON,
        nullable=True
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="UPLOADED"
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )
 
    chunks: Mapped[list[Document_Chunks]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan"
    )
