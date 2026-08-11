import uuid
from datetime import datetime

from sqlalchemy import String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


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
    
    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=True
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
 
