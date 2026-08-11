import uuid
from datetime import datetime

from sqlalchemy import String, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base  

class Jobs(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), 
        primary_key=True, 
        default=uuid.uuid4
    )
    
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False
    )
    
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PENDING"
    )
    
    attempt_count: Mapped[int] = mapped_column(
        nullable=False,
        default=0
    )
    
    error_message: Mapped[str] = mapped_column(
        String(500),
        nullable=True
    )
    
    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=True
    )
    
    completed_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=True
    )
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow
    )
