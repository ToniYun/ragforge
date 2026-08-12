from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence


@dataclass(frozen=True)
class Page:
    page_number: int
    text: str
    
@dataclass
class Chunk:
    document_id: str
    chunk_index: int
    page_number: int
    content: str
    token_count: int
    created_at: datetime
    chunk_id: str
    page_numbers: list[int]
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "content": self.content,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            "chunk_id": self.chunk_id,
            "page_numbers": self.page_numbers,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": self.metadata,
        }
    
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def make_chunk_id(document_id: str, strategy: str, content: str, start: int, end: int) -> str:
    """Generate a unique chunk ID based on document ID and chunk index."""
    digest = hashlib.sha256(
        f"{document_id}:{strategy}:{content}:{start}:{end}".encode("utf-8")   
    ).hexdigest()
    return f"{document_id}:{digest[:16]}"

class BaseChunker(ABC):
    strategy: str = "base"
    
    @abstractmethod
    def chunk(
        self,
        pages: Sequence[Page],
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:...