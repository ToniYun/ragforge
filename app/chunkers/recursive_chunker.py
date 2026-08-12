from __future__ import annotations
 
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence
 
from .base import BaseChunker, Chunk, Page, make_chunk_id, utcnow
from .normalize import normalize_pages
from .tokenizer import get_token_counter

DEFAULT_SEPARATORS = ["\n\n","\n",". ","? ","! ","; ",", "," "]
PAGE_JOIN = "\n\n"


@dataclass
class _Fragment:
    text: str
    start: int
    tokens: int
    

class RecursiveChunker(BaseChunker):
    strategy = "recursive-v1"
 
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 75,
        min_chunk_size: int = 100,
        separators: Sequence[str] | None = None,
        encoding_name: str = "cl100k_base",
        length_fn: Callable[[str], int] | None = None,
    ) -> None:
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.separators = list(separators) if separators else list(DEFAULT_SEPARATORS)
        self._len = length_fn or get_token_counter(encoding_name)
 
    # ---------- entry points ----------
 
    def chunk_raw_pages(
        self,
        pages: Sequence[str] | Sequence[dict] | Sequence[Page],
        document_id: str,
        metadata: dict[str, Any] | None = None,
        **normalize_kwargs: Any,
    ) -> list[Chunk]:
        """Normalize then chunk. This is the one you usually want."""
        return self.chunk(normalize_pages(pages, **normalize_kwargs), document_id, metadata)
 
    def chunk(
        self,
        pages: Sequence[Page],
        document_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Chunk already-normalized pages."""
        if not pages:
            return []
 
        text, spans = self._assemble(pages)
        fragments = self._to_fragments(text)
        groups = self._absorb_runt(self._merge(fragments))
        created_at = utcnow()
 
        chunks: list[Chunk] = []
        for group in groups:
            raw = "".join(f.text for f in group)
            content = raw.strip()
            if not content:
                continue
 
            start = group[0].start + (len(raw) - len(raw.lstrip()))
            end = start + len(content)
            page_numbers = [n for (s, e, n) in spans if s < end and start < e]
 
            chunks.append(
                Chunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    page_number=page_numbers[0] if page_numbers else pages[0].page_number,
                    content=content,
                    token_count=self._len(content),
                    created_at=created_at,
                    chunk_id=make_chunk_id(document_id, self.strategy, content, start, end),
                    page_numbers=page_numbers,
                    char_start=start,
                    char_end=end,
                    metadata=dict(metadata or {}),
                )
            )
        return chunks
 
    # ---------- assembly ----------
 
    @staticmethod
    def _assemble(pages: Sequence[Page]) -> tuple[str, list[tuple[int, int, int]]]:
        """Join pages, recording the exact span each one occupies.
 
        Joining is what lets a sentence broken by a page break stay in one
        chunk; the span map is what lets that chunk still name its pages.
        """
        bodies: list[str] = []
        spans: list[tuple[int, int, int]] = []
        cursor = 0
        for page in pages:
            bodies.append(page.text)
            spans.append((cursor, cursor + len(page.text), page.page_number))
            cursor += len(page.text) + len(PAGE_JOIN)
        return PAGE_JOIN.join(bodies), spans
 
    # ---------- splitting ----------
 
    def _to_fragments(self, text: str) -> list[_Fragment]:
        fragments: list[_Fragment] = []
        cursor = 0
        for piece in self._split(text, self.separators):
            fragments.append(_Fragment(piece, cursor, self._len(piece)))
            cursor += len(piece)
        return fragments
 
    def _split(self, text: str, separators: Sequence[str]) -> list[str]:
        """Split until every fragment fits chunk_size. Concatenation is lossless."""
        if self._len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return self._hard_split(text)
 
        parts = self._split_keeping(text, separators[0])
        if len(parts) == 1:
            return self._split(text, separators[1:])
 
        out: list[str] = []
        for part in parts:
            if self._len(part) <= self.chunk_size:
                out.append(part)
            else:
                out.extend(self._split(part, separators[1:]))
        return out
 
    @staticmethod
    def _split_keeping(text: str, separator: str) -> list[str]:
        """Split on `separator`, keeping it attached to the preceding fragment."""
        parts = re.split(f"({re.escape(separator)})", text)
        out = []
        for i in range(0, len(parts), 2):
            fragment = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            if fragment:
                out.append(fragment)
        return out
 
    def _hard_split(self, text: str) -> list[str]:
        """Last resort: a table row or base64 blob with no separators in it."""
        ratio = len(text) / max(1, self._len(text))
        window = max(1, int(self.chunk_size * ratio))
        return [text[i : i + window] for i in range(0, len(text), window)]
 
    # ---------- merging ----------
 
    def _merge(self, fragments: list[_Fragment]) -> list[list[_Fragment]]:
        groups: list[list[_Fragment]] = []
        current: list[_Fragment] = []
        total = 0
 
        for fragment in fragments:
            if current and total + fragment.tokens > self.chunk_size:
                groups.append(current)
                current, total = self._overlap_tail(current)
            current.append(fragment)
            total += fragment.tokens
 
        if current:
            groups.append(current)
        return groups
 
    def _overlap_tail(self, group: list[_Fragment]) -> tuple[list[_Fragment], int]:
        """Seed the next chunk with the tail of this one."""
        if self.chunk_overlap <= 0:
            return [], 0
        tail: list[_Fragment] = []
        total = 0
        for fragment in reversed(group):
            if tail and total + fragment.tokens > self.chunk_overlap:
                break
            tail.insert(0, fragment)
            total += fragment.tokens
            if total >= self.chunk_overlap:
                break
        if total >= self.chunk_size:  # a single oversized fragment: don't repeat it
            return [], 0
        return tail, total
 
    def _absorb_runt(self, groups: list[list[_Fragment]]) -> list[list[_Fragment]]:
        """Fold a too-small trailing chunk into its neighbour."""
        if len(groups) < 2:
            return groups
        if sum(f.tokens for f in groups[-1]) >= self.min_chunk_size:
            return groups
        previous = groups[-2]
        seen = {(f.start, f.text) for f in previous}
        previous.extend(f for f in groups[-1] if (f.start, f.text) not in seen)
        return groups[:-1]


def chunk_pages(
    pages: Sequence[str] | Sequence[dict] | Sequence[Page],
    document_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Module-level convenience wrapper around the default RecursiveChunker."""
    return RecursiveChunker().chunk_raw_pages(pages, document_id, metadata)