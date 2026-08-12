from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Sequence

from .base import Page

_INVISIBLE = dict.fromkeys(map(ord, "\ufeff\u200b\u200c\u200d\u2060"), None)
_SPACES = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_HYPHEN_BREAK = re.compile(r"(\w)[-\u2010\u2011]\n(\w)")
_BULLET = re.compile(r"^\s*(?:[-*\u2022\u2023\u25e6\u00b7]|(?:\w{1,3}[.)]))\s+")
_SENTENCE_END = re.compile(r"[.!?:;\"')\]]$")


def normalize_page(text: str, dehyphenate:bool = True, unwrap_lines:bool = True):
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    
    if dehyphenate:
        text = _HYPHEN_BREAK.sub(r"\1\2", text)
        
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()

def _unwrap(text:str)->str:
    out: list[str] = []
    for line in text.split("\n"):
        if out and _is_wrap(out[-1], line):
            out[-1] += f"{out[-1]} {line}"
        else:
            out.append(line)
    return "\n".join(out)

def _is_wrap(prev:str, following:str)->bool:
    if not prev or not following:
        return False
    if _SENTENCE_END.search(prev):
        return False
    if _BULLET.search(following):
        return False
    
    if len(following) < 40 and not following[0].islower():
        return False
    return True

def drop_repeated_lines(pages:Sequence[str], min_ratio: float = 0.6, max_length: int = 90, edge_lines: int = 3):
    if len(pages)<3:
        return list(pages)
    
    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
        edges = lines[:edge_lines] + lines[-edge_lines:]
        counts.update(ln for ln in edges if len(ln) <= max_length)
        
    threshold = max(2, int(len(pages) * min_ratio))
    boilerplate = {
        line
        for line, count in counts.items()
        if count >= threshold and not _is_page_number(line)
    }
    
    cleaned = []
    for page in pages:
        lines = page.split("\n")
        indexed = [(i, ln) for i, ln in enumerate(lines) if ln.strip()]
    
        edge_idx = {i for i, ln in indexed[:edge_lines] + indexed[-edge_lines:]}
        kept = [
            ln
            for i, ln in enumerate(lines)
            if ln.strip() not in boilerplate
            and not (i in edge_idx and _is_page_number(ln))
        ]  
        cleaned.append("\n".join(kept))
    return cleaned

_PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*/\s*\d{1,4})?$", re.IGNORECASE)

def _is_page_number(line: str):
    return bool(_PAGE_NUMBER.match(line.strip()))

def normalize_pages(
    pages:Sequence | Sequence[Page] | Sequence[dict],
    strip_boilerplate: bool = True,
    dehyphenate: bool = True,
    unwrap_lines: bool = True
):
    numbered: list[tuple[int,str]] = []
    for i,page in enumerate(pages, start=1):
        if isinstance(page, Page):
            numbered.append((page.page_number, page.text))
        elif isinstance(page,dict):
            number = page.get("page_number", page.get("page", i))
            numbered.append((number, page.get("text", "")))
        else:
            numbered.append((i,page))
    
    bodies = [text for _, text in numbered]
    if strip_boilerplate:
        bodies = drop_repeated_lines(bodies)
        
    result: list[Page] = []
    for (number, _), body in zip(numbered, bodies):
        cleaned = normalize_page(body, dehyphenate=dehyphenate, unwrap_lines=unwrap_lines)
        if cleaned:
            result.append(Page(page_number=number, text=cleaned))
    return result
