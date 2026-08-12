from __future__ import annotations

from functools import lru_cache
from typing import Callable

@lru_cache(maxsize=8)
def get_token_counter(encoding_name: str = "cl100k_base"):
    try:
        import tiktoken
        
        endoing = tiktoken
        
        encoding = tiktoken.get_encoding(encoding_name)
        
        def count(text: str):
            return len(encoding.encode(text,disallowed_special=()))
        
        return count
    except Exception:
        return _estimate_tokens
    
def _estimate_tokens(text:str):
    return (len(text)+3) // 4