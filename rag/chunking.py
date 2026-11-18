"""Structure-aware, sentence-friendly chunking.

Good chunking is the single biggest lever on retrieval quality:

* we split on paragraph -> line -> sentence -> word boundaries, so chunks
  never cut a sentence in half unless the sentence is bigger than the chunk;
* consecutive chunks overlap a little, so an answer that spans a boundary is
  still fully present in at least one chunk;
* every chunk carries a "context header" (file, page, section) that is
  embedded together with the text - a chunk named "Refunds > Time window" is
  far easier to retrieve than the same words without any context.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from . import config
from .loaders import Page

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


@dataclass
class Chunk:
    id: str
    text: str                 # what the user/model sees
    embed_text: str           # what we send to the embedding model
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Rough token count - good enough for sizing, no tokenizer download."""
    return max(1, len(text) // 4)


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators or separators == [""]:
        return _hard_split(text, size)

    separator, rest = separators[0], separators[1:]
    parts = text.split(separator)
    # keep the separator attached so sentences stay readable
    pieces = [p + separator for p in parts[:-1]] + [parts[-1]]

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > size:
            if current.strip():
                chunks.append(current)
                current = ""
            chunks.extend(_recursive_split(piece, size, rest))
        elif len(current) + len(piece) <= size:
            current += piece
        else:
            if current.strip():
                chunks.append(current)
            current = piece
    if current.strip():
        chunks.append(current)
    return chunks


def _overlap_tail(previous: str, overlap: int) -> str:
    """Take the tail of the previous chunk, cut on a word boundary."""
    if overlap <= 0 or not previous:
        return ""
    tail = previous[-overlap:]
    # avoid starting mid-word
    match = re.search(r"\s", tail)
    if match and match.start() < len(tail) - 1:
        tail = tail[match.start() + 1 :]
    return tail.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    raw_chunks = [c.strip() for c in _recursive_split(text, size, _SEPARATORS)]
    raw_chunks = [c for c in raw_chunks if c]

    if overlap <= 0:
        return raw_chunks

    result: list[str] = []
    for index, chunk in enumerate(raw_chunks):
        if index == 0:
            result.append(chunk)
            continue
        tail = _overlap_tail(raw_chunks[index - 1], overlap)
        if tail and not chunk.startswith(tail):
            result.append(f"{tail}\n{chunk}")
        else:
            result.append(chunk)
    return result


def _context_header(page: Page) -> str:
    bits = [page.source]
    if page.page:
        bits.append(f"page {page.page}")
    if page.section:
        bits.append(f"section: {page.section}")
    if page.extra.get("row_start"):
        bits.append(f"rows {page.extra['row_start']}-{page.extra.get('row_end')}")
    return " | ".join(bits)


def _chunk_id(path: str, page: int | None, index: int, text: str) -> str:
    payload = f"{path}|{page}|{index}|{text[:200]}"
    return hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()[:24]


def chunk_pages(
    pages: list[Page],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP

    chunks: list[Chunk] = []
    for page in pages:
        header = _context_header(page)
        for index, piece in enumerate(split_text(page.text, size, overlap)):
            if len(piece.strip()) < 20:
                continue
            metadata: dict[str, Any] = {
                "source": page.source,
                "path": page.path,
                "chunk_index": index,
                "chars": len(piece),
                "tokens": estimate_tokens(piece),
            }
            if page.page is not None:
                metadata["page"] = page.page
            if page.section:
                metadata["section"] = page.section
            if page.extra.get("row_start"):
                metadata["row_start"] = page.extra["row_start"]
                metadata["row_end"] = page.extra["row_end"]
            chunks.append(
                Chunk(
                    id=_chunk_id(page.path, page.page, index, piece),
                    text=piece,
                    embed_text=f"{header}\n{piece}",
                    metadata=metadata,
                )
            )
    return chunks
