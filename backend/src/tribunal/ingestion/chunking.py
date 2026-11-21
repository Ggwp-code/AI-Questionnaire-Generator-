"""Page-bounded text chunking with overlap."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    page_number: int
    char_start: int
    char_end: int
    text: str


def chunk_pages(pages: dict[int, str], chunk_chars: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_number in sorted(pages):
        text = pages[page_number]
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            if end < len(text):
                floor = start + int(chunk_chars * 0.8)
                for i in range(end, floor, -1):
                    if text[i - 1].isspace():
                        end = i
                        break
            if text[start:end].strip():
                chunks.append(Chunk(page_number, start, end, text[start:end]))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks
