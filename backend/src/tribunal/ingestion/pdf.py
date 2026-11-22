"""PDF reading: per-page text, image coverage and bookmarks via pypdfium2."""

from __future__ import annotations

from dataclasses import dataclass

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from pypdfium2 import PdfiumError

from tribunal.errors import ValidationFailed


@dataclass
class PageInfo:
    page_number: int
    text: str
    image_coverage: float


def _open(data: bytes) -> pdfium.PdfDocument:
    try:
        return pdfium.PdfDocument(data)
    except (PdfiumError, ValueError, TypeError) as exc:
        raise ValidationFailed("file is not a readable PDF") from exc


def _text(page) -> str:
    tp = page.get_textpage()
    try:
        return tp.get_text_range()
    finally:
        tp.close()


def _coverage(page) -> float:
    width, height = page.get_size()
    area = width * height
    if area <= 0:
        return 0.0
    total = 0.0
    for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]):
        left, bottom, right, top = obj.get_bounds()
        total += max(0.0, right - left) * max(0.0, top - bottom)
    return min(1.0, total / area)


def read_pdf(data: bytes, max_pages: int) -> list[PageInfo]:
    doc = _open(data)
    try:
        n = len(doc)
        if n == 0:  # defensive: pdfium refuses zero-page files in _open
            raise ValidationFailed("PDF has no pages")
        if n > max_pages:
            raise ValidationFailed(f"PDF has {n} pages; the limit is {max_pages}")
        out = []
        for i in range(n):
            page = doc[i]
            try:
                out.append(PageInfo(i + 1, _text(page), _coverage(page)))
            finally:
                page.close()
        return out
    finally:
        doc.close()


def read_toc(data: bytes) -> list[tuple[str, int]]:
    doc = _open(data)
    try:
        out = []
        for item in doc.get_toc():
            dest = item.get_dest()
            index = dest.get_index() if dest else None
            if index is not None:
                out.append((item.get_title(), index + 1))
        return out
    finally:
        doc.close()


def page_texts(data: bytes, pages: list[int]) -> dict[int, str]:
    doc = _open(data)
    try:
        out = {}
        for p in pages:
            page = doc[p - 1]
            try:
                out[p] = _text(page)
            finally:
                page.close()
        return out
    finally:
        doc.close()
