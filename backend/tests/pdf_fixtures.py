"""Helpers that build small PDFs for tests (fpdf2 core fonts and Pillow's default font only)."""

from __future__ import annotations

import io

from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont


def _text_image(label: str, size: tuple[int, int] = (850, 1100)) -> bytes:
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=40)
    for i in range(12):
        draw.text((40, 40 + i * 80), f"{label} line {i}", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def digital_pdf(pages: list[str], bookmarks: list[tuple[str, int]] | None = None) -> bytes:
    pdf = FPDF()
    for number, text in enumerate(pages, start=1):
        pdf.add_page()
        for title, page in bookmarks or []:
            if page == number:
                pdf.start_section(title)
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 6, text)
    return bytes(pdf.output())


def scanned_pdf(page_count: int) -> bytes:
    pdf = FPDF()
    for n in range(page_count):
        pdf.add_page()
        pdf.image(io.BytesIO(_text_image(f"scan {n + 1}")), x=0, y=0, w=210, h=297)
    return bytes(pdf.output())


def mixed_pdf() -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=14)
    pdf.cell(0, 10, "Native header line")
    pdf.image(io.BytesIO(_text_image("figure")), x=10, y=30, w=190, h=250)
    return bytes(pdf.output())


class FakeOcrEngine:
    def __init__(self, texts: dict[int, str]):
        self.texts = texts
        self.calls: list[tuple[list[int], list[str]]] = []

    def ocr_pages(self, data: bytes, pages: list[int], languages: list[str]) -> dict[int, str]:
        self.calls.append((list(pages), list(languages)))
        return {p: self.texts[p] for p in pages if p in self.texts}
