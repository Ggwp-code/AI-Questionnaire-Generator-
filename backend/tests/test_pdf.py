import pytest
from pdf_fixtures import digital_pdf, encrypted_pdf, mixed_pdf, scanned_pdf, zero_page_pdf

from tribunal.errors import ValidationFailed
from tribunal.ingestion.pdf import page_texts, read_pdf, read_toc


def test_read_digital():
    pages = read_pdf(digital_pdf(["Alpha beta gamma " * 10, "Second page text " * 10]), 10)
    assert len(pages) == 2
    assert "Alpha beta gamma" in pages[0].text
    assert pages[0].page_number == 1
    assert pages[0].image_coverage == 0


def test_read_scanned():
    pages = read_pdf(scanned_pdf(2), 10)
    assert all(p.text.strip() == "" and p.image_coverage > 0.9 for p in pages)


def test_read_mixed():
    p = read_pdf(mixed_pdf(), 10)[0]
    assert p.text.strip() != "" and p.image_coverage >= 0.5


def test_not_pdf():
    with pytest.raises(ValidationFailed) as e:
        read_pdf(b"hello", 10)
    assert e.value.message == "file is not a readable PDF"


def test_page_limit():
    with pytest.raises(ValidationFailed) as e:
        read_pdf(digital_pdf(["a", "b", "c"]), 2)
    assert e.value.message == "PDF has 3 pages; the limit is 2"


def test_toc():
    data = digital_pdf(["x", "y"], bookmarks=[("One", 1), ("Two", 2)])
    assert read_toc(data) == [("One", 1), ("Two", 2)]


def test_page_texts():
    out = page_texts(digital_pdf(["first page", "second page"]), [2])
    assert set(out) == {2} and "second" in out[2]


def test_encrypted_pdf_unreadable():
    with pytest.raises(ValidationFailed) as e:
        read_pdf(encrypted_pdf(), 10)
    assert e.value.message == "file is not a readable PDF"


def test_zero_page_pdf_refused_by_pdfium():
    # pdfium refuses zero-page files at load time, so "PDF has no pages" is only a defensive check
    with pytest.raises(ValidationFailed) as e:
        read_pdf(zero_page_pdf(), 10)
    assert e.value.message == "file is not a readable PDF"
