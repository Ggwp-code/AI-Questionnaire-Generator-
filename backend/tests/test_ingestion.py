from __future__ import annotations

import pytest
from conftest import make_subject, make_user
from pdf_fixtures import FakeOcrEngine, digital_pdf, encrypted_pdf, mixed_pdf, scanned_pdf, zero_page_pdf
from sqlalchemy import func, select

from tribunal.context import build_context
from tribunal.db.models import Document, Page, Passage, Topic
from tribunal.ingestion.ocr import OcrUnavailable
from tribunal.ingestion.pipeline import extraction_version, ingest_document
from tribunal.runtime_config import ConfigSnapshot
from tribunal.settings import load_settings, reset_settings_cache


def _doc(ctx, data: bytes, languages=None) -> str:
    ctx.store.put("docs/x.pdf", data)
    with ctx.session_factory.begin() as s:
        user = make_user(s, "T@Example.com")
        subject = make_subject(s, user, "S")
        doc = Document(subject_id=subject.id, uploaded_by=user.id, filename="x.pdf", content_hash="h",
                       storage_key="docs/x.pdf", size_bytes=len(data),
                       ocr_languages=languages or ["eng"])
        s.add(doc)
        s.flush()
        return doc.id


def _run(ctx, doc_id):
    ingest_document(ctx, ctx.snapshot, doc_id)
    with ctx.session_factory() as s:
        return s.get(Document, doc_id)


def _count(ctx, model):
    with ctx.session_factory() as s:
        return s.scalar(select(func.count()).select_from(model))


def test_digital_document_ready(ctx, fake_ocr):
    pdf = digital_pdf(["Decision trees split data. " * 20, "Linear regression fits lines. " * 20],
                      [("Decision Trees", 1), ("Linear Regression", 2)])
    doc = _run(ctx, _doc(ctx, pdf))
    assert doc.status == "ready" and doc.page_count == 2 and doc.extraction_version.startswith("x-")
    with ctx.session_factory() as s:
        assert {p.method for p in s.scalars(select(Page))} == {"native"}
        passages = list(s.scalars(select(Passage)))
        assert passages and {p.page_number for p in passages} <= {1, 2}
        topics = s.scalars(select(Topic).order_by(Topic.page_start)).all()
        assert [t.title for t in topics] == ["Decision Trees", "Linear Regression"]
    assert fake_ocr.calls == []


def test_scanned_document_uses_ocr(ctx, fake_ocr):
    fake_ocr.texts = {1: "1 The French Revolution\n" + "text " * 40, 2: "2 The Industrial Revolution\n" + "text " * 40}
    doc = _run(ctx, _doc(ctx, scanned_pdf(2)))
    assert doc.status == "ready"
    assert fake_ocr.calls == [([1, 2], ["eng"])]
    with ctx.session_factory() as s:
        assert [p.method for p in s.scalars(select(Page).order_by(Page.page_number))] == ["ocr", "ocr"]
        titles = [t.title for t in s.scalars(select(Topic).order_by(Topic.page_start))]
    assert any("French Revolution" in t for t in titles) and any("Industrial Revolution" in t for t in titles)


def test_mixed_page_gets_ocr(ctx, fake_ocr):
    fake_ocr.texts = {1: "OCR BODY TEXT"}
    doc = _run(ctx, _doc(ctx, mixed_pdf()))
    assert doc.status == "ready"
    with ctx.session_factory() as s:
        page = s.scalars(select(Page).where(Page.page_number == 1)).one()
    assert page.method == "native+ocr" and "OCR BODY TEXT" in page.text and len(page.text) > len("OCR BODY TEXT") + 1


def test_not_a_pdf_fails_visibly(ctx):
    doc = _run(ctx, _doc(ctx, b"not a pdf"))
    assert doc.status == "failed" and doc.error == "file is not a readable PDF"


def test_encrypted_pdf_fails_visibly(ctx):
    doc = _run(ctx, _doc(ctx, encrypted_pdf()))
    assert doc.status == "failed" and doc.error == "file is not a readable PDF"


def test_ocr_unavailable_fails_visibly(ctx):
    class Missing:
        def ocr_pages(self, data, pages, languages):
            raise OcrUnavailable("OCR engine 'ocrmypdf' is not installed; scanned pages cannot be read")

    ctx.ocr_override = Missing()
    doc = _run(ctx, _doc(ctx, scanned_pdf(1)))
    assert doc.status == "failed" and "not installed" in doc.error


def test_unexpected_error_fails_and_reraises(ctx):
    class Broken:
        def ocr_pages(self, data, pages, languages):
            raise RuntimeError("disk on fire")

    ctx.ocr_override = Broken()
    doc_id = _doc(ctx, scanned_pdf(1))
    with pytest.raises(RuntimeError):
        ingest_document(ctx, ctx.snapshot, doc_id)
    with ctx.session_factory() as s:
        assert s.get(Document, doc_id).status == "failed"


def test_reingest_replaces_rows(ctx):
    doc_id = _doc(ctx, digital_pdf(["Alpha beta gamma. " * 30, "Delta epsilon. " * 30], [("Intro", 1)]))
    _run(ctx, doc_id)
    before = (_count(ctx, Page), _count(ctx, Passage), _count(ctx, Topic))
    _run(ctx, doc_id)
    assert (_count(ctx, Page), _count(ctx, Passage), _count(ctx, Topic)) == before


def test_ocr_language_from_config(test_env, monkeypatch):
    monkeypatch.setenv("TRIBUNAL_OCR_LANGUAGE", "hin")
    reset_settings_cache()
    fake = FakeOcrEngine({1: "page text " * 20})
    ctx = build_context(load_settings(), ocr_override=fake, migrate=True)
    languages = ctx.snapshot.config.ingestion.ocr.languages
    assert languages == ["hin"]
    doc = _run(ctx, _doc(ctx, scanned_pdf(1), languages))
    assert doc.status == "ready" and fake.calls[0][1] == ["hin"]


def test_extraction_version_changes_with_language(cfg):
    assert extraction_version(cfg, ["eng"], "e") != extraction_version(cfg, ["hin"], "e")
    assert extraction_version(cfg, ["eng"], "e") == extraction_version(cfg, ["eng"], "e")


def _limited(ctx, **limits):
    cfg = ctx.snapshot.config
    cfg = cfg.model_copy(update={"ingestion": cfg.ingestion.model_copy(update=limits)})
    return ConfigSnapshot(config=cfg, content_hash="x")


def test_over_size_limit_fails(ctx):
    doc_id = _doc(ctx, digital_pdf(["Alpha beta. " * 20]))
    ingest_document(ctx, _limited(ctx, max_upload_bytes=10), doc_id)
    with ctx.session_factory() as s:
        d = s.get(Document, doc_id)
    assert d.status == "failed" and d.error == "file is larger than the 10 byte limit"


def test_over_page_limit_fails(ctx):
    doc_id = _doc(ctx, digital_pdf(["Alpha beta. " * 20] * 3))
    ingest_document(ctx, _limited(ctx, max_pages=2), doc_id)
    with ctx.session_factory() as s:
        d = s.get(Document, doc_id)
    assert d.status == "failed" and d.error == "PDF has 3 pages; the limit is 2"


def test_zero_page_pdf_fails(ctx):
    doc = _run(ctx, _doc(ctx, zero_page_pdf()))
    assert doc.status == "failed" and doc.error == "file is not a readable PDF"
