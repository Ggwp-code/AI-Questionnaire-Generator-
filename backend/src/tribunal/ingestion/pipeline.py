"""Document ingestion: read, OCR where needed, chunk, embed, outline."""

from __future__ import annotations

from importlib.metadata import version

from sqlalchemy import delete, select

from tribunal.context import AppContext
from tribunal.db.models import Document, Page, Passage, Topic
from tribunal.db.types import utcnow
from tribunal.errors import TribunalError, ValidationFailed
from tribunal.ingestion.chunking import chunk_pages
from tribunal.ingestion.outline import build_outline
from tribunal.ingestion.pdf import read_pdf, read_toc
from tribunal.runtime_config import ConfigSnapshot, RuntimeConfig, canonical_hash


def extraction_version(cfg: RuntimeConfig, languages: list[str], embedder_id: str) -> str:
    ing = cfg.ingestion
    return "x-" + canonical_hash({
        "parser": "pypdfium2", "parser_version": version("pypdfium2"),
        "ocr_engine": ing.ocr.engine, "languages": languages,
        "rotate_pages": ing.ocr.rotate_pages, "deskew": ing.ocr.deskew,
        "native_min_chars": ing.native_min_chars, "mixed_image_coverage": ing.mixed_image_coverage,
        "chunk_chars": cfg.retrieval.chunk_chars, "chunk_overlap_chars": cfg.retrieval.chunk_overlap_chars,
        "embedder": embedder_id,
    })[:12]


def _fail(ctx: AppContext, document_id: str, error: str) -> None:
    with ctx.session_factory.begin() as session:
        doc = session.get(Document, document_id)
        doc.status = "failed"
        doc.error = error


def ingest_document(ctx: AppContext, snapshot: ConfigSnapshot, document_id: str) -> None:
    with ctx.session_factory() as session:
        doc = session.get(Document, document_id)
        storage_key, languages, subject_id = doc.storage_key, list(doc.ocr_languages), doc.subject_id
        ready_version = doc.extraction_version if doc.status == "ready" else None
    try:
        xv = extraction_version(snapshot.config, languages, ctx.embedder(snapshot).model_id)
        if xv == ready_version:
            return  # this index version is already built; cited passages stay as they are
        with ctx.session_factory.begin() as session:
            doc = session.get(Document, document_id)
            doc.status, doc.error = "processing", None
        _ingest(ctx, snapshot, document_id, storage_key, languages, subject_id, xv)
    except TribunalError as exc:
        _fail(ctx, document_id, exc.message)
    except Exception as exc:
        _fail(ctx, document_id, f"unexpected error: {exc}")
        raise


def _ingest(ctx, snapshot, document_id, storage_key, languages, subject_id, xv) -> None:
    cfg = snapshot.config
    ing = cfg.ingestion
    data = ctx.store.get(storage_key)
    if len(data) > ing.max_upload_bytes:
        raise ValidationFailed(f"file is larger than the {ing.max_upload_bytes} byte limit")
    infos = read_pdf(data, ing.max_pages)
    texts = {p.page_number: p.text for p in infos}
    methods = {p.page_number: "native" for p in infos}
    # An image-heavy page that still has native text keeps it (native+ocr), even when short.
    mixed = [p.page_number for p in infos if p.text.strip() and p.image_coverage >= ing.mixed_image_coverage]
    need_ocr = [p.page_number for p in infos
                if p.page_number not in mixed and len(p.text.strip()) < ing.native_min_chars]
    if need_ocr or mixed:
        wanted = sorted(need_ocr + mixed)
        ocr_text = ctx.ocr(snapshot).ocr_pages(data, wanted, languages)
        for n in need_ocr:
            texts[n], methods[n] = ocr_text.get(n, ""), "ocr"
        for n in mixed:
            texts[n], methods[n] = texts[n] + "\n" + ocr_text.get(n, ""), "native+ocr"

    embedder = ctx.embedder(snapshot)
    chunks = chunk_pages(texts, cfg.retrieval.chunk_chars, cfg.retrieval.chunk_overlap_chars)
    vectors = embedder.embed([c.text for c in chunks]) if chunks else []
    outline = build_outline(read_toc(data), texts, ing.outline, len(infos))

    with ctx.session_factory.begin() as session:
        # Passages and topics are never deleted: citations and slots reference them.
        session.execute(delete(Page).where(Page.document_id == document_id))
        for n in sorted(texts):
            session.add(Page(document_id=document_id, page_number=n, method=methods[n],
                             char_count=len(texts[n]), text=texts[n]))
        built = session.scalar(select(Passage.id).where(Passage.document_id == document_id,
                                                        Passage.extraction_version == xv).limit(1))
        if built is None:  # otherwise an earlier run built this version: reuse it, never duplicate it
            for c, v in zip(chunks, vectors, strict=True):
                session.add(Passage(document_id=document_id, subject_id=subject_id, page_number=c.page_number,
                                    char_start=c.char_start, char_end=c.char_end, text=c.text,
                                    embedding=v, extraction_version=xv))
        existing: dict[str, list[Topic]] = {}
        for t in session.scalars(select(Topic).where(Topic.document_id == document_id, Topic.source != "faculty")
                                 .order_by(Topic.page_start, Topic.created_at)):
            existing.setdefault(t.title.lower(), []).append(t)
        for e in outline:
            same = existing.get(e.title.lower())
            if same:
                topic = same.pop(0)
                topic.page_start, topic.page_end = e.page_start, e.page_end
            else:
                session.add(Topic(subject_id=subject_id, document_id=document_id, title=e.title,
                                  page_start=e.page_start, page_end=e.page_end, source=e.source))
        doc = session.get(Document, document_id)
        doc.page_count, doc.extraction_version = len(infos), xv
        doc.status, doc.ready_at, doc.error = "ready", utcnow(), None
