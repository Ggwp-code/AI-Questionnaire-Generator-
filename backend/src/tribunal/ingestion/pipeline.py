"""Document ingestion: read, OCR where needed, chunk, embed, outline."""

from __future__ import annotations

import re
from importlib.metadata import version

from sqlalchemy import delete, select

from tribunal.context import AppContext
from tribunal.db.models import Document, Page, Passage, Topic
from tribunal.db.types import utcnow
from tribunal.errors import TribunalError, ValidationFailed
from tribunal.ingestion.chunking import chunk_pages
from tribunal.ingestion.outline import build_outline
from tribunal.ingestion.pdf import read_pdf, read_toc
from tribunal.jobs.queue import ClaimedJob, owned_transaction
from tribunal.runtime_config import ConfigSnapshot, RuntimeConfig, canonical_hash

FLAG_RULES = 2  # bump when page_flags changes, so documents are extracted again
_FORMULA = re.compile(
    r"[=<>\u2212≠≈≤≥±×÷√∑∏∫∂∇∞^²³¹⁰-₟]"
    r"|\w[)\]]? [+/] [\w(]"  # spaced plus or slash between operands
    r"|\w\*\w"
    r"|(?<![\w-])[A-Za-z0-9] - [A-Za-z0-9](?![\w-])")  # spaced hyphen only between single-character operands
_GREEK = re.compile("[\u0391-\u03a9\u03b1-\u03c9]")
_GREEK_LANGUAGES = {"ell", "grc"}  # in these books Greek letters are ordinary text
_CELL_GAP = re.compile(r" {2,}|\t|\|")


def page_flags(text: str, languages: list[str] | tuple = ()) -> set[str]:
    """Kinds of content whose OCR is unreliable enough to keep out of evidence."""
    # shortcut: symbol and column-gap heuristics, replace with layout-parser regions when one is configured
    greek = _GREEK.search(text) and not _GREEK_LANGUAGES & set(languages)
    flags = {"formula"} if _FORMULA.search(text) or greek else set()
    rows = sum(len([c for c in _CELL_GAP.split(line) if c.strip()]) >= 3 for line in text.splitlines())
    return flags | {"table"} if rows >= 3 else flags


def extraction_version(cfg: RuntimeConfig, languages: list[str], embedder_id: str) -> str:
    ing = cfg.ingestion
    return "x-" + canonical_hash({
        "parser": "pypdfium2", "parser_version": version("pypdfium2"),
        "ocr_engine": ing.ocr.engine, "languages": languages,
        "rotate_pages": ing.ocr.rotate_pages, "deskew": ing.ocr.deskew,
        "native_min_chars": ing.native_min_chars, "mixed_image_coverage": ing.mixed_image_coverage,
        "chunk_chars": cfg.retrieval.chunk_chars, "chunk_overlap_chars": cfg.retrieval.chunk_overlap_chars,
        "embedder": embedder_id,
        "flag_rules": FLAG_RULES, "flag_formulas_on": sorted(ing.flag_formulas_on),
        "flag_tables_on": sorted(ing.flag_tables_on), "flagged_page_policy": ing.flagged_page_policy,
        "outline": ing.outline.model_dump(mode="json"),
    })[:12]


def _fail(ctx: AppContext, document_id: str, error: str) -> None:
    with ctx.session_factory.begin() as session:
        doc = session.get(Document, document_id)
        doc.status = "ready" if doc.extraction_version else "failed"
        doc.error = error


def ingest_document(ctx: AppContext, snapshot: ConfigSnapshot, document_id: str, *,
                    owner: ClaimedJob | None = None) -> None:
    with ctx.session_factory() as session:
        doc = session.get(Document, document_id)
        storage_key, languages, subject_id = doc.storage_key, list(doc.ocr_languages), doc.subject_id
    try:
        xv = extraction_version(snapshot.config, languages, ctx.embedder(snapshot).model_id)
        with owned_transaction(ctx.session_factory, owner) as session:
            doc = session.get(Document, document_id)
            doc.error = None
            doc.status = "ready" if doc.extraction_version else "processing"
            if xv == doc.extraction_version:
                doc.embedder_id = ctx.embedder(snapshot).model_id  # the version includes it: backfills 0005
                return
        _ingest(ctx, snapshot, document_id, storage_key, languages, subject_id, xv, owner)
    except TribunalError as exc:
        if owner is not None:
            raise
        _fail(ctx, document_id, exc.message)
    except Exception as exc:
        if owner is not None:
            raise
        _fail(ctx, document_id, f"unexpected error: {exc}")
        raise


def _ingest(ctx, snapshot, document_id, storage_key, languages, subject_id, xv, owner) -> None:
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

    flagged_on = {"formula": set(ing.flag_formulas_on), "table": set(ing.flag_tables_on)}
    excluded = sorted(n for n in texts if ing.flagged_page_policy == "exclude" and any(
        flagged_on[kind] & set(methods[n].split("+")) for kind in page_flags(texts[n], languages)))

    embedder = ctx.embedder(snapshot)
    evidence = {n: t for n, t in texts.items() if n not in excluded}
    chunks = chunk_pages(evidence, cfg.retrieval.chunk_chars, cfg.retrieval.chunk_overlap_chars)
    vectors = embedder.embed([c.text for c in chunks]) if chunks else []
    outline = build_outline(read_toc(data), texts, ing.outline, len(infos))

    with owned_transaction(ctx.session_factory, owner) as session:
        # Passages and topics are never deleted: citations and slots reference them.
        session.execute(delete(Page).where(Page.document_id == document_id))
        for n in sorted(texts):
            session.add(Page(document_id=document_id, page_number=n, method=methods[n],
                             char_count=len(texts[n]), text=texts[n], excluded=n in excluded))
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
                topic.page_start, topic.page_end, topic.retired_at = e.page_start, e.page_end, None
            else:
                session.add(Topic(subject_id=subject_id, document_id=document_id, title=e.title,
                                  page_start=e.page_start, page_end=e.page_end, source=e.source))
        for topic in (t for left in existing.values() for t in left):  # no longer in this outline
            topic.retired_at = topic.retired_at or utcnow()
        doc = session.get(Document, document_id)
        doc.page_count, doc.extraction_version = len(infos), xv
        doc.embedder_id = embedder.model_id
        doc.status, doc.ready_at, doc.error = "ready", utcnow(), None
