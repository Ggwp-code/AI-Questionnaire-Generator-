"""Document upload and download, topics and passages."""

from __future__ import annotations

import hashlib

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from tribunal.api.deps import DB, current_user, get_ctx, require_member
from tribunal.context import AppContext
from tribunal.db.models import Document, Page, Passage, Topic, User
from tribunal.db.types import new_id
from tribunal.errors import Conflict, NotFound, TribunalError, ValidationFailed
from tribunal.jobs.queue import enqueue

router = APIRouter(prefix="/api")


class UploadTooLarge(TribunalError):
    code = "upload_too_large"
    status = 413


class TitleIn(BaseModel, extra="forbid"):
    title: str = Field(min_length=1, max_length=500)


def _doc_json(d: Document, excluded: list[int] = ()) -> dict:
    # Upload to searchable: from the accepted upload to the index becoming ready.
    searchable = (d.ready_at - d.created_at).total_seconds() if d.ready_at else None
    return {"id": d.id, "filename": d.filename, "status": d.status, "error": d.error,
            "page_count": d.page_count, "created_at": d.created_at, "ready_at": d.ready_at,
            "searchable_seconds": searchable, "excluded_pages": list(excluded)}


def _topic_json(t: Topic) -> dict:
    return {"id": t.id, "title": t.title, "document_id": t.document_id, "page_start": t.page_start,
            "page_end": t.page_end, "source": t.source}


def page_url(document_id: str, page: int) -> str:
    return f"/api/documents/{document_id}/file#page={page}"


@router.post("/subjects/{sid}/documents", status_code=202)
def upload(sid: str, file: UploadFile = File(...), languages: str | None = Form(None),
           user: User = Depends(current_user), session: Session = DB,
           ctx: AppContext = Depends(get_ctx)) -> dict:
    require_member(session, sid, user)
    ing = ctx.snapshot.config.ingestion
    data = file.file.read(ing.max_upload_bytes + 1)
    if len(data) > ing.max_upload_bytes:
        raise UploadTooLarge(f"file is larger than {ing.max_upload_bytes} bytes")
    if not data.startswith(b"%PDF-"):
        raise ValidationFailed("file is not a PDF", details=["the file does not start with %PDF-"])
    digest = hashlib.sha256(data).hexdigest()
    if session.scalar(select(Document.id).where(Document.subject_id == sid, Document.content_hash == digest)):
        raise Conflict("this file was already uploaded to the subject")
    langs = [x.strip() for x in languages.split(",") if x.strip()] if languages else []
    doc_id = new_id()
    key = f"documents/{sid}/{doc_id}.pdf"
    ctx.store.put(key, data)
    doc = Document(id=doc_id, subject_id=sid, uploaded_by=user.id, filename=(file.filename or "upload.pdf")[:500],
                   content_hash=digest, storage_key=key, size_bytes=len(data),
                   ocr_languages=langs or list(ing.ocr.languages))
    session.add(doc)
    session.flush()
    job = enqueue(session, "ingest", doc.id, ctx.snapshot.snapshot_id)
    return {"document": _doc_json(doc), "job_id": job.id}


@router.post("/documents/{did}/reprocess", status_code=202)
def reprocess(did: str, user: User = Depends(current_user), session: Session = DB,
              ctx: AppContext = Depends(get_ctx)) -> dict:
    doc = session.get(Document, did)
    if doc is None:
        raise NotFound("document not found")
    require_member(session, doc.subject_id, user)
    if doc.status in ("queued", "processing"):
        raise Conflict("document is already being processed")
    doc.status = "queued"
    job = enqueue(session, "ingest", doc.id, ctx.snapshot.snapshot_id)
    return {"document": _doc_json(doc), "job_id": job.id}


@router.get("/subjects/{sid}/documents")
def list_documents(sid: str, user: User = Depends(current_user), session: Session = DB) -> list[dict]:
    require_member(session, sid, user)
    docs = session.scalars(select(Document).where(Document.subject_id == sid).order_by(Document.created_at, Document.id))
    docs = list(docs)
    excluded: dict[str, list[int]] = {}
    for did, n in session.execute(select(Page.document_id, Page.page_number)
                                  .where(Page.document_id.in_([d.id for d in docs]), Page.excluded)
                                  .order_by(Page.page_number)):
        excluded.setdefault(did, []).append(n)
    return [_doc_json(d, excluded.get(d.id, [])) for d in docs]


@router.get("/documents/{did}/file")
def download(did: str, user: User = Depends(current_user), session: Session = DB,
             ctx: AppContext = Depends(get_ctx)) -> Response:
    doc = session.get(Document, did)
    if doc is None:
        raise NotFound("document not found")
    require_member(session, doc.subject_id, user)
    return Response(ctx.store.get(doc.storage_key), media_type="application/pdf")


@router.get("/subjects/{sid}/topics")
def list_topics(sid: str, user: User = Depends(current_user), session: Session = DB) -> list[dict]:
    require_member(session, sid, user)
    return [_topic_json(t) for t in session.scalars(
        select(Topic).where(Topic.subject_id == sid, Topic.retired_at.is_(None)).order_by(Topic.created_at, Topic.page_start, Topic.id))]


@router.post("/subjects/{sid}/topics", status_code=201)
def create_topic(sid: str, body: TitleIn, user: User = Depends(current_user),
                 session: Session = DB) -> dict:
    require_member(session, sid, user)
    topic = Topic(subject_id=sid, title=body.title.strip(), source="faculty")
    session.add(topic)
    session.flush()
    return _topic_json(topic)


@router.patch("/topics/{tid}")
def rename_topic(tid: str, body: TitleIn, user: User = Depends(current_user),
                 session: Session = DB) -> dict:
    topic = session.get(Topic, tid)
    if topic is None:
        raise NotFound("topic not found")
    require_member(session, topic.subject_id, user)
    # A renamed topic is faculty-owned from now on, so an outline rebuild never retires it.
    topic.title, topic.source = body.title.strip(), "faculty"
    return _topic_json(topic)


@router.get("/passages/{pid}")
def get_passage(pid: str, user: User = Depends(current_user), session: Session = DB) -> dict:
    passage = session.get(Passage, pid)
    if passage is None:
        raise NotFound("passage not found")
    require_member(session, passage.subject_id, user)
    doc = session.get(Document, passage.document_id)
    return {"id": passage.id, "document_id": doc.id, "filename": doc.filename, "page_number": passage.page_number,
            "text": passage.text, "url": page_url(doc.id, passage.page_number)}
