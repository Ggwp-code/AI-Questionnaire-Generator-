"""Templates, papers, slot editing and approval."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from tribunal.api.deps import DB, current_user, get_ctx, require_member
from tribunal.api.routers.documents import page_url
from tribunal.context import AppContext
from tribunal.db.models import Document, Paper, PaperTemplate, Passage, Revision, Slot, Topic, User
from tribunal.db.types import utcnow
from tribunal.errors import Conflict, NotFound, ValidationFailed
from tribunal.export.render import build_export, render_pdf
from tribunal.generation.blueprint import expand_blueprint, parse_template_spec
from tribunal.generation.draft import Draft, check_draft
from tribunal.jobs.queue import enqueue

router = APIRouter(prefix="/api")


class TemplateIn(BaseModel, extra="forbid"):
    name: str = Field(min_length=1, max_length=300)
    spec: dict


class PaperIn(BaseModel, extra="forbid"):
    subject_id: str
    template_id: str
    topic_ids: list[str]
    document_ids: list[str] | None = None
    title: str = Field(min_length=1, max_length=500)


class EditIn(BaseModel, extra="forbid"):
    content: dict


def _tpl_json(t: PaperTemplate) -> dict:
    return {"id": t.id, "name": t.name, "version": t.version, "spec": t.spec}


@router.post("/templates", status_code=201)
def create_template(body: TemplateIn, user: User = Depends(current_user), session: Session = DB,
                    ctx: AppContext = Depends(get_ctx)) -> dict:
    expand_blueprint(parse_template_spec(body.spec), ["probe"], ctx.snapshot.config)  # surface problems now
    latest = session.scalar(select(func.max(PaperTemplate.version)).where(
        PaperTemplate.owner_id == user.id, PaperTemplate.name == body.name)) or 0
    tpl = PaperTemplate(owner_id=user.id, name=body.name, version=latest + 1, spec=body.spec)
    session.add(tpl)
    session.flush()
    return _tpl_json(tpl)


@router.get("/templates")
def list_templates(user: User = Depends(current_user), session: Session = DB) -> list[dict]:
    return [_tpl_json(t) for t in session.scalars(
        select(PaperTemplate).where(PaperTemplate.owner_id == user.id)
        .order_by(PaperTemplate.name, PaperTemplate.version))]


@router.post("/papers", status_code=202)
def create_paper(body: PaperIn, user: User = Depends(current_user), session: Session = DB,
                 ctx: AppContext = Depends(get_ctx)) -> dict:
    require_member(session, body.subject_id, user)
    tpl = session.get(PaperTemplate, body.template_id)
    if tpl is None or tpl.owner_id != user.id:
        raise NotFound("template not found")
    topic_ids = list(dict.fromkeys(body.topic_ids))
    topics = {t.id: t for t in session.scalars(
        select(Topic).where(Topic.id.in_(topic_ids), Topic.subject_id == body.subject_id,
                            Topic.retired_at.is_(None)))}
    if len(topics) != len(topic_ids):
        raise ValidationFailed("unknown topic", details=["every topic must be a current topic of the subject"])
    ready = {d.id: d for d in session.scalars(select(Document).where(
        Document.subject_id == body.subject_id, Document.status == "ready").order_by(Document.created_at))}
    if body.document_ids is None:
        doc_ids = list(ready)
    else:
        doc_ids = list(dict.fromkeys(body.document_ids))
        if any(d not in ready for d in doc_ids):
            raise ValidationFailed("unknown document", details=["every document must be a ready document of the subject"])
    if not doc_ids:
        raise ValidationFailed("the subject has no ready documents")
    specs = expand_blueprint(parse_template_spec(tpl.spec), topic_ids, ctx.snapshot.config)
    paper = Paper(subject_id=body.subject_id, created_by=user.id, template_id=tpl.id, title=body.title.strip(),
                  config_snapshot_id=ctx.snapshot.snapshot_id, document_ids=doc_ids, topic_ids=topic_ids)
    session.add(paper)
    session.flush()
    for sp in specs:
        session.add(Slot(paper_id=paper.id, number=sp.number, section=sp.section, question_type=sp.question_type,
                         marks=sp.marks, bloom=sp.bloom, difficulty=sp.difficulty, topic_id=sp.topic_id,
                         topic_title=topics[sp.topic_id].title))
    job = enqueue(session, "generate_paper", paper.id, ctx.snapshot.snapshot_id)
    return {"paper_id": paper.id, "job_id": job.id}


def load_paper(session: Session, user: User, pid: str) -> Paper:
    paper = session.get(Paper, pid)
    if paper is None:
        raise NotFound("paper not found")
    require_member(session, paper.subject_id, user)
    return paper


def _slot(session: Session, paper: Paper, number: int) -> Slot:
    slot = session.scalar(select(Slot).where(Slot.paper_id == paper.id, Slot.number == number))
    if slot is None:
        raise NotFound("slot not found")
    return slot


def _rev_json(r: Revision) -> dict:
    return {"id": r.id, "number": r.number, "author": r.author, "content": r.content, "review": r.review,
            "model": r.model, "prompt_version": r.prompt_version, "approved_by": r.approved_by,
            "approved_at": r.approved_at}


def _seconds(start: datetime, end: datetime | None) -> float | None:
    return None if end is None else (end - start).total_seconds()


@router.get("/papers/{pid}")
def get_paper(pid: str, user: User = Depends(current_user), session: Session = DB) -> dict:
    paper = load_paper(session, user, pid)
    tpl = session.get(PaperTemplate, paper.template_id)
    slots = session.scalars(select(Slot).where(Slot.paper_id == pid).order_by(Slot.number)).all()
    revs = {r.id: r for r in session.scalars(select(Revision).where(
        Revision.id.in_([s.current_revision_id for s in slots if s.current_revision_id])))}
    cited = {c for r in revs.values() for c in r.content.get("citations", [])}
    passages = {p.id: p for p in session.scalars(select(Passage).where(
        Passage.id.in_(cited), Passage.document_id.in_(paper.document_ids)))}
    out = []
    for s in slots:
        rev = revs.get(s.current_revision_id)
        cites = [{"passage_id": c, "document_id": passages[c].document_id, "page_number": passages[c].page_number,
                  "url": page_url(passages[c].document_id, passages[c].page_number)}
                 for c in (rev.content.get("citations", []) if rev else []) if c in passages]
        out.append({"id": s.id, "number": s.number, "section": s.section, "question_type": s.question_type,
                    "marks": s.marks, "topic_title": s.topic_title, "status": s.status,
                    "status_detail": s.status_detail, "revision": _rev_json(rev) if rev else None,
                    "citations": cites})
    return {"id": paper.id, "title": paper.title, "status": paper.status, "error": paper.error,
            "subject_id": paper.subject_id, "template": {"id": tpl.id, "name": tpl.name, "version": tpl.version},
            "config_snapshot_id": paper.config_snapshot_id, "document_ids": paper.document_ids,
            "tokens_used": paper.tokens_used,
            "timings": {"first_question_seconds": _seconds(paper.created_at, paper.first_question_at),
                        "total_seconds": _seconds(paper.created_at, paper.completed_at)},
            "slots": out}


@router.post("/papers/{pid}/slots/{n}/regenerate", status_code=202)
def regenerate(pid: str, n: int, user: User = Depends(current_user), session: Session = DB,
               ctx: AppContext = Depends(get_ctx)) -> dict:
    slot = _slot(session, load_paper(session, user, pid), n)
    if slot.status in ("pending", "running"):
        raise Conflict("the slot is already being generated")
    rev = session.get(Revision, slot.current_revision_id) if slot.current_revision_id else None
    if rev is not None:
        rev.approved_by = rev.approved_at = None
    slot.status, slot.status_detail = "pending", None
    return {"job_id": enqueue(session, "regenerate_slot", slot.id, ctx.snapshot.snapshot_id).id}


@router.put("/papers/{pid}/slots/{n}")
def edit_slot(pid: str, n: int, body: EditIn, user: User = Depends(current_user),
              session: Session = DB, ctx: AppContext = Depends(get_ctx)) -> dict:
    paper = load_paper(session, user, pid)
    slot = _slot(session, paper, n)
    if slot.status in ("pending", "running"):
        raise Conflict("the slot is being generated")
    snap = ctx.snapshot_for(paper.config_snapshot_id)
    content, _ = ctx.policy(snap).clean_value(body.content)
    try:
        draft = Draft.model_validate(content)
    except ValidationError as exc:
        raise ValidationFailed("Draft is invalid", details=[
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]) from None
    g = snap.config.generation
    problems = check_draft(
        draft, marks=slot.marks, question_type=slot.question_type,
        passage_ids=set(session.scalars(select(Passage.id).where(Passage.document_id.in_(paper.document_ids)))),
        placeholder_patterns=g.placeholder_patterns, mcq_option_count=g.mcq_option_count)
    if problems:
        raise ValidationFailed("Draft is invalid", details=problems)
    number = (session.scalar(select(func.max(Revision.number)).where(Revision.slot_id == slot.id)) or 0) + 1
    rev = Revision(slot_id=slot.id, number=number, author="faculty", content=draft.model_dump(), created_by=user.id)
    session.add(rev)
    session.flush()
    slot.current_revision_id, slot.status, slot.status_detail = rev.id, "drafted", None
    return _rev_json(rev)


@router.post("/papers/{pid}/slots/{n}/approve")
def approve(pid: str, n: int, user: User = Depends(current_user), session: Session = DB) -> dict:
    slot = _slot(session, load_paper(session, user, pid), n)
    if slot.status not in ("drafted", "needs_review") or not slot.current_revision_id:
        raise Conflict("only a drafted or needs_review slot with a revision can be approved")
    rev = session.get(Revision, slot.current_revision_id)
    rev.approved_by, rev.approved_at = user.id, utcnow()
    slot.status = "approved"
    return {"slot_id": slot.id, "number": slot.number, "status": slot.status, "revision": _rev_json(rev)}


@router.get("/papers/{pid}/export")
def export(pid: str, part: Literal["paper", "key"], format: Literal["json", "pdf"] = "json",
           user: User = Depends(current_user), session: Session = DB,
           ctx: AppContext = Depends(get_ctx)):
    paper = load_paper(session, user, pid)
    snap = ctx.snapshot_for(paper.config_snapshot_id)
    doc = build_export(session, paper, part, snap, ctx.policy(snap))
    if format == "json":
        return doc
    st = ctx.settings
    return Response(render_pdf(doc, font_regular=st.pdf_font_regular, font_bold=st.pdf_font_bold,
                               fallbacks=st.pdf_font_fallbacks, cfg=snap.config.export), media_type="application/pdf")
