"""Identity, subjects and their members."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from tribunal.api.auth import get_or_create_user
from tribunal.api.deps import DB, current_user, get_ctx, require_member
from tribunal.context import AppContext
from tribunal.db.models import Paper, PaperTemplate, Subject, SubjectMember, User
from tribunal.errors import Forbidden

router = APIRouter(prefix="/api")


class Named(BaseModel, extra="forbid"):
    name: str = Field(min_length=1, max_length=300)


class MemberIn(BaseModel, extra="forbid"):
    email: str = Field(min_length=3, max_length=320)


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict:
    return {"id": user.id, "email": user.email, "is_admin": user.is_admin}


@router.post("/subjects", status_code=201)
def create_subject(body: Named, user: User = Depends(current_user), session: Session = DB) -> dict:
    subject = Subject(name=body.name.strip(), owner_id=user.id)
    session.add(subject)
    session.flush()
    session.add(SubjectMember(subject_id=subject.id, user_id=user.id, role="owner"))
    return {"id": subject.id, "name": subject.name, "role": "owner"}


@router.get("/subjects")
def list_subjects(user: User = Depends(current_user), session: Session = DB) -> list[dict]:
    rows = session.execute(
        select(Subject, SubjectMember.role).join(SubjectMember, SubjectMember.subject_id == Subject.id)
        .where(SubjectMember.user_id == user.id).order_by(Subject.created_at, Subject.id)
    )
    return [{"id": s.id, "name": s.name, "role": role} for s, role in rows]


@router.post("/subjects/{sid}/members", status_code=201)
def add_member(sid: str, body: MemberIn, user: User = Depends(current_user), session: Session = DB,
               ctx: AppContext = Depends(get_ctx)) -> dict:
    subject = require_member(session, sid, user)
    mine = session.get(SubjectMember, (sid, user.id))
    if not (mine is not None and mine.role == "owner") and not user.is_admin:
        raise Forbidden("only the subject owner can add members")
    member = get_or_create_user(session, ctx, body.email)
    if session.get(SubjectMember, (sid, member.id)) is None:
        session.add(SubjectMember(subject_id=subject.id, user_id=member.id, role="member"))
    return {"subject_id": sid, "email": member.email}


@router.get("/subjects/{sid}/papers")
def list_papers(sid: str, user: User = Depends(current_user), session: Session = DB) -> list[dict]:
    require_member(session, sid, user)
    rows = session.execute(
        select(Paper, PaperTemplate).join(PaperTemplate, PaperTemplate.id == Paper.template_id)
        .where(Paper.subject_id == sid).order_by(Paper.created_at.desc(), Paper.id.desc()))
    return [{"id": p.id, "title": p.title, "status": p.status, "created_at": p.created_at,
             "template": {"name": t.name, "version": t.version}} for p, t in rows]


@router.get("/options")
def options(_: User = Depends(current_user), ctx: AppContext = Depends(get_ctx)) -> dict:
    cfg = ctx.snapshot.config
    return {
        "question_types": [{"key": k, "label": v.label} for k, v in cfg.generation.question_types.items() if v.enabled],
        "bloom_levels": [{"key": b.key, "label": b.label} for b in cfg.generation.bloom_levels],
        "difficulty_levels": [{"key": d.key, "label": d.label} for d in cfg.generation.difficulty_levels],
        "default_ocr_languages": cfg.ingestion.ocr.languages,
    }
