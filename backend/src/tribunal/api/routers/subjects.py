"""Identity, subjects and their members."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from tribunal.api.auth import get_or_create_user
from tribunal.api.deps import DB, current_user, get_ctx, require_member
from tribunal.context import AppContext
from tribunal.db.models import Subject, SubjectMember, User
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
