"""Request dependencies: context, transaction, signed-in user, subject membership."""

from __future__ import annotations

from collections.abc import Iterator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from tribunal.api.auth import authenticate
from tribunal.context import AppContext
from tribunal.db.models import Subject, SubjectMember, User
from tribunal.errors import NotFound


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_session(ctx: AppContext = Depends(get_ctx)) -> Iterator[Session]:
    with ctx.session_factory.begin() as session:
        yield session


def current_user(request: Request, ctx: AppContext = Depends(get_ctx), session: Session = Depends(get_session)) -> User:
    return authenticate(request, ctx, session)


def require_member(session: Session, subject_id: str, user: User) -> Subject:
    subject = session.get(Subject, subject_id)
    if subject is not None and (user.is_admin or session.get(SubjectMember, (subject_id, user.id)) is not None):
        return subject
    raise NotFound("subject not found")  # same answer for unknown and foreign subjects
