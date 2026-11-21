"""ORM models."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from tribunal.db.types import Embedding, JsonType, UTCDateTime, new_id, utcnow


class Base(DeclarativeBase):
    pass


def _id():
    return mapped_column(String(32), primary_key=True, default=new_id)


def _fk(target: str, **kw):
    return mapped_column(String(32), ForeignKey(target), **kw)


def _created():
    return mapped_column(UTCDateTime, default=utcnow)


def _opt_time():
    return mapped_column(UTCDateTime, nullable=True)


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = _id()
    email: Mapped[str] = mapped_column(String(320), unique=True)
    display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_admin: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = _created()


class Subject(Base):
    __tablename__ = "subjects"
    id: Mapped[str] = _id()
    name: Mapped[str] = mapped_column(String(300))
    owner_id: Mapped[str] = _fk("users.id")
    created_at: Mapped[datetime] = _created()


class SubjectMember(Base):
    __tablename__ = "subject_members"
    subject_id: Mapped[str] = _fk("subjects.id", primary_key=True)
    user_id: Mapped[str] = _fk("users.id", primary_key=True)
    role: Mapped[str] = mapped_column(String(20))


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("subject_id", "content_hash"),)
    id: Mapped[str] = _id()
    subject_id: Mapped[str] = _fk("subjects.id")
    uploaded_by: Mapped[str] = _fk("users.id")
    filename: Mapped[str] = mapped_column(String(500))
    content_hash: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(500))
    size_bytes: Mapped[int]
    page_count: Mapped[int | None] = mapped_column(nullable=True)
    ocr_languages: Mapped[list] = mapped_column(JsonType)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = _created()
    ready_at: Mapped[datetime | None] = _opt_time()


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("document_id", "page_number"),)
    id: Mapped[str] = _id()
    document_id: Mapped[str] = _fk("documents.id")
    page_number: Mapped[int]
    method: Mapped[str] = mapped_column(String(20))
    char_count: Mapped[int]
    text: Mapped[str] = mapped_column(Text)


class Passage(Base):
    __tablename__ = "passages"
    id: Mapped[str] = _id()
    document_id: Mapped[str] = _fk("documents.id", index=True)
    subject_id: Mapped[str] = _fk("subjects.id", index=True)
    page_number: Mapped[int]
    char_start: Mapped[int]
    char_end: Mapped[int]
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[object] = mapped_column(Embedding)
    extraction_version: Mapped[str] = mapped_column(String(64))


class Topic(Base):
    __tablename__ = "topics"
    id: Mapped[str] = _id()
    subject_id: Mapped[str] = _fk("subjects.id")
    document_id: Mapped[str | None] = _fk("documents.id", nullable=True)
    title: Mapped[str] = mapped_column(String(500))
    page_start: Mapped[int | None] = mapped_column(nullable=True)
    page_end: Mapped[int | None] = mapped_column(nullable=True)
    source: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = _created()


class PaperTemplate(Base):
    __tablename__ = "paper_templates"
    __table_args__ = (UniqueConstraint("owner_id", "name", "version"),)
    id: Mapped[str] = _id()
    owner_id: Mapped[str] = _fk("users.id")
    name: Mapped[str] = mapped_column(String(300))
    version: Mapped[int]
    spec: Mapped[dict] = mapped_column(JsonType)
    created_at: Mapped[datetime] = _created()


class ConfigSnapshotRow(Base):
    __tablename__ = "config_snapshots"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    content: Mapped[dict] = mapped_column(JsonType)
    created_at: Mapped[datetime] = _created()


class Paper(Base):
    __tablename__ = "papers"
    id: Mapped[str] = _id()
    subject_id: Mapped[str] = _fk("subjects.id")
    created_by: Mapped[str] = _fk("users.id")
    template_id: Mapped[str] = _fk("paper_templates.id")
    title: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    config_snapshot_id: Mapped[str] = mapped_column(String(64), ForeignKey("config_snapshots.id"))
    document_ids: Mapped[list] = mapped_column(JsonType)
    topic_ids: Mapped[list] = mapped_column(JsonType)
    tokens_used: Mapped[int] = mapped_column(default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created()
    started_at: Mapped[datetime | None] = _opt_time()
    first_question_at: Mapped[datetime | None] = _opt_time()
    completed_at: Mapped[datetime | None] = _opt_time()


class Slot(Base):
    __tablename__ = "slots"
    __table_args__ = (UniqueConstraint("paper_id", "number"),)
    id: Mapped[str] = _id()
    paper_id: Mapped[str] = _fk("papers.id")
    number: Mapped[int]
    section: Mapped[str] = mapped_column(String(100))
    question_type: Mapped[str] = mapped_column(String(50))
    marks: Mapped[int]
    bloom: Mapped[str] = mapped_column(String(50))
    difficulty: Mapped[str] = mapped_column(String(50))
    topic_id: Mapped[str | None] = _fk("topics.id", nullable=True)
    topic_title: Mapped[str] = mapped_column(String(500))
    status: Mapped[str] = mapped_column(String(20), default="pending")
    status_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_revision_id: Mapped[str | None] = mapped_column(String(32), nullable=True)  # no FK: cycle


class Revision(Base):
    __tablename__ = "revisions"
    __table_args__ = (UniqueConstraint("slot_id", "number"),)
    id: Mapped[str] = _id()
    slot_id: Mapped[str] = _fk("slots.id")
    number: Mapped[int]
    author: Mapped[str] = mapped_column(String(20))
    content: Mapped[dict] = mapped_column(JsonType)
    review: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    retrieval_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    created_by: Mapped[str | None] = _fk("users.id", nullable=True)
    approved_by: Mapped[str | None] = _fk("users.id", nullable=True)
    approved_at: Mapped[datetime | None] = _opt_time()
    created_at: Mapped[datetime] = _created()


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = _id()
    kind: Mapped[str] = mapped_column(String(30))
    target_id: Mapped[str] = mapped_column(String(32))
    config_snapshot_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default="queued")
    attempts: Mapped[int] = mapped_column(default=0)
    reserved_tokens: Mapped[int] = mapped_column(default=0, server_default='0')
    lease_until: Mapped[datetime | None] = _opt_time()
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created()
    finished_at: Mapped[datetime | None] = _opt_time()
