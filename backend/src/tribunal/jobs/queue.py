"""Database-backed job queue."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from tribunal.db.models import Job
from tribunal.db.types import utcnow


@dataclass
class ClaimedJob:
    id: str
    kind: str
    target_id: str
    config_snapshot_id: str
    attempts: int


def enqueue(session: Session, kind: str, target_id: str, snapshot_id: str) -> Job:
    job = Job(kind=kind, target_id=target_id, config_snapshot_id=snapshot_id)
    session.add(job)
    session.flush()
    return job


def claim_next(factory: sessionmaker[Session], lease_seconds: int, now: datetime | None = None) -> ClaimedJob | None:
    now = now or utcnow()
    with factory.begin() as session:
        job = session.scalars(
            select(Job)
            .where(or_(Job.status == "queued", (Job.status == "running") & (Job.lease_until < now)))
            .order_by(Job.created_at, Job.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).first()
        if job is None:
            return None
        job.status = "running"
        job.attempts += 1
        job.lease_until = now + timedelta(seconds=lease_seconds)
        return ClaimedJob(job.id, job.kind, job.target_id, job.config_snapshot_id, job.attempts)


def finish_job(factory: sessionmaker[Session], job_id: str) -> None:
    with factory.begin() as session:
        job = session.get(Job, job_id)
        job.status = "done"
        job.finished_at = utcnow()


def fail_job(factory: sessionmaker[Session], job_id: str, error: str, max_attempts: int) -> bool:
    with factory.begin() as session:
        job = session.get(Job, job_id)
        job.error = error
        if job.attempts < max_attempts:
            job.status = "queued"
            job.lease_until = None
            return True
        job.status = "failed"
        job.finished_at = utcnow()
        return False
