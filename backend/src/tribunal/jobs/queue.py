"""Database-backed job queue."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from tribunal.db.models import Job, Paper, Slot
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
        eligible = or_(Job.status == "queued", (Job.status == "running") & (Job.lease_until < now))
        candidate = (select(Job.id).where(eligible).order_by(Job.created_at, Job.id)
                     .limit(1).with_for_update(skip_locked=True).scalar_subquery())
        job = session.scalar(update(Job).where(Job.id == candidate, eligible)
                             .values(status="running", attempts=Job.attempts + 1,
                                     lease_until=now + timedelta(seconds=lease_seconds))
                             .returning(Job).execution_options(synchronize_session=False))
        if job is None:
            return None
        _release_reservations(session, job)
        return ClaimedJob(job.id, job.kind, job.target_id, job.config_snapshot_id, job.attempts)


def _owned(job_id: str, attempts: int):
    """`attempts` grows on every claim, so it fences out a worker whose lease was taken over."""
    return update(Job).where(Job.id == job_id, Job.status == "running", Job.attempts == attempts)


class OwnershipLost(Exception):
    """A newer attempt has taken responsibility for the target."""


@contextmanager
def owned_transaction(factory, owner: ClaimedJob | None):
    with factory.begin() as session:
        if owner is not None:
            locked = session.execute(_owned(owner.id, owner.attempts).values(attempts=Job.attempts))
            if locked.rowcount != 1:
                raise OwnershipLost(owner.id)
        yield session


def _release_reservations(session, job):
    if job.reserved_tokens:
        paper_id = (job.target_id if job.kind == "generate_paper"
                    else session.get(Slot, job.target_id).paper_id)
        session.execute(update(Paper).where(Paper.id == paper_id)
                        .values(tokens_used=Paper.tokens_used - job.reserved_tokens))
        job.reserved_tokens = 0


def finish_job(factory: sessionmaker[Session], job_id: str, attempts: int) -> bool:
    """Mark the job done. False when the caller no longer owns it."""
    with factory.begin() as session:
        result = session.execute(_owned(job_id, attempts).values(status="done", error=None, finished_at=utcnow()))
        if result.rowcount == 1:
            _release_reservations(session, session.get(Job, job_id))
            return True
        return False


def fail_job(factory: sessionmaker[Session], job_id: str, attempts: int, error: str,
             max_attempts: int, on_terminal: Callable[[Session], None] | None = None) -> bool | None:
    """True: queued for another attempt. False: failed for good. None: the caller no longer owns it."""
    retry = attempts < max_attempts
    values = {"status": "queued", "lease_until": None} if retry else {"status": "failed", "finished_at": utcnow()}
    with factory.begin() as session:
        owned = session.execute(_owned(job_id, attempts).values(error=error, **values)).rowcount == 1
        if owned:
            _release_reservations(session, session.get(Job, job_id))
            if not retry and on_terminal:
                on_terminal(session)
    return retry if owned else None


def renew_lease(factory: sessionmaker[Session], job_id: str, attempts: int, lease_seconds: float) -> bool:
    """Extend the lease of a running job. False when the caller no longer owns it."""
    with factory.begin() as session:
        until = utcnow() + timedelta(seconds=lease_seconds)
        return session.execute(_owned(job_id, attempts).values(lease_until=until)).rowcount == 1
