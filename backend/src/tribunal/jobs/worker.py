"""Job worker loop."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

from sqlalchemy import select
from sqlalchemy.orm import Session

from tribunal.context import AppContext
from tribunal.db.models import Document, Paper, Slot
from tribunal.errors import TribunalError
from tribunal.generation.papers import regenerate, run_paper
from tribunal.ingestion.pipeline import ingest_document
from tribunal.jobs.queue import ClaimedJob, OwnershipLost, claim_next, fail_job, finish_job, renew_lease

log = logging.getLogger(__name__)


def _ingest(ctx: AppContext, job: ClaimedJob) -> None:
    try:
        ingest_document(ctx, ctx.snapshot_for(job.config_snapshot_id), job.target_id, owner=job)
    except TribunalError as exc:
        raise _PermanentFailure(exc.message) from exc


def _ingest_failed(session, job: ClaimedJob, error: str) -> None:
    doc = session.get(Document, job.target_id)
    if doc is not None:
        doc.status = "ready" if doc.extraction_version else "failed"
        doc.error = error


def _generate_paper(ctx: AppContext, job: ClaimedJob) -> None:
    run_paper(ctx, ctx.snapshot_for(job.config_snapshot_id), job.target_id, owner=job)


def _regenerate_slot(ctx: AppContext, job: ClaimedJob) -> None:
    regenerate(ctx, ctx.snapshot_for(job.config_snapshot_id), job.target_id, owner=job)


def _paper_failed(session, job: ClaimedJob, error: str) -> None:
    paper = session.get(Paper, job.target_id)
    if paper is not None:
        paper.status = "failed"
        paper.error = error
    for slot in session.scalars(select(Slot).where(Slot.paper_id == job.target_id,
                                                   Slot.status.in_(("pending", "running")))):
        slot.status = "failed"
        slot.status_detail = f"paper generation failed: {error}"


def _slot_failed(session, job: ClaimedJob, error: str) -> None:
    slot = session.get(Slot, job.target_id)
    if slot is not None:
        slot.status = "failed"
        slot.status_detail = f"regeneration failed: {error}"


HANDLERS: dict[str, Callable[[AppContext, ClaimedJob], None]] = {
    "ingest": _ingest,
    "generate_paper": _generate_paper,
    "regenerate_slot": _regenerate_slot,
}
FINAL_FAILURE: dict[str, Callable[[Session, ClaimedJob, str], None]] = {
    "ingest": _ingest_failed,
    "generate_paper": _paper_failed,
    "regenerate_slot": _slot_failed,
}


@contextmanager
def _lease_kept(ctx: AppContext, job: ClaimedJob, lease_seconds: float) -> Iterator[None]:
    """Renew the job's lease every third of its length until the block ends."""
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(lease_seconds / 3):
            try:
                if not renew_lease(ctx.session_factory, job.id, job.attempts, lease_seconds):
                    return  # another worker owns the job now
            except Exception:
                log.exception("could not renew the lease of job %s", job.id)

    thread = threading.Thread(target=beat, name=f"lease-{job.id}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def run_once(ctx: AppContext, kinds: Sequence[str] | None = None) -> bool:
    retries = ctx.snapshot.config.retries
    job = claim_next(ctx.session_factory, retries.lease_seconds, kinds=kinds)
    if job is None:
        return False
    try:
        if job.attempts > retries.max_attempts:
            # Reclaimed after lease expiry too often: the worker likely dies on this job.
            raise _PermanentFailure(
                f"job abandoned after {job.attempts - 1} attempts; the worker stopped while processing it")
        handler = HANDLERS.get(job.kind)
        if handler is None:
            raise _PermanentFailure(f"unknown job kind '{job.kind}'")
        with _lease_kept(ctx, job, retries.lease_seconds):
            handler(ctx, job)
    except OwnershipLost:
        owned = False
    except Exception as exc:
        log.exception("job %s (%s) failed", job.id, job.kind)
        max_attempts = 0 if isinstance(exc, _PermanentFailure) else retries.max_attempts
        hook = FINAL_FAILURE.get(job.kind)
        # Only our own errors carry text safe to show; anything else may hold paths or SQL.
        error = (str(exc) if isinstance(exc, (TribunalError, _PermanentFailure))
                 else f"internal error in job {job.id}; see the worker log")
        requeued = fail_job(ctx.session_factory, job.id, job.attempts, error, max_attempts,
                            lambda session: hook(session, job, error) if hook else None,
                            retries.job_retry_delay_seconds)
        owned = requeued is not None
    else:
        owned = finish_job(ctx.session_factory, job.id, job.attempts)
    if not owned:
        log.warning("job %s (%s) was reclaimed by another worker after its lease expired; result discarded",
                    job.id, job.kind)
    return True


def run_until_idle(ctx: AppContext, kinds: Sequence[str] | None, max_seconds: float) -> int:
    """Process claimable jobs until none is left or the time is up; returns how many were processed."""
    deadline = time.monotonic() + max_seconds
    processed = 0
    while time.monotonic() < deadline and run_once(ctx, kinds):
        processed += 1
    return processed


class _PermanentFailure(Exception):
    """A job that must fail now, without further retries."""


def run_forever(ctx: AppContext, stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    while not stop.is_set():
        try:
            busy = run_once(ctx)
        except Exception:
            log.exception("worker iteration failed")
            busy = False
        if not busy:
            stop.wait(ctx.snapshot.config.retries.poll_interval_seconds)
