"""Job worker loop."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from sqlalchemy import select

from tribunal.context import AppContext
from tribunal.db.models import Document, Paper, Slot
from tribunal.generation.papers import regenerate, run_paper
from tribunal.ingestion.pipeline import ingest_document
from tribunal.jobs.queue import ClaimedJob, claim_next, fail_job, finish_job, renew_lease

log = logging.getLogger(__name__)


def _ingest(ctx: AppContext, job: ClaimedJob) -> None:
    ingest_document(ctx, ctx.snapshot_for(job.config_snapshot_id), job.target_id)


def _ingest_failed(ctx: AppContext, job: ClaimedJob, error: str) -> None:
    with ctx.session_factory.begin() as session:
        doc = session.get(Document, job.target_id)
        if doc is not None:
            doc.status = "failed"
            doc.error = error


def _generate_paper(ctx: AppContext, job: ClaimedJob) -> None:
    run_paper(ctx, ctx.snapshot_for(job.config_snapshot_id), job.target_id)


def _regenerate_slot(ctx: AppContext, job: ClaimedJob) -> None:
    regenerate(ctx, ctx.snapshot_for(job.config_snapshot_id), job.target_id)


def _paper_failed(ctx: AppContext, job: ClaimedJob, error: str) -> None:
    with ctx.session_factory.begin() as session:
        paper = session.get(Paper, job.target_id)
        if paper is not None:
            paper.status = "failed"
            paper.error = error
        for slot in session.scalars(select(Slot).where(Slot.paper_id == job.target_id,
                                                       Slot.status.in_(("pending", "running")))):
            slot.status = "failed"
            slot.status_detail = f"paper generation failed: {error}"


def _slot_failed(ctx: AppContext, job: ClaimedJob, error: str) -> None:
    with ctx.session_factory.begin() as session:
        slot = session.get(Slot, job.target_id)
        if slot is not None:
            slot.status = "failed"
            slot.status_detail = f"regeneration failed: {error}"


HANDLERS: dict[str, Callable[[AppContext, ClaimedJob], None]] = {
    "ingest": _ingest,
    "generate_paper": _generate_paper,
    "regenerate_slot": _regenerate_slot,
}
FINAL_FAILURE: dict[str, Callable[[AppContext, ClaimedJob, str], None]] = {
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


def run_once(ctx: AppContext) -> bool:
    retries = ctx.snapshot.config.retries
    job = claim_next(ctx.session_factory, retries.lease_seconds)
    if job is None:
        return False
    try:
        if job.attempts > retries.max_attempts:
            # Reclaimed after lease expiry too often: the worker likely dies on this job.
            raise _Abandoned(f"job abandoned after {job.attempts - 1} attempts; the worker stopped while processing it")
        handler = HANDLERS.get(job.kind)
        if handler is None:
            raise _Abandoned(f"unknown job kind '{job.kind}'")
        with _lease_kept(ctx, job, retries.lease_seconds):
            handler(ctx, job)
    except Exception as exc:
        log.exception("job %s (%s) failed", job.id, job.kind)
        max_attempts = 0 if isinstance(exc, _Abandoned) else retries.max_attempts
        requeued = fail_job(ctx.session_factory, job.id, job.attempts, str(exc), max_attempts)
        if requeued is False and (hook := FINAL_FAILURE.get(job.kind)):
            hook(ctx, job, str(exc))
        owned = requeued is not None
    else:
        owned = finish_job(ctx.session_factory, job.id, job.attempts)
    if not owned:
        log.warning("job %s (%s) was reclaimed by another worker after its lease expired; result discarded",
                    job.id, job.kind)
    return True


class _Abandoned(Exception):
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
