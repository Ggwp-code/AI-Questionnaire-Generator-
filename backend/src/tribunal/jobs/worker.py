"""Job worker loop."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from tribunal.context import AppContext
from tribunal.db.models import Document
from tribunal.ingestion.pipeline import ingest_document
from tribunal.jobs.queue import ClaimedJob, claim_next, fail_job, finish_job

log = logging.getLogger(__name__)


def _ingest(ctx: AppContext, job: ClaimedJob) -> None:
    ingest_document(ctx, ctx.snapshot_for(job.config_snapshot_id), job.target_id)


def _ingest_failed(ctx: AppContext, job: ClaimedJob, error: str) -> None:
    with ctx.session_factory.begin() as session:
        doc = session.get(Document, job.target_id)
        if doc is not None:
            doc.status = "failed"
            doc.error = error


HANDLERS: dict[str, Callable[[AppContext, ClaimedJob], None]] = {"ingest": _ingest}
FINAL_FAILURE: dict[str, Callable[[AppContext, ClaimedJob, str], None]] = {"ingest": _ingest_failed}


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
        handler(ctx, job)
    except Exception as exc:
        log.exception("job %s (%s) failed", job.id, job.kind)
        max_attempts = 0 if isinstance(exc, _Abandoned) else retries.max_attempts
        if not fail_job(ctx.session_factory, job.id, str(exc), max_attempts):
            hook = FINAL_FAILURE.get(job.kind)
            if hook:
                hook(ctx, job, str(exc))
    else:
        finish_job(ctx.session_factory, job.id)
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
