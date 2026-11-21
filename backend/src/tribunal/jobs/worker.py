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
        HANDLERS[job.kind](ctx, job)
    except Exception as exc:
        log.exception("job %s (%s) failed", job.id, job.kind)
        if not fail_job(ctx.session_factory, job.id, str(exc), retries.max_attempts):
            FINAL_FAILURE[job.kind](ctx, job, str(exc))
    else:
        finish_job(ctx.session_factory, job.id)
    return True


def run_forever(ctx: AppContext, stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    while not stop.is_set():
        if not run_once(ctx):
            stop.wait(ctx.snapshot.config.retries.poll_interval_seconds)
