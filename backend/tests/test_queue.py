from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest
from conftest import make_subject, make_user
from pdf_fixtures import digital_pdf

from tribunal.db.models import Document, Job
from tribunal.db.types import utcnow
from tribunal.jobs import worker
from tribunal.jobs.queue import claim_next, enqueue, fail_job, finish_job, renew_lease
from tribunal.runtime_config import ConfigSnapshot


def _enqueue(ctx, *targets):
    with ctx.session_factory.begin() as s:
        return [enqueue(s, "ingest", t, ctx.snapshot.snapshot_id).id for t in targets]


def test_claim_order_and_lease(ctx):
    a, b = _enqueue(ctx, "a", "b")
    assert claim_next(ctx.session_factory, 60).id == a
    assert claim_next(ctx.session_factory, 60).id == b
    assert claim_next(ctx.session_factory, 60) is None


def test_expired_lease_reclaimed(ctx):
    (a,) = _enqueue(ctx, "a")
    claim_next(ctx.session_factory, 60)
    again = claim_next(ctx.session_factory, 60, now=utcnow() + timedelta(seconds=61))
    assert again.id == a and again.attempts == 2


def test_fail_requeues_then_fails(ctx):
    (a,) = _enqueue(ctx, "a")
    first = claim_next(ctx.session_factory, 60)
    assert fail_job(ctx.session_factory, a, first.attempts, "boom", 2) is True
    second = claim_next(ctx.session_factory, 60)
    assert fail_job(ctx.session_factory, a, second.attempts, "boom again", 2) is False
    with ctx.session_factory() as s:
        job = s.get(Job, a)
        assert job.status == "failed" and job.error == "boom again"


def test_run_once_marks_document_failed(ctx, monkeypatch):
    with ctx.session_factory.begin() as s:
        user = make_user(s, "T@Example.com")
        subject = make_subject(s, user, "S")
        doc = Document(subject_id=subject.id, uploaded_by=user.id, filename="a.pdf", content_hash="h",
                       storage_key="k", size_bytes=1, ocr_languages=["eng"])
        s.add(doc)
        s.flush()
        doc_id = doc.id
        job_id = enqueue(s, "ingest", doc_id, ctx.snapshot.snapshot_id).id

    def boom(_ctx, _job):
        raise RuntimeError("/srv/x.py SELECT secret")

    monkeypatch.setitem(worker.HANDLERS, "ingest", boom)
    for _ in range(ctx.snapshot.config.retries.max_attempts):
        assert worker.run_once(ctx) is True
    assert worker.run_once(ctx) is False
    with ctx.session_factory() as s:
        d = s.get(Document, doc_id)
        assert d.status == "failed" and d.error == f"internal error in job {job_id}; see the worker log"
        assert "/srv" not in d.error


test_unexpected_error_text_is_generic = test_run_once_marks_document_failed


def _doc(ctx):
    with ctx.session_factory.begin() as s:
        user = make_user(s, "T@Example.com")
        subject = make_subject(s, user, "S")
        doc = Document(subject_id=subject.id, uploaded_by=user.id, filename="a.pdf", content_hash="h",
                       storage_key="k", size_bytes=1, ocr_languages=["eng"])
        s.add(doc)
        s.flush()
        return doc.id


def _job_status(ctx, job_id):
    with ctx.session_factory() as s:
        job = s.get(Job, job_id)
        return job.status, job.error


def test_unknown_kind_fails(ctx):
    with ctx.session_factory.begin() as s:
        job_id = enqueue(s, "nope", "t", ctx.snapshot.snapshot_id).id
    assert worker.run_once(ctx) is True
    assert _job_status(ctx, job_id) == ("failed", "unknown job kind 'nope'")


def test_run_forever_survives_claim_error(ctx, monkeypatch):
    stop = threading.Event()
    calls = []

    def flaky(*_a, **_k):
        calls.append(1)
        stop.set()
        raise RuntimeError("db down")

    monkeypatch.setattr(worker, "claim_next", flaky)
    worker.run_forever(ctx, stop)
    assert calls == [1]


def test_job_abandoned_after_repeated_lease_expiry(ctx, monkeypatch):
    doc_id = _doc(ctx)
    with ctx.session_factory.begin() as s:
        job_id = enqueue(s, "ingest", doc_id, ctx.snapshot.snapshot_id).id
    cap = ctx.snapshot.config.retries.max_attempts
    for _ in range(cap):  # the worker "dies" each time: claimed, never finished
        assert claim_next(ctx.session_factory, 60) is not None
        with ctx.session_factory.begin() as s:
            s.get(Job, job_id).lease_until = utcnow() - timedelta(seconds=1)
    monkeypatch.setitem(worker.HANDLERS, "ingest", lambda *_: (_ for _ in ()).throw(AssertionError("ran")))
    assert worker.run_once(ctx) is True
    expected = f"job abandoned after {cap} attempts; the worker stopped while processing it"
    assert _job_status(ctx, job_id) == ("failed", expected)
    with ctx.session_factory() as s:
        d = s.get(Document, doc_id)
        assert d.status == "failed" and d.error == expected


def test_reclaimed_job_is_fenced_from_its_old_worker(ctx):
    (a,) = _enqueue(ctx, "a")
    first = claim_next(ctx.session_factory, 60)
    second = claim_next(ctx.session_factory, 60, now=utcnow() + timedelta(seconds=61))
    assert (first.attempts, second.attempts) == (1, 2)
    assert finish_job(ctx.session_factory, a, first.attempts) is False
    assert fail_job(ctx.session_factory, a, first.attempts, "late", 3) is None
    assert renew_lease(ctx.session_factory, a, first.attempts, 60) is False
    assert _job_status(ctx, a) == ("running", None)
    assert finish_job(ctx.session_factory, a, second.attempts) is True
    assert _job_status(ctx, a) == ("done", None)


@pytest.mark.parametrize("outcome", ["returns", "raises"])
def test_worker_that_lost_its_lease_leaves_job_alone(ctx, monkeypatch, caplog, outcome):
    (a,) = _enqueue(ctx, "a")
    later = utcnow() + timedelta(seconds=ctx.snapshot.config.retries.lease_seconds + 1)
    taken, hooks = [], []

    def slow(_ctx, _job):  # another worker reclaims the job while this one is still busy
        taken.append(claim_next(ctx.session_factory, 60, now=later))
        if outcome == "raises":
            raise RuntimeError("late failure")

    monkeypatch.setitem(worker.HANDLERS, "ingest", slow)
    monkeypatch.setitem(worker.FINAL_FAILURE, "ingest", lambda *args: hooks.append(args))
    assert worker.run_once(ctx) is True
    assert taken[0].attempts == 2 and hooks == []
    assert _job_status(ctx, a) == ("running", None)
    assert f"job {a} (ingest) was reclaimed by another worker" in caplog.text


def test_heartbeat_keeps_long_job_leased(ctx, monkeypatch):
    lease = 0.5
    cfg = ctx.snapshot.config
    # built without validation, which would refuse a lease this short
    ctx.snapshot = ConfigSnapshot.model_construct(config=cfg.model_copy(update={
        "retries": cfg.retries.model_copy(update={"lease_seconds": lease})}), content_hash="short-lease")
    (a,) = _enqueue(ctx, "a")
    stolen = []

    def long_job(_ctx, _job):
        end = time.monotonic() + 3 * lease
        while time.monotonic() < end:
            stolen.append(claim_next(ctx.session_factory, lease))
            time.sleep(lease / 5)

    monkeypatch.setitem(worker.HANDLERS, "ingest", long_job)
    assert worker.run_once(ctx) is True
    assert stolen and not any(stolen)
    assert _job_status(ctx, a) == ("done", None)


@pytest.mark.parametrize("readable", [False, True])
def test_ingest_job_reports_document_outcome(ctx, readable):
    ctx.store.put("k", digital_pdf(["Decision trees split data. " * 20]) if readable else b"not a pdf")
    doc_id = _doc(ctx)
    with ctx.session_factory.begin() as s:
        job_id = enqueue(s, "ingest", doc_id, ctx.snapshot.snapshot_id).id
    assert worker.run_once(ctx) is True
    with ctx.session_factory() as s:
        job, doc = s.get(Job, job_id), s.get(Document, doc_id)
        error = None if readable else "file is not a readable PDF"
        assert (job.status, job.error, job.attempts) == ("done" if readable else "failed", error, 1)
        assert (doc.status, doc.error) == ("ready" if readable else "failed", error)


def test_simultaneous_claims(ctx):
    from concurrent.futures import ThreadPoolExecutor

    (job_id,) = _enqueue(ctx, "a")
    ready = threading.Barrier(2)

    def claim():
        ready.wait(timeout=5)
        return claim_next(ctx.session_factory, 60)

    with ThreadPoolExecutor(2) as pool:
        claims = [pool.submit(claim) for _ in range(2)]
        results = [f.result() for f in claims]
    assert [(j.id, j.attempts) for j in results if j] == [(job_id, 1)]


@pytest.mark.parametrize("late_failure", [False, True])
def test_reclaimed_ocr_preserves_winning_index(ctx, fake_ocr, monkeypatch, late_failure):
    from pdf_fixtures import scanned_pdf

    from tribunal.db.models import Page, Passage

    ctx.store.put("k", scanned_pdf(1))
    doc_id = _doc(ctx)
    (job_id,) = _enqueue(ctx, doc_id)
    winning_text = "Decision trees classify samples. " * 8

    def paused_ocr(*args):
        with ctx.session_factory.begin() as s:
            s.get(Job, job_id).lease_until = utcnow() - timedelta(seconds=1)
        monkeypatch.setattr(fake_ocr, "ocr_pages", lambda *args: {1: winning_text})
        assert worker.run_once(ctx)
        if late_failure:
            raise RuntimeError("late OCR error")
        return {1: "stale OCR"}

    monkeypatch.setattr(fake_ocr, "ocr_pages", paused_ocr)
    assert worker.run_once(ctx)
    with ctx.session_factory() as s:
        doc = s.get(Document, doc_id)
        assert (doc.status, doc.error) == ("ready", None)
        assert s.query(Page).one().text == winning_text
        assert s.query(Passage).one().text == winning_text
    assert _job_status(ctx, job_id) == ("done", None)


def test_terminal_job_and_target_failure_roll_back_together(ctx):
    doc_id = _doc(ctx)
    (job_id,) = _enqueue(ctx, doc_id)
    job = claim_next(ctx.session_factory, 60)

    def broken_hook(session):
        session.get(Document, doc_id).error = "partial"
        session.flush()
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        fail_job(ctx.session_factory, job_id, job.attempts, "failure", 0, broken_hook)
    assert _job_status(ctx, job_id) == ("running", None)
    with ctx.session_factory() as s:
        assert s.get(Document, doc_id).error is None


def test_worker_that_lost_its_lease_does_not_mark_document_ready(ctx, fake_ocr, monkeypatch):
    from pdf_fixtures import scanned_pdf

    ctx.store.put("k", scanned_pdf(1))
    doc_id = _doc(ctx)
    (job_id,) = _enqueue(ctx, doc_id)
    second_busy, release_second = threading.Event(), threading.Event()
    second = threading.Thread(target=worker.run_once, args=(ctx,))

    def second_ocr(*_args):
        second_busy.set()
        assert release_second.wait(5)
        return {1: "Decision trees classify samples. " * 8}

    def first_ocr(*_args):  # the lease expires and a second worker starts while this OCR runs
        with ctx.session_factory.begin() as s:
            s.get(Job, job_id).lease_until = utcnow() - timedelta(seconds=1)
        monkeypatch.setattr(fake_ocr, "ocr_pages", second_ocr)
        second.start()
        assert second_busy.wait(5)
        return {1: "stale OCR " * 8}

    monkeypatch.setattr(fake_ocr, "ocr_pages", first_ocr)
    assert worker.run_once(ctx)
    with ctx.session_factory() as s:
        assert s.get(Document, doc_id).status == "processing"
    release_second.set()
    second.join(5)
    with ctx.session_factory() as s:
        assert s.get(Document, doc_id).status == "ready"
    assert _job_status(ctx, job_id) == ("done", None)


def test_requeued_job_waits_out_its_retry_delay(ctx):
    (a,) = _enqueue(ctx, "a")
    first = claim_next(ctx.session_factory, 60)
    assert fail_job(ctx.session_factory, a, first.attempts, "boom", 3, retry_delay_seconds=30) is True
    assert claim_next(ctx.session_factory, 60) is None
    assert claim_next(ctx.session_factory, 60, now=utcnow() + timedelta(seconds=29)) is None
    assert claim_next(ctx.session_factory, 60, now=utcnow() + timedelta(seconds=31)).id == a


def test_claim_respects_kinds(ctx):
    with ctx.session_factory.begin() as s:
        a = enqueue(s, "ingest", "a", ctx.snapshot.snapshot_id).id
        b = enqueue(s, "generate_paper", "b", ctx.snapshot.snapshot_id).id
    assert claim_next(ctx.session_factory, 60, kinds=["generate_paper"]).id == b
    assert claim_next(ctx.session_factory, 60, kinds=["generate_paper"]) is None
    assert claim_next(ctx.session_factory, 60, kinds=[]) is None
    assert claim_next(ctx.session_factory, 60).id == a
