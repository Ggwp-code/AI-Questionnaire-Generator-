from __future__ import annotations

import threading
from datetime import timedelta

from conftest import make_subject, make_user

from tribunal.db.models import Document, Job
from tribunal.db.types import utcnow
from tribunal.jobs import worker
from tribunal.jobs.queue import claim_next, enqueue, fail_job


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
    claim_next(ctx.session_factory, 60)
    assert fail_job(ctx.session_factory, a, "boom", 2) is True
    claim_next(ctx.session_factory, 60)
    assert fail_job(ctx.session_factory, a, "boom again", 2) is False
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
        enqueue(s, "ingest", doc_id, ctx.snapshot.snapshot_id)

    def boom(_ctx, _job):
        raise RuntimeError("kaput")

    monkeypatch.setitem(worker.HANDLERS, "ingest", boom)
    for _ in range(ctx.snapshot.config.retries.max_attempts):
        assert worker.run_once(ctx) is True
    assert worker.run_once(ctx) is False
    with ctx.session_factory() as s:
        d = s.get(Document, doc_id)
        assert d.status == "failed" and d.error == "kaput"


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
