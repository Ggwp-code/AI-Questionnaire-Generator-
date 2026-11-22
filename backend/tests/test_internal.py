from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from tribunal.api.app import create_app
from tribunal.db.models import Job
from tribunal.db.types import utcnow
from tribunal.jobs import worker
from tribunal.jobs.queue import claim_next, enqueue, finish_job
from tribunal.jobs.worker import run_until_idle

AUTH = {"Authorization": "Bearer dispatch-secret"}


@pytest.fixture
def token_env(monkeypatch):
    monkeypatch.setenv("INTERNAL_DISPATCH_TOKEN", "dispatch-secret")


@pytest.fixture
def client(token_env, ctx):
    return TestClient(create_app(ctx))


def _enqueue(ctx, kind, *targets):
    with ctx.session_factory.begin() as s:
        return [enqueue(s, kind, t, ctx.snapshot.snapshot_id).id for t in targets]


def _statuses(ctx):
    with ctx.session_factory() as s:
        return {j.target_id: j.status for j in s.query(Job)}


def _stub(monkeypatch, ran, sleep=0.0):
    def handler(_ctx, job):
        ran.append(job.target_id)
        time.sleep(sleep)

    monkeypatch.setattr(worker, "HANDLERS", {"ingest": handler, "generate_paper": handler})


def _drain(client, kinds=("ingest", "generate_paper"), headers=AUTH, max_seconds=5):
    return client.post("/internal/drain", json={"kinds": list(kinds), "max_seconds": max_seconds}, headers=headers)


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "dispatch-secret"}])
def test_drain_requires_token(client, ctx, monkeypatch, headers):
    ran = []
    _stub(monkeypatch, ran)
    _enqueue(ctx, "ingest", "a")
    response = _drain(client, headers=headers)
    assert response.status_code == 401 and response.json()["error"]["message"] == "invalid dispatch token"
    assert ran == [] and _statuses(ctx) == {"a": "queued"}


def test_drain_not_mounted_without_token(ctx, monkeypatch):
    monkeypatch.delenv("INTERNAL_DISPATCH_TOKEN", raising=False)
    assert _drain(TestClient(create_app(ctx))).status_code == 404


def test_drain_runs_only_requested_kinds(client, ctx, monkeypatch):
    ran = []
    _stub(monkeypatch, ran)
    _enqueue(ctx, "ingest", "i1")
    _enqueue(ctx, "generate_paper", "g1", "g2")
    response = _drain(client, kinds=["generate_paper"])
    assert response.status_code == 200 and response.json() == {"processed": 2}
    assert sorted(ran) == ["g1", "g2"]
    assert _statuses(ctx) == {"i1": "queued", "g1": "done", "g2": "done"}


def test_drain_rejects_bad_body(client):
    assert client.post("/internal/drain", json={"kinds": ["nope"], "max_seconds": 5}, headers=AUTH).status_code == 422
    assert _drain(client, max_seconds=841).status_code == 422


def test_overlapping_drains_run_each_job_once(ctx, monkeypatch):
    ran = []
    _stub(monkeypatch, ran, sleep=0.02)
    targets = [f"t{i}" for i in range(6)]
    _enqueue(ctx, "ingest", *targets)
    start = threading.Barrier(2)
    counts = []

    def drain():
        start.wait(timeout=5)
        counts.append(run_until_idle(ctx, ["ingest"], 10))

    threads = [threading.Thread(target=drain) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(ran) == targets and sum(counts) == 6
    assert set(_statuses(ctx).values()) == {"done"}


def test_drain_resumes_abandoned_job(ctx, monkeypatch):
    ran = []
    _stub(monkeypatch, ran)
    (a,) = _enqueue(ctx, "ingest", "a")
    stale = claim_next(ctx.session_factory, 60)  # a container that then stopped
    with ctx.session_factory.begin() as s:
        s.get(Job, a).lease_until = utcnow() - timedelta(seconds=1)
    assert run_until_idle(ctx, ["ingest"], 5) == 1
    assert ran == ["a"] and _statuses(ctx) == {"a": "done"}
    assert finish_job(ctx.session_factory, a, stale.attempts) is False  # stale worker is fenced out


def test_drain_respects_max_seconds(ctx, monkeypatch):
    ran = []
    _stub(monkeypatch, ran, sleep=0.3)
    _enqueue(ctx, "ingest", *"abcde")
    processed = run_until_idle(ctx, ["ingest"], 0.5)
    assert processed == len(ran) and 1 <= processed < 5
    assert list(_statuses(ctx).values()).count("queued") == 5 - processed
