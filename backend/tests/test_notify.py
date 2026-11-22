from __future__ import annotations

import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import digital_pdf
from test_api import PAGES, SPEC, A, h

from tribunal.api.app import create_app
from tribunal.context import build_context
from tribunal.db.models import Job
from tribunal.jobs.notify import CloudflareQueueNotifier, NullNotifier
from tribunal.jobs.queue import enqueue
from tribunal.jobs.worker import run_once
from tribunal.settings import reset_settings_cache

TOKEN = "cf-token-very-secret"
BASE = "https://api.cloudflare.com/client/v4/accounts/acct1/queues"


@pytest.fixture
def cf_env(test_env, monkeypatch):
    for k, v in {"JOB_NOTIFY": "cloudflare_queue", "CF_ACCOUNT_ID": "acct1", "CF_API_TOKEN": TOKEN,
                 "CF_QUEUE_INGESTION_ID": "qing", "CF_QUEUE_GENERATION_ID": "qgen", "INTERNAL_DISPATCH_TOKEN": "dispatch-secret"}.items():
        monkeypatch.setenv(k, v)
    reset_settings_cache()


def make(handler):
    from tribunal.settings import load_settings

    ctx = build_context(load_settings(), migrate=True)
    assert isinstance(ctx.notifier, CloudflareQueueNotifier)
    ctx.notifier = CloudflareQueueNotifier(ctx.settings, httpx.Client(transport=httpx.MockTransport(handler)))
    return ctx


def upload(client, sid):
    return client.post(f"/api/subjects/{sid}/documents", headers=h(A),
                       files={"file": ("d.pdf", digital_pdf(PAGES), "application/pdf")})


def test_upload_notifies_ingestion_queue_after_commit(cf_env):
    seen = []
    ctx = make(lambda r: seen.append(r) or httpx.Response(200, json={"success": True}))
    client = TestClient(create_app(ctx))
    sid = client.post("/api/subjects", json={"name": "S"}, headers=h(A)).json()["id"]
    assert upload(client, sid).status_code == 202
    assert len(seen) == 1
    with ctx.session_factory() as s:
        job = s.query(Job).one()
    assert str(seen[0].url) == f"{BASE}/qing/messages" and seen[0].method == "POST"
    assert seen[0].headers["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(seen[0].content) == {"content_type": "json", "body": {"job_id": job.id, "kind": "ingest"}}


def test_paper_notifies_generation_queue(cf_env):
    seen = []
    ctx = make(lambda r: seen.append(r) or httpx.Response(200, json={"success": True}))
    client = TestClient(create_app(ctx))
    sid = client.post("/api/subjects", json={"name": "S"}, headers=h(A)).json()["id"]
    assert upload(client, sid).status_code == 202
    while run_once(ctx):
        pass
    tid = client.post(f"/api/subjects/{sid}/topics", json={"title": "Decision trees"}, headers=h(A)).json()["id"]
    tpl = client.post("/api/templates", json={"name": "t", "spec": SPEC}, headers=h(A)).json()["id"]
    seen.clear()
    r = client.post("/api/papers", headers=h(A), json={
        "subject_id": sid, "template_id": tpl, "topic_ids": [tid], "title": "P"})
    assert r.status_code == 202
    assert [str(q.url) for q in seen] == [f"{BASE}/qgen/messages"]
    assert json.loads(seen[0].content)["body"]["kind"] == "generate_paper"


def test_rolled_back_request_sends_nothing(cf_env):
    seen = []
    ctx = make(lambda r: seen.append(r) or httpx.Response(200))
    with pytest.raises(RuntimeError):
        with ctx.session_factory.begin() as s:
            enqueue(s, "ingest", "t1", ctx.snapshot.snapshot_id)
            raise RuntimeError("boom")
    assert seen == []
    with ctx.session_factory.begin() as s:
        enqueue(s, "ingest", "t2", ctx.snapshot.snapshot_id)
    assert len(seen) == 1 and json.loads(seen[0].content)["body"]["kind"] == "ingest"


@pytest.mark.parametrize("failure", ["raise", "401"])
def test_notify_failure_keeps_request_and_job(cf_env, failure):
    def handler(request):
        if failure == "raise":
            raise httpx.ConnectError(f"down {request.headers['Authorization']}")
        return httpx.Response(401, json={"errors": [{"message": "bad token"}]})

    ctx = make(handler)
    client = TestClient(create_app(ctx))
    sid = client.post("/api/subjects", json={"name": "S"}, headers=h(A)).json()["id"]
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    notify_log = logging.getLogger("tribunal.jobs.notify")
    notify_log.addHandler(handler)  # the app's logging setup does not propagate to caplog
    try:
        assert upload(client, sid).status_code == 202
    finally:
        notify_log.removeHandler(handler)
    with ctx.session_factory() as s:
        assert s.query(Job).one().status == "queued"
    warnings = [r.getMessage() for r in records if r.levelno == logging.WARNING]
    assert warnings and not any(TOKEN in w for w in warnings)


def test_null_notifier_by_default(ctx):
    assert isinstance(ctx.notifier, NullNotifier)
    with ctx.session_factory.begin() as s:
        enqueue(s, "ingest", "t", ctx.snapshot.snapshot_id)
        assert s.info["tribunal_notify"]
