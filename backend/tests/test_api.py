from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import digital_pdf

from tribunal.api.app import create_app
from tribunal.db.models import Page, Passage, Slot
from tribunal.jobs.worker import run_once

PAGES = ["Decision trees split data into branches. " * 5, "Linear regression fits lines to data. " * 5]
SPEC = {"sections": [{"title": "A", "question_type": "short_answer", "count": 3, "marks_each": 2,
                      "bloom": "understand", "difficulty": "easy"}], "total_marks": 6}
A, B = "a@example.com", "b@example.com"


def h(email):
    return {"X-Dev-User": email}


@pytest.fixture
def client(ctx):
    return TestClient(create_app(ctx))


def jobs(ctx):
    while run_once(ctx):
        pass


def mk_subject(client, who=A, name="S"):
    r = client.post("/api/subjects", json={"name": name}, headers=h(who))
    assert r.status_code == 201
    return r.json()["id"]


def upload(client, sid, data=None, who=A):
    return client.post(f"/api/subjects/{sid}/documents", headers=h(who),
                       files={"file": ("d.pdf", data if data is not None else digital_pdf(PAGES), "application/pdf")})


@pytest.fixture
def ready(client, ctx):
    sid = mk_subject(client)
    r = upload(client, sid)
    assert r.status_code == 202
    jobs(ctx)
    return sid, r.json()["document"]["id"]


@pytest.fixture
def paper(client, ctx, ready):
    sid, _ = ready
    tid = client.post(f"/api/subjects/{sid}/topics", json={"title": "Decision trees"}, headers=h(A)).json()["id"]
    tpl = client.post("/api/templates", json={"name": "t", "spec": SPEC}, headers=h(A))
    assert tpl.status_code == 201 and tpl.json()["version"] == 1
    r = client.post("/api/papers", headers=h(A), json={
        "subject_id": sid, "template_id": tpl.json()["id"], "topic_ids": [tid], "title": "P"})
    assert r.status_code == 202
    jobs(ctx)
    return r.json()["paper_id"]


def test_healthz_and_me(client):
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/api/me", headers=h("A@Example.com")).json()["email"] == "a@example.com"


def test_requires_sign_in(client):
    r = client.get("/api/subjects")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"


def test_subject_isolation(client):
    sid = mk_subject(client)
    assert client.get("/api/subjects", headers=h(B)).json() == []
    assert client.get(f"/api/subjects/{sid}/documents", headers=h(B)).status_code == 404


def test_members(client):
    sid = mk_subject(client)
    assert client.post(f"/api/subjects/{sid}/members", json={"email": "B@example.com"}, headers=h(A)).status_code == 201
    assert [s["id"] for s in client.get("/api/subjects", headers=h(B)).json()] == [sid]
    assert client.post(f"/api/subjects/{sid}/members", json={"email": "c@example.com"}, headers=h(B)).status_code == 403


def test_upload_and_ingest(client, ready):
    sid, _ = ready
    docs = client.get(f"/api/subjects/{sid}/documents", headers=h(A)).json()
    assert docs[0]["status"] == "ready" and docs[0]["page_count"] == 2
    assert isinstance(client.get(f"/api/subjects/{sid}/topics", headers=h(A)).json(), list)


def test_upload_rejects_non_pdf(client):
    assert upload(client, mk_subject(client), b"hello").status_code == 422


def test_upload_too_large(client, ctx):
    ing = ctx.snapshot.config.ingestion.model_copy(update={"max_upload_bytes": 1000})
    cfg = ctx.snapshot.config.model_copy(update={"ingestion": ing})
    ctx.snapshot = ctx.snapshot.model_copy(update={"config": cfg})
    r = upload(client, mk_subject(client), digital_pdf(PAGES))
    assert r.status_code == 413 and r.json()["error"]["code"] == "upload_too_large"


def test_duplicate_upload(client, ctx):
    data = digital_pdf(PAGES)
    s1, s2 = mk_subject(client), mk_subject(client, name="S2")
    assert upload(client, s1, data).status_code == 202
    assert upload(client, s1, data).status_code == 409
    assert upload(client, s2, data).status_code == 202
    jobs(ctx)
    with ctx.session_factory() as s:
        for sid in (s1, s2):
            ps = s.query(Passage).filter_by(subject_id=sid).all()
            assert ps and {p.document_id for p in ps} == {
                d["id"] for d in client.get(f"/api/subjects/{sid}/documents", headers=h(A)).json()}


def test_file_download_scoped(client, ready):
    _, did = ready
    assert client.get(f"/api/documents/{did}/file", headers=h(B)).status_code == 404
    r = client.get(f"/api/documents/{did}/file", headers=h(A))
    assert r.status_code == 200 and r.content.startswith(b"%PDF")


def test_topics_and_passages_scoped(client, ctx, ready):
    sid, _ = ready
    t = client.post(f"/api/subjects/{sid}/topics", json={"title": "X"}, headers=h(A)).json()
    assert t["source"] == "faculty"
    assert client.patch(f"/api/topics/{t['id']}", json={"title": "Y"}, headers=h(A)).json()["title"] == "Y"
    assert client.patch(f"/api/topics/{t['id']}", json={"title": "Z"}, headers=h(B)).status_code == 404
    with ctx.session_factory() as s:
        pid = s.query(Passage).first().id
    assert client.get(f"/api/passages/{pid}", headers=h(A)).status_code == 200
    assert client.get(f"/api/passages/{pid}", headers=h(B)).status_code == 404


def test_template_problems_immediate(client):
    bad = {"sections": [dict(SPEC["sections"][0], question_type="essay")], "total_marks": 6}
    r = client.post("/api/templates", json={"name": "t", "spec": bad}, headers=h(A))
    assert r.status_code == 422
    assert "sections[0].question_type 'essay' is not enabled" in r.json()["error"]["details"]


def test_paper_flow(client, ctx, paper):
    p = client.get(f"/api/papers/{paper}", headers=h(A)).json()
    assert [s["status"] for s in p["slots"]] == ["drafted"] * 3
    assert p["timings"]["first_question_seconds"] is not None and p["timings"]["total_seconds"] is not None
    for s in p["slots"]:
        assert s["citations"]
        for c in s["citations"]:
            pas = client.get(f"/api/passages/{c['passage_id']}", headers=h(A)).json()
            assert c["url"] == pas["url"]
            with ctx.session_factory() as db:
                page = db.query(Page).filter_by(document_id=c["document_id"], page_number=c["page_number"]).one()
            assert pas["text"] in page.text


def test_regenerate_one_slot(client, ctx, paper):
    before = [s["revision"]["id"] for s in client.get(f"/api/papers/{paper}", headers=h(A)).json()["slots"]]
    assert client.post(f"/api/papers/{paper}/slots/2/regenerate", headers=h(A)).status_code == 202
    jobs(ctx)
    after = [s["revision"]["id"] for s in client.get(f"/api/papers/{paper}", headers=h(A)).json()["slots"]]
    assert after[0] == before[0] and after[2] == before[2] and after[1] != before[1]


def test_edit_and_approve(client, paper):
    slot = client.get(f"/api/papers/{paper}", headers=h(A)).json()["slots"][0]
    content = dict(slot["revision"]["content"])
    url = f"/api/papers/{paper}/slots/1"
    assert client.put(url, json={"content": dict(content, answer="  ")}, headers=h(A)).status_code == 422
    r = client.put(url, json={"content": dict(content, answer="Edited answer")}, headers=h(A))
    assert r.status_code == 200 and r.json()["author"] == "faculty"
    for n in (1, 2, 3):
        assert client.post(f"/api/papers/{paper}/slots/{n}/approve", headers=h(A)).status_code == 200
    slots = client.get(f"/api/papers/{paper}", headers=h(A)).json()["slots"]
    assert [s["status"] for s in slots] == ["approved"] * 3
    assert client.post(f"/api/papers/{paper}/slots/1/approve", headers=h(A)).status_code == 409


def test_foreign_paper_hidden(client, paper):
    assert client.get(f"/api/papers/{paper}", headers=h(B)).status_code == 404
    assert client.put(f"/api/papers/{paper}/slots/1", json={"content": {}}, headers=h(B)).status_code == 404
    assert client.post(f"/api/papers/{paper}/slots/1/approve", headers=h(B)).status_code == 404


def test_no_ready_documents(client):
    sid = mk_subject(client)
    tid = client.post(f"/api/subjects/{sid}/topics", json={"title": "T"}, headers=h(A)).json()["id"]
    tpl = client.post("/api/templates", json={"name": "t", "spec": SPEC}, headers=h(A)).json()["id"]
    r = client.post("/api/papers", headers=h(A), json={
        "subject_id": sid, "template_id": tpl, "topic_ids": [tid], "title": "P"})
    assert r.status_code == 422 and r.json()["error"]["message"] == "the subject has no ready documents"


def test_slot_rows_exist(ctx, paper):
    with ctx.session_factory() as s:
        assert s.query(Slot).filter_by(paper_id=paper).count() == 3
