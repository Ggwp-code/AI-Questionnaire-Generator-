from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pdf_fixtures import digital_pdf
from sqlalchemy import select

from tribunal.api.app import create_app
from tribunal.db.models import ConfigSnapshotRow, Document, Job, Page, Paper, Passage, Slot
from tribunal.ingestion.pipeline import ingest_document
from tribunal.jobs.worker import run_once
from tribunal.runtime_config import ConfigSnapshot

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


def approve_slot(client, paper, n, who=A):
    rev = client.get(f"/api/papers/{paper}", headers=h(who)).json()["slots"][n - 1]["revision"]["id"]
    return client.post(f"/api/papers/{paper}/slots/{n}/approve", json={"revision_id": rev}, headers=h(who))


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
    blocked = client.get(f"/api/papers/{paper}/export?part=key&format=json", headers=h(A))
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "export_blocked"
    assert blocked.json()["error"]["details"]
    # the stub writes near-identical questions, which export refuses, so give 2 and 3 distinct wording
    for n, q in ((2, "Describe how pruning limits tree depth"), (3, "Explain least squares fitting of a regression line")):
        c = dict(client.get(f"/api/papers/{paper}", headers=h(A)).json()["slots"][n - 1]["revision"]["content"], question=q)
        assert client.put(f"/api/papers/{paper}/slots/{n}", json={"content": c}, headers=h(A)).status_code == 200
    c = dict(content, answer="Edited answer", question="Why do decision trees overfit noisy training data")
    assert client.put(url, json={"content": c}, headers=h(A)).status_code == 200
    for n in (1, 2, 3):
        assert approve_slot(client, paper, n).status_code == 200
    slots = client.get(f"/api/papers/{paper}", headers=h(A)).json()["slots"]
    assert [s["status"] for s in slots] == ["approved"] * 3
    assert approve_slot(client, paper, 1).status_code == 409
    key = client.get(f"/api/papers/{paper}/export?part=key&format=json", headers=h(A))
    assert key.status_code == 200 and str(key.json()).count("Edited answer") >= 1
    pdf = client.get(f"/api/papers/{paper}/export?part=paper&format=pdf", headers=h(A))
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf" and pdf.content.startswith(b"%PDF")
    assert client.get(f"/api/papers/{paper}/export?part=bad", headers=h(A)).status_code == 422


def test_foreign_paper_hidden(client, paper):
    assert client.get(f"/api/papers/{paper}", headers=h(B)).status_code == 404
    assert client.get(f"/api/papers/{paper}/export?part=key", headers=h(B)).status_code == 404
    assert client.put(f"/api/papers/{paper}/slots/1", json={"content": {}}, headers=h(B)).status_code == 404
    assert client.post(f"/api/papers/{paper}/slots/1/approve", json={"revision_id": "x"},
                       headers=h(B)).status_code == 404


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


def test_oversized_body_rejected_before_sign_in(ctx):
    ing = ctx.snapshot.config.ingestion.model_copy(update={"max_upload_bytes": 1000})
    cfg = ctx.snapshot.config.model_copy(update={"ingestion": ing})
    ctx.snapshot = ctx.snapshot.model_copy(update={"config": cfg})
    c = TestClient(create_app(ctx))
    r = c.post("/api/subjects/x/documents", files={"file": ("d.pdf", b"%PDF-" + b"0" * (1024 * 1024 + 2000))})
    assert r.status_code == 413  # not 401: the limit runs first


def test_commit_failure_is_visible(client, ctx, request):
    # the commit must happen before the response is sent, so a failure cannot hide behind a 2xx
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    def mark(session, _ctx):
        if any(type(o).__name__ == "Subject" for o in session.new):
            session.info["boom"] = True

    def boom(session):
        if session.info.get("boom"):
            raise RuntimeError("commit failed")

    event.listen(Session, "after_flush", mark)
    event.listen(Session, "before_commit", boom)
    request.addfinalizer(lambda: (event.remove(Session, "after_flush", mark),
                                  event.remove(Session, "before_commit", boom)))
    c = TestClient(create_app(ctx), raise_server_exceptions=False)
    assert c.post("/api/subjects", json={"name": "S"}, headers=h(A)).status_code == 500


def test_admin_revoked_when_removed_from_admin_emails(client, ctx, monkeypatch):
    from tribunal.context import build_context

    sid = mk_subject(client, A)
    monkeypatch.setenv("ADMIN_EMAILS", B)
    from tribunal.settings import load_settings

    admin_ctx = build_context(load_settings(), ocr_override=ctx.ocr_override)
    c = TestClient(create_app(admin_ctx))
    assert c.get(f"/api/subjects/{sid}/documents", headers=h(B)).status_code == 200
    monkeypatch.delenv("ADMIN_EMAILS")
    plain = TestClient(create_app(build_context(load_settings(), ocr_override=ctx.ocr_override)))
    assert plain.get(f"/api/subjects/{sid}/documents", headers=h(B)).status_code == 404


def test_retired_outline_topic_hidden_and_refused(client, ctx):
    from tribunal.ingestion.pipeline import ingest_document
    from tribunal.runtime_config import ConfigSnapshot

    sid = mk_subject(client)
    did = upload(client, sid, digital_pdf(["1 Decision Trees\n" + "Trees split data. " * 20,
                                           "2 Linear Regression\n" + "Lines fit data. " * 20])).json()["document"]["id"]
    jobs(ctx)
    topics = {t["title"]: t["id"] for t in client.get(f"/api/subjects/{sid}/topics", headers=h(A)).json()}
    client.patch(f"/api/topics/{topics['2 Linear Regression']}", json={"title": "Regression"}, headers=h(A))
    cfg = ctx.snapshot.config
    ing = cfg.ingestion.model_copy(update={"outline": cfg.ingestion.outline.model_copy(update={"use_headings": False})})
    ingest_document(ctx, ConfigSnapshot(config=cfg.model_copy(update={"ingestion": ing}), content_hash="nh"), did)
    listed = client.get(f"/api/subjects/{sid}/topics", headers=h(A)).json()
    assert [t["title"] for t in listed] == ["Regression"]  # a faculty rename survives the rebuild
    tpl = client.post("/api/templates", json={"name": "t", "spec": SPEC}, headers=h(A)).json()
    r = client.post("/api/papers", headers=h(A), json={
        "subject_id": sid, "template_id": tpl["id"], "topic_ids": [topics["1 Decision Trees"]], "title": "P"})
    assert r.status_code == 422 and r.json()["error"]["message"] == "unknown topic"


def test_documents_list_reports_excluded_pages(client, ctx, fake_ocr):
    from pdf_fixtures import scanned_pdf

    fake_ocr.texts = {1: "plain prose " * 20, 2: "H = -\u03a3 p log p\n" + "entropy measures impurity " * 10}
    sid = mk_subject(client)
    upload(client, sid, scanned_pdf(2))
    jobs(ctx)
    [d] = client.get(f"/api/subjects/{sid}/documents", headers=h(A)).json()
    assert d["status"] == "ready" and d["error"] is None and d["excluded_pages"] == [2]


def test_reprocess_reingests_a_failed_document(client, ctx):
    sid = mk_subject(client)
    did = upload(client, sid, b"%PDF-not really a pdf").json()["document"]["id"]
    jobs(ctx)
    [d] = client.get(f"/api/subjects/{sid}/documents", headers=h(A)).json()
    assert d["status"] == "failed"
    with ctx.session_factory() as s:
        key = s.get(Document, did).storage_key
    ctx.store.put(key, digital_pdf(PAGES))  # the file was replaced or the fault cleared
    r = client.post(f"/api/documents/{did}/reprocess", headers=h(A))
    assert r.status_code == 202 and r.json()["document"]["status"] == "queued" and r.json()["job_id"]
    jobs(ctx)
    [d] = client.get(f"/api/subjects/{sid}/documents", headers=h(A)).json()
    assert d["status"] == "ready" and d["error"] is None


def test_reprocess_refused_while_queued_and_for_non_members(client, ctx):
    sid = mk_subject(client)
    did = upload(client, sid).json()["document"]["id"]
    r = client.post(f"/api/documents/{did}/reprocess", headers=h(A))
    assert r.status_code == 409 and r.json()["error"]["message"] == "document is already being processed"
    jobs(ctx)
    assert client.post(f"/api/documents/{did}/reprocess", headers=h(B)).status_code == 404
    assert client.post("/api/documents/nope/reprocess", headers=h(A)).status_code == 404


def _oidc_client(ctx, monkeypatch, *, require_verified=True, header="Authorization"):
    import time

    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ctx.settings = ctx.settings.model_copy(update={
        "auth_mode": "oidc", "oidc_issuer": "https://idp.example", "oidc_audience": "tribunal",
        "oidc_jwks_url": "https://idp.example/jwks", "oidc_require_email_verified": require_verified,
        "auth_jwt_header": header})
    app = create_app(ctx)
    app.state.jwks_client = type("K", (), {"get_signing_key_from_jwt": lambda self, t: type(
        "S", (), {"key": key.public_key()})()})()

    def call(**claims):
        base = {"iss": "https://idp.example", "aud": "tribunal", "exp": int(time.time()) + 600,
                "email": "o@example.com", "email_verified": True}
        base.update({k: v for k, v in claims.items() if v is not None})
        for k in [k for k, v in claims.items() if v is None]:
            base.pop(k)
        token = jwt.encode(base, key, algorithm="RS256")
        value = f"Bearer {token}" if header == "Authorization" else token
        return TestClient(app).get("/api/me", headers={header: value})
    call.app, call.sign = app, lambda: jwt.encode(
        {"iss": "https://idp.example", "aud": "tribunal", "exp": int(time.time()) + 600, "email": "o@example.com"},
        key, algorithm="RS256")
    return call


def test_oidc_token_without_exp_is_refused(ctx, monkeypatch):
    call = _oidc_client(ctx, monkeypatch)
    assert call().status_code == 200
    assert call(exp=None).status_code == 401


@pytest.mark.parametrize("claim", [False, None, "true"])
def test_oidc_unverified_email_is_refused(ctx, monkeypatch, claim):
    r = _oidc_client(ctx, monkeypatch)(email_verified=claim)
    assert r.status_code == 401 and r.json()["error"]["message"] == "email is not verified"


@pytest.mark.parametrize("claim", [False, None])
def test_oidc_unverified_email_accepted_when_not_required(ctx, monkeypatch, claim):
    assert _oidc_client(ctx, monkeypatch, require_verified=False)(email_verified=claim).status_code == 200


def _slot_json(client, paper, n):
    return client.get(f"/api/papers/{paper}", headers=h(A)).json()["slots"][n - 1]


def test_approve_refuses_a_stale_revision(client, paper):
    first = _slot_json(client, paper, 1)["revision"]
    edited = client.put(f"/api/papers/{paper}/slots/1", headers=h(A),
                        json={"content": dict(first["content"], answer="A second instructor's answer")})
    assert edited.status_code == 200
    url = f"/api/papers/{paper}/slots/1/approve"
    r = client.post(url, json={"revision_id": first["id"]}, headers=h(A))
    assert r.status_code == 409
    assert r.json()["error"]["message"] == "the question changed since you opened it; review the current revision"
    now = _slot_json(client, paper, 1)
    assert now["status"] == "drafted" and now["revision"]["id"] == edited.json()["id"]
    assert now["revision"]["approved_by"] is None
    assert client.post(url, headers=h(A)).status_code == 422  # the revision id is required
    assert client.post(url, json={"revision_id": edited.json()["id"]}, headers=h(A)).status_code == 200


def test_regenerate_enqueues_with_the_papers_own_snapshot(client, ctx, paper):
    newer = ctx.snapshot.model_copy(update={"content_hash": "f" * 64})
    with ctx.session_factory.begin() as s:
        s.add(ConfigSnapshotRow(id=newer.snapshot_id, content=newer.as_json()))
    ctx.snapshot = newer
    r = client.post(f"/api/papers/{paper}/slots/2/regenerate", headers=h(A))
    assert r.status_code == 202
    with ctx.session_factory() as s:
        assert s.get(Job, r.json()["job_id"]).config_snapshot_id == s.get(Paper, paper).config_snapshot_id
        assert s.get(Paper, paper).config_snapshot_id != newer.snapshot_id


def test_edit_may_not_cite_a_superseded_extraction(client, ctx, paper):
    first = _slot_json(client, paper, 1)["revision"]
    cited = first["content"]["citations"]
    with ctx.session_factory() as s:
        did = s.get(Paper, paper).document_ids[0]
        old = set(s.scalars(select(Passage.id).where(Passage.document_id == did)))
    cfg = ctx.snapshot.config
    ing = cfg.ingestion.model_copy(update={"outline": cfg.ingestion.outline.model_copy(update={"use_headings": False})})
    ingest_document(ctx, ConfigSnapshot(config=cfg.model_copy(update={"ingestion": ing}), content_hash="nh"), did)
    with ctx.session_factory() as s:
        assert set(s.scalars(select(Passage.id).where(Passage.document_id == did))) > old  # re-indexed
    stale = sorted(old - set(cited))
    assert stale, "the fixture needs a passage the current revision does not cite"
    url = f"/api/papers/{paper}/slots/1"
    refused = client.put(url, headers=h(A), json={"content": dict(first["content"], citations=cited + stale[:1])})
    assert refused.status_code == 422
    kept = client.put(url, headers=h(A), json={"content": dict(first["content"], answer="Edited answer")})
    assert kept.status_code == 200


def test_oidc_reads_configured_header(ctx, monkeypatch):
    call = _oidc_client(ctx, monkeypatch, require_verified=False, header="Cf-Access-Jwt-Assertion")
    assert call().status_code == 200
    r = TestClient(call.app).get("/api/me", headers={"Authorization": f"Bearer {call.sign()}"})
    assert r.status_code == 401


def test_subject_papers_listed_newest_first_and_scoped(client, ctx, paper, ready):
    sid, _ = ready
    tpl = client.post("/api/templates", json={"name": "t", "spec": SPEC}, headers=h(A)).json()["id"]
    tid = client.get(f"/api/subjects/{sid}/topics", headers=h(A)).json()[0]["id"]
    second = client.post("/api/papers", headers=h(A), json={
        "subject_id": sid, "template_id": tpl, "topic_ids": [tid], "title": "Second"}).json()["paper_id"]
    rows = client.get(f"/api/subjects/{sid}/papers", headers=h(A)).json()
    assert [r["id"] for r in rows] == [second, paper]
    assert set(rows[0]) == {"id", "title", "status", "created_at", "template"}
    assert rows[0]["template"] == {"name": "t", "version": 2}
    assert client.get(f"/api/subjects/{sid}/papers", headers=h(B)).status_code == 404


def test_options_lists_configured_values(client, ctx):
    cfg = ctx.snapshot.config
    r = client.get("/api/options", headers=h(A))
    assert r.status_code == 200
    body = r.json()
    assert body["question_types"] == [{"key": k, "label": v.label}
                                      for k, v in cfg.generation.question_types.items() if v.enabled]
    assert body["bloom_levels"] == [{"key": b.key, "label": b.label} for b in cfg.generation.bloom_levels]
    assert body["difficulty_levels"] == [{"key": d.key, "label": d.label} for d in cfg.generation.difficulty_levels]
    assert body["default_ocr_languages"] == cfg.ingestion.ocr.languages
    assert client.get("/api/options").status_code == 401


def test_options_omits_disabled_question_types(client, ctx):
    cfg = ctx.snapshot.config
    off, *rest = cfg.generation.question_types
    qts = cfg.generation.question_types
    qts[off] = qts[off].model_copy(update={"enabled": False})
    keys = [q["key"] for q in client.get("/api/options", headers=h(A)).json()["question_types"]]
    assert off not in keys and keys == rest
