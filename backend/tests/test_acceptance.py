"""First-ticket acceptance: the whole product through the HTTP API with offline fakes."""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from pdf_fixtures import FakeOcrEngine, digital_pdf, scanned_pdf

from tribunal.api.app import create_app
from tribunal.cli import main
from tribunal.context import build_context
from tribunal.db.models import Job, Page, Paper, Passage
from tribunal.generation.providers.base import ProviderResult
from tribunal.generation.providers.stub import StubProvider
from tribunal.jobs.queue import enqueue
from tribunal.jobs.worker import run_once
from tribunal.settings import load_settings, reset_settings_cache
from tribunal.textpolicy import TextPolicy

A, B = "a@example.com", "b@example.com"
ROOT = Path(__file__).resolve().parents[1]

ML_PAGES = [
    "Decision trees split the training data into branches by asking one feature question at each node. "
    "Each leaf then predicts the majority label of the examples that reach it.",
    "Linear regression fits a straight line to the data by minimising the squared error of its predictions. "
    "The slope shows how strongly the input moves the output.",
    "Neural networks stack layers of weighted sums and nonlinear activations to learn complex mappings. "
    "Training adjusts the weights step by step to reduce the loss.",
]
ML_BOOKMARKS = [("Decision Trees", 1), ("Linear Regression", 2), ("Neural Networks", 3)]
HISTORY_OCR = {
    1: "1 The French Revolution\nThe Bastille was stormed by the Parisian crowd and royal power began to collapse.",
    2: "2 The Industrial Revolution\nSteam engines and mechanised mills moved production from cottages into factories.",
}


def spec(marks_each: int = 2) -> dict:
    return {"sections": [{"title": "A", "question_type": "short_answer", "count": 3, "marks_each": marks_each,
                          "bloom": "understand", "difficulty": "easy"}], "total_marks": 3 * marks_each}


def h(email):
    return {"X-Dev-User": email}


def drain(ctx):
    while run_once(ctx):
        pass


def new_subject(client, who, name):
    r = client.post("/api/subjects", json={"name": name}, headers=h(who))
    assert r.status_code == 201
    return r.json()["id"]


def ingest(client, ctx, sid, who, data):
    r = client.post(f"/api/subjects/{sid}/documents", headers=h(who),
                    files={"file": ("book.pdf", data, "application/pdf")})
    assert r.status_code == 202
    drain(ctx)
    docs = client.get(f"/api/subjects/{sid}/documents", headers=h(who)).json()
    assert [d["status"] for d in docs] == ["ready"], docs
    return docs[0]["id"]


def topics(client, sid, who):
    return client.get(f"/api/subjects/{sid}/topics", headers=h(who)).json()


def make_paper(client, ctx, sid, who, topic_ids, marks_each=2, queue_wait=0.0):
    tpl = client.post("/api/templates", json={"name": "t", "spec": spec(marks_each)}, headers=h(who))
    assert tpl.status_code == 201
    r = client.post("/api/papers", headers=h(who), json={
        "subject_id": sid, "template_id": tpl.json()["id"], "topic_ids": topic_ids, "title": "P"})
    assert r.status_code == 202
    time.sleep(queue_wait)
    drain(ctx)
    return r.json()["paper_id"]


def get_paper(client, pid, who):
    r = client.get(f"/api/papers/{pid}", headers=h(who))
    assert r.status_code == 200
    return r.json()


def approve_all(client, pid, who):
    for n in (1, 2, 3):
        assert client.post(f"/api/papers/{pid}/slots/{n}/approve", headers=h(who)).status_code == 200


def test_first_ticket(test_env, settings, fake_ocr, ctx):
    fake_ocr.texts.update(HISTORY_OCR)
    client = TestClient(create_app(ctx))
    sa, sb = new_subject(client, A, "Machine Learning"), new_subject(client, B, "Modern History")
    doc_a = ingest(client, ctx, sa, A, digital_pdf(ML_PAGES, ML_BOOKMARKS))
    doc_b = ingest(client, ctx, sb, B, scanned_pdf(2))

    with ctx.session_factory() as s:
        assert {p.method for p in s.query(Page).filter_by(document_id=doc_a)} == {"native"}
        assert {p.method for p in s.query(Page).filter_by(document_id=doc_b)} == {"ocr"}
    ta, tb = topics(client, sa, A), topics(client, sb, B)
    assert [t["title"] for t in ta] == ["Decision Trees", "Linear Regression", "Neural Networks"]
    assert [t["source"] for t in ta] == ["bookmark"] * 3
    assert [t["title"] for t in tb] == ["1 The French Revolution", "2 The Industrial Revolution"]
    assert {t["source"] for t in tb} == {"heading"}

    pa = make_paper(client, ctx, sa, A, [t["id"] for t in ta], queue_wait=0.2)
    pb = make_paper(client, ctx, sb, B, [tb[0]["id"], tb[1]["id"]], queue_wait=0.2)

    paper_a, paper_b = get_paper(client, pa, A), get_paper(client, pb, B)
    for paper, who, doc in ((paper_a, A, doc_a), (paper_b, B, doc_b)):
        assert [s["status"] for s in paper["slots"]] == ["drafted"] * 3
        assert {s["marks"] for s in paper["slots"]} == {2}
        for slot in paper["slots"]:
            assert slot["citations"]
            for c in slot["citations"]:
                assert c["document_id"] == doc
                passage = client.get(f"/api/passages/{c['passage_id']}", headers=h(who)).json()
                with ctx.session_factory() as s:
                    page = s.query(Page).filter_by(document_id=doc, page_number=passage["page_number"]).one()
                assert passage["text"] in page.text
                assert c["page_number"] == passage["page_number"]
                assert passage["url"].endswith(f"#page={page.page_number}")
                assert c["url"] == passage["url"]

    # Faculty B asks, in a separate paper, for a topic that only A's book covers.
    trees = client.post(f"/api/subjects/{sb}/topics", json={"title": "Decision Trees"}, headers=h(B)).json()
    gap = get_paper(client, make_paper(client, ctx, sb, B, [trees["id"]]), B)
    for slot in gap["slots"]:
        assert (slot["status"], slot["revision"], slot["citations"]) == ("evidence_gap", None, [])
        assert "Decision Trees" in slot["status_detail"]
    with ctx.session_factory() as s:
        a_ids = {doc_a} | {p.id for p in s.query(Passage).filter_by(document_id=doc_a)}
    gap_json = json.dumps(gap)
    assert not [i for i in a_ids if i in gap_json]

    # The mirror: A's book has no French Revolution, and "The" in the title must not fake a match.
    fr = client.post(f"/api/subjects/{sa}/topics", json={"title": "1 The French Revolution"}, headers=h(A)).json()
    gap_a = get_paper(client, make_paper(client, ctx, sa, A, [fr["id"]]), A)
    assert [s["status"] for s in gap_a["slots"]] == ["evidence_gap"] * 3

    before = [s["revision"]["id"] for s in paper_a["slots"]]
    assert client.post(f"/api/papers/{pa}/slots/2/regenerate", headers=h(A)).status_code == 202
    drain(ctx)
    after = [s["revision"]["id"] for s in get_paper(client, pa, A)["slots"]]
    assert after[0] == before[0] and after[2] == before[2] and after[1] != before[1]

    approve_all(client, pa, A)
    key = client.get(f"/api/papers/{pa}/export?part=key&format=json", headers=h(A))
    assert key.status_code == 200
    text = json.dumps(key.json())
    assert len(re.findall(r'"answer"', text)) == 3 and "page" in text
    pdf = client.get(f"/api/papers/{pa}/export?part=paper&format=pdf", headers=h(A))
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")

    cite = paper_a["slots"][0]["citations"][0]["passage_id"]
    for url in (f"/api/papers/{pa}", f"/api/passages/{cite}", f"/api/documents/{doc_a}/file",
                f"/api/papers/{pa}/export?part=key&format=json"):
        assert client.get(url, headers=h(B)).status_code == 404, url

    ta_t = get_paper(client, pa, A)["timings"]
    tb_t = get_paper(client, pb, B)["timings"]
    for t in (ta_t, tb_t):
        assert 0.2 <= t["first_question_seconds"] <= t["total_seconds"]  # queue wait is counted
    searchable = {who: client.get(f"/api/subjects/{sid}/documents", headers=h(who)).json()[0]["searchable_seconds"]
                  for sid, who in ((sa, A), (sb, B))}
    assert None not in searchable.values()
    print(f"timings A={ta_t} B={tb_t} searchable A={searchable[A]} B={searchable[B]}")


def test_return_later(test_env, settings, ctx):
    client = TestClient(create_app(ctx))
    sid = new_subject(client, A, "Machine Learning")
    doc = ingest(client, ctx, sid, A, digital_pdf(ML_PAGES, ML_BOOKMARKS))
    pid = make_paper(client, ctx, sid, A, [t["id"] for t in topics(client, sid, A)])
    approve_all(client, pid, A)

    def kept(slot):
        rev = slot["revision"]
        return rev["id"], slot["citations"], slot["status"], rev["approved_by"], rev["approved_at"]

    def key_citations():
        key = client.get(f"/api/papers/{pid}/export?part=key&format=json", headers=h(A))
        assert key.status_code == 200
        return [c["passage_id"] for sec in key.json()["sections"] for q in sec["questions"] for c in q["citations"]]

    before = get_paper(client, pid, A)["slots"]
    key_citations()
    with ctx.session_factory.begin() as s:  # the faculty member comes back the next day
        paper = s.get(Paper, pid)
        paper.created_at -= timedelta(days=1)
        paper.started_at -= timedelta(days=1)

    assert client.post(f"/api/papers/{pid}/slots/2/regenerate", headers=h(A)).status_code == 202
    drain(ctx)
    after = get_paper(client, pid, A)["slots"]
    assert [kept(after[0]), kept(after[2])] == [kept(before[0]), kept(before[2])]
    assert after[1]["revision"]["id"] != before[1]["revision"]["id"]
    assert after[1]["status"] in ("drafted", "needs_review")
    assert (after[1]["revision"]["approved_by"], after[1]["revision"]["approved_at"]) == (None, None)
    blocked = client.get(f"/api/papers/{pid}/export?part=key&format=json", headers=h(A))
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "export_blocked"
    assert "question 2 is not approved" in blocked.json()["error"]["details"]
    assert client.post(f"/api/papers/{pid}/slots/2/approve", headers=h(A)).status_code == 200
    cited = key_citations()

    with ctx.session_factory.begin() as s:  # reprocess the identical book
        job_id = enqueue(s, "ingest", doc, ctx.snapshot.snapshot_id).id
    drain(ctx)
    with ctx.session_factory() as s:
        assert s.get(Job, job_id).status == "done"
    assert [d["status"] for d in client.get(f"/api/subjects/{sid}/documents", headers=h(A)).json()] == ["ready"]
    assert key_citations() == cited
    for passage_id in cited:
        assert client.get(f"/api/passages/{passage_id}", headers=h(A)).status_code == 200


def test_new_subject_from_headings_only(test_env, settings, ctx):
    pages = ["Coral reefs shelter a quarter of marine species although they cover a tiny part of the seabed. "
             "Warm shallow water keeps the reef builders alive."]
    client = TestClient(create_app(ctx))
    sid = new_subject(client, A, "Marine Biology")
    ingest(client, ctx, sid, A, digital_pdf(pages))
    assert topics(client, sid, A) == []
    t = client.post(f"/api/subjects/{sid}/topics", json={"title": "Coral Reefs"}, headers=h(A)).json()
    pid = make_paper(client, ctx, sid, A, [t["id"]])
    assert {s["status"] for s in get_paper(client, pid, A)["slots"]} == {"drafted"}


class _EmojiProvider:
    def __init__(self, cfg):
        self._stub = StubProvider(cfg)

    def complete(self, role, system, user):
        res = self._stub.complete(role, system, user)
        if role.prompt != "writer":
            return res
        data = json.loads(res.text)
        data["question"] = ("Explain \U0001F333 entropy: H = −Σ p log₂ p (café, 日本語) "
                            "\U0001F44D\U0001F3FD " + data["question"])
        data["answer"] += " \U0001F680"
        return ProviderResult(json.dumps(data), res.model, res.input_tokens, res.output_tokens)


def test_emoji_response_through_generation_and_export(test_env, settings, cfg, policy, fake_ocr):
    ctx = build_context(settings, ocr_override=fake_ocr, migrate=True, provider_factory=lambda n, c: _EmojiProvider(c))
    client = TestClient(create_app(ctx))
    sid = new_subject(client, A, "Machine Learning")
    ingest(client, ctx, sid, A, digital_pdf(ML_PAGES, ML_BOOKMARKS))
    pid = make_paper(client, ctx, sid, A, [t["id"] for t in topics(client, sid, A)])
    slots = get_paper(client, pid, A)["slots"]
    assert [s["status"] for s in slots] == ["drafted"] * 3
    for s in slots:
        q = s["revision"]["content"]["question"]
        assert "H = −Σ p log₂ p (café, 日本語)" in q
        assert policy.findings(q) == []
    approve_all(client, pid, A)
    key = client.get(f"/api/papers/{pid}/export?part=key&format=json", headers=h(A))
    assert key.status_code == 200
    text = json.dumps(key.json(), ensure_ascii=False)
    assert TextPolicy(cfg.text_policy.preserve_codepoints).findings(text) == []
    for needle in ("−Σ", "log₂", "café", "日本語"):
        assert needle in text


class _CountingProvider:
    """Stub provider that records the peak number of concurrent writer calls."""

    def __init__(self, cfg, tally):
        self._stub, self._tally = StubProvider(cfg), tally

    def complete(self, role, system, user):
        if role.prompt != "writer":
            return self._stub.complete(role, system, user)
        with self._tally["lock"]:
            self._tally["now"] += 1
            self._tally["max"] = max(self._tally["max"], self._tally["now"])
        try:
            time.sleep(0.05)
            return self._stub.complete(role, system, user)
        finally:
            with self._tally["lock"]:
                self._tally["now"] -= 1


def _profile(tmp_path, monkeypatch, n, model, per_paper, lang, marks):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / f'p{n}.db'}")
    monkeypatch.setenv("STORAGE_LOCAL_ROOT", str(tmp_path / f"store{n}"))
    monkeypatch.setenv("TRIBUNAL_WRITER_MODEL", model)
    monkeypatch.setenv("TRIBUNAL_PER_PAPER", str(per_paper))
    monkeypatch.setenv("TRIBUNAL_OCR_LANGUAGE", lang)
    reset_settings_cache()
    ocr = FakeOcrEngine(HISTORY_OCR)
    tally = {"lock": threading.Lock(), "now": 0, "max": 0}
    ctx = build_context(load_settings(), ocr_override=ocr, migrate=True,
                        provider_factory=lambda n, c: _CountingProvider(c, tally))
    assert ctx.snapshot.config.concurrency.per_paper == per_paper
    client = TestClient(create_app(ctx))
    sid = new_subject(client, A, "Modern History")
    ingest(client, ctx, sid, A, scanned_pdf(2))
    pid = make_paper(client, ctx, sid, A, [t["id"] for t in topics(client, sid, A)], marks)
    return ocr, get_paper(client, pid, A), tally["max"]


def test_two_deployment_profiles(test_env, tmp_path, monkeypatch):
    ocr1, p1, peak1 = _profile(tmp_path, monkeypatch, 1, "stub-writer-a", 1, "eng", 2)
    ocr2, p2, _ = _profile(tmp_path, monkeypatch, 2, "stub-writer-b", 3, "hin", 5)
    assert peak1 == 1
    assert {s["revision"]["model"] for s in p1["slots"]} == {"stub-writer-a"}
    assert {s["revision"]["model"] for s in p2["slots"]} == {"stub-writer-b"}
    assert {tuple(lang) for _, lang in ocr1.calls} == {("eng",)}
    assert {tuple(lang) for _, lang in ocr2.calls} == {("hin",)}
    assert p1["config_snapshot_id"] != p2["config_snapshot_id"]
    assert {s["marks"] for s in p1["slots"]} == {2} and {s["marks"] for s in p2["slots"]} == {5}


def test_missing_values_are_actionable(test_env, monkeypatch, capsys):
    monkeypatch.setenv("RUNTIME_CONFIG", str(ROOT / "config" / "runtime.example.yaml"))
    monkeypatch.delenv("TRIBUNAL_WRITER_MODEL", raising=False)
    reset_settings_cache()
    assert main(["check-config"]) == 2
    assert "TRIBUNAL_WRITER_MODEL" in capsys.readouterr().err


def test_no_sample_course_data_in_product_code():
    banned = re.compile(r"AIML|machine learning|modern history|gpt-|claude-", re.I)
    co1 = re.compile(r"\bCO1\b")
    hits, scanned = [], 0
    for base in (ROOT / "src", ROOT / "config"):
        for p in base.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                scanned += 1
                text = p.read_text(encoding="utf-8", errors="ignore")
                if banned.search(text) or co1.search(text):
                    hits.append(str(p))
    assert scanned > 0
    assert hits == []
