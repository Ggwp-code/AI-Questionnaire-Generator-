from __future__ import annotations

import json
import re
from datetime import timedelta

import pytest
from conftest import make_subject, make_user
from pdf_fixtures import digital_pdf

from tribunal.context import build_context
from tribunal.db.models import Document, Paper, PaperTemplate, Passage, Revision, Slot, Topic
from tribunal.db.types import utcnow
from tribunal.generation.blueprint import expand_blueprint, parse_template_spec
from tribunal.generation.papers import regenerate, run_paper
from tribunal.generation.providers.base import ProviderError
from tribunal.generation.providers.stub import StubProvider
from tribunal.generation.slots import generate_slot
from tribunal.ingestion.pipeline import ingest_document
from tribunal.jobs.queue import enqueue
from tribunal.jobs.worker import run_once
from tribunal.settings import reset_settings_cache

PAGES = ["Decision trees split data into branches. " * 5,
         "Linear regression fits lines to data. " * 5,
         "Neural networks learn weights from data. " * 5]
SPEC = {"sections": [{"title": "A", "question_type": "short_answer", "count": 3, "marks_each": 2,
                      "bloom": "understand", "difficulty": "easy"}], "total_marks": 6}


class Fake:
    """Dispatches on role; unset behaviours fall through to the real stub."""

    def __init__(self, cfg):
        self.stub = StubProvider(cfg)
        self.writer = self.reviewer = self.reasoner = None
        self.prompts: list[str] = []

    def complete(self, role, system, user):
        beh = self.reasoner if role.model == "stub-reasoner" else (
            self.writer if role.prompt == "writer" else self.reviewer)
        if role.prompt == "writer":
            self.prompts.append(user)
        if beh is None:
            return self.stub.complete(role, system, user)
        out = beh(user)
        from tribunal.generation.providers.base import ProviderResult
        return ProviderResult(out, role.model, 10, 10)


def verdict(v):
    return lambda user: json.dumps({"verdict": v, "reasons": ["because"]})


def _raise(user):
    raise ProviderError("down", transient=False)


@pytest.fixture
def fake_holder():
    return {}


def make_ctx(settings, fake_ocr, holder):
    def factory(name, cfg):
        holder["fake"] = Fake(cfg)
        return holder["fake"]
    ctx = build_context(settings, ocr_override=fake_ocr, provider_factory=factory, migrate=True)
    ctx.provider(ctx.snapshot, "writer")  # creates the shared fake so tests can configure it
    return ctx


def make_paper(ctx, topics=("Decision trees", "Linear regression", "Neural networks"), n=3):
    spec = dict(SPEC, sections=[dict(SPEC["sections"][0], count=n)], total_marks=2 * n)
    with ctx.session_factory.begin() as s:
        user = make_user(s, "g@example.com")
        subj = make_subject(s, user, "Digital")
        data = digital_pdf(PAGES, [])
        ctx.store.put("d.pdf", data)
        doc = Document(subject_id=subj.id, uploaded_by=user.id, filename="d.pdf", content_hash="h",
                       storage_key="d.pdf", size_bytes=len(data), ocr_languages=["eng"])
        s.add(doc)
        tpl = PaperTemplate(owner_id=user.id, name="t", version=1, spec=spec)
        s.add(tpl)
        s.flush()
        tops = [Topic(subject_id=subj.id, title=t, source="manual") for t in topics]
        s.add_all(tops)
        s.flush()
        paper = Paper(subject_id=subj.id, created_by=user.id, template_id=tpl.id, title="P",
                      config_snapshot_id=ctx.snapshot.snapshot_id, document_ids=[doc.id],
                      topic_ids=[t.id for t in tops])
        s.add(paper)
        s.flush()
        titles = {t.id: t.title for t in tops}
        for sp in expand_blueprint(parse_template_spec(spec), paper.topic_ids, ctx.snapshot.config):
            s.add(Slot(paper_id=paper.id, number=sp.number, section=sp.section, question_type=sp.question_type,
                       marks=sp.marks, bloom=sp.bloom, difficulty=sp.difficulty, topic_id=sp.topic_id,
                       topic_title=titles[sp.topic_id]))
        ids = (paper.id, doc.id)
    ingest_document(ctx, ctx.snapshot, ids[1])
    return ids


@pytest.fixture
def gen(settings, fake_ocr, fake_holder):
    ctx = make_ctx(settings, fake_ocr, fake_holder)
    return ctx, make_paper(ctx)[0], fake_holder


def slots(ctx, pid):
    with ctx.session_factory() as s:
        return s.query(Slot).filter_by(paper_id=pid).order_by(Slot.number).all()


def revs(ctx, slot_id):
    with ctx.session_factory() as s:
        return s.query(Revision).filter_by(slot_id=slot_id).order_by(Revision.number).all()


def go(ctx, pid):
    run_paper(ctx, ctx.snapshot, pid)
    return slots(ctx, pid)


def test_three_slots_drafted(gen):
    ctx, pid, _ = gen
    sl = go(ctx, pid)
    assert [s.status for s in sl] == ["drafted"] * 3
    with ctx.session_factory() as s:
        paper = s.get(Paper, pid)
        valid = {p.id for p in s.query(Passage).filter(Passage.document_id.in_(paper.document_ids))}
        assert paper.status == "drafted" and paper.first_question_at and paper.completed_at
        assert paper.tokens_used > 0
    for slot in sl:
        rev = revs(ctx, slot.id)[-1]
        assert slot.current_revision_id == rev.id and set(rev.content["citations"]) <= valid
        assert rev.review["verdict"] == "pass"


def test_revision_records_versions(gen):
    ctx, pid, _ = gen
    rev = revs(ctx, go(ctx, pid)[0].id)[0]
    assert (rev.model, rev.prompt_version) == ("stub-writer", "writer@1")
    assert rev.retrieval_version.startswith("r-")


def test_writer_model_from_config(test_env, monkeypatch, fake_ocr, fake_holder):
    monkeypatch.setenv("TRIBUNAL_WRITER_MODEL", "stub-writer-b")
    reset_settings_cache()
    from tribunal.settings import load_settings
    ctx = make_ctx(load_settings(), fake_ocr, fake_holder)
    pid = make_paper(ctx)[0]
    assert revs(ctx, go(ctx, pid)[0].id)[0].model == "stub-writer-b"


def test_evidence_gap_slot(settings, fake_ocr, fake_holder):
    ctx = make_ctx(settings, fake_ocr, fake_holder)
    pid = make_paper(ctx, topics=("Decision trees", "Photosynthesis", "Neural networks"))[0]
    sl = go(ctx, pid)
    assert [s.status for s in sl] == ["drafted", "evidence_gap", "drafted"]
    assert "Photosynthesis" in sl[1].status_detail


def test_reviewer_exception_never_passes(gen):
    ctx, pid, h = gen
    h["fake"].reviewer = _raise
    for s in go(ctx, pid):
        assert s.status == "needs_review"
        assert revs(ctx, s.id)[-1].review["verdict"] == "faculty_review"


def test_invalid_output_fails_slot(gen):
    ctx, pid, h = gen
    h["fake"].writer = lambda u: "not json"
    for s in go(ctx, pid):
        assert s.status == "failed" and "not valid JSON" in s.status_detail
        assert revs(ctx, s.id) == []


def _good(user):
    return StubProvider._answer(user)


def test_revise_then_pass(settings, fake_ocr, fake_holder):
    ctx = make_ctx(settings, fake_ocr, fake_holder)
    pid = make_paper(ctx, n=1, topics=("Decision trees",))[0]
    calls = []

    def writer(user):
        calls.append(user)
        out = json.loads(_good(user))
        if len(calls) == 1:
            out["citations"] = []
        return json.dumps(out)

    fake_holder["fake"].writer = writer
    (slot,) = go(ctx, pid)
    assert slot.status == "drafted" and len(revs(ctx, slot.id)) == 2
    assert "- no citations" in calls[1] and "- no citations" not in calls[0]


@pytest.mark.parametrize("reasoner, expected", [("pass", "drafted"), ("revise", "needs_review")])
def test_disputed_review_escalates(gen, reasoner, expected):
    ctx, pid, h = gen
    h["fake"].reviewer = verdict("revise")
    h["fake"].reasoner = verdict(reasoner)
    assert {s.status for s in go(ctx, pid)} == {expected}


def test_budget_exhausted(test_env, monkeypatch, fake_ocr, fake_holder):
    src = test_env["runtime"]
    text = re.sub(r"file: \.\./\.\./config", f"file: {src.parent.parent.parent}/config", src.read_text())
    text = text.replace("paper_tokens: 200000", "paper_tokens: 100")
    new = test_env["tmp"] / "runtime.yaml"
    new.write_text(text)
    monkeypatch.setenv("RUNTIME_CONFIG", str(new))
    reset_settings_cache()
    from tribunal.settings import load_settings
    ctx = make_ctx(load_settings(), fake_ocr, fake_holder)
    pid = make_paper(ctx)[0]
    for s in go(ctx, pid):
        assert s.status == "failed" and "budget" in s.status_detail


def test_regenerate_preserves_others(gen):
    ctx, pid, _ = gen
    before = {s.number: s.current_revision_id for s in go(ctx, pid)}
    regenerate(ctx, ctx.snapshot, slots(ctx, pid)[1].id)
    after = {s.number: s.current_revision_id for s in slots(ctx, pid)}
    assert after[1] == before[1] and after[3] == before[3] and after[2] != before[2]
    assert [r.number for r in revs(ctx, slots(ctx, pid)[1].id)] == [1, 2]


def test_regenerate_after_original_paper_deadline(gen):
    ctx, pid, _ = gen
    go(ctx, pid)
    with ctx.session_factory.begin() as s:
        s.get(Paper, pid).created_at = utcnow() - timedelta(days=2)
    regenerate(ctx, ctx.snapshot, slots(ctx, pid)[1].id)
    slot = slots(ctx, pid)[1]
    assert (slot.status, slot.status_detail) == ("drafted", None)


def test_passed_deadline_fails_slot(gen):
    ctx, pid, _ = gen
    slot_id = slots(ctx, pid)[0].id
    generate_slot(ctx, ctx.snapshot, slot_id, deadline=utcnow() - timedelta(seconds=1))
    slot = slots(ctx, pid)[0]
    assert (slot.status, slot.status_detail) == ("failed", "paper deadline exceeded")


def test_concurrent_slots_sqlite(test_env, monkeypatch, fake_ocr, fake_holder):
    monkeypatch.setenv("TRIBUNAL_PER_PAPER", "3")
    reset_settings_cache()
    from tribunal.settings import load_settings
    ctx = make_ctx(load_settings(), fake_ocr, fake_holder)
    pid = make_paper(ctx)[0]
    assert {s.status for s in go(ctx, pid)} == {"drafted"}


def test_worker_runs_generate_job(gen):
    ctx, pid, _ = gen
    with ctx.session_factory.begin() as s:
        enqueue(s, "generate_paper", pid, ctx.snapshot.snapshot_id)
    assert run_once(ctx) is True
    with ctx.session_factory() as s:
        assert s.get(Paper, pid).status == "drafted"


def test_no_reasoner_role(test_env, monkeypatch, fake_ocr, fake_holder):
    src = test_env["runtime"]
    text = src.read_text().replace("../../config", f"{src.parent.parent.parent}/config")
    text = re.sub(r"  reasoner:\n(    .*\n)+", "", text)
    text = text.replace("escalate_disputed_reviews: true", "escalate_disputed_reviews: false")
    new = test_env["tmp"] / "runtime.yaml"
    new.write_text(text)
    monkeypatch.setenv("RUNTIME_CONFIG", str(new))
    reset_settings_cache()
    from tribunal.settings import load_settings
    ctx = make_ctx(load_settings(), fake_ocr, fake_holder)
    assert "reasoner" not in ctx.snapshot.config.roles
    assert {s.status for s in go(ctx, make_paper(ctx)[0])} == {"drafted"}


def test_running_slot_is_rerun(gen):
    ctx, pid, _ = gen
    with ctx.session_factory.begin() as s:
        s.query(Slot).filter_by(paper_id=pid, number=1).one().status = "running"
    assert {s.status for s in go(ctx, pid)} == {"drafted"}


def test_final_failure_fails_open_slots(gen):
    from tribunal.jobs.queue import ClaimedJob
    from tribunal.jobs.worker import FINAL_FAILURE
    ctx, pid, _ = gen
    with ctx.session_factory.begin() as s:
        s.query(Slot).filter_by(paper_id=pid, number=1).one().status = "running"
    FINAL_FAILURE["generate_paper"](ctx, ClaimedJob("j", "generate_paper", pid, ctx.snapshot.snapshot_id, 3), "boom")
    sl = slots(ctx, pid)
    assert {s.status for s in sl} == {"failed"} and all("boom" in s.status_detail for s in sl)
    with ctx.session_factory() as s:
        assert s.get(Paper, pid).status == "failed"
