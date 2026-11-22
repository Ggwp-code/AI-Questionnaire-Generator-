from __future__ import annotations

import json
import re
from datetime import timedelta

import pytest
import yaml
from conftest import make_subject, make_user
from pdf_fixtures import digital_pdf

from tribunal.context import build_context
from tribunal.db.models import Document, Job, Paper, PaperTemplate, Passage, Revision, Slot, Topic
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
        self.writer = self.reviewer = self.reasoner = self.codegen = None
        self.prompts: list[str] = []

    def complete(self, role, system, user):
        beh = self.reasoner if role.model == "stub-reasoner" else (
            self.writer if role.prompt == "writer" else self.reviewer)
        if role.prompt == "codegen":
            beh = self.codegen
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


def make_paper(ctx, topics=("Decision trees", "Linear regression", "Neural networks"), n=3, question_type="short_answer"):
    section = dict(SPEC["sections"][0], count=n, question_type=question_type)
    spec = dict(SPEC, sections=[section], total_marks=2 * n)
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
    template = ctx.snapshot.config.prompts["writer"]
    assert (rev.model, rev.prompt_version) == ("stub-writer", f"{template.id}@{template.version}")
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
    done = go(ctx, pid)
    assert {s.status for s in done} == {expected}
    for slot in done:
        review = revs(ctx, slot.id)[-1].review
        assert review["disputed"] == {"verdict": "revise", "reasons": ["because"], "model": review["disputed"]["model"]}
        assert review["verdict"] == reasoner
        detail = "the reasoner overruled the reviewer: because" if reasoner == "pass" else "because"
        assert slot.status_detail == detail


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
        FINAL_FAILURE["generate_paper"](s, ClaimedJob("j", "generate_paper", pid, ctx.snapshot.snapshot_id, 3), "boom")
    sl = slots(ctx, pid)
    assert {s.status for s in sl} == {"failed"} and all("boom" in s.status_detail for s in sl)
    with ctx.session_factory() as s:
        assert s.get(Paper, pid).status == "failed"


@pytest.mark.parametrize("late_failure", [False, True])
def test_reclaimed_generation_preserves_winning_result(settings, fake_ocr, fake_holder, late_failure):
    ctx = make_ctx(settings, fake_ocr, fake_holder)
    pid, _ = make_paper(ctx, topics=("Decision trees",), n=1)
    with ctx.session_factory.begin() as s:
        jid = enqueue(s, "generate_paper", pid, ctx.snapshot.snapshot_id).id

    def result():
        with ctx.session_factory() as s:
            slot = s.query(Slot).filter_by(paper_id=pid).one()
            paper = s.get(Paper, pid)
            job = s.get(Job, jid)
            return (slot.status, paper.status, job.status, paper.tokens_used, job.reserved_tokens,
                    slot.current_revision_id, paper.completed_at, s.query(Revision).count())

    winner = []

    def paused_writer(user):
        with ctx.session_factory.begin() as s:
            job = s.get(Job, jid)
            assert job.reserved_tokens > 0 and job.reserved_tokens == s.get(Paper, pid).tokens_used
            job.lease_until = utcnow() - timedelta(seconds=1)
        fake_holder["fake"].writer = None
        assert run_once(ctx)
        winner.append(result())
        if late_failure:
            raise RuntimeError("stale provider failure")
        return _good(user)

    fake_holder["fake"].writer = paused_writer
    assert run_once(ctx)
    assert [result()] == winner
    slot, paper, job, used, reserved, _, _, count = winner[0]
    assert (slot, paper, job, reserved, count) == ("drafted", "drafted", "done", 0, 1)
    assert 0 < used < ctx.snapshot.config.budgets.paper_tokens  # settled to real usage


def test_replayed_regeneration_job_keeps_the_saved_result_and_the_approval(gen):
    from tribunal.jobs.queue import claim_next

    ctx, pid, _ = gen
    go(ctx, pid)
    slot = slots(ctx, pid)[1]
    with ctx.session_factory.begin() as s:
        jid = enqueue(s, "regenerate_slot", slot.id, ctx.snapshot.snapshot_id).id
    job = claim_next(ctx.session_factory, 60)
    assert job.id == jid
    regenerate(ctx, ctx.snapshot, slot.id, owner=job)  # saves its revision, then the worker dies
    saved = slots(ctx, pid)[1]
    with ctx.session_factory.begin() as s:  # faculty approve the saved revision meanwhile
        rev = s.get(Revision, saved.current_revision_id)
        rev.approved_by, rev.approved_at = s.get(Paper, pid).created_by, utcnow()
        s.get(Slot, slot.id).status = "approved"
    count = len(revs(ctx, slot.id))
    regenerate(ctx, ctx.snapshot, slot.id, owner=job)  # the reclaimed job replays
    after = slots(ctx, pid)[1]
    assert (after.status, after.current_revision_id) == ("approved", saved.current_revision_id)
    assert len(revs(ctx, slot.id)) == count


def _enqueue_regeneration(ctx, pid):
    go(ctx, pid)
    slot = slots(ctx, pid)[1]
    with ctx.session_factory.begin() as s:
        jid = enqueue(s, "regenerate_slot", slot.id, ctx.snapshot.snapshot_id).id
    return slot, jid


def _assert_retried(ctx, pid, slot, jid):
    with ctx.session_factory() as s:
        assert s.get(Job, jid).status == "done"
    after = slots(ctx, pid)[1]
    assert after.status in ("drafted", "needs_review")
    assert after.current_revision_id == revs(ctx, slot.id)[-1].id


def test_regeneration_that_fails_after_saving_its_draft_is_retried(gen):
    ctx, pid, h = gen
    slot, jid = _enqueue_regeneration(ctx, pid)

    def boom(user):
        raise RuntimeError("dies during review")

    h["fake"].reviewer = boom
    assert run_once(ctx)  # the writer saved a revision, then the attempt raised and was requeued
    h["fake"].reviewer = None
    assert run_once(ctx)
    _assert_retried(ctx, pid, slot, jid)


def test_regeneration_whose_worker_died_mid_review_is_retried(gen):
    from tribunal.jobs.queue import claim_next

    ctx, pid, h = gen
    slot, jid = _enqueue_regeneration(ctx, pid)
    later = utcnow() + timedelta(seconds=ctx.snapshot.config.retries.lease_seconds + 1)
    taken = []

    def reclaimed(user):  # another worker takes the job while this one is in review
        taken.append(claim_next(ctx.session_factory, 60, now=later))
        return verdict("pass")(user)

    h["fake"].reviewer = reclaimed
    assert run_once(ctx)  # the old worker finds it lost ownership and stops
    assert taken[0].id == jid
    h["fake"].reviewer = None
    regenerate(ctx, ctx.snapshot, slot.id, owner=taken[0])  # the reclaiming worker runs the job
    with ctx.session_factory.begin() as s:
        s.get(Job, jid).status = "done"  # the worker would now finish it
    _assert_retried(ctx, pid, slot, jid)


def _tight_budget(test_env, fake_ocr, fake_holder, monkeypatch, tokens):
    src = test_env["runtime"]
    text = re.sub(r"file: \.\./\.\./config", f"file: {src.parent.parent.parent}/config", src.read_text())
    new = test_env["tmp"] / "runtime.yaml"
    new.write_text(text.replace("paper_tokens: 200000", f"paper_tokens: {tokens}"))
    monkeypatch.setenv("RUNTIME_CONFIG", str(new))
    reset_settings_cache()
    from tribunal.settings import load_settings
    return make_ctx(load_settings(), fake_ocr, fake_holder)


def test_call_larger_than_remaining_budget_is_refused_before_the_provider(
        test_env, monkeypatch, fake_ocr, fake_holder):
    ctx = _tight_budget(test_env, fake_ocr, fake_holder, monkeypatch, 3000)  # the writer bound is about 3700
    pid = make_paper(ctx)[0]
    for s in go(ctx, pid):
        assert s.status == "failed" and "budget" in s.status_detail
    assert fake_holder["fake"].prompts == []
    with ctx.session_factory() as s:
        assert s.get(Paper, pid).tokens_used == 0


def test_tokens_used_never_exceeds_the_paper_budget(test_env, monkeypatch, fake_ocr, fake_holder):
    ctx = _tight_budget(test_env, fake_ocr, fake_holder, monkeypatch, 9000)
    pid = make_paper(ctx)[0]
    go(ctx, pid)
    with ctx.session_factory() as s:
        assert 0 < s.get(Paper, pid).tokens_used <= 9000


def test_call_that_fits_the_remaining_budget_is_allowed(test_env, monkeypatch, fake_ocr, fake_holder):
    ctx = _tight_budget(test_env, fake_ocr, fake_holder, monkeypatch, 3800)  # below the old flat reserve of 4000
    pid = make_paper(ctx, n=1, topics=("Decision trees",))[0]
    (slot,) = go(ctx, pid)
    assert slot.status == "drafted"


@pytest.fixture
def computational(settings, test_env, fake_ocr, fake_holder):
    def setup(kind="numerical", *, codegen=True, reviewer=True):
        src = test_env["runtime"]
        runtime = yaml.safe_load(src.read_text())
        prompt_dir = src.parents[2] / "config" / "prompts"
        if codegen:
            runtime["roles"]["codegen"] = dict(provider="stub", model="stub-codegen", prompt="codegen",
                                                max_output_tokens=500)
            runtime["prompts"]["codegen"] = {}
        for name in runtime["prompts"]:
            runtime["prompts"][name]["file"] = str(prompt_dir / f"{name}.yaml")
        runtime["roles"]["math_writer"] = dict(runtime["roles"]["writer"], model="stub-math-writer")
        runtime["generation"]["question_types"][kind] = dict(label=kind, enabled=True, writer_role="math_writer")
        runtime["generation"]["reviewer_enabled"] = reviewer
        path = test_env["tmp"] / "computational.yaml"
        path.write_text(yaml.safe_dump(runtime))
        ctx = make_ctx(settings.model_copy(update={"runtime_config": path}), fake_ocr, fake_holder)
        pid, _ = make_paper(ctx, topics=("Decision trees",), n=1, question_type=kind)
        fake = fake_holder["fake"]

        def writer(user):
            draft = json.loads(_good(user))
            draft.update(question="What is twice two?", answer="4")
            if kind == "computational_mcq":
                draft.update(options=["0", "2", "4", "8"], correct_option=2)
            return json.dumps(draft)

        fake.writer = writer
        return ctx, pid, fake

    return setup


@pytest.mark.parametrize("kind", ["numerical", "calculation", "trace", "computational_mcq"])
def test_independent_computation_survives_disputed_review(computational, kind):
    ctx, pid, fake = computational(kind)
    verifier_calls = []

    def codegen(user):
        verifier_calls.append(user)
        assert "What is twice two?" in user
        assert '"answer":' not in user and "[P:" not in user and '"rubric":' not in user
        assert '"correct_option":' not in user
        if kind == "computational_mcq":
            assert '"options":' in user
        return '{"expression":"2*2"}'

    def reviewer(user):
        assert '"verification":' in user and '"expression": "2*2"' in user
        return verdict("revise")(user)

    fake.codegen, fake.reviewer, fake.reasoner = codegen, reviewer, verdict("pass")
    (slot,) = go(ctx, pid)
    assert slot.status == "drafted"
    revisions = revs(ctx, slot.id)
    assert len(revisions) == len(verifier_calls) == 2
    for rev in revisions:
        assert rev.review["verification"]["status"] == "pass"
        assert rev.model == "stub-math-writer"
    assert revisions[-1].review["disputed"]["verdict"] == "revise"


def test_mismatch_revises_writer_and_fails_when_exhausted(computational):
    ctx, pid, fake = computational(reviewer=False)
    fake.codegen = lambda _: '{"expression":"2+3"}'
    (slot,) = go(ctx, pid)
    assert slot.status == "failed"
    revisions = revs(ctx, slot.id)
    assert [r.review["verification"]["status"] for r in revisions] == ["mismatch", "mismatch"]
    assert revisions[0].review["verification"]["detail"] in fake.prompts[1]


@pytest.mark.parametrize("fault,expected,call_count", [
    ("missing", "unavailable", 0), ("provider", "unavailable", 1),
    ("invalid", "invalid", 2), ("unsupported", "unsupported", 1), ("repair", "pass", 2),
])
def test_verification_failures_and_bounded_repair(computational, fault, expected, call_count):
    ctx, pid, fake = computational(codegen=fault != "missing", reviewer=False)
    calls = []

    def codegen(user):
        calls.append(user)
        if fault == "provider":
            _raise(user)
        if fault == "unsupported":
            return '{"unsupported":true}'
        if fault == "repair" and len(calls) == 2:
            return '{"expression":"2+2"}'
        return "invalid JSON"

    fake.codegen = codegen
    (slot,) = go(ctx, pid)
    assert slot.status == ("drafted" if expected == "pass" else "needs_review")
    (rev,) = revs(ctx, slot.id)
    assert rev.review["verification"]["status"] == expected
    assert len(calls) == call_count
    if call_count == 2:
        assert "Correct this problem" in calls[1]


def test_verification_follows_structural_checks(computational):
    ctx, pid, fake = computational()
    original_writer = fake.writer

    def writer(user):
        draft = json.loads(original_writer(user))
        draft["citations"] = []
        return json.dumps(draft)

    fake.writer = writer
    fake.codegen = lambda _: pytest.fail("structurally invalid draft reached codegen")
    (slot,) = go(ctx, pid)
    assert slot.status == "failed" and "no citations" in slot.status_detail
    assert all(not r.review for r in revs(ctx, slot.id))
