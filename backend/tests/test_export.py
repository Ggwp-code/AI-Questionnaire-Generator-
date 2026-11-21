from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from conftest import make_subject, make_user
from pdf_fixtures import digital_pdf

from tribunal.context import build_context
from tribunal.db.models import Document, Paper, PaperTemplate, Passage, Revision, Slot, Topic, User
from tribunal.errors import ValidationFailed
from tribunal.export.checks import ExportBlocked
from tribunal.export.render import build_export, render_pdf
from tribunal.generation.blueprint import expand_blueprint, parse_template_spec
from tribunal.generation.papers import run_paper
from tribunal.generation.providers.stub import StubProvider
from tribunal.ingestion.pipeline import ingest_document

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
def ex(settings, fake_ocr):
    ctx = make_ctx(settings, fake_ocr, {})
    pid, doc = make_paper(ctx)
    run_paper(ctx, ctx.snapshot, pid)
    return ctx, pid, doc


def approve(ctx, pid, numbers=(1, 2, 3)):
    with ctx.session_factory.begin() as s:
        uid = s.query(User).one().id
        for slot in s.query(Slot).filter_by(paper_id=pid):
            if slot.number in numbers:
                rev = s.get(Revision, slot.current_revision_id)
                rev.approved_by, rev.approved_at = uid, datetime.now(UTC)
                slot.status = "approved"


def export(ctx, pid, part):
    with ctx.session_factory() as s:
        return build_export(s, s.get(Paper, pid), part, ctx.snapshot, ctx.policy(ctx.snapshot))


def set_content(ctx, pid, number, **kw):
    with ctx.session_factory.begin() as s:
        slot = s.query(Slot).filter_by(paper_id=pid, number=number).one()
        rev = s.get(Revision, slot.current_revision_id)
        rev.content = {**rev.content, **kw}


def test_unapproved_blocked(ex):
    ctx, pid, _ = ex
    with pytest.raises(ExportBlocked) as e:
        export(ctx, pid, "paper")
    assert "question 1 is not approved" in e.value.details and e.value.code == "export_blocked"


def test_key_has_answers_and_citations(ex):
    ctx, pid, doc = ex
    approve(ctx, pid)
    key = export(ctx, pid, "key")
    q = key["sections"][0]["questions"][0]
    assert q["answer"] and q["rubric"]
    with ctx.session_factory() as s:
        p = s.get(Passage, q["citations"][0]["passage_id"])
        assert q["citations"][0]["url"] == f"/api/documents/{doc}/file#page={p.page_number}"
    assert len(key["approvals"]) == 3 and key["approvals"][0]["approved_by"] == "g@example.com"
    assert all("answer" not in x for x in export(ctx, pid, "paper")["sections"][0]["questions"])


def test_key_survives_reingest_of_same_pdf(ex):
    ctx, pid, doc = ex
    approve(ctx, pid)
    cited = [c["passage_id"] for q in export(ctx, pid, "key")["sections"][0]["questions"] for c in q["citations"]]
    ingest_document(ctx, ctx.snapshot, doc)
    again = [c["passage_id"] for q in export(ctx, pid, "key")["sections"][0]["questions"] for c in q["citations"]]
    assert cited and again == cited


def test_duplicate_detected(ex):
    ctx, pid, _ = ex
    set_content(ctx, pid, 2, question="Explain how decision trees split data into branches.")
    set_content(ctx, pid, 1, question="Explain how decision trees split data into branches.")
    approve(ctx, pid)
    with pytest.raises(ExportBlocked) as e:
        export(ctx, pid, "key")
    assert "questions 1 and 2 are near duplicates" in e.value.details


def test_rubric_and_foreign_citation_blocked(ex):
    ctx, pid, _ = ex
    set_content(ctx, pid, 1, rubric=[{"criterion": "x", "marks": 1}], citations=["nope"])
    approve(ctx, pid)
    with pytest.raises(ExportBlocked) as e:
        export(ctx, pid, "key")
    assert "question 1 rubric totals 1, expected 2" in e.value.details
    assert "question 1 cites a passage outside this paper's sources" in e.value.details


def test_export_cleans_emoji(ex):
    ctx, pid, _ = ex
    set_content(ctx, pid, 1, question="Tree \U0001F333: \u2212\u03a3 log\u2082 caf\u00e9 \u65e5\u672c\u8a9e")
    approve(ctx, pid)
    text = json.dumps(export(ctx, pid, "key"), ensure_ascii=False)
    assert "\U0001F333" not in text
    assert "\u2212\u03a3 log\u2082 caf\u00e9 \u65e5\u672c\u8a9e" in text


def doc_with(text):
    return {"title": "T", "subject": "S", "total_marks": 2, "instructions": ["Be brief"],
            "sections": [{"title": "A", "answer_any": None,
                          "questions": [{"number": 1, "marks": 2, "question": text, "options": None}]}]}


def render(doc, cfg, font=None):
    return render_pdf(doc, font_regular=font, font_bold=None, fallbacks=[], cfg=cfg.export)


def test_pdf_core_font(cfg):
    assert render(doc_with("caf\u00e9"), cfg).startswith(b"%PDF")


def test_pdf_needs_font_for_unicode(cfg):
    with pytest.raises(ValidationFailed):
        render(doc_with("\u65e5\u672c\u8a9e"), cfg)


def test_pdf_with_font(cfg):
    font = Path("/usr/share/fonts/TTF/DejaVuSans.ttf")
    if not font.exists():
        pytest.skip("font not installed")
    assert render(doc_with("\u2212\u03a3 log\u2082"), cfg, font).startswith(b"%PDF")


def test_failed_refresh_preserves_approved_key(ex, monkeypatch):
    ctx, pid, doc = ex
    approve(ctx, pid)
    expected = export(ctx, pid, "key")

    def failed_refresh(*_args):
        assert export(ctx, pid, "key")["sections"] == expected["sections"]
        raise ValidationFailed("could not refresh index")

    monkeypatch.setattr("tribunal.ingestion.pipeline.extraction_version", lambda *_args: "changed")
    monkeypatch.setattr("tribunal.ingestion.pipeline.read_pdf", failed_refresh)
    ingest_document(ctx, ctx.snapshot, doc)
    key = export(ctx, pid, "key")
    assert key["sections"] == expected["sections"]
    assert key["approvals"] == expected["approvals"]
