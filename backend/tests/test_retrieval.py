from __future__ import annotations

import pytest
from conftest import make_subject, make_user
from pdf_fixtures import digital_pdf, scanned_pdf

from tribunal.db.models import Document, Subject, Topic, User
from tribunal.ingestion.pipeline import ingest_document
from tribunal.retrieval.search import retrieve


def _ingest(ctx, session, user, subject, key, data):
    ctx.store.put(key, data)
    doc = Document(subject_id=subject.id, uploaded_by=user.id, filename=key, content_hash=key,
                   storage_key=key, size_bytes=len(data), ocr_languages=["eng"])
    session.add(doc)
    session.flush()
    return doc.id


@pytest.fixture
def world(ctx, fake_ocr):
    fake_ocr.texts = {1: "The French Revolution overthrew the monarchy. " * 5,
                      2: "The Industrial Revolution changed factories. " * 5}
    with ctx.session_factory.begin() as s:
        user = make_user(s, "r@example.com")
        a = make_subject(s, user, "A")
        b = make_subject(s, user, "B")
        pdf = digital_pdf(["Decision trees split data into branches. " * 5,
                           "Linear regression fits lines to data. " * 5], [])
        doc_a = _ingest(ctx, s, user, a, "a.pdf", pdf)
        doc_b = _ingest(ctx, s, user, b, "b.pdf", scanned_pdf(2))
        ids = (a.id, b.id)
    for d in (doc_a, doc_b):
        ingest_document(ctx, ctx.snapshot, d)
    return ids + (doc_a, doc_b)


def _go(ctx, subject, docs, query, topic=None, cfg=None):
    cfg = cfg or ctx.snapshot.config.retrieval
    with ctx.session_factory() as s:
        return retrieve(s, ctx.embedder(ctx.snapshot), cfg, subject_id=subject, document_ids=docs,
                        query=query, topic=topic)


def test_trees_in_subject_a(ctx, world):
    a, _, doc_a, _ = world
    r = _go(ctx, a, [doc_a], "trees")
    assert r.gap is None and r.passages[0].page_number == 1 and "tree" in r.passages[0].text.lower()


def test_trees_in_subject_b_is_gap(ctx, world):
    _, b, _, doc_b = world
    r = _go(ctx, b, [doc_b], "trees")
    assert r.passages == [] and "trees" in r.gap


def test_other_subject_documents_ignored(ctx, world):
    _, b, doc_a, _ = world
    assert _go(ctx, b, [doc_a], "trees").gap


def test_unready_document_ignored(ctx, world):
    a, _, doc_a, _ = world
    with ctx.session_factory.begin() as s:
        s.get(Document, doc_a).status = "processing"
    assert _go(ctx, a, [doc_a], "trees").gap


def test_topic_boost_prefers_topic_pages(ctx, fake_ocr):
    with ctx.session_factory.begin() as s:
        user = make_user(s, "t@example.com")
        a = make_subject(s, user, "A")
        doc = _ingest(ctx, s, user, a, "t.pdf", digital_pdf(["Shared data topic words here. " * 5] * 2, []))
        sid = a.id
    ingest_document(ctx, ctx.snapshot, doc)
    topic = Topic(subject_id=sid, document_id=doc, title="t", page_start=2, page_end=2, source="outline")
    r = _go(ctx, sid, [doc], "data", topic=topic)
    assert r.gap is None and r.passages[0].page_number == 2


def test_version_stable(ctx, world):
    a, _, doc_a, _ = world
    cfg = ctx.snapshot.config.retrieval
    v = _go(ctx, a, [doc_a], "trees").version
    assert v == _go(ctx, a, [doc_a], "trees").version
    other = cfg.model_copy(update={"min_score": min(cfg.min_score + 0.01, 1.0) if cfg.min_score < 1 else 0.0})
    assert v != _go(ctx, a, [doc_a], "trees", cfg=other).version


@pytest.mark.parametrize("page, query, found", [
    ("The pH of a solution measures how acidic it is. ", "pH", True),
    ("The French Revolution overthrew the monarchy of France. ", "Theory of Computation", False),
])
def test_short_terms_match_but_function_words_do_not(ctx, page, query, found):
    with ctx.session_factory.begin() as s:
        user = make_user(s, "p@example.com")
        a = make_subject(s, user, "A")
        doc = _ingest(ctx, s, user, a, "p.pdf", digital_pdf([page * 5], []))
        sid = a.id
    ingest_document(ctx, ctx.snapshot, doc)
    r = _go(ctx, sid, [doc], query, topic=Topic(subject_id=sid, title=query, source="faculty"))
    assert (r.gap is None) == found


def test_common_words_in_topic_title_do_not_fake_evidence(ctx, world):
    a, _, doc_a, _ = world
    assert _go(ctx, a, [doc_a], "The Cold War and the Space Race").gap
    assert _go(ctx, a, [doc_a], "The Decision Trees").gap is None


def test_min_term_coverage_is_enforced(ctx, world):
    a, _, doc_a, _ = world
    cfg = ctx.snapshot.config.retrieval.model_copy(update={"min_term_coverage": 1.0})
    assert _go(ctx, a, [doc_a], "trees pottery", cfg=cfg).gap
    assert _go(ctx, a, [doc_a], "trees data", cfg=cfg).gap is None


def test_document_outside_selection_never_retrieved(ctx, world):
    a, _, doc_a, _ = world
    with ctx.session_factory.begin() as s:
        first = s.get(Document, doc_a)
        other = _ingest(ctx, s, s.get(User, first.uploaded_by), s.get(Subject, first.subject_id), "other.pdf",
                        digital_pdf(["Quantum entanglement links particles. " * 5], []))
    ingest_document(ctx, ctx.snapshot, other)
    assert _go(ctx, a, [other], "entanglement").gap is None
    assert _go(ctx, a, [doc_a], "entanglement").gap


def test_documents_of_another_embedder_are_left_out_and_named(ctx, world):
    a, _, doc_a, _ = world
    with ctx.session_factory.begin() as s:
        s.get(Document, doc_a).embedder_id = "some-other-model"
    r = _go(ctx, a, [doc_a], "trees")
    assert r.passages == [] and r.gap.endswith("; re-index for the current embedding model: a.pdf")
    with ctx.session_factory.begin() as s:
        s.get(Document, doc_a).embedder_id = None
    assert "re-index for the current embedding model: a.pdf" in _go(ctx, a, [doc_a], "trees").gap
    ingest_document(ctx, ctx.snapshot, doc_a)  # reprocessing records the current embedder again
    assert _go(ctx, a, [doc_a], "trees").gap is None
