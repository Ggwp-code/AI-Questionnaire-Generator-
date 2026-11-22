"""Course-scoped two-stage retrieval that reports evidence gaps instead of guessing."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from tribunal.db.models import Document, Passage, Topic
from tribunal.retrieval.embedding import Embedder, terms
from tribunal.runtime_config import RetrievalConfig, canonical_hash


@dataclass
class ScoredPassage:
    passage_id: str
    document_id: str
    page_number: int
    text: str
    score: float


@dataclass
class RetrievalResult:
    passages: list[ScoredPassage]
    gap: str | None
    version: str


def retrieve(session: Session, embedder: Embedder, cfg: RetrievalConfig, *, subject_id: str,
             document_ids: list[str], query: str, topic: Topic | None = None) -> RetrievalResult:
    rows, stale = searchable_passages(session, subject_id, document_ids, embedder.model_id)
    return rank_passages(rows, embedder, cfg, query, topic, stale)


def searchable_passages(session: Session, subject_id: str, document_ids: list[str], embedder_id: str):
    """Passages of the ready, current documents indexed with this embedder, and the filenames of
    selected ready documents left out because they were indexed with another one (or before it was recorded)."""
    rows = session.execute(
        select(Passage).join(Document, Document.id == Passage.document_id)
        .where(Passage.subject_id == subject_id, Passage.document_id.in_(document_ids),
               Document.status == "ready", Passage.extraction_version == Document.extraction_version,
               Document.embedder_id == embedder_id)
    ).scalars().all()
    stale = session.scalars(
        select(Document.filename).where(
            Document.subject_id == subject_id, Document.id.in_(document_ids), Document.status == "ready",
            or_(Document.embedder_id.is_(None), Document.embedder_id != embedder_id))
        .order_by(Document.filename)).all()
    return rows, list(stale)


def rank_passages(rows, embedder: Embedder, cfg: RetrievalConfig, query: str,
                  topic: Topic | None, stale: list[str] = ()) -> RetrievalResult:
    version = "r-" + canonical_hash({
        "retrieval": cfg.model_dump(mode="json"),
        "extractions": sorted({p.extraction_version for p in rows}),
    })[:12]

    scored: list[ScoredPassage] = []
    if rows:
        q = embedder.embed([query])[0]
        cos = np.stack([p.embedding for p in rows]) @ q
        # shortcut: full scan in Python, switch to pgvector ordering when libraries get large
        keep = np.argsort(-cos)[: cfg.candidate_k]
        q_terms = terms(query)
        # Rare words count more: a word that every passage has cannot show the topic is covered.
        row_terms = [terms(p.text) for p in rows]
        n = len(rows)
        idf = {t: math.log(1 + (n - df + 0.5) / (df + 0.5))
               for t in q_terms for df in [sum(t in rt for rt in row_terms)]}
        total = sum(idf.values())
        denom = cfg.vector_weight + cfg.lexical_weight + cfg.topic_page_boost
        for i in keep:
            p = rows[int(i)]
            coverage = sum(idf[t] for t in q_terms & row_terms[int(i)]) / total if total else 0.0
            if cfg.require_term_match and coverage < cfg.min_term_coverage:
                continue
            in_topic = (topic is not None and topic.document_id == p.document_id
                        and topic.page_start is not None and topic.page_end is not None
                        and topic.page_start <= p.page_number <= topic.page_end)
            score = (cfg.vector_weight * float(cos[i])
                     + cfg.lexical_weight * coverage
                     + cfg.topic_page_boost * in_topic) / denom
            if score >= cfg.min_score:
                scored.append(ScoredPassage(p.id, p.document_id, p.page_number, p.text, score))
    scored.sort(key=lambda s: -s.score)
    scored = scored[: cfg.top_k]
    if len(scored) < cfg.min_passages:
        gap = f"The selected books have no passage that supports '{query}'"
        if stale:
            gap += f"; re-index for the current embedding model: {', '.join(stale)}"
        return RetrievalResult([], gap, version)
    return RetrievalResult(scored, None, version)
