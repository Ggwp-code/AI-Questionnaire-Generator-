"""Course-scoped two-stage retrieval that reports evidence gaps instead of guessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
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
    rows = session.execute(
        select(Passage).join(Document, Document.id == Passage.document_id)
        .where(Passage.subject_id == subject_id, Passage.document_id.in_(document_ids),
               Document.status == "ready", Passage.extraction_version == Document.extraction_version)
    ).scalars().all()
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
        denom = cfg.vector_weight + cfg.lexical_weight + cfg.topic_page_boost
        for i in keep:
            p = rows[int(i)]
            matched = len(q_terms & terms(p.text))
            if cfg.require_term_match and not matched:
                continue
            in_topic = (topic is not None and topic.document_id == p.document_id
                        and topic.page_start is not None and topic.page_end is not None
                        and topic.page_start <= p.page_number <= topic.page_end)
            score = (cfg.vector_weight * float(cos[i])
                     + cfg.lexical_weight * (matched / len(q_terms) if q_terms else 0.0)
                     + cfg.topic_page_boost * in_topic) / denom
            if score >= cfg.min_score:
                scored.append(ScoredPassage(p.id, p.document_id, p.page_number, p.text, score))
    scored.sort(key=lambda s: -s.score)
    scored = scored[: cfg.top_k]
    if len(scored) < cfg.min_passages:
        return RetrievalResult([], f"The selected books have no passage that supports '{query}'", version)
    return RetrievalResult(scored, None, version)
