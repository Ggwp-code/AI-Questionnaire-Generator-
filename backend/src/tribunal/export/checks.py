"""Pre-export checks: every export must pass structural and approval checks."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from tribunal.db.models import Passage, Revision, Slot
from tribunal.errors import Conflict
from tribunal.retrieval.embedding import terms
from tribunal.runtime_config import RuntimeConfig


class ExportBlocked(Conflict):
    code = "export_blocked"


def current_revisions(session: Session, paper) -> list[tuple[Slot, Revision | None]]:
    slots = session.scalars(select(Slot).where(Slot.paper_id == paper.id).order_by(Slot.number)).all()
    return [(s, session.get(Revision, s.current_revision_id) if s.current_revision_id else None) for s in slots]


def export_problems(session: Session, paper, cfg: RuntimeConfig, *,
                    revisions: list[tuple[Slot, Revision | None]] | None = None) -> list[str]:
    valid = set(session.scalars(select(Passage.id).where(Passage.document_id.in_(paper.document_ids))))
    problems: list[str] = []
    texts: list[tuple[int, set[str]]] = []
    for slot, rev in revisions if revisions is not None else current_revisions(session, paper):
        n = slot.number
        if slot.status != "approved" or rev is None or not rev.approved_by:
            problems.append(f"question {n} is not approved")
            continue
        c = rev.content
        if not str(c.get("question", "")).strip():
            problems.append(f"question {n} has a blank question")
        if c.get("parts"):
            for pt, m in zip(c["parts"], slot.parts or [], strict=False):
                tag = f"question {n} part {pt['label']}"
                if not str(pt.get("answer", "")).strip():
                    problems.append(f"{tag} has a blank answer")
                total = sum(r["marks"] for r in pt.get("rubric", []))
                if abs(total - m) > 1e-9:
                    problems.append(f"{tag} rubric totals {total:g}, expected {m}")
                if any(cid not in valid for cid in pt.get("citations", [])):
                    problems.append(f"{tag} cites a passage outside this paper's sources")
        else:
            if not str(c.get("answer", "")).strip():
                problems.append(f"question {n} has a blank answer")
            total = sum(r["marks"] for r in c.get("rubric", []))
            if abs(total - slot.marks) > 1e-9:
                problems.append(f"question {n} rubric totals {total:g}, expected {slot.marks}")
            if any(cid not in valid for cid in c.get("citations", [])):
                problems.append(f"question {n} cites a passage outside this paper's sources")
        texts.append((n, terms(str(c.get("question", "")))))
    threshold = cfg.generation.duplicate_similarity
    for i, (a, ta) in enumerate(texts):
        for b, tb in texts[i + 1:]:
            union = ta | tb
            if union and len(ta & tb) / len(union) >= threshold:
                problems.append(f"questions {a} and {b} are near duplicates")
    return problems
