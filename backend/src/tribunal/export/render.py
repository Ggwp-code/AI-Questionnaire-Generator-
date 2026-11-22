"""Assemble the export document from approved revisions and render it to PDF."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fpdf import FPDF
from sqlalchemy.orm import Session

from tribunal.db.models import Document, PaperTemplate, Passage, Subject, User
from tribunal.errors import ValidationFailed
from tribunal.export.checks import ExportBlocked, current_revisions, export_problems
from tribunal.generation.blueprint import parse_template_spec
from tribunal.runtime_config import ConfigSnapshot, ExportConfig
from tribunal.textpolicy import TextPolicy

__all__ = ["ExportBlocked", "build_export", "render_pdf"]


def _resolve(session: Session, ids: list[str]) -> list[dict]:
    out = []
    for pid in ids:
        p = session.get(Passage, pid)
        doc = session.get(Document, p.document_id)
        out.append({"passage_id": pid, "document": doc.filename, "page": p.page_number,
                    "url": f"/api/documents/{p.document_id}/file#page={p.page_number}"})
    return out


def build_export(session: Session, paper, part: Literal["paper", "key"], snapshot: ConfigSnapshot,
                 policy: TextPolicy) -> dict:
    revisions = current_revisions(session, paper)  # read once: what is checked is what is exported
    problems = export_problems(session, paper, snapshot.config, revisions=revisions)
    if problems:
        raise ExportBlocked("Paper cannot be exported", details=problems)
    tpl = session.get(PaperTemplate, paper.template_id)
    spec = parse_template_spec(tpl.spec)
    answer_any = {s.title: s.answer_any for s in spec.sections}
    key = part == "key"
    cite = key and snapshot.config.export.include_citations_in_answer_key
    sections: dict[str, dict] = {}
    approvals = []
    for slot, rev in revisions:
        c = rev.content
        q = {"number": slot.number, "marks": slot.marks, "question": c["question"], "options": c.get("options")}
        if c.get("parts"):
            q["parts"] = [{k: v for k, v in pt.items() if k in ("label", "question", "options")}
                          for pt in c["parts"]]
        if key:
            q["answer"] = c["answer"]
            q["rubric"] = c.get("rubric", [])
            for qp, pt in zip(q.get("parts", []), c.get("parts") or [], strict=True):
                qp["answer"], qp["rubric"] = pt["answer"], pt.get("rubric", [])
                if cite:
                    qp["citations"] = _resolve(session, pt.get("citations", []))
            if cite:
                q["citations"] = _resolve(session, c.get("citations", []))
            approvals.append({"number": slot.number, "revision_id": rev.id,
                              "approved_by": session.get(User, rev.approved_by).email,
                              "approved_at": rev.approved_at.isoformat()})
        sec = sections.setdefault(slot.section, {"title": slot.section, "answer_any": answer_any.get(slot.section),
                                                 "questions": []})
        sec["questions"].append(q)
    doc = {"paper_id": paper.id, "title": paper.title, "subject": session.get(Subject, paper.subject_id).name,
           "template": {"name": tpl.name, "version": tpl.version}, "total_marks": spec.total_marks,
           "instructions": spec.instructions, "sections": list(sections.values())}
    if key:
        doc["approvals"] = approvals
    return policy.clean_value(doc)[0]


def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def render_pdf(doc: dict, *, font_regular: Path | None, font_bold: Path | None, fallbacks: list[str],
               cfg: ExportConfig) -> bytes:
    if font_regular is None:
        for s in _strings(doc):
            try:
                s.encode("latin-1")
            except UnicodeEncodeError:
                raise ValidationFailed(
                    "PDF_FONT_REGULAR is required to export this paper's characters") from None
    pdf = FPDF()
    pdf.set_margins(cfg.pdf_margin_mm, cfg.pdf_margin_mm)
    pdf.set_auto_page_break(True, cfg.pdf_margin_mm)
    if font_regular is None:
        family = "Helvetica"
    else:
        family = "body"
        pdf.add_font(family, "", str(font_regular))
        pdf.add_font(family, "B", str(font_bold or font_regular))
        pdf.set_fallback_fonts(fallbacks)
    pdf.add_page()
    size = cfg.pdf_font_size

    def line(text: str, style: str = "", scale: float = 1.0):
        pdf.set_font(family, style, size * scale)
        pdf.multi_cell(0, size * scale * 0.5, text, new_x="LMARGIN", new_y="NEXT")

    line(doc["title"], "B", 1.5)
    line(f"{doc['subject']} - Total marks: {doc['total_marks']:g}")
    for ins in doc["instructions"]:
        line(ins)
    for sec in doc["sections"]:
        pdf.ln(size * 0.3)
        any_note = f" (answer any {sec['answer_any']})" if sec["answer_any"] else ""
        line(sec["title"] + any_note, "B", 1.2)
        for q in sec["questions"]:
            line(f"{q['number']}. {q['question']} [{q['marks']:g}]")
            for i, opt in enumerate(q["options"] or []):
                line(f"    {chr(65 + i)}. {opt}")
            for pt in q.get("parts", []):
                line(f"    ({pt['label']}) {pt['question']}")
                for i, opt in enumerate(pt.get("options") or []):
                    line(f"        {chr(65 + i)}. {opt}")
            if "answer" in q:
                line("Answer: " + q["answer"], "B")
                for pt in q.get("parts", []):
                    line(f"    ({pt['label']}) {pt['answer']}", "B")
                    for r in pt.get("rubric", []):
                        line(f"        - {r['criterion']} ({r['marks']:g})")
                for r in q["rubric"]:
                    line(f"    - {r['criterion']} ({r['marks']:g})")
                for c in q.get("citations", []):
                    line(f"    Source: {c['document']} p.{c['page']}")
    return bytes(pdf.output())
