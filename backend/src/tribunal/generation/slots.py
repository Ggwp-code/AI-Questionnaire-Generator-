"""Generate one slot: retrieve, write, check, review, revise, record every attempt."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select

from tribunal.context import AppContext
from tribunal.db.models import Job, Paper, Revision, Slot, Topic
from tribunal.db.types import utcnow
from tribunal.generation.calls import BudgetExhausted, call_with_retries, reserve_tokens, settle_tokens
from tribunal.generation.draft import (
    Draft,
    DraftInvalid,
    Review,
    check_draft,
    parse_draft,
    parse_json_object,
    parse_review,
)
from tribunal.generation.prompts import draft_block, passages_block, render, slot_block
from tribunal.generation.providers.base import ProviderError, ProviderResult
from tribunal.generation.verification import COMPUTATIONAL_TYPES, verify_solution
from tribunal.jobs.queue import ClaimedJob, owned_transaction
from tribunal.retrieval.search import rank_passages, searchable_passages
from tribunal.runtime_config import ConfigSnapshot


class _Stop(Exception):
    """End the slot as failed with this message."""


def _finish(ctx: AppContext, slot_id: str, status: str, detail: str | None, owner: ClaimedJob | None = None) -> None:
    with owned_transaction(ctx.session_factory, owner) as s:
        slot = s.get(Slot, slot_id)
        slot.status, slot.status_detail = status, detail
        if status in ("drafted", "needs_review"):
            paper = s.get(Paper, slot.paper_id)
            if paper.first_question_at is None:
                paper.first_question_at = utcnow()


def generate_slot(ctx: AppContext, snapshot: ConfigSnapshot, slot_id: str, *,
                  deadline: datetime | None = None, owner: ClaimedJob | None = None) -> None:
    """`deadline` belongs to the current generation run; by default it starts with this slot."""
    deadline = deadline or utcnow() + timedelta(seconds=snapshot.config.budgets.paper_deadline_seconds)
    try:
        status, detail = _generate(ctx, snapshot, slot_id, deadline, owner)
    except (ProviderError, BudgetExhausted, _Stop) as exc:
        status, detail = "failed", str(exc)
    _finish(ctx, slot_id, status, detail, owner)


def _generate(ctx: AppContext, snapshot: ConfigSnapshot, slot_id: str,
              deadline: datetime, owner: ClaimedJob | None) -> tuple[str, str | None]:
    cfg = snapshot.config
    g, budgets = cfg.generation, cfg.budgets
    policy = ctx.policy(snapshot)
    with owned_transaction(ctx.session_factory, owner) as s:
        slot = s.get(Slot, slot_id)
        slot.status = "running"
        paper = s.get(Paper, slot.paper_id)
        topic = s.get(Topic, slot.topic_id) if slot.topic_id else None
        paper_id, subject_id, doc_ids = paper.id, paper.subject_id, list(paper.document_ids)

    def call(role_name: str, system: str, user: str) -> tuple[ProviderResult, str]:
        if utcnow() > deadline:
            raise _Stop("paper deadline exceeded")
        provider, role = ctx.provider(snapshot, role_name)
        # shortcut: the bound assumes at least one byte per token, true of byte-level tokenizers
        reserve = len((system + user).encode("utf-8")) + role.max_output_tokens
        with owned_transaction(ctx.session_factory, owner) as s:
            reserve_tokens(s, paper_id, reserve, budgets.paper_tokens)
            if owner:
                s.get(Job, owner.id).reserved_tokens += reserve
        try:
            res = call_with_retries(lambda: provider.complete(role, system, user), cfg.retries)
        except BaseException:
            with owned_transaction(ctx.session_factory, owner) as s:
                settle_tokens(s, paper_id, -reserve)
                if owner:
                    s.get(Job, owner.id).reserved_tokens -= reserve
            raise
        with owned_transaction(ctx.session_factory, owner) as s:
            settle_tokens(s, paper_id, res.input_tokens + res.output_tokens - reserve)
            if owner:
                s.get(Job, owner.id).reserved_tokens -= reserve
        return res, role.model

    if utcnow() > deadline:
        raise _Stop("paper deadline exceeded")
    with ctx.session_factory() as s:
        embedder = ctx.embedder(snapshot)
        rows, stale = searchable_passages(s, subject_id, doc_ids, embedder.model_id)
    found = rank_passages(rows, embedder, cfg.retrieval, slot.topic_title, topic, stale)
    if found.gap:
        return "evidence_gap", found.gap

    writer = g.question_types[slot.question_type].writer_role
    wt = cfg.prompts[cfg.roles[writer].prompt]
    sblock, pblock = slot_block(slot, cfg), passages_block(found.passages)
    passage_ids = {p.passage_id for p in found.passages}
    attempts = 1 + g.max_revisions
    problems: list[str] = []

    def verify(draft: Draft) -> dict:
        role = cfg.roles.get("codegen")
        if role is None:
            return {"status": "unavailable", "detail": "codegen role is missing from this configuration"}
        template = cfg.prompts[role.prompt]
        question = draft.model_dump(include={"question", "options"}, exclude_none=True)
        system, user = render(template, question_block=draft_block(question))
        for attempt in range(2):
            try:
                result, model = call("codegen", system, user)
            except ProviderError:
                return {"status": "unavailable", "detail": "The codegen provider was unavailable.", "model": role.model}
            status, detail = verify_solution(result.text, draft, slot.question_type, g.numeric_relative_tolerance)
            if status != "invalid" or attempt == 1:
                break
            user += f"\nYour previous verification was invalid. Correct this problem: {detail}"
        record = {"status": status, "detail": detail, "model": model, "prompt_version": f"{template.id}@{template.version}"}
        if status in ("pass", "mismatch"):
            record.update(parse_json_object(result.text))
        return record

    def review_draft(role_name: str, content: dict) -> tuple[Review, str]:
        """A failing or unparseable reviewer never passes: it asks for a person."""
        template = cfg.prompts[ctx.provider(snapshot, role_name)[1].prompt]
        system, user = render(template, slot_block=sblock, passages_block=pblock, draft_block=draft_block(content))
        try:
            res, model = call(role_name, system, user)
            return parse_review(res.text, policy), model
        except (ProviderError, DraftInvalid) as exc:
            return Review(verdict="faculty_review", reasons=[f"{role_name} unavailable: {exc}"]), role_name

    for i in range(attempts):
        system, user = render(
            wt, slot_block=sblock, passages_block=pblock,
            type_instructions=wt.type_instructions.get(slot.question_type, ""),
            revision_notes="\n".join(f"- {p}" for p in problems), mcq_option_count=g.mcq_option_count)
        res, model = call(writer, system, user)
        try:
            draft = parse_draft(res.text, policy)
        except DraftInvalid as exc:
            problems = exc.problems
            continue
        with owned_transaction(ctx.session_factory, owner) as s:
            nxt = (s.scalar(select(func.max(Revision.number)).where(Revision.slot_id == slot_id)) or 0) + 1
            rev = Revision(slot_id=slot_id, number=nxt, author="model", content=draft.model_dump(), model=model,
                           prompt_version=f"{wt.id}@{wt.version}", retrieval_version=found.version,
                           job_id=owner.id if owner else None,
                           input_tokens=res.input_tokens, output_tokens=res.output_tokens)
            s.add(rev)
            s.flush()
            rev_id = rev.id
            s.get(Slot, slot_id).current_revision_id = rev_id
        problems = check_draft(draft, marks=slot.marks, question_type=slot.question_type, passage_ids=passage_ids,
                               placeholder_patterns=g.placeholder_patterns, mcq_option_count=g.mcq_option_count,
                               parts=slot.parts)
        if problems:
            continue
        content = draft.model_dump()
        if slot.question_type in COMPUTATIONAL_TYPES:
            verification = verify(draft)
            with owned_transaction(ctx.session_factory, owner) as s:
                s.get(Revision, rev_id).review = {"verification": verification}
            if verification["status"] == "mismatch":
                problems = [verification["detail"]]
                continue
            if verification["status"] != "pass":
                return "needs_review", verification["detail"]
            content["verification"] = verification
        if not g.reviewer_enabled:
            return "drafted", None

        def save(review: Review, rmodel: str, rev_id: str = rev_id, disputed: dict | None = None) -> None:
            with owned_transaction(ctx.session_factory, owner) as s:
                revision = s.get(Revision, rev_id)
                revision.review = {**(revision.review or {}), "verdict": review.verdict, "reasons": review.reasons,
                                   "model": rmodel, **({"disputed": disputed} if disputed else {})}

        review, rmodel = review_draft("reviewer", content)
        save(review, rmodel)
        if review.verdict == "pass":
            return "drafted", None
        if review.verdict == "faculty_review":
            return "needs_review", "; ".join(review.reasons) or None
        if i < attempts - 1:
            problems = review.reasons
            continue
        if not g.escalate_disputed_reviews:
            return "needs_review", "; ".join(review.reasons) or None
        final, fmodel = review_draft("reasoner", content)
        save(final, fmodel, disputed={"verdict": review.verdict, "reasons": review.reasons, "model": rmodel})
        if final.verdict == "pass":
            return "drafted", "the reasoner overruled the reviewer: " + "; ".join(review.reasons)
        return "needs_review", "; ".join(final.reasons) or None
    return "failed", "; ".join(problems)
