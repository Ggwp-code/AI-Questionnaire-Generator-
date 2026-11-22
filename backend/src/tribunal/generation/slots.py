"""Generate one slot: retrieve, write, check, review, revise, record every attempt."""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func, select

from tribunal.context import AppContext
from tribunal.db.models import Job, Paper, Revision, Slot, Topic
from tribunal.db.types import utcnow
from tribunal.generation.calls import BudgetExhausted, call_with_retries, reserve_tokens, settle_tokens
from tribunal.generation.draft import DraftInvalid, Review, check_draft, parse_draft, parse_review
from tribunal.generation.prompts import draft_block, passages_block, render, slot_block
from tribunal.generation.providers.base import ProviderError, ProviderResult
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
        with owned_transaction(ctx.session_factory, owner) as s:
            reserve_tokens(s, paper_id, budgets.reserve_per_call, budgets.paper_tokens)
            if owner:
                s.get(Job, owner.id).reserved_tokens += budgets.reserve_per_call
        try:
            res = call_with_retries(lambda: provider.complete(role, system, user), cfg.retries)
        except BaseException:
            with owned_transaction(ctx.session_factory, owner) as s:
                settle_tokens(s, paper_id, -budgets.reserve_per_call)
                if owner:
                    s.get(Job, owner.id).reserved_tokens -= budgets.reserve_per_call
            raise
        with owned_transaction(ctx.session_factory, owner) as s:
            settle_tokens(s, paper_id, res.input_tokens + res.output_tokens - budgets.reserve_per_call)
            if owner:
                s.get(Job, owner.id).reserved_tokens -= budgets.reserve_per_call
        return res, role.model

    if utcnow() > deadline:
        raise _Stop("paper deadline exceeded")
    with ctx.session_factory() as s:
        embedder = ctx.embedder(snapshot)
        rows, stale = searchable_passages(s, subject_id, doc_ids, embedder.model_id)
    found = rank_passages(rows, embedder, cfg.retrieval, slot.topic_title, topic, stale)
    if found.gap:
        return "evidence_gap", found.gap

    _, writer_role = ctx.provider(snapshot, "writer")
    wt = cfg.prompts[writer_role.prompt]
    sblock, pblock = slot_block(slot, cfg), passages_block(found.passages)
    passage_ids = {p.passage_id for p in found.passages}
    attempts = 1 + g.max_revisions
    problems: list[str] = []

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
        res, model = call("writer", system, user)
        try:
            draft = parse_draft(res.text, policy)
        except DraftInvalid as exc:
            problems = exc.problems
            continue
        with owned_transaction(ctx.session_factory, owner) as s:
            nxt = (s.scalar(select(func.max(Revision.number)).where(Revision.slot_id == slot_id)) or 0) + 1
            rev = Revision(slot_id=slot_id, number=nxt, author="model", content=draft.model_dump(), model=model,
                           prompt_version=f"{wt.id}@{wt.version}", retrieval_version=found.version,
                           input_tokens=res.input_tokens, output_tokens=res.output_tokens)
            s.add(rev)
            s.flush()
            rev_id = rev.id
            s.get(Slot, slot_id).current_revision_id = rev_id
        problems = check_draft(draft, marks=slot.marks, question_type=slot.question_type, passage_ids=passage_ids,
                               placeholder_patterns=g.placeholder_patterns, mcq_option_count=g.mcq_option_count)
        if problems:
            continue
        if not g.reviewer_enabled:
            return "drafted", None

        def save(review: Review, rmodel: str, rev_id: str = rev_id, disputed: dict | None = None) -> None:
            with owned_transaction(ctx.session_factory, owner) as s:
                s.get(Revision, rev_id).review = {"verdict": review.verdict, "reasons": review.reasons,
                                                  "model": rmodel, **({"disputed": disputed} if disputed else {})}

        content = draft.model_dump()
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
