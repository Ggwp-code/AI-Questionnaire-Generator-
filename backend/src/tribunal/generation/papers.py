"""Run all slots of a paper concurrently and regenerate single slots."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

from sqlalchemy import select

from tribunal.context import AppContext
from tribunal.db.models import Paper, Revision, Slot
from tribunal.db.types import utcnow
from tribunal.generation.slots import _finish, generate_slot
from tribunal.jobs.queue import ClaimedJob, OwnershipLost, owned_transaction
from tribunal.runtime_config import ConfigSnapshot


def run_paper(ctx: AppContext, snapshot: ConfigSnapshot, paper_id: str, *, owner: ClaimedJob | None = None) -> None:
    deadline = utcnow() + timedelta(seconds=snapshot.config.budgets.paper_deadline_seconds)
    with owned_transaction(ctx.session_factory, owner) as s:
        # running slots are orphans of an earlier attempt of this job
        for orphan in s.scalars(select(Slot).where(Slot.paper_id == paper_id, Slot.status == "running")):
            orphan.status = "pending"
        paper = s.get(Paper, paper_id)
        paper.status = "running"
        if paper.started_at is None:
            paper.started_at = utcnow()
        slot_ids = list(s.scalars(select(Slot.id).where(Slot.paper_id == paper_id, Slot.status == "pending")
                                  .order_by(Slot.number)))
    # fill the unlocked caches before threads share them
    ctx.embedder(snapshot)
    for role in snapshot.config.roles:
        ctx.provider(snapshot, role)

    def one(slot_id: str) -> None:
        try:
            generate_slot(ctx, snapshot, slot_id, deadline=deadline, owner=owner)
        except OwnershipLost:
            raise
        except Exception as exc:
            _finish(ctx, slot_id, "failed", f"{type(exc).__name__}: {exc}", owner)

    workers = max(1, min(snapshot.config.concurrency.per_paper, len(slot_ids)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, slot_ids))
    with owned_transaction(ctx.session_factory, owner) as s:
        paper = s.get(Paper, paper_id)
        paper.status, paper.completed_at = "drafted", utcnow()


def regenerate(ctx: AppContext, snapshot: ConfigSnapshot, slot_id: str, *, owner: ClaimedJob | None = None) -> None:
    with owned_transaction(ctx.session_factory, owner) as s:
        if owner and s.scalar(select(Revision.id).where(Revision.slot_id == slot_id, Revision.job_id == owner.id)):
            return  # redelivered after this job saved its result: leave the slot (and any approval) alone
        slot = s.get(Slot, slot_id)
        slot.status, slot.status_detail = "pending", None
    generate_slot(ctx, snapshot, slot_id, owner=owner)
