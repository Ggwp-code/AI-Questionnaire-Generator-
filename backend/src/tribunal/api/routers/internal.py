"""Routes for the platform that wakes workers; not part of the public API."""

from __future__ import annotations

import hmac
from typing import Literal

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field

from tribunal.api.deps import get_ctx
from tribunal.context import AppContext
from tribunal.errors import Unauthorized
from tribunal.jobs.worker import run_until_idle

router = APIRouter(prefix="/internal")


class DrainIn(BaseModel, extra="forbid"):
    kinds: list[Literal["ingest", "generate_paper", "regenerate_slot"]]
    max_seconds: float = Field(ge=1, le=840)


@router.post("/drain")
def drain(body: DrainIn, authorization: str | None = Header(default=None),
          ctx: AppContext = Depends(get_ctx)) -> dict:
    token = ctx.settings.internal_dispatch_token.get_secret_value()
    expected = f"Bearer {token}".encode()
    if not hmac.compare_digest((authorization or "").encode(), expected):
        raise Unauthorized("invalid dispatch token")
    return {"processed": run_until_idle(ctx, body.kinds, body.max_seconds)}
