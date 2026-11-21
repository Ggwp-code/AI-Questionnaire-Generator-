"""Retry wrapper and atomic per-paper token budget."""

from __future__ import annotations

import time
from collections.abc import Callable

from sqlalchemy import text

from tribunal.errors import TribunalError
from tribunal.generation.providers.base import ProviderError, ProviderResult
from tribunal.runtime_config import RetryConfig


class BudgetExhausted(TribunalError):
    code = "budget_exhausted"


def call_with_retries(
    fn: Callable[[], ProviderResult], retries: RetryConfig, sleep: Callable[[float], None] = time.sleep
) -> ProviderResult:
    attempt = 1
    while True:
        try:
            return fn()
        except ProviderError as exc:
            if not exc.transient or attempt >= retries.max_attempts:
                raise
        sleep(min(retries.max_delay_seconds, retries.base_delay_seconds * 2 ** (attempt - 1)))
        attempt += 1


def reserve_tokens(session, paper_id: str, amount: int, limit: int) -> None:
    res = session.execute(
        text("UPDATE papers SET tokens_used = tokens_used + :amount WHERE id = :id AND tokens_used + :amount <= :limit"),
        {"amount": amount, "id": paper_id, "limit": limit},
    )
    if res.rowcount == 0:
        raise BudgetExhausted(f"paper token budget of {limit} is exhausted")


def settle_tokens(session, paper_id: str, delta: int) -> None:
    session.execute(text("UPDATE papers SET tokens_used = tokens_used + :d WHERE id = :id"), {"d": delta, "id": paper_id})
