from __future__ import annotations

import pytest

from tribunal.db.migrate import upgrade
from tribunal.db.models import ConfigSnapshotRow, Paper, PaperTemplate, Subject, User
from tribunal.db.session import make_engine, make_session_factory
from tribunal.generation.calls import BudgetExhausted, call_with_retries, reserve_tokens, settle_tokens
from tribunal.generation.providers.base import ProviderError, ProviderResult
from tribunal.runtime_config import RetryConfig

RETRY = RetryConfig(
    max_attempts=3, base_delay_seconds=0.01, max_delay_seconds=1, lease_seconds=60, poll_interval_seconds=1
)
OK = ProviderResult("x", "m", 1, 1)


def _flaky(n, transient=True):
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= n:
            raise ProviderError("e", transient=transient)
        return OK

    return fn, calls


def test_retries_transient_then_succeeds():
    fn, calls = _flaky(2)
    sleeps = []
    assert call_with_retries(fn, RETRY, sleep=sleeps.append) is OK
    assert sleeps == [0.01, 0.02] and len(calls) == 3


def test_gives_up_after_max_attempts():
    fn, calls = _flaky(5)
    with pytest.raises(ProviderError):
        call_with_retries(fn, RETRY, sleep=lambda s: None)
    assert len(calls) == 3


def test_no_retry_on_permanent():
    fn, calls = _flaky(1, transient=False)
    with pytest.raises(ProviderError):
        call_with_retries(fn, RETRY, sleep=lambda s: None)
    assert len(calls) == 1


def test_budget_reserve_and_exhaust(tmp_path):
    url = f"sqlite:///{tmp_path}/b.db"
    upgrade(url)
    factory = make_session_factory(make_engine(url))
    with factory() as s:
        s.add_all([User(id="u", email="u@x.test"), ConfigSnapshotRow(id="c", content={})])
        s.flush()
        s.add_all([Subject(id="s", name="n", owner_id="u"), PaperTemplate(id="t", owner_id="u", name="n", version=1, spec={})])
        s.flush()
        s.add(Paper(id="p", subject_id="s", created_by="u", template_id="t", title="T", config_snapshot_id="c",
                    document_ids=[], topic_ids=[]))
        s.commit()
        reserve_tokens(s, "p", 150, 200)
        with pytest.raises(BudgetExhausted):
            reserve_tokens(s, "p", 100, 200)
        settle_tokens(s, "p", -50)
        s.commit()
        assert s.get(Paper, "p").tokens_used == 100
