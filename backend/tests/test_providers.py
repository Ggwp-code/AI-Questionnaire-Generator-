from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import openai
import pytest

from tribunal.errors import ConfigurationError
from tribunal.generation.prompts import passages_block
from tribunal.generation.providers import make_provider
from tribunal.generation.providers.base import ProviderError
from tribunal.generation.providers.openai_provider import OpenAIProvider
from tribunal.generation.providers.stub import StubProvider
from tribunal.runtime_config import ProviderConfig, RoleConfig


@dataclass
class P:
    passage_id: str
    document_id: str
    page_number: int
    text: str
    score: float


PCFG = ProviderConfig(kind="stub", max_concurrent=2, timeout_seconds=5)
ROLE = RoleConfig(provider="stub", model="m-1", prompt="writer", max_output_tokens=100)
PASSAGES = [P("a1", "d", 1, "Alpha is first. More alpha.", 0.9), P("b2", "d", 2, "Beta is second. More beta.", 0.8)]
USER = "Question number: 2\nMarks: 4\n\n" + passages_block(PASSAGES) + "\n\nQuestion type guidance:\nx"


def test_stub_cites_supplied_passage():
    res = StubProvider(PCFG).complete(ROLE, "sys", USER)
    data = json.loads(res.text)
    assert data["citations"] == ["b2"]
    assert sum(r["marks"] for r in data["rubric"]) == 4
    assert data["question"].endswith("Beta is second.")
    assert res.model == "m-1" and data["verdict"] == "pass"


def test_stub_deterministic():
    s = StubProvider(PCFG)
    assert s.complete(ROLE, "s", USER) == s.complete(ROLE, "s", USER)


def test_stub_without_passages():
    assert json.loads(StubProvider(PCFG).complete(ROLE, "s", "nothing").text) == {"verdict": "pass", "reasons": []}


class FakeClient:
    def __init__(self, exc=None):
        self.kwargs = None
        self.exc = exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5),
        )


def test_openai_request_shape():
    client = FakeClient()
    res = OpenAIProvider(PCFG, client).complete(ROLE, "sys", "usr")
    kw = client.kwargs
    assert kw["model"] == "m-1" and kw["response_format"] == {"type": "json_object"}
    assert kw["max_completion_tokens"] == 100
    assert "temperature" not in kw and "reasoning_effort" not in kw
    assert (res.text, res.input_tokens, res.output_tokens) == ("{}", 3, 5)
    role = ROLE.model_copy(update={"temperature": 0.2, "effort": "low"})
    OpenAIProvider(PCFG, client).complete(role, "s", "u")
    assert client.kwargs["temperature"] == 0.2 and client.kwargs["reasoning_effort"] == "low"


def _resp(code):
    return httpx.Response(code, request=httpx.Request("POST", "http://x"))


@pytest.mark.parametrize(
    "exc,transient",
    [
        (openai.RateLimitError("slow", response=_resp(429), body=None), True),
        (openai.APITimeoutError(request=httpx.Request("POST", "http://x")), True),
        (openai.InternalServerError("boom", response=_resp(500), body=None), True),
        (openai.BadRequestError("bad", response=_resp(400), body=None), False),
    ],
)
def test_openai_error_mapping(exc, transient):
    with pytest.raises(ProviderError) as ei:
        OpenAIProvider(PCFG, FakeClient(exc)).complete(ROLE, "s", "u")
    assert ei.value.transient is transient


def test_anthropic_not_available():
    with pytest.raises(ConfigurationError):
        make_provider("claude", ProviderConfig(kind="anthropic", max_concurrent=1, timeout_seconds=5))


def test_make_stub():
    assert isinstance(make_provider("stub", PCFG), StubProvider)


def test_context_provider_cached(ctx):
    a, role = ctx.provider(ctx.snapshot, "writer")
    assert a is ctx.provider(ctx.snapshot, "writer")[0]
    assert role.model == ctx.snapshot.config.roles["writer"].model


def test_context_uses_provider_factory(settings, fake_ocr):
    from tribunal.context import build_context

    calls = []
    sentinel = object()
    c = build_context(settings, ocr_override=fake_ocr, migrate=True,
                      provider_factory=lambda name, cfg: calls.append(name) or sentinel)
    assert c.provider(c.snapshot, "writer")[0] is sentinel
    c.provider(c.snapshot, "reviewer")
    assert calls == ["stub"]
