"""OpenAI-compatible chat completions provider."""

from __future__ import annotations

import os
import threading

import openai

from tribunal.generation.providers.base import ProviderError, ProviderResult
from tribunal.runtime_config import ProviderConfig, RoleConfig

_TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)


class OpenAIProvider:
    def __init__(self, cfg: ProviderConfig, client=None):
        self._sem = threading.BoundedSemaphore(cfg.max_concurrent)
        self._client = client or openai.OpenAI(
            api_key=os.environ[cfg.api_key_env],
            base_url=cfg.base_url,
            timeout=cfg.timeout_seconds,
            max_retries=0,
        )

    def complete(self, role: RoleConfig, system: str, user: str) -> ProviderResult:
        kwargs: dict = {
            "model": role.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": role.max_output_tokens,
        }
        if role.temperature is not None:
            kwargs["temperature"] = role.temperature
        if role.effort:
            kwargs["reasoning_effort"] = role.effort
        try:
            with self._sem:
                resp = self._client.chat.completions.create(**kwargs)
        except _TRANSIENT as exc:
            raise ProviderError(str(exc), transient=True) from exc
        except openai.APIError as exc:
            raise ProviderError(str(exc), transient=False) from exc
        return ProviderResult(
            resp.choices[0].message.content or "", role.model, resp.usage.prompt_tokens, resp.usage.completion_tokens
        )
