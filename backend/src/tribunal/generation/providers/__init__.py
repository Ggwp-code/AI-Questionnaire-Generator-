from __future__ import annotations

from tribunal.errors import ConfigurationError
from tribunal.generation.providers.base import ModelProvider
from tribunal.runtime_config import ProviderConfig


def make_provider(name: str, cfg: ProviderConfig) -> ModelProvider:
    if cfg.kind == "stub":
        from tribunal.generation.providers.stub import StubProvider

        return StubProvider(cfg)
    if cfg.kind == "openai":
        from tribunal.generation.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(cfg)
    raise ConfigurationError([f"providers.{name}: kind '{cfg.kind}' is not available in this build"])
