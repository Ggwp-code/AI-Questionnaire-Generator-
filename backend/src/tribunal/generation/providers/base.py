"""Provider protocol and shared result and error types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tribunal.errors import TribunalError
from tribunal.runtime_config import RoleConfig


@dataclass
class ProviderResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int


class ProviderError(TribunalError):
    code = "provider_error"
    status = 502

    def __init__(self, message: str, *, transient: bool):
        super().__init__(message)
        self.transient = transient


class ModelProvider(Protocol):
    def complete(self, role: RoleConfig, system: str, user: str) -> ProviderResult: ...
