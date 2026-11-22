"""Shared exception types with stable, user-facing error codes."""

from __future__ import annotations

from typing import Any


class TribunalError(Exception):
    """Base class for errors that carry a stable code and an HTTP status."""

    code = "internal_error"
    status = 500

    def __init__(self, message: str, *, details: Any = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.details = details
        if code:
            self.code = code


class ConfigurationError(TribunalError):
    """Raised when deployment settings or runtime configuration are missing or invalid."""

    code = "configuration_error"
    status = 500

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Configuration is incomplete or invalid: " + "; ".join(problems), details=problems)


class NotFound(TribunalError):
    code = "not_found"
    status = 404


class Forbidden(TribunalError):
    code = "forbidden"
    status = 403


class Unauthorized(TribunalError):
    code = "unauthorized"
    status = 401


class Conflict(TribunalError):
    code = "conflict"
    status = 409


class ValidationFailed(TribunalError):
    """Request is well formed but violates a domain rule. Details list each problem."""

    code = "validation_failed"
    status = 422


class ServiceUnavailable(TribunalError):
    code = "service_unavailable"
    status = 503
