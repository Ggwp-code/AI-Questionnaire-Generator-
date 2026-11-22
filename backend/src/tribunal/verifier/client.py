"""Clients the generation worker uses to run verifier programs."""

from __future__ import annotations

from dataclasses import asdict
from typing import Protocol

import httpx
from pydantic import SecretStr

from tribunal.verifier.runner import Limits, RunResult, run_program


class VerifierClient(Protocol):
    def run(self, program: str) -> RunResult: ...


class SubprocessVerifier:
    """Development and tests only: runs the program next to the worker."""

    def __init__(self, limits: Limits):
        self.limits = limits

    def run(self, program: str) -> RunResult:
        return run_program(program, self.limits)


class HttpVerifier:
    def __init__(self, url: str, token: SecretStr | None, limits: Limits, client: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.token = token
        self.limits = limits
        # Service wall clock plus headroom for transport.
        self.client = client or httpx.Client(timeout=limits.wall_seconds + 10)

    def run(self, program: str) -> RunResult:
        headers = {"Authorization": f"Bearer {self.token.get_secret_value()}"} if self.token else {}
        try:
            resp = self.client.post(
                f"{self.url}/run", json={"program": program, "limits": asdict(self.limits)}, headers=headers
            )
            if not resp.is_success:
                return _unavailable(str(resp.status_code))
            try:
                body = resp.json()
            except (ValueError, RecursionError):
                return _unavailable("malformed response")
            if (
                not isinstance(body, dict)
                or body.get("status") not in ("ok", "error", "timeout", "too_large")
                or "last_line" not in body
                or not isinstance(body["last_line"], (str, type(None)))
                or not isinstance(body.get("stderr_tail"), str)
            ):
                return _unavailable("malformed response")
            return RunResult(body["status"], body["last_line"], body["stderr_tail"])
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError) as exc:
            return _unavailable(type(exc).__name__)


def _unavailable(why: str) -> RunResult:
    return RunResult("error", None, f"verifier service unavailable: {why}")
