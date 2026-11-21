"""Deterministic offline provider used by tests and acceptance runs."""

from __future__ import annotations

import json
import re
import threading

from tribunal.generation.providers.base import ProviderResult
from tribunal.runtime_config import ProviderConfig, RoleConfig

# shortcut: a passage's text ends at its first blank line, fine for the stub, not for real parsing.
_PASSAGE = re.compile(r"^\[P:([^\]]+)\] \(page \d+\)\n(.*?)(?:\n\s*\n|\Z)", re.S | re.M)


class StubProvider:
    def __init__(self, cfg: ProviderConfig):
        self._sem = threading.BoundedSemaphore(cfg.max_concurrent)

    def complete(self, role: RoleConfig, system: str, user: str) -> ProviderResult:
        with self._sem:
            text = self._answer(user)
        return ProviderResult(text, role.model, len(user.split()), len(text.split()))

    @staticmethod
    def _answer(user: str) -> str:
        passages = _PASSAGE.findall(user)
        if not passages:
            return json.dumps({"verdict": "pass", "reasons": []})
        number = re.search(r"^Question number: (\d+)", user, re.M)
        marks = re.search(r"^Marks: (\d+)", user, re.M)
        pid, body = passages[((int(number.group(1)) if number else 1) - 1) % len(passages)]
        first = re.split(r"(?<=[.!?])\s", body.strip(), maxsplit=1)[0][:200]
        return json.dumps(
            {
                "question": f"Explain the following point from the source material: {first}",
                "answer": body.strip()[:400],
                "rubric": [
                    {
                        "criterion": "Accurate explanation grounded in the source",
                        "marks": int(marks.group(1)) if marks else 1,
                    }
                ],
                "citations": [pid],
                "verdict": "pass",
                "reasons": [],
            }
        )
