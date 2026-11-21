"""Parse model replies into drafts and reviews, and check drafts against a slot."""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ValidationError

from tribunal.errors import ValidationFailed
from tribunal.textpolicy import TextPolicy


class RubricItem(BaseModel):
    criterion: str
    marks: float


class Draft(BaseModel, extra="ignore"):
    question: str
    answer: str
    rubric: list[RubricItem] = []
    citations: list[str] = []
    options: list[str] | None = None
    correct_option: int | None = None


class Review(BaseModel, extra="ignore"):
    verdict: Literal["pass", "revise", "faculty_review"]
    reasons: list[str] = []


class DraftInvalid(ValidationFailed):
    code = "draft_invalid"

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Draft is invalid: " + "; ".join(problems), details=problems)


_FENCE = re.compile(r"^```[\w-]*[ \t]*\n?(.*?)\n?```$", re.DOTALL)


def parse_json_object(text: str) -> dict:
    text = text.strip()
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        raise DraftInvalid(["model output is not valid JSON"]) from None
    if not isinstance(data, dict):
        raise DraftInvalid(["model output is not a JSON object"])
    return data


def _parse[T: BaseModel](text: str, policy: TextPolicy, model: type[T]) -> T:
    data, _ = policy.clean_value(parse_json_object(text))
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise DraftInvalid(
            [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        ) from None


def parse_draft(text: str, policy: TextPolicy) -> Draft:
    return _parse(text, policy, Draft)


def parse_review(text: str, policy: TextPolicy) -> Review:
    return _parse(text, policy, Review)


def check_draft(
    draft: Draft,
    *,
    marks: int,
    question_type: str,
    passage_ids: set[str],
    placeholder_patterns: list[str],
    mcq_option_count: int,
) -> list[str]:
    problems: list[str] = []
    if not draft.question.strip():
        problems.append("question is blank")
    if not draft.answer.strip():
        problems.append("answer is blank")
    if not draft.rubric:
        problems.append("rubric is empty")
    else:
        total = sum(r.marks for r in draft.rubric)
        if abs(total - marks) > 1e-6:
            problems.append(f"rubric marks total {total:g} but the slot is worth {marks}")
    if not draft.citations:
        problems.append("no citations")
    problems += [
        f"citation '{c}' is not one of the supplied passages"
        for c in draft.citations
        if c not in passage_ids
    ]
    for p in placeholder_patterns:
        if re.search(p, draft.question):
            problems.append(f"question matches placeholder pattern '{p}'")
        if re.search(p, draft.answer):
            problems.append(f"answer matches placeholder pattern '{p}'")
    if question_type == "mcq":
        k = len(draft.options or [])
        if k != mcq_option_count:
            problems.append(f"expected {mcq_option_count} options, got {k}")
        if draft.correct_option is None or not 0 <= draft.correct_option < k:
            problems.append("correct_option is missing or out of range")
    return problems
