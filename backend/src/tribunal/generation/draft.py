"""Parse model replies into drafts and reviews, and check drafts against a slot."""

from __future__ import annotations

import json
import re
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ValidationError

from tribunal.errors import ValidationFailed
from tribunal.textpolicy import TextPolicy


class RubricItem(BaseModel):
    criterion: str
    marks: float


class DraftPart(BaseModel, extra="ignore"):
    label: str
    question: str
    answer: str
    rubric: list[RubricItem] = []
    citations: list[str] = []
    options: list[str] | None = None
    correct_option: int | None = None


class Draft(BaseModel, extra="ignore"):
    question: str
    answer: str
    parts: list[DraftPart] | None = None
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
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE]([+-]?[0-9]+))?")
NUMERICAL_ANSWER_TYPES = frozenset({"numerical", "numeric", "calculation"})


def parse_number(text: str) -> Fraction:
    """Parse a plain decimal or scientific notation within the verification limits."""
    if len(text) > 80:
        raise ValueError
    text = text.strip()
    match = _NUMBER.fullmatch(text)
    if not match or abs(int(match.group(1) or 0)) > 100:
        raise ValueError
    return Fraction(text)


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


def _choice_problems(options_in: list[str] | None, correct: int | None, answer: str, question_type: str,
                     mcq_option_count: int) -> list[str]:
    problems: list[str] = []
    options = [" ".join(option.split()) for option in options_in or []]
    k = len(options)
    if k != mcq_option_count:
        problems.append(f"expected {mcq_option_count} options, got {k}")
    nonblank = all(options)
    distinct = len({option.casefold() for option in options}) == k
    if not nonblank:
        problems.append("options must be nonblank")
    if not distinct:
        problems.append("options must be distinct")
    if correct is None or not 0 <= correct < k:
        problems.append("correct_option is missing or out of range")
    elif " ".join(answer.split()) != options[correct]:
        problems.append("answer must match the selected option text")
    if question_type == "computational_mcq" and nonblank and distinct:
        try:
            values = {parse_number(option) for option in options_in or []}
        except ValueError:
            problems.append("computational options must be plain numbers")
        else:
            if len(values) != k:
                problems.append("computational option values must be distinct")
    return problems


def check_draft(
    draft: Draft,
    *,
    marks: int,
    question_type: str,
    passage_ids: set[str],
    placeholder_patterns: list[str],
    mcq_option_count: int,
    parts: list[int] | None = None,
) -> list[str]:
    problems: list[str] = []
    if not draft.question.strip():
        problems.append("question is blank")
    is_choice = question_type in {"mcq", "computational_mcq"}
    if parts:
        # the stem is shared; answer, rubric, citations and options live in the parts
        got = draft.parts or []
        if len(got) != len(parts):
            problems.append(f"expected {len(parts)} parts, got {len(got)}")
        for part, m in zip(got, parts, strict=False):
            if not part.answer.strip():
                problems.append(f"part {part.label} answer is blank")
            total = sum(r.marks for r in part.rubric)
            if abs(total - m) > 1e-6:
                problems.append(f"part {part.label} rubric totals {total:g}, expected {m}")
            problems += [f"citation '{c}' is not one of the supplied passages"
                         for c in part.citations if c not in passage_ids]
            for p in placeholder_patterns:
                if re.search(p, part.question):
                    problems.append(f"part {part.label} question matches placeholder pattern '{p}'")
                if re.search(p, part.answer):
                    problems.append(f"part {part.label} answer matches placeholder pattern '{p}'")
            if is_choice:
                problems += [f"part {part.label}: {x}" for x in _choice_problems(
                    part.options, part.correct_option, part.answer, question_type, mcq_option_count)]
        for p in placeholder_patterns:
            if re.search(p, draft.question):
                problems.append(f"question matches placeholder pattern '{p}'")
        return problems
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
    if is_choice:
        problems += _choice_problems(draft.options, draft.correct_option, draft.answer, question_type,
                                     mcq_option_count)
    if question_type in NUMERICAL_ANSWER_TYPES and draft.answer.strip():
        try:
            parse_number(draft.answer)
        except ValueError:
            problems.append("answer must be a plain number")
    return problems
