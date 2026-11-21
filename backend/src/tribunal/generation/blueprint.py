"""Paper template parsing and expansion into one slot spec per question."""

from dataclasses import dataclass

from pydantic import BaseModel, PositiveInt, ValidationError

from tribunal.errors import ValidationFailed
from tribunal.runtime_config import RuntimeConfig


class TemplateSection(BaseModel, extra="forbid"):
    title: str
    question_type: str
    count: PositiveInt
    marks_each: PositiveInt
    bloom: str
    difficulty: str
    answer_any: PositiveInt | None = None


class TemplateSpec(BaseModel, extra="forbid"):
    sections: list[TemplateSection]
    total_marks: PositiveInt
    instructions: list[str] = []


@dataclass
class SlotSpec:
    number: int
    section: str
    question_type: str
    marks: int
    bloom: str
    difficulty: str
    topic_id: str


def parse_template_spec(data: dict) -> TemplateSpec:
    try:
        spec = TemplateSpec.model_validate(data)
    except ValidationError as e:
        problems = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
        raise ValidationFailed("Template is invalid", details=problems) from e
    if not spec.sections:
        raise ValidationFailed("Template is invalid", details=["sections: at least one section is required"])
    return spec


def expand_blueprint(spec: TemplateSpec, topic_ids: list[str], cfg: RuntimeConfig) -> list[SlotSpec]:
    problems: list[str] = []
    types, blooms, diffs = cfg.question_type_keys(), cfg.bloom_keys(), cfg.difficulty_keys()
    total = 0
    for i, s in enumerate(spec.sections):
        if s.question_type not in types:
            problems.append(f"sections[{i}].question_type '{s.question_type}' is not enabled")
        if s.bloom not in blooms:
            problems.append(f"sections[{i}].bloom '{s.bloom}' is not a configured Bloom level")
        if s.difficulty not in diffs:
            problems.append(f"sections[{i}].difficulty '{s.difficulty}' is not a configured difficulty level")
        if s.answer_any is not None and s.answer_any > s.count:
            problems.append(f"sections[{i}].answer_any {s.answer_any} exceeds count {s.count}")
        total += s.marks_each * (s.answer_any or s.count)
    if total != spec.total_marks:
        problems.append(f"total_marks is {spec.total_marks} but the sections add up to {total}")
    if not topic_ids:
        problems.append("at least one topic is required")
    if problems:
        raise ValidationFailed("Template is invalid", details=problems)

    slots: list[SlotSpec] = []
    for s in spec.sections:
        for _ in range(s.count):
            n = len(slots)
            slots.append(SlotSpec(n + 1, s.title, s.question_type, s.marks_each, s.bloom, s.difficulty,
                                  topic_ids[n % len(topic_ids)]))
    return slots
