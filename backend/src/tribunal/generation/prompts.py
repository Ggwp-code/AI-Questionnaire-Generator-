"""Prompt rendering and the code-rendered blocks the templates embed."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from jinja2 import Environment, StrictUndefined

from tribunal.db.models import Slot
from tribunal.runtime_config import PromptTemplate, RuntimeConfig

if TYPE_CHECKING:
    from tribunal.retrieval.search import ScoredPassage

_ENV = Environment(undefined=StrictUndefined, autoescape=False)


def render(template: PromptTemplate, **variables) -> tuple[str, str]:
    return (
        _ENV.from_string(template.system).render(**variables),
        _ENV.from_string(template.user).render(**variables),
    )


def slot_block(slot: Slot, cfg: RuntimeConfig) -> str:
    g = cfg.generation
    bloom = {lv.key: lv.label for lv in g.bloom_levels}.get(slot.bloom, slot.bloom)
    diff = {lv.key: lv.label for lv in g.difficulty_levels}.get(slot.difficulty, slot.difficulty)
    parts = [f"Parts: {len(slot.parts)}, with marks " + ", ".join(str(m) for m in slot.parts)] if slot.parts else []
    return "\n".join(
        [
            f"Question number: {slot.number}",
            f"Question type: {slot.question_type} ({g.question_types[slot.question_type].label})",
            f"Marks: {slot.marks}",
            f"Bloom level: {bloom}",
            f"Difficulty: {diff}",
            f"Topic: {slot.topic_title}",
            *parts,
        ]
    )


def passages_block(passages: list[ScoredPassage]) -> str:
    return "\n\n".join(f"[P:{p.passage_id}] (page {p.page_number})\n{p.text}" for p in passages)


def draft_block(content: dict) -> str:
    return json.dumps(content, indent=2, ensure_ascii=False)
