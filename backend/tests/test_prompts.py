from __future__ import annotations

from dataclasses import dataclass

import jinja2
import pytest

from tribunal.db.models import Slot
from tribunal.generation.prompts import draft_block, passages_block, render, slot_block
from tribunal.runtime_config import PromptTemplate


@dataclass
class P:
    passage_id: str
    document_id: str
    page_number: int
    text: str
    score: float


def test_render_strict():
    t = PromptTemplate(id="x", version="1", system="hi {{ a }}", user="{{ missing }}")
    with pytest.raises(jinja2.UndefinedError):
        render(t, a=1)
    assert render(PromptTemplate(id="x", version="1", system="s {{ a }}", user="u"), a=1) == ("s 1", "u")


def test_slot_block_lines(cfg):
    slot = Slot(number=3, question_type="short_answer", marks=2, bloom="apply", difficulty="easy", topic_title="Graphs")
    assert slot_block(slot, cfg).split("\n") == [
        "Question number: 3",
        "Question type: short_answer (Short answer)",
        "Marks: 2",
        "Bloom level: Apply",
        "Difficulty: Easy",
        "Topic: Graphs",
    ]


def test_passages_and_draft_blocks():
    out = passages_block([P("a", "d", 1, "one", 1.0), P("b", "d", 7, "two", 1.0)])
    assert out == "[P:a] (page 1)\none\n\n[P:b] (page 7)\ntwo"
    assert draft_block({"q": "é"}) == '{\n  "q": "é"\n}'
