import json
from pathlib import Path

import pytest

from tribunal.generation.draft import Draft, RubricItem, check_draft

LEGACY = Path(__file__).resolve().parents[2] / "data" / "generated_papers"


@pytest.mark.skipif(not LEGACY.is_dir(), reason="legacy papers missing")
def test_legacy_blank_and_error_slots_rejected(cfg):
    blank = error = 0
    for f in sorted(LEGACY.glob("*.json")):
        for s in json.loads(f.read_text(encoding="utf-8")).get("sections", []):
            for q in s.get("questions", []):
                d = Draft(
                    question=q["question_text"],
                    answer=q["answer"],
                    rubric=[RubricItem(criterion="c", marks=q["marks"])],
                    citations=["p1"],
                )
                problems = check_draft(
                    d, marks=q["marks"], question_type="short_answer", passage_ids={"p1"},
                    placeholder_patterns=cfg.generation.placeholder_patterns, mcq_option_count=4,
                )
                if not q["question_text"].strip():
                    blank += 1
                    assert "question is blank" in problems
                if not q["answer"].strip():
                    assert "answer is blank" in problems
                if q["question_text"].startswith("[Error generating question"):
                    error += 1
                    assert "question matches placeholder pattern '(?i)error generating'" in problems
    assert blank >= 6 and error >= 5
