import pytest

from tribunal.errors import ValidationFailed
from tribunal.generation.blueprint import expand_blueprint, parse_template_spec

SEC = {"title": "A", "question_type": "short_answer", "count": 3, "marks_each": 2,
       "bloom": "understand", "difficulty": "medium"}
SPEC = {"sections": [SEC], "total_marks": 6}


def test_expand_three_slots(cfg):
    slots = expand_blueprint(parse_template_spec(SPEC), ["t1", "t2"], cfg)
    assert [s.number for s in slots] == [1, 2, 3]
    assert [s.topic_id for s in slots] == ["t1", "t2", "t1"]
    assert all(s.marks == 2 for s in slots)


def test_all_problems_reported(cfg):
    sec = {**SEC, "question_type": "essay", "bloom": "x", "difficulty": "y"}
    spec = parse_template_spec({"sections": [sec], "total_marks": 7})
    with pytest.raises(ValidationFailed) as ei:
        expand_blueprint(spec, [], cfg)
    d = ei.value.details
    for p in [
        "sections[0].question_type 'essay' is not enabled",
        "sections[0].bloom 'x' is not a configured Bloom level",
        "sections[0].difficulty 'y' is not a configured difficulty level",
        "total_marks is 7 but the sections add up to 6",
        "at least one topic is required",
    ]:
        assert p in d


def test_answer_any_counts_marks(cfg):
    sec = {**SEC, "count": 4, "answer_any": 3, "marks_each": 5}
    slots = expand_blueprint(parse_template_spec({"sections": [sec], "total_marks": 15}), ["t"], cfg)
    assert len(slots) == 4


def test_answer_any_exceeds_count(cfg):
    sec = {**SEC, "answer_any": 4}
    with pytest.raises(ValidationFailed) as ei:
        expand_blueprint(parse_template_spec({"sections": [sec], "total_marks": 8}), ["t"], cfg)
    assert "sections[0].answer_any 4 exceeds count 3" in ei.value.details


def test_unknown_field_rejected():
    with pytest.raises(ValidationFailed):
        parse_template_spec({**SPEC, "colour": "red"})


def test_parts_must_sum_to_marks(cfg):
    ok = expand_blueprint(parse_template_spec({"sections": [{**SEC, "parts": [1, 1]}], "total_marks": 6}), ["t"], cfg)
    assert all(s.parts == [1, 1] for s in ok)
    assert expand_blueprint(parse_template_spec(SPEC), ["t"], cfg)[0].parts is None
    with pytest.raises(ValidationFailed) as ei:
        expand_blueprint(parse_template_spec({"sections": [{**SEC, "parts": [1, 2]}], "total_marks": 6}), ["t"], cfg)
    assert "sections[0].parts sum to 3, expected marks_each 2" in ei.value.details
