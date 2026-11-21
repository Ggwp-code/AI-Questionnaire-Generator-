import pytest

from tribunal.generation.draft import (
    Draft,
    DraftInvalid,
    Review,
    RubricItem,
    check_draft,
    parse_draft,
    parse_json_object,
    parse_review,
)


def _check(d, **kw):
    args = dict(marks=2, question_type="short_answer", passage_ids={"p1"}, placeholder_patterns=[], mcq_option_count=4)
    return check_draft(d, **{**args, **kw})


def test_fenced_json_parsed(policy):
    assert parse_draft('```json\n{"question": "Q?", "answer": "A"}\n```', policy).question == "Q?"
    assert parse_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_not_json(policy):
    with pytest.raises(DraftInvalid) as e:
        parse_draft("nope", policy)
    assert e.value.problems == ["model output is not valid JSON"]
    with pytest.raises(DraftInvalid) as e:
        parse_json_object("[1]")
    assert e.value.problems == ["model output is not a JSON object"]


def test_wrong_types(policy):
    with pytest.raises(DraftInvalid) as e:
        parse_draft('{"question": 5, "answer": "a", "rubric": "x"}', policy)
    assert any(p.startswith("rubric") for p in e.value.problems)


def test_emoji_removed_before_validation(policy):
    d = parse_draft('{"question": "Why \U0001F333 E = −Σ p?", "answer": "a"}', policy)
    assert d.question == "Why E = −Σ p?"


def test_good_draft_has_no_problems():
    d = Draft(question="Q", answer="A", rubric=[RubricItem(criterion="x", marks=2)], citations=["p1"])
    assert _check(d) == []


def test_problem_strings():
    d = Draft(question=" ", answer="", citations=["zz"])
    assert _check(d) == [
        "question is blank",
        "answer is blank",
        "rubric is empty",
        "citation 'zz' is not one of the supplied passages",
    ]
    assert "no citations" in _check(Draft(question="Q", answer="A"))


def test_rubric_total():
    d = Draft(question="Q", answer="A", rubric=[RubricItem(criterion="x", marks=1.5)], citations=["p1"])
    assert _check(d) == ["rubric marks total 1.5 but the slot is worth 2"]


def test_placeholder_and_mcq():
    d = Draft(question="TODO", answer="TODO", rubric=[RubricItem(criterion="x", marks=2)], citations=["p1"],
              options=["a", "b"], correct_option=2)
    assert _check(d, placeholder_patterns=["TODO"], question_type="mcq") == [
        "question matches placeholder pattern 'TODO'",
        "answer matches placeholder pattern 'TODO'",
        "expected 4 options, got 2",
        "correct_option is missing or out of range",
    ]


def test_review_parse(policy):
    assert parse_review('{"verdict": "revise", "reasons": ["cite page"]}', policy) == Review(
        verdict="revise", reasons=["cite page"]
    )


def test_review_bad_verdict(policy):
    with pytest.raises(DraftInvalid):
        parse_review('{"verdict": "ok"}', policy)
