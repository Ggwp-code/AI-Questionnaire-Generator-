import json

import pytest

from tribunal.generation.draft import Draft
from tribunal.generation.verification import verify_solution


def check(expression, answer="4", *, kind="numerical", tolerance=0.01, **kwargs):
    draft = Draft(question="Calculate.", answer=answer, **kwargs)
    return verify_solution(json.dumps({"expression": expression}), draft, kind, tolerance)[0]


@pytest.mark.parametrize(("expression", "answer"), [
    ("3 * (2 + -1) + 1", "4"),
    ("0.1 + .2", "3E-1"),
    ("2**-2", ".25"),
    ("-(2 + 2)", "-4"),
    ("13 // 4 + 13 % 4", "4"),
    ("9007199254740993 - 9007199254740992", "1"),
])
def test_calculation(expression, answer):
    assert check(expression, answer, tolerance=0) == "pass"


@pytest.mark.parametrize("expression", [
    "__import__('os')", "range(1)", "[4]", "(1, 4)", "True", "2**0.5", "2**10000000",
    "1e1000000000", "(1e100**64)**64", "1/0", "++", "4 # hi", "1+" * 40 + "1",
    "0" * 257, None, 4,
])
def test_invalid_arithmetic(expression):
    assert check(expression) == "invalid"


@pytest.mark.parametrize(("expression", "answer"), [
    ("4", "4"), ("(4)", "4"), ("+4", "4"), ("-4", "-4"), ("--4", "4"), ("4e0", "4"),
])
def test_calculation_requires_a_binary_operation(expression, answer):
    assert check(expression, answer) == "invalid"


def test_relative_tolerance_includes_boundary():
    assert check("0-100", "-99") == "pass"
    assert check("0-100", "-98.999") == "mismatch"
    assert check("1-1", "1e-12") == "mismatch"


def test_bad_writer_answer_requires_a_rewrite():
    assert check("2+2", "four") == "mismatch"
    assert check("2+2", kind="computational_mcq", options=["three", "4"], correct_option=1) == "mismatch"


def test_trace_numeric_output():
    assert check("2+2", kind="trace") == "pass"
    assert check("2+2", "4\n4", kind="trace") == "unsupported"


def test_mcq_agreement():
    draft = Draft(question="Sum?", answer="4", options=["3", "4", "5"], correct_option=1)
    for fields, status in [
        ({}, "pass"), ({"option_index": 1}, "pass"), ({"option_index": True}, "invalid"),
        ({"option_index": 2}, "invalid"),
    ]:
        assert verify_solution(json.dumps({"expression": "2+2", **fields}), draft, "computational_mcq", 0.01)[0] == status
    for changes in [{"correct_option": 0}, {"answer": "5"}, {"answer": "4.01"}, {"options": ["4", "4.01"]}]:
        assert verify_solution('{"expression":"2+2"}', draft.model_copy(update=changes), "computational_mcq", 0.01)[0] == "mismatch"


@pytest.mark.parametrize(("reply", "status"), [
    ("nonsense", "invalid"), ('{"unsupported":true}', "unsupported"),
    ('{"unsupported":1}', "invalid"), ('{"unsupported":true,"expression":"1"}', "invalid"),
    ('{"expression":"4","option_index":1}', "invalid"), ("[" * 8000, "invalid"),
    (" " * 8193, "invalid"),
])
def test_verifier_response(reply, status):
    assert verify_solution(reply, Draft(question="Q", answer="4"), "numeric", 0.01)[0] == status
