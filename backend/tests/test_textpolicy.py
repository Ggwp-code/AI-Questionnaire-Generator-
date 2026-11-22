import pytest


@pytest.mark.parametrize("raw,expected", [
    ("tree \U0001F333 here", "tree here"),
    ("E = −Σ p log₂ p", "E = −Σ p log₂ p"),
    ("café 日本語 नमस्ते", "café 日本語 नमस्ते"),
    ("ok \U0001F44D\U0001F3FD", "ok"),
    ("family \U0001F468\u200d\U0001F469\u200d\U0001F467 end", "family end"),
    ("flag \U0001F1EE\U0001F1F3", "flag"),
    ("step 1\ufe0f\u20e3", "step 1"),
    ("©\ufe0f 2026", "© 2026"),
    ("a → b, x² + y² ≤ 1, 5 °C, 3 µm, A ♥ B", "a → b, x² + y² ≤ 1, 5 °C, 3 µm, A ♥ B"),
])
def test_clean(policy, raw, expected):
    assert policy.clean(raw)[0] == expected


def test_clean_value_nested(policy):
    value, n = policy.clean_value({"q": ["a \U0001F333", {"b": "x\U0001F600"}], "n": 3})
    assert value == {"q": ["a", {"b": "x"}], "n": 3} and n == 2
