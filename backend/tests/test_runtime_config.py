from pathlib import Path

import pytest

from tribunal.errors import ConfigurationError
from tribunal.runtime_config import load_runtime_config

EXAMPLE = Path(__file__).parent.parent / "config" / "runtime.example.yaml"


def test_fixture_loads(test_env):
    assert load_runtime_config(test_env["runtime"]).snapshot_id.startswith("test-1+")


def test_hash_stable(test_env):
    a = load_runtime_config(test_env["runtime"])
    b = load_runtime_config(test_env["runtime"])
    assert a.content_hash == b.content_hash


def test_writer_model_from_env(test_env, monkeypatch):
    default = load_runtime_config(test_env["runtime"])
    monkeypatch.setenv("TRIBUNAL_WRITER_MODEL", "other-writer")
    other = load_runtime_config(test_env["runtime"])
    assert other.config.roles["writer"].model == "other-writer"
    assert other.content_hash != default.content_hash


def test_per_paper_typed_from_env(test_env, monkeypatch):
    monkeypatch.setenv("TRIBUNAL_PER_PAPER", "3")
    assert load_runtime_config(test_env["runtime"]).config.concurrency.per_paper == 3


def test_example_requires_model_env(monkeypatch):
    for name in ("TRIBUNAL_WRITER_MODEL", "TRIBUNAL_REVIEWER_MODEL", "TRIBUNAL_REASONER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigurationError) as exc:
        load_runtime_config(EXAMPLE)
    assert any("TRIBUNAL_WRITER_MODEL" in p and "model id for the writer role" in p for p in exc.value.problems)


def test_unknown_role_provider(test_env):
    text = test_env["runtime"].read_text()
    # keep prompt file references valid after the copy
    text = text.replace("../../config/prompts", str(test_env["runtime"].parent.parent.parent / "config" / "prompts"))
    text = text.replace("  writer:\n    provider: stub", "  writer:\n    provider: missing", 1)
    copy = test_env["tmp"] / "runtime.yaml"
    copy.write_text(text)
    with pytest.raises(ConfigurationError) as exc:
        load_runtime_config(copy)
    assert any("roles.writer.provider 'missing' is not defined under providers" in p for p in exc.value.problems)
