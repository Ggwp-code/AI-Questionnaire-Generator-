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


def test_lease_must_exceed_deadline(test_env):
    text = test_env["runtime"].read_text()
    text = text.replace("../../config/prompts", str(test_env["runtime"].parent.parent.parent / "config" / "prompts"))
    copy = test_env["tmp"] / "runtime.yaml"
    copy.write_text(text.replace("lease_seconds: 900", "lease_seconds: 600"))
    with pytest.raises(ConfigurationError) as exc:
        load_runtime_config(copy)
    assert any("retries.lease_seconds must be greater than budgets.paper_deadline_seconds" in p
               for p in exc.value.problems)


def test_example_question_formats_and_codegen_prompt(monkeypatch):
    from tribunal.generation.prompts import render

    for role in ("WRITER", "REVIEWER", "REASONER"):
        monkeypatch.setenv(f"TRIBUNAL_{role}_MODEL", role.lower())
    config = load_runtime_config(EXAMPLE).config
    expected = {"short_answer", "long_answer", "mcq", "numerical", "trace", "computational_mcq"}
    assert expected <= set(config.question_type_keys())
    assert expected <= set(config.prompts["writer"].type_instructions)
    codegen = config.roles["codegen"]
    assert codegen.model == config.roles["reviewer"].model
    _, user = render(config.prompts[codegen.prompt], question_block="Compute the area.")
    assert user.strip() == "Compute the area."


def test_older_short_answer_snapshot_without_codegen_loads(snapshot):
    from tribunal.runtime_config import snapshot_from_json

    content = snapshot.as_json()
    content["roles"].pop("codegen", None)
    content["prompts"].pop("codegen", None)
    content["generation"]["question_types"] = {"short_answer": {"label": "Short answer", "enabled": True}}
    restored = snapshot_from_json(content).config
    assert restored.question_type_keys() == ["short_answer"]
    assert restored.generation.question_types["short_answer"].writer_role == "writer"


def test_snapshot_without_verifier_section_loads(snapshot):
    from tribunal.runtime_config import snapshot_from_json

    content = snapshot.as_json()
    content.pop("verifier")
    assert snapshot_from_json(content).config.verifier is None


def test_runtime_file_without_verifier_is_a_problem(tmp_path, test_env):
    import pytest
    import yaml

    from tribunal.errors import ConfigurationError
    from tribunal.runtime_config import load_runtime_config

    raw = yaml.safe_load(test_env["runtime"].read_text())
    raw.pop("verifier")
    p = tmp_path / "r.yaml"
    raw["prompts"] = {k: {"file": str(test_env["runtime"].parent / v["file"])} for k, v in raw["prompts"].items()}
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigurationError) as exc:
        load_runtime_config(p)
    assert any("verifier" in x for x in exc.value.problems)
