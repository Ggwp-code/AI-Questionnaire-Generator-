from __future__ import annotations

from pathlib import Path

import pytest

from tribunal.runtime_config import ConfigSnapshot, RuntimeConfig, load_runtime_config
from tribunal.settings import Settings, load_settings, reset_settings_cache
from tribunal.textpolicy import TextPolicy

FIXTURE_RUNTIME = Path(__file__).parent / "fixtures" / "runtime.yaml"


@pytest.fixture
def test_env(tmp_path, monkeypatch) -> dict[str, Path]:
    store = tmp_path / "store"
    db = tmp_path / "tribunal.db"
    monkeypatch.setenv("APP_PROFILE", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("RUNTIME_CONFIG", str(FIXTURE_RUNTIME))
    monkeypatch.setenv("AUTH_MODE", "dev")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_LOCAL_ROOT", str(store))
    monkeypatch.setenv("SIGNING_SECRET", "s" * 40)
    monkeypatch.delenv("TRIBUNAL_ENV_FILE", raising=False)
    return {"tmp": tmp_path, "db": db, "store": store, "runtime": FIXTURE_RUNTIME}


@pytest.fixture
def settings(test_env) -> Settings:
    reset_settings_cache()
    return load_settings()


@pytest.fixture
def snapshot(test_env) -> ConfigSnapshot:
    return load_runtime_config(test_env["runtime"])


@pytest.fixture
def cfg(snapshot) -> RuntimeConfig:
    return snapshot.config


@pytest.fixture
def policy(cfg) -> TextPolicy:
    return TextPolicy(cfg.text_policy.preserve_codepoints)


@pytest.fixture
def fake_ocr():
    from pdf_fixtures import FakeOcrEngine

    return FakeOcrEngine({})


@pytest.fixture
def ctx(settings, fake_ocr):
    from tribunal.context import build_context

    return build_context(settings, ocr_override=fake_ocr, migrate=True)


def make_user(session, email):
    from tribunal.db.models import User

    user = User(email=email.lower())
    session.add(user)
    session.flush()
    return user


def make_subject(session, owner, name):
    from tribunal.db.models import Subject, SubjectMember

    subject = Subject(name=name, owner_id=owner.id)
    session.add(subject)
    session.flush()
    session.add(SubjectMember(subject_id=subject.id, user_id=owner.id, role="owner"))
    session.flush()
    return subject
