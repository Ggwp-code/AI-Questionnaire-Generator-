import logging

import pytest

from tribunal.errors import ConfigurationError
from tribunal.logging_setup import RedactingFilter
from tribunal.settings import load_settings

REQUIRED = ("APP_PROFILE", "DATABASE_URL", "RUNTIME_CONFIG", "AUTH_MODE", "STORAGE_BACKEND", "SIGNING_SECRET")


def problems_of(**env):
    with pytest.raises(ConfigurationError) as exc:
        load_settings()
    return exc.value.problems


def test_missing_settings_named(monkeypatch):
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TRIBUNAL_ENV_FILE", raising=False)
    problems = problems_of()
    for name in REQUIRED:
        assert f"{name} is required but not set" in problems


def test_hosted_profile_rejects_dev_auth(test_env, monkeypatch):
    monkeypatch.setenv("APP_PROFILE", "staging")
    assert "AUTH_MODE must be 'oidc' for the staging profile" in problems_of()


def test_wildcard_origin_rejected(test_env, monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "*")
    assert "ALLOWED_ORIGINS must list explicit origins; '*' is not accepted" in problems_of()


def test_short_signing_secret(test_env, monkeypatch):
    monkeypatch.setenv("SIGNING_SECRET", "short")
    assert "SIGNING_SECRET must be at least 32 characters" in problems_of()


def test_test_profile_loads(settings):
    assert settings.app_profile == "test" and not settings.is_hosted


def test_redacting_filter_hides_secret():
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "key=supersecretvalue", None, None)
    RedactingFilter(["supersecretvalue"]).filter(record)
    assert record.getMessage() == "key=[redacted]"
