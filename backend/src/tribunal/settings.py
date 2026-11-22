"""Deployment settings.

Every value here describes *where and how* this deployment runs: service addresses,
allowed origins, storage locations, identity provider and credentials. Values come
from the process environment (optionally an env file named by TRIBUNAL_ENV_FILE).
Course content and tunable generation policy never live here; see runtime_config.py.

Missing or inconsistent settings are collected and reported together, by name.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from sqlalchemy.engine import make_url

from tribunal.errors import ConfigurationError

Profile = Literal["local", "test", "staging", "production"]
HOSTED_PROFILES = ("staging", "production")


def _split_csv(value: object) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


CsvList = Annotated[list[str], NoDecode]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # Identity of this deployment
    app_profile: Profile
    app_name: str = "Tribunal"
    public_base_url: str | None = None
    institution_name: str | None = None

    # Network
    host: str | None = None  # 127.0.0.1 on local and test, 0.0.0.0 on hosted profiles (see _default_host)
    port: int = 8000
    allowed_origins: CsvList = Field(default_factory=list)

    # Persistence
    database_url: str
    db_pool_size: int = 5
    db_max_overflow: int = 5

    # Versioned runtime configuration (models, prompts, OCR, retrieval, limits)
    runtime_config: Path

    # Identity provider
    auth_mode: Literal["dev", "oidc"]
    oidc_issuer: str | None = None
    oidc_audience: str | None = None
    oidc_jwks_url: str | None = None
    oidc_require_email_verified: bool = True
    auth_jwt_header: str = "Authorization"
    admin_emails: CsvList = Field(default_factory=list)

    # Object storage
    storage_backend: Literal["local", "s3"]
    storage_local_root: Path | None = None
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    s3_bucket_uploads: str | None = None
    s3_bucket_exports: str | None = None
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    presign_ttl_seconds: int = 600

    # Secrets used by this service itself
    signing_secret: SecretStr
    internal_dispatch_token: SecretStr | None = None

    # Optional wake-up notifications to queue consumers (database remains the source of truth)
    job_notify: Literal["none", "cloudflare_queue"] = "none"
    cf_account_id: str | None = None
    cf_api_token: SecretStr | None = None
    cf_queue_ingestion_id: str | None = None
    cf_queue_generation_id: str | None = None

    # Exports
    pdf_font_regular: Path | None = None
    pdf_font_bold: Path | None = None
    pdf_font_fallbacks: CsvList = Field(default_factory=list)

    # Static frontend served by the API process (compose / single-host installs only)
    serve_frontend_dir: Path | None = None

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"

    # Guard rail: stub model providers are refused on hosted profiles unless explicitly allowed
    allow_stub_provider: bool = False

    @field_validator("allowed_origins", "admin_emails", "pdf_font_fallbacks", mode="before")
    @classmethod
    def _csv(cls, value: object) -> list[str]:
        return _split_csv(value)

    @field_validator("admin_emails")
    @classmethod
    def _lower_emails(cls, value: list[str]) -> list[str]:
        return [v.lower() for v in value]

    @model_validator(mode="after")
    def _default_host(self) -> Settings:
        if self.host is None:
            self.host = "0.0.0.0" if self.app_profile in HOSTED_PROFILES else "127.0.0.1"
        return self

    @property
    def is_hosted(self) -> bool:
        return self.app_profile in HOSTED_PROFILES

    def secret_values(self) -> list[str]:
        """Plain secret values, used only by the log redaction filter."""
        values: list[str] = []
        for name in ("signing_secret", "internal_dispatch_token", "s3_access_key_id", "s3_secret_access_key", "cf_api_token"):
            secret = getattr(self, name)
            if secret is not None and secret.get_secret_value():
                values.append(secret.get_secret_value())
        password = make_url(self.database_url).password
        if password:
            values.append(password)
        return values


def consistency_problems(s: Settings) -> list[str]:
    """Cross-field rules. Each problem names the setting (environment variable) involved."""
    problems: list[str] = []

    if s.is_hosted:
        if s.auth_mode != "oidc":
            problems.append(f"AUTH_MODE must be 'oidc' for the {s.app_profile} profile")
        if not s.allowed_origins:
            problems.append(f"ALLOWED_ORIGINS is required for the {s.app_profile} profile")
        if not s.public_base_url or not s.public_base_url.startswith("https://"):
            problems.append("PUBLIC_BASE_URL must be an https:// address on hosted profiles")
        if s.storage_backend != "s3":
            problems.append("STORAGE_BACKEND must be 's3' on hosted profiles (container disk is temporary)")
        if not s.database_url.startswith(("postgresql", "postgres")):
            problems.append("DATABASE_URL must point to PostgreSQL on hosted profiles")
        if not s.pdf_font_regular:
            problems.append("PDF_FONT_REGULAR is required on hosted profiles")

    if "*" in s.allowed_origins:
        problems.append("ALLOWED_ORIGINS must list explicit origins; '*' is not accepted")
    for origin in s.allowed_origins:
        if not origin.startswith(("http://", "https://")):
            problems.append(f"ALLOWED_ORIGINS entry '{origin}' must start with http:// or https://")

    if len(s.signing_secret.get_secret_value()) < 32:
        problems.append("SIGNING_SECRET must be at least 32 characters")

    if s.auth_mode == "oidc":
        for name in ("oidc_issuer", "oidc_audience", "oidc_jwks_url"):
            if not getattr(s, name):
                problems.append(f"{name.upper()} is required when AUTH_MODE is 'oidc'")

    if s.storage_backend == "local" and not s.storage_local_root:
        problems.append("STORAGE_LOCAL_ROOT is required when STORAGE_BACKEND is 'local'")
    if s.storage_backend == "s3":
        for name in ("s3_bucket_uploads", "s3_bucket_exports", "s3_access_key_id", "s3_secret_access_key"):
            if not getattr(s, name):
                problems.append(f"{name.upper()} is required when STORAGE_BACKEND is 's3'")

    if s.job_notify == "cloudflare_queue":
        for name in ("cf_account_id", "cf_api_token", "cf_queue_ingestion_id", "cf_queue_generation_id",
                     "internal_dispatch_token"):
            if not getattr(s, name):
                problems.append(f"{name.upper()} is required when JOB_NOTIFY is 'cloudflare_queue'")

    for name in ("pdf_font_regular", "pdf_font_bold"):
        path = getattr(s, name)
        if path is not None and not Path(path).is_file():
            problems.append(f"{name.upper()} points to '{path}', which is not a readable file")

    if not Path(s.runtime_config).is_file():
        problems.append(f"RUNTIME_CONFIG points to '{s.runtime_config}', which is not a readable file")

    return problems


def load_settings(env_file: str | None = None) -> Settings:
    """Load and validate settings. Raises ConfigurationError naming every problem at once."""
    env_file = env_file or os.environ.get("TRIBUNAL_ENV_FILE")
    try:
        settings = Settings(_env_file=env_file) if env_file else Settings()
    except ValidationError as exc:
        problems = []
        for err in exc.errors():
            name = str(err["loc"][0]).upper() if err.get("loc") else "UNKNOWN"
            if err.get("type") == "missing":
                problems.append(f"{name} is required but not set")
            else:
                problems.append(f"{name}: {err.get('msg')}")
        raise ConfigurationError(problems) from None
    problems = consistency_problems(settings)
    if problems:
        raise ConfigurationError(problems)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
