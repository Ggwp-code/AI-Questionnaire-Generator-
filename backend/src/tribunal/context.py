"""Process-wide runtime context shared by the API and the worker."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session, sessionmaker

from tribunal.db.migrate import upgrade
from tribunal.db.models import ConfigSnapshotRow
from tribunal.db.session import make_engine, make_session_factory
from tribunal.errors import ConfigurationError, NotFound
from tribunal.generation.providers import make_provider
from tribunal.generation.providers.base import ModelProvider
from tribunal.ingestion.ocr import OcrEngine, make_ocr_engine
from tribunal.logging_setup import collect_provider_secrets, configure_logging
from tribunal.retrieval.embedding import Embedder, make_embedder
from tribunal.runtime_config import (
    ConfigSnapshot,
    ProviderConfig,
    RoleConfig,
    hosted_profile_problems,
    load_runtime_config,
    provider_credential_problems,
    snapshot_from_json,
)
from tribunal.settings import Settings
from tribunal.storage import LocalStore, S3Store, make_store
from tribunal.textpolicy import TextPolicy


@dataclass
class AppContext:
    settings: Settings
    snapshot: ConfigSnapshot
    session_factory: sessionmaker[Session]
    store: LocalStore | S3Store
    ocr_override: OcrEngine | None = None
    provider_factory: Callable[[str, ProviderConfig], ModelProvider] | None = None
    _providers: dict[tuple[str, str], ModelProvider] = field(default_factory=dict)
    _snapshots: dict[str, ConfigSnapshot] = field(default_factory=dict)
    _embedders: dict[str, Embedder] = field(default_factory=dict)

    def snapshot_for(self, snapshot_id: str) -> ConfigSnapshot:
        if snapshot_id == self.snapshot.snapshot_id:
            return self.snapshot
        if snapshot_id not in self._snapshots:
            with self.session_factory() as session:
                row = session.get(ConfigSnapshotRow, snapshot_id)
                if row is None:
                    raise NotFound(f"config snapshot '{snapshot_id}' not found")
                self._snapshots[snapshot_id] = snapshot_from_json(row.content)
        return self._snapshots[snapshot_id]

    def embedder(self, snapshot: ConfigSnapshot) -> Embedder:
        emb = snapshot.config.retrieval.embedding
        key = emb.model_dump_json()
        if key not in self._embedders:
            self._embedders[key] = make_embedder(emb)
        return self._embedders[key]

    def provider(self, snapshot: ConfigSnapshot, role_name: str) -> tuple[ModelProvider, RoleConfig]:
        role = snapshot.config.roles[role_name]
        key = (snapshot.snapshot_id, role.provider)
        if key not in self._providers:
            make = self.provider_factory or make_provider
            self._providers[key] = make(role.provider, snapshot.config.providers[role.provider])
        return self._providers[key], role

    def ocr(self, snapshot: ConfigSnapshot) -> OcrEngine:
        return self.ocr_override or make_ocr_engine(snapshot.config.ingestion.ocr)

    def policy(self, snapshot: ConfigSnapshot) -> TextPolicy:
        return TextPolicy(snapshot.config.text_policy.preserve_codepoints)


def build_context(
    settings: Settings,
    *,
    ocr_override: OcrEngine | None = None,
    migrate: bool = False,
    provider_factory: Callable[[str, ProviderConfig], ModelProvider] | None = None,
) -> AppContext:
    snapshot = load_runtime_config(settings.runtime_config)
    problems = hosted_profile_problems(
        snapshot, hosted=settings.is_hosted, allow_stub=settings.allow_stub_provider
    ) + provider_credential_problems(snapshot)
    if problems:
        raise ConfigurationError(problems)
    envs = [p.api_key_env for p in snapshot.config.providers.values() if p.api_key_env]
    configure_logging(settings.log_level, settings.log_format, settings.secret_values() + collect_provider_secrets(envs))
    if migrate:
        upgrade(settings.database_url)
    engine = make_engine(settings.database_url, pool_size=settings.db_pool_size, max_overflow=settings.db_max_overflow)
    factory = make_session_factory(engine)
    with factory.begin() as session:
        if session.get(ConfigSnapshotRow, snapshot.snapshot_id) is None:
            session.add(ConfigSnapshotRow(id=snapshot.snapshot_id, content=snapshot.as_json()))
    return AppContext(settings, snapshot, factory, make_store(settings), ocr_override, provider_factory)
