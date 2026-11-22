"""Programmatic alembic upgrade."""

from __future__ import annotations

from alembic import command
from alembic.config import Config


def upgrade(database_url: str) -> None:
    cfg = Config()
    cfg.set_main_option("script_location", "tribunal:migrations")
    cfg.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(cfg, "head")
