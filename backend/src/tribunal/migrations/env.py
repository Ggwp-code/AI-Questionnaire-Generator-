"""Alembic environment."""

from __future__ import annotations

from alembic import context
from sqlalchemy import create_engine

from tribunal.db.models import Base

config = context.config
url = config.get_main_option("sqlalchemy.url")
sqlite = url.startswith("sqlite")

if context.is_offline_mode():
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True, render_as_batch=sqlite)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(url)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=sqlite)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
