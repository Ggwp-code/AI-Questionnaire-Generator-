"""Portable column types."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import JSON, DateTime, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator

JsonType = JSON().with_variant(JSONB(), "postgresql")


def new_id() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Embedding(TypeDecorator):
    """Dense vector. Uses pgvector on PostgreSQL and float32 bytes elsewhere."""

    impl = LargeBinary
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector())
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if dialect.name == "postgresql":
            return arr
        return arr.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray, memoryview)):
            return np.frombuffer(bytes(value), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC timestamps on every backend (SQLite stores naive values)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime passed to UTCDateTime; use timezone-aware values")
        value = value.astimezone(UTC)
        if dialect.name == "sqlite":
            return value.replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
