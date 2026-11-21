"""Object storage. Only the local filesystem backend is available in this build."""

from __future__ import annotations

from pathlib import Path

from tribunal.errors import ConfigurationError
from tribunal.settings import Settings


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        p = Path(key)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"invalid storage key: {key!r}")
        return self.root / p

    def put(self, key: str, data: bytes) -> None:
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self.path(key).read_bytes()


def make_store(settings: Settings) -> LocalStore:
    if settings.storage_backend == "s3":
        raise ConfigurationError(["STORAGE_BACKEND 's3' is not available in this build"])
    root = Path(settings.storage_local_root)
    root.mkdir(parents=True, exist_ok=True)
    return LocalStore(root)
