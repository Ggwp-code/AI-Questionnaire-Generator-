"""Object storage: local filesystem or S3-compatible (AWS S3, Cloudflare R2)."""

from __future__ import annotations

from pathlib import Path

from tribunal.errors import ConfigurationError
from tribunal.settings import Settings


def _check_key(key: str) -> Path:
    p = Path(key)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"invalid storage key: {key!r}")
    return p


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / _check_key(key)

    def put(self, key: str, data: bytes) -> None:
        target = self.path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def get(self, key: str) -> bytes:
        return self.path(key).read_bytes()


class S3Store:
    def __init__(self, client, bucket: str):
        self.client = client
        self.bucket = bucket

    def put(self, key: str, data: bytes) -> None:
        _check_key(key)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        _check_key(key)
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()


def make_store(settings: Settings) -> LocalStore | S3Store:
    if settings.storage_backend == "s3":
        try:
            import boto3
        except ImportError:
            raise ConfigurationError(
                ["STORAGE_BACKEND 's3' needs boto3: install the s3 extra (pip install -e \"backend[s3]\")"]
            ) from None
        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id.get_secret_value(),
            aws_secret_access_key=settings.s3_secret_access_key.get_secret_value(),
        )
        return S3Store(client, settings.s3_bucket_uploads)
    root = Path(settings.storage_local_root)
    root.mkdir(parents=True, exist_ok=True)
    return LocalStore(root)
