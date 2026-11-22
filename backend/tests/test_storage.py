import io
import sys

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from pydantic import SecretStr

from tribunal.errors import ConfigurationError
from tribunal.storage import LocalStore, S3Store, make_store


def test_store_roundtrip(tmp_path):
    s = LocalStore(tmp_path)
    s.put("documents/a/b.pdf", b"x")
    assert s.get("documents/a/b.pdf") == b"x"


@pytest.mark.parametrize("key", ["../x", "/etc/passwd", "a/../../x"])
def test_store_rejects_escape(tmp_path, key):
    with pytest.raises(ValueError):
        LocalStore(tmp_path).put(key, b"")


def test_make_store_local_creates_root(settings):
    store = make_store(settings)
    assert settings.storage_local_root.is_dir()
    store.put("k", b"v")
    assert store.get("k") == b"v"


def _s3_client():
    import boto3

    return boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="k", aws_secret_access_key="s"
    )


def test_s3_store_roundtrip():
    client = _s3_client()
    body = StreamingBody(io.BytesIO(b"%PDF-1"), len(b"%PDF-1"))
    with Stubber(client) as stubber:
        stubber.add_response(
            "put_object", {}, {"Bucket": "uploads", "Key": "documents/s/d.pdf", "Body": b"%PDF-1"}
        )
        stubber.add_response(
            "get_object", {"Body": body}, {"Bucket": "uploads", "Key": "documents/s/d.pdf"}
        )
        store = S3Store(client, "uploads")
        store.put("documents/s/d.pdf", b"%PDF-1")
        assert store.get("documents/s/d.pdf") == b"%PDF-1"
        stubber.assert_no_pending_responses()


@pytest.mark.parametrize("key", ["../x", "/etc/passwd", "a/../../x"])
def test_s3_store_rejects_escape(key):
    client = _s3_client()
    with Stubber(client) as stubber:
        store = S3Store(client, "uploads")
        with pytest.raises(ValueError):
            store.put(key, b"")
        with pytest.raises(ValueError):
            store.get(key)
        stubber.assert_no_pending_responses()


def test_make_store_s3_from_settings(settings):
    s3_settings = settings.model_copy(
        update={
            "storage_backend": "s3",
            "s3_endpoint_url": "https://acct.r2.cloudflarestorage.com",
            "s3_region": "auto",
            "s3_bucket_uploads": "uploads",
            "s3_access_key_id": SecretStr("k"),
            "s3_secret_access_key": SecretStr("s"),
        }
    )
    store = make_store(s3_settings)
    assert isinstance(store, S3Store)
    assert store.bucket == "uploads"
    assert store.client.meta.endpoint_url == "https://acct.r2.cloudflarestorage.com"
    assert store.client.meta.region_name == "auto"


def test_make_store_s3_without_boto3(settings, monkeypatch):
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(ConfigurationError, match="install the s3 extra"):
        make_store(settings.model_copy(update={"storage_backend": "s3"}))
