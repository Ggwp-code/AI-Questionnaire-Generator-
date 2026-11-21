import pytest

from tribunal.errors import ConfigurationError
from tribunal.storage import LocalStore, make_store


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


def test_make_store_s3_not_available(settings):
    with pytest.raises(ConfigurationError, match="STORAGE_BACKEND 's3'"):
        make_store(settings.model_copy(update={"storage_backend": "s3"}))
