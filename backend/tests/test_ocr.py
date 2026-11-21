import subprocess

import pytest
from pdf_fixtures import digital_pdf, scanned_pdf

from tribunal.errors import ConfigurationError
from tribunal.ingestion import ocr
from tribunal.ingestion.ocr import OcrFailed, OcrmypdfEngine, OcrUnavailable, make_ocr_engine


def _which(monkeypatch, value="/usr/bin/ocrmypdf"):
    monkeypatch.setattr(ocr.shutil, "which", lambda name: value)


def test_ocrmypdf_command(monkeypatch, cfg):
    _which(monkeypatch)
    seen = {}

    def fake(argv, **kw):
        seen["argv"], seen["kw"] = argv, kw
        with open(argv[-1], "wb") as f:
            f.write(digital_pdf(["one one", "two two", "three three"]))
        return subprocess.CompletedProcess(argv, 0, "", "")

    c = cfg.ingestion.ocr.model_copy(update={"rotate_pages": True, "deskew": False, "jobs": 2})
    out = OcrmypdfEngine(c, runner=fake).ocr_pages(scanned_pdf(3), [1, 3], ["eng", "hin"])
    argv = seen["argv"]
    assert argv[:10] == ["ocrmypdf", "--force-ocr", "--pages", "1,3", "-l", "eng+hin", "--jobs", "2", "-q", "--rotate-pages"]
    assert "--deskew" not in argv
    assert seen["kw"]["timeout"] == c.timeout_seconds
    assert set(out) == {1, 3}


def test_ocrmypdf_missing(monkeypatch, cfg):
    _which(monkeypatch, None)
    with pytest.raises(OcrUnavailable, match="not installed"):
        OcrmypdfEngine(cfg.ingestion.ocr).ocr_pages(b"", [1], ["eng"])


def test_ocrmypdf_nonzero(monkeypatch, cfg):
    _which(monkeypatch)
    fake = lambda argv, **kw: subprocess.CompletedProcess(argv, 2, "", "bad")  # noqa: E731
    with pytest.raises(OcrFailed, match="bad"):
        OcrmypdfEngine(cfg.ingestion.ocr, runner=fake).ocr_pages(b"", [1], ["eng"])


def test_ocrmypdf_timeout(monkeypatch, cfg):
    _which(monkeypatch)

    def fake(argv, **kw):
        raise subprocess.TimeoutExpired(argv, kw["timeout"])

    with pytest.raises(OcrFailed, match="timed out after 600 seconds"):
        OcrmypdfEngine(cfg.ingestion.ocr, runner=fake).ocr_pages(b"", [1], ["eng"])


def test_make_ocr_engine_unknown(cfg):
    with pytest.raises(ConfigurationError):
        make_ocr_engine(cfg.ingestion.ocr.model_copy(update={"engine": "nope"}))
    assert isinstance(make_ocr_engine(cfg.ingestion.ocr), OcrmypdfEngine)
