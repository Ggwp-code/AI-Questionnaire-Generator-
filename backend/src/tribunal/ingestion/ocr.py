"""OCR engines. ocrmypdf is the only supported engine."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from tribunal.errors import ConfigurationError, ServiceUnavailable, TribunalError
from tribunal.ingestion.pdf import page_texts
from tribunal.runtime_config import OcrConfig


class OcrUnavailable(ServiceUnavailable):
    code = "ocr_unavailable"


class OcrFailed(TribunalError):
    code = "ocr_failed"


class OcrEngine(Protocol):
    def ocr_pages(self, data: bytes, pages: list[int], languages: list[str]) -> dict[int, str]: ...


class OcrmypdfEngine:
    def __init__(self, cfg: OcrConfig, runner=subprocess.run):
        self.cfg = cfg
        self.runner = runner

    def ocr_pages(self, data: bytes, pages: list[int], languages: list[str]) -> dict[int, str]:
        if shutil.which("ocrmypdf") is None:
            raise OcrUnavailable("OCR engine 'ocrmypdf' is not installed; scanned pages cannot be read")
        cfg = self.cfg
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "in.pdf"), Path(tmp, "out.pdf")
            src.write_bytes(data)
            argv = [
                "ocrmypdf", "--force-ocr", "--pages", ",".join(str(p) for p in pages),
                "-l", "+".join(languages), "--jobs", str(cfg.jobs), "-q",
            ]
            if cfg.rotate_pages:
                argv.append("--rotate-pages")
            if cfg.deskew:
                argv.append("--deskew")
            argv += [str(src), str(dst)]
            try:
                res = self.runner(argv, capture_output=True, text=True, timeout=cfg.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise OcrFailed(f"OCR timed out after {cfg.timeout_seconds} seconds") from exc
            if res.returncode != 0:
                raise OcrFailed(f"OCR failed: {(res.stderr or '')[-500:]}")
            return page_texts(dst.read_bytes(), pages)


def make_ocr_engine(cfg: OcrConfig) -> OcrEngine:
    if cfg.engine != "ocrmypdf":
        raise ConfigurationError([f"ingestion.ocr.engine '{cfg.engine}' is not supported"])
    return OcrmypdfEngine(cfg)
