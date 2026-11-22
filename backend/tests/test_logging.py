import io
import json
import logging

from tribunal.logging_setup import configure_logging


def test_traceback_is_redacted(monkeypatch):
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    configure_logging("INFO", "json", ["signing-secret-value", "dbpassword123"])
    try:
        raise RuntimeError("bad signing-secret-value for postgresql://u:dbpassword123@h/db")
    except RuntimeError:
        logging.getLogger("t").exception("failed")
    line = buf.getvalue()
    json.loads(line)
    assert "signing-secret-value" not in line and "dbpassword123" not in line
    assert "[REDACTED]" in line
