"""Opt-in acceptance on a real scanned book with real OCR and the configured hosted model.

Skipped unless TRIBUNAL_LIVE_SCAN_PDF, TRIBUNAL_LIVE_RUNTIME_CONFIG, the provider credential and
OCRmyPDF are all available. See "Pilot acceptance" in backend/README.md.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tribunal.api.app import create_app
from tribunal.context import build_context
from tribunal.db.models import Page
from tribunal.jobs.worker import run_once
from tribunal.runtime_config import load_runtime_config, provider_credential_problems
from tribunal.settings import load_settings, reset_settings_cache

WHO = {"X-Dev-User": "live-acceptance@example.com"}
SPEC = {"sections": [{"title": "A", "question_type": "short_answer", "count": 3, "marks_each": 2,
                      "bloom": "understand", "difficulty": "easy"}], "total_marks": 6}


def missing_requirements() -> list[str]:
    missing = []
    scan = os.environ.get("TRIBUNAL_LIVE_SCAN_PDF")
    if not scan or not os.access(scan, os.R_OK) or not Path(scan).is_file():
        missing.append("TRIBUNAL_LIVE_SCAN_PDF must name a readable PDF")
    if shutil.which("ocrmypdf") is None:
        missing.append("ocrmypdf is not installed")
    config = os.environ.get("TRIBUNAL_LIVE_RUNTIME_CONFIG")
    if not config or not Path(config).is_file():
        return [*missing, "TRIBUNAL_LIVE_RUNTIME_CONFIG must name a runtime config file"]
    try:
        snapshot = load_runtime_config(config)
    except Exception as exc:
        return [*missing, f"TRIBUNAL_LIVE_RUNTIME_CONFIG does not load: {exc}"]
    cfg = snapshot.config
    stub = sorted(name for name, role in cfg.roles.items() if cfg.providers[role.provider].kind == "stub")
    if stub:
        missing.append(f"these roles use the stub provider: {', '.join(stub)}")
    return missing + provider_credential_problems(snapshot)


def drain(ctx):
    while run_once(ctx):
        pass


def review_sheet(client, scan: str, paper: dict, key: dict) -> str:
    lines = ["# Review sheet: live scanned-book acceptance", "",
             f"Source scan: {scan}",
             f"Paper {paper['id']}, runtime config snapshot {paper['config_snapshot_id']}", "",
             "Open each cited page in the source scan and decide whether it supports the answer. "
             "A citation on the right page can still support a wrong answer, for example when OCR "
             "misread a minus sign or an exponent.", ""]
    for section in key["sections"]:
        for q in section["questions"]:
            lines += [f"## Question {q['number']} ({q['marks']} marks)", "", f"Question: {q['question']}", "",
                      f"Answer: {q['answer']}", ""]
            for n, cite in enumerate(q["citations"], 1):
                text = client.get(f"/api/passages/{cite['passage_id']}", headers=WHO).json()["text"]
                lines += [f"Citation {n}: page {cite['page']} of the source scan", ""]
                lines += [f"> {line}" for line in text.splitlines()] + [""]
            lines += ["Reviewer: supports the answer? yes / no", ""]
    return "\n".join(lines)


def test_live_scanned_book(test_env, monkeypatch, tmp_path):
    missing = missing_requirements()
    if missing:
        pytest.skip("live acceptance needs: " + "; ".join(missing))
    scan = os.environ["TRIBUNAL_LIVE_SCAN_PDF"]
    # test_env already points the database and store at the pytest tmp dir
    monkeypatch.setenv("RUNTIME_CONFIG", os.environ["TRIBUNAL_LIVE_RUNTIME_CONFIG"])
    reset_settings_cache()
    ctx = build_context(load_settings(), migrate=True)
    client = TestClient(create_app(ctx))

    sid = client.post("/api/subjects", json={"name": "Live acceptance"}, headers=WHO).json()["id"]
    r = client.post(f"/api/subjects/{sid}/documents", headers=WHO,
                    files={"file": (Path(scan).name, Path(scan).read_bytes(), "application/pdf")})
    assert r.status_code == 202, r.text
    drain(ctx)
    (doc,) = client.get(f"/api/subjects/{sid}/documents", headers=WHO).json()
    assert doc["status"] == "ready", doc
    with ctx.session_factory() as s:
        assert {p.method for p in s.query(Page).filter_by(document_id=doc["id"])} == {"ocr"}

    topic_ids = [t["id"] for t in client.get(f"/api/subjects/{sid}/topics", headers=WHO).json()][:3]
    assert topic_ids, "the scan produced no outline topics"
    tpl = client.post("/api/templates", json={"name": "live", "spec": SPEC}, headers=WHO).json()
    r = client.post("/api/papers", headers=WHO, json={
        "subject_id": sid, "template_id": tpl["id"], "topic_ids": topic_ids, "title": "Live acceptance"})
    assert r.status_code == 202, r.text
    drain(ctx)
    paper = client.get(f"/api/papers/{r.json()['paper_id']}", headers=WHO).json()
    assert [s["status"] in ("drafted", "needs_review") for s in paper["slots"]] == [True] * 3, paper["slots"]
    for slot in paper["slots"]:
        assert client.post(f"/api/papers/{paper['id']}/slots/{slot['number']}/approve", headers=WHO).status_code == 200
    key = client.get(f"/api/papers/{paper['id']}/export?part=key&format=json", headers=WHO)
    assert key.status_code == 200, key.text

    sheet = Path(os.environ.get("TRIBUNAL_LIVE_REVIEW_SHEET") or tmp_path / "review-sheet.md")
    sheet.write_text(review_sheet(client, scan, paper, key.json()), encoding="utf-8")
    print(f"review sheet: {sheet}")


def test_live_requirements_refuse_the_stub_provider(test_env, monkeypatch):
    monkeypatch.setenv("TRIBUNAL_LIVE_RUNTIME_CONFIG", str(test_env["runtime"]))
    assert any("use the stub provider" in m for m in missing_requirements())
