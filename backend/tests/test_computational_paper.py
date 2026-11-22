"""Exercise the shipped computational workflow through the API and leased worker."""

import json
import re
from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from pdf_fixtures import digital_pdf

from tribunal.api.app import create_app
from tribunal.context import build_context
from tribunal.generation.providers.base import ProviderResult
from tribunal.jobs.worker import run_once


def test_numerical_paper_through_approval_and_export(settings, tmp_path, monkeypatch):
    config_dir = Path(__file__).resolve().parents[1] / "config"
    config = yaml.safe_load((config_dir / "runtime.example.yaml").read_text())
    for name in ("WRITER", "REVIEWER", "REASONER"):
        monkeypatch.setenv(f"TRIBUNAL_{name}_MODEL", name.lower())
    for ref in config["prompts"].values():
        ref["file"] = str(config_dir / ref["file"])
    config["providers"]["openai"]["kind"] = "stub"
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(yaml.safe_dump(config))
    calls = []

    class Provider:
        def complete(self, role, system, user):
            calls.append(role.prompt)
            if role.prompt == "writer":
                passage = re.search(r"\[P:([^\]]+)\]", user)[1]
                body = {"question": "A rectangle is 7 by 6. What is its area?",
                        "answer": "42", "rubric": [{"criterion": "Correct area", "marks": 2}],
                        "citations": [passage]}
            elif role.prompt == "codegen":
                assert "7 by 6" in user and "42" not in user
                body = {"expression": "7 * 6"}
            else:
                body = {"verdict": "pass", "reasons": []}
            return ProviderResult(json.dumps(body), role.model, 10, 10)

    ctx = build_context(settings.model_copy(update={"runtime_config": runtime}), migrate=True,
                        provider_factory=lambda *_: Provider())
    client = TestClient(create_app(ctx), headers={"X-Dev-User": "math@example.com"})
    sid = client.post("/api/subjects", json={"name": "Geometry"}).json()["id"]
    data = digital_pdf(["Rectangle area equals length multiplied by width. " * 4])
    upload = client.post(f"/api/subjects/{sid}/documents", files={"file": ("geometry.pdf", data)})
    assert upload.status_code == 202 and run_once(ctx)
    tid = client.post(f"/api/subjects/{sid}/topics", json={"title": "Rectangle area"}).json()["id"]
    spec = {"total_marks": 2, "sections": [{"title": "A", "question_type": "numerical", "count": 1,
            "marks_each": 2, "bloom": "apply", "difficulty": "easy"}]}
    tpl = client.post("/api/templates", json={"name": "Area", "spec": spec}).json()
    paper = client.post("/api/papers", json={"subject_id": sid, "template_id": tpl["id"],
                        "topic_ids": [tid], "title": "Geometry"}).json()
    path = f"/api/papers/{paper['paper_id']}"
    assert run_once(ctx)
    slot, = client.get(path).json()["slots"]
    assert slot["status"] == "drafted", slot
    assert calls == ["writer", "codegen", "reviewer"]
    assert client.get(f"{path}/export?part=key").status_code == 409
    assert client.post(f"{path}/slots/1/approve", json={"revision_id": slot["revision"]["id"]}).status_code == 200
    key = client.get(f"{path}/export?part=key").json()
    assert key["sections"][0]["questions"][0]["answer"] == "42"
    edited = client.put(f"{path}/slots/1", json={"content": {**slot["revision"]["content"], "answer": "43"}})
    assert edited.status_code == 200
    current, = client.get(path).json()["slots"]
    assert current["status"] == "needs_review" and current["revision"]["review"] is None
    assert client.get(f"{path}/export?part=key").status_code == 409
