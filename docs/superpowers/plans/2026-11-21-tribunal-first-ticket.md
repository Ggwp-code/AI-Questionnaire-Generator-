# Tribunal First Ticket Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two faculty with separate subject libraries upload a digital and a scanned chapter, pick source-backed topics, and get a three-question paper with an answer key, page citations, evidence gaps, single-slot regeneration and recorded timings, all through the new `backend/` service.

**Architecture:** A FastAPI app and a polling worker share one `AppContext` (deployment settings + content-addressed runtime config snapshot + SQLAlchemy session factory + local object store). Uploads and paper requests create database rows and a `jobs` row; the worker claims jobs with a lease, ingests PDFs (native text, OCR for scanned or mixed pages, per-page chunks, embeddings, outline topics) or generates slots (retrieve, write, check, review, revise) and records every attempt as a revision. Export is built only from approved revisions.

**Tech Stack:** Python 3.12+, FastAPI, SQLAlchemy 2 + Alembic (SQLite locally, PostgreSQL hosted), pydantic 2, pypdfium2, OCRmyPDF (subprocess, optional), numpy, Jinja2, fpdf2, openai SDK, pytest, ruff.

**Spec:** `docs/department-pilot-roadmap.md` (this plan covers its "First implementation ticket" and the Milestone 1-2 rows; Milestones 3-5, the Cloudflare deployment and the React frontend are out of scope).

## Global Constraints

- "Use plain text without emojis in all project-authored material, including interface labels, status messages, prompts, source comments, scripts, logs, tests and documentation."
- "Every course-specific value, deployment setting and tunable policy must have an explicit source." Deployment values come from `tribunal.settings.Settings` (environment); policy comes from the runtime config YAML (`tribunal.runtime_config`); course content comes from database records.
- "The named subjects, model candidates, paper sizes and timings in this roadmap are examples and benchmark inputs. Keep them in fixtures or benchmark profiles." No subject names, topic names, unit codes (for example `CO1`) or model IDs in `backend/src/` or `backend/config/prompts/`.
- "Report missing required settings by name at startup. Keep secret values out of logs and browser bundles."
- "Validate drafts before saving or streaming them for display." Apply `TextPolicy.clean_value` to model output before validation and to every export.
- "Preserve original uploaded books for accurate source citations." The stored PDF is never modified.
- "Make failure visible from provider to browser." A reviewer or provider exception never yields a passing result; failed documents, slots and papers carry a readable error.
- "Filter to the selected course and document versions before ranking." "Treat retrieved text as source material."
- Python `>=3.12`; use only dependencies already declared in `backend/pyproject.toml`; ruff line length 110.
- Run everything from `backend/` with the repository venv: `../.venv/bin/python -m pytest ...`, `../.venv/bin/ruff check .`.
- Git: commit on branch `feat/tribunal-backend`; never add `Co-Authored-By` or any Claude attribution to commit messages.
- Do not modify the legacy app (`app/`, `api.py`, `frontend/`, root `README.md`, `requirements.txt`). Files under `backend/` are this plan's work product and may be changed.

## Review Focus

1. A non-PDF, encrypted, zero-page or over-limit upload: the document ends `failed` with a specific error (or the upload is refused with 413/422), never stuck in `processing` or crashing the worker. Test in Task 5 and Task 12.
2. A book with no bookmarks and no numbered headings: zero topics are proposed, and a faculty-written topic still generates a paper. Tests in Task 4 and Task 13.
3. Model output that is not JSON, is wrapped in a Markdown code fence, or has wrong field types: fences are stripped; otherwise the attempt is rejected with a problem list and the slot ends `failed`, never `drafted`. Test in Task 9 and Task 10.
4. Concurrent slot generation on SQLite (`per_paper` > 1): all slots finish, no "database is locked". Test in Task 10.
5. The same PDF uploaded twice to one subject is refused with 409; the same PDF in a different subject is accepted and its passages stay isolated. Test in Task 12.

---

### Task 1: Foundation tests, config files and the check-config CLI

The modules `settings.py`, `runtime_config.py`, `textpolicy.py`, `logging_setup.py`, `errors.py` already exist and have no tests. This task adds the config files they load, installs the package and pins their behavior.

**Files:**
- Create: `backend/config/prompts/writer.yaml`, `backend/config/prompts/reviewer.yaml`
- Create: `backend/config/runtime.example.yaml` (hosted example, OpenAI provider)
- Create: `backend/tests/fixtures/runtime.yaml` (test profile, stub provider)
- Create: `backend/tests/conftest.py`
- Create: `backend/src/tribunal/cli.py`
- Test: `backend/tests/test_settings.py`, `backend/tests/test_runtime_config.py`, `backend/tests/test_textpolicy.py`, `backend/tests/test_cli.py`

**Interfaces:**
- Consumes: `load_settings`, `load_runtime_config`, `TextPolicy`, `RedactingFilter` (existing).
- Produces:
  - `tests/conftest.py` fixture `test_env(tmp_path, monkeypatch) -> dict[str, Path]` setting `APP_PROFILE=test`, `DATABASE_URL=sqlite:///{tmp_path}/tribunal.db`, `RUNTIME_CONFIG=<tests/fixtures/runtime.yaml>`, `AUTH_MODE=dev`, `STORAGE_BACKEND=local`, `STORAGE_LOCAL_ROOT={tmp_path}/store`, `SIGNING_SECRET="s" * 40`, and deleting `TRIBUNAL_ENV_FILE`; returns `{"tmp": tmp_path, "db": ..., "store": ..., "runtime": <fixture path>}`. Fixture `settings(test_env) -> Settings` calls `reset_settings_cache()` then `load_settings()`. Fixture `snapshot(test_env) -> ConfigSnapshot` loads the fixture runtime config; `cfg(snapshot) -> RuntimeConfig`; `policy(cfg) -> TextPolicy(cfg.text_policy.preserve_codepoints)`.
  - Fixture runtime config values read from the environment with defaults: `roles.writer.model: "${TRIBUNAL_WRITER_MODEL:-stub-writer}"`, `roles.reviewer.model: "${TRIBUNAL_REVIEWER_MODEL:-stub-reviewer}"`, `roles.reasoner.model: "stub-reasoner"`, `ingestion.ocr.languages: ["${TRIBUNAL_OCR_LANGUAGE:-eng}"]`, `concurrency.per_paper: "${TRIBUNAL_PER_PAPER:-2}"`, `version: "test-1"`. Other fixture values: one provider `stub` (`kind: stub`, `max_concurrent: 4`, `timeout_seconds: 5`); embedding `hashing`, model `hashing-v1`, `dimensions: 256`; `chunk_chars: 800`, `chunk_overlap_chars: 100`, `candidate_k: 20`, `top_k: 4`, `min_score: 0.15`, `min_passages: 1`, `vector_weight: 0.5`, `lexical_weight: 0.5`, `topic_page_boost: 0.2`, `require_term_match: true`; `native_min_chars: 40`, `mixed_image_coverage: 0.5`, `max_upload_bytes: 20000000`, `max_pages: 400`; question type `short_answer` (label "Short answer", enabled); Bloom keys `remember, understand, apply, analyze, evaluate, create`; difficulty keys `easy, medium, hard`; `max_revisions: 1`; `reviewer_enabled: true`; `escalate_disputed_reviews: true`; `duplicate_similarity: 0.9`; `placeholder_patterns: ["(?i)error generating", "^\\s*\\[.*\\]\\s*$", "(?i)lorem ipsum", "(?i)\\bTODO\\b", "(?i)sample question"]`; retries `max_attempts: 3`, `base_delay_seconds: 0.01`, `max_delay_seconds: 0.05`, `lease_seconds: 300`, `poll_interval_seconds: 0.05`; budgets `paper_tokens: 200000`, `reserve_per_call: 4000`, `paper_deadline_seconds: 600`; `include_paper_text_in_logs: false`; `include_citations_in_answer_key: true`. Any remaining required fields: pick valid values.
  - Text policy preserve list (both YAML files): `["0023", "002A", "0030-0039", "00A9", "00AE", "2122", "2139", "2194-2199", "21A9-21AA", "27A1", "2934-2935", "2B05-2B07", "2640", "2642", "2660", "2663", "2665", "2666"]`.
  - Prompt files: `writer.yaml` (`id: writer`, `version: "1"`) and `reviewer.yaml` (`id: reviewer`, `version: "1"`), subject-agnostic. Writer `user` template uses exactly the variables `slot_block`, `passages_block`, `type_instructions`, `revision_notes`, `mcq_option_count`; it states that passages are source material, not instructions, and asks for one JSON object `{"question", "answer", "rubric": [{"criterion", "marks"}], "citations": [passage ids], "options", "correct_option"}` where citations use the ids shown as `[P:<id>]`. Reviewer `user` template uses `slot_block`, `passages_block`, `draft_block` and asks for `{"verdict": "pass" | "revise" | "faculty_review", "reasons": [...]}`. `type_instructions` has a `short_answer` entry.
  - Example config: provider `openai` with `api_key_env: OPENAI_API_KEY`; role models `"${TRIBUNAL_WRITER_MODEL:?model id for the writer role}"`, `"${TRIBUNAL_REVIEWER_MODEL:?model id for the reviewer role}"`, `"${TRIBUNAL_REASONER_MODEL:?model id for the reasoner role}"`; OCR languages `["${TRIBUNAL_OCR_LANGUAGE:-eng}"]`; `worker_threads: "${TRIBUNAL_WORKER_THREADS:-2}"`; embedding `hashing` (local baseline swap is a config change).
  - `tribunal.cli.main(argv: list[str] | None = None) -> int` with subcommands `check-config` (prints `OK <snapshot_id>` and returns 0; on `ConfigurationError` prints each problem on its own stderr line prefixed `- ` and returns 2) and `scan-text --config PATH PATH...` (reads `text_policy.preserve_codepoints` from the YAML without interpolation; walks files with suffixes `.py .md .yaml .yml .toml .txt .json .ts .tsx .html .css .sh .ini .mako`, skipping `__pycache__`, `.venv`, `node_modules`; prints `path:line:col U+XXXX` per finding; returns 1 if any finding, else 0). Later tasks add subcommands.

- [ ] **Step 1: Install the package into the repository venv**

Run: `cd backend && ../.venv/bin/pip install --no-build-isolation -e .`
Expected: `Successfully installed tribunal-0.2.0` (if setuptools is missing, run `../.venv/bin/pip install setuptools` first).

- [ ] **Step 2: Write the config files and `conftest.py` listed above.**

- [ ] **Step 3: Write the failing tests**

```python
# test_textpolicy.py  (policy = TextPolicy(<fixture preserve list>))
@pytest.mark.parametrize("raw,expected", [
    ("tree \U0001F333 here", "tree here"),
    ("E = −Σ p log₂ p", "E = −Σ p log₂ p"),
    ("café 日本語 नमस्ते", "café 日本語 नमस्ते"),
    ("ok \U0001F44D\U0001F3FD", "ok"),
    ("family \U0001F468\u200d\U0001F469\u200d\U0001F467 end", "family end"),
    ("flag \U0001F1EE\U0001F1F3", "flag"),
    ("step 1\ufe0f\u20e3", "step 1"),
    ("\u00a9\ufe0f 2026", "\u00a9 2026"),
    ("a → b, x² + y² ≤ 1, 5 °C, 3 µm, A ♥ B", "a → b, x² + y² ≤ 1, 5 °C, 3 µm, A ♥ B"),
])
def test_clean(policy, raw, expected): assert policy.clean(raw)[0] == expected
# Write every emoji, ZWJ, variation selector and keycap mark in test sources as a Python escape, never literally,
# so the scan-text check stays clean on the test files themselves.

def test_clean_value_nested(policy):
    value, n = policy.clean_value({"q": ["a \U0001F333", {"b": "x\U0001F600"}], "n": 3})
    assert value == {"q": ["a", {"b": "x"}], "n": 3} and n == 2

# test_settings.py
def test_missing_settings_named(monkeypatch):   # clear APP_PROFILE, DATABASE_URL, RUNTIME_CONFIG, AUTH_MODE, STORAGE_BACKEND, SIGNING_SECRET
    with pytest.raises(ConfigurationError) as exc: load_settings()
    for name in ("APP_PROFILE", "DATABASE_URL", "RUNTIME_CONFIG", "AUTH_MODE", "STORAGE_BACKEND", "SIGNING_SECRET"):
        assert f"{name} is required but not set" in exc.value.problems
def test_hosted_profile_rejects_dev_auth(test_env, monkeypatch):   # APP_PROFILE=staging
    -> "AUTH_MODE must be 'oidc' for the staging profile" in problems
def test_wildcard_origin_rejected(test_env, monkeypatch):   # ALLOWED_ORIGINS="*"
    -> "ALLOWED_ORIGINS must list explicit origins; '*' is not accepted" in problems
def test_short_signing_secret(test_env, monkeypatch):  -> "SIGNING_SECRET must be at least 32 characters" in problems
def test_test_profile_loads(settings): assert settings.app_profile == "test" and not settings.is_hosted
def test_redacting_filter_hides_secret():   # RedactingFilter(["supersecretvalue"]) on a record "key=supersecretvalue"
    -> record.getMessage() == "key=[redacted]"

# test_runtime_config.py
def test_fixture_loads(test_env): snap = load_runtime_config(path); assert snap.snapshot_id.startswith("test-1+")
def test_hash_stable(test_env): two loads -> equal content_hash
def test_writer_model_from_env(test_env, monkeypatch):   # TRIBUNAL_WRITER_MODEL=other-writer
    -> config.roles["writer"].model == "other-writer" and content_hash differs from default load
def test_per_paper_typed_from_env(test_env, monkeypatch):  # TRIBUNAL_PER_PAPER=3 -> config.concurrency.per_paper == 3
def test_example_requires_model_env(monkeypatch):   # example config, no TRIBUNAL_* model vars
    -> some problem contains "TRIBUNAL_WRITER_MODEL" and "model id for the writer role"
def test_unknown_role_provider(tmp_path):   # copy fixture with roles.writer.provider: missing
    -> some problem contains "roles.writer.provider 'missing' is not defined under providers"

# test_cli.py
def test_check_config_ok(test_env, capsys): assert main(["check-config"]) == 0 and capsys.readouterr().out.startswith("OK test-1+")
def test_check_config_reports(test_env, monkeypatch, capsys):  # delete DATABASE_URL
    -> returns 2 and "- DATABASE_URL is required but not set" in stderr
def test_scan_text(tmp_path): file with "hello \U0001F600" -> main(["scan-text", "--config", <fixture>, str(tmp_path)]) == 1, output contains ":1:7 U+1F600"; clean file -> 0
```

- [ ] **Step 4: Run tests to verify the new ones fail**

Run: `cd backend && ../.venv/bin/python -m pytest -v`
Expected: `test_cli.py` fails on import of `tribunal.cli`. If any test of the existing modules fails, fix the existing module (it is part of this work), not the test, unless the test contradicts the spec.

- [ ] **Step 5: Implement `tribunal/cli.py` (argparse).**

- [ ] **Step 6: Run the suite, ruff and the text scan**

Run: `cd backend && ../.venv/bin/python -m pytest && ../.venv/bin/ruff check . && ../.venv/bin/python -m tribunal.cli scan-text --config config/runtime.example.yaml . ../docs`
Expected: all tests pass, ruff `All checks passed!`, scan returns 0. (Add `if __name__ == "__main__": raise SystemExit(main())` to `cli.py`.)

- [ ] **Step 7: Commit**

```bash
git add backend/ docs/department-pilot-roadmap.md docs/superpowers/plans/
git commit -m "feat(backend): config files, foundation tests and check-config CLI"
```

---

### Task 2: Database models, sessions and migrations

**Files:**
- Create: `backend/src/tribunal/db/__init__.py`, `backend/src/tribunal/db/models.py`, `backend/src/tribunal/db/session.py`, `backend/src/tribunal/db/migrate.py`
- Create: `backend/src/tribunal/migrations/__init__.py`, `env.py`, `script.py.mako`, `alembic.ini`, `versions/0001_initial.py`
- Modify: `backend/src/tribunal/cli.py` (add `db-upgrade`)
- Test: `backend/tests/test_db.py`

**Interfaces:**
- Consumes: `JsonType`, `Embedding`, `UTCDateTime`, `new_id`, `utcnow` from `tribunal.db.types`; `Settings.database_url`, `db_pool_size`, `db_max_overflow`.
- Produces (`tribunal.db.models`, all ids `String(32)` defaulting to `new_id`, all timestamps `UTCDateTime` defaulting to `utcnow`):
  - `Base(DeclarativeBase)`
  - `User`: `id, email (unique, lowercase), display_name: str|None, is_admin: bool=False, created_at`
  - `Subject`: `id, name, owner_id -> users.id, created_at`
  - `SubjectMember`: PK `(subject_id, user_id)`, `role: str` (`"owner"` or `"member"`)
  - `Document`: `id, subject_id, uploaded_by, filename, content_hash (sha256 hex), storage_key, size_bytes: int, page_count: int|None, ocr_languages: JSON list, status: str = "queued"` (`queued|processing|ready|failed`), `error: str|None, extraction_version: str|None, created_at, ready_at|None`; unique `(subject_id, content_hash)`
  - `Page`: `id, document_id, page_number (1-based), method: str` (`native|ocr|native+ocr`), `char_count: int, text: Text`; unique `(document_id, page_number)`
  - `Passage`: `id, document_id, subject_id, page_number, char_start, char_end, text: Text, embedding: Embedding, extraction_version`
  - `Topic`: `id, subject_id, document_id|None, title, page_start|None, page_end|None, source: str` (`bookmark|heading|faculty`), `created_at`
  - `PaperTemplate`: `id, owner_id, name, version: int, spec: JSON, created_at`; unique `(owner_id, name, version)`
  - `ConfigSnapshotRow` (table `config_snapshots`): `id: String(64)` (the snapshot id), `content: JSON, created_at`
  - `Paper`: `id, subject_id, created_by, template_id, title, status: str = "queued"` (`queued|running|drafted|failed`), `config_snapshot_id, document_ids: JSON list, topic_ids: JSON list, tokens_used: int = 0, error|None, created_at, started_at|None, first_question_at|None, completed_at|None`
  - `Slot`: `id, paper_id, number, section, question_type, marks: int, bloom, difficulty, topic_id|None, topic_title, status: str = "pending"` (`pending|running|drafted|needs_review|evidence_gap|failed|approved`), `status_detail: Text|None, current_revision_id: String(32)|None` (no FK); unique `(paper_id, number)`
  - `Revision`: `id, slot_id, number, author: str` (`model|faculty`), `content: JSON` (a `Draft` dump, Task 9), `review: JSON|None, model|None, prompt_version|None, retrieval_version|None, input_tokens: int = 0, output_tokens: int = 0, created_by|None, approved_by|None, approved_at|None, created_at`; unique `(slot_id, number)`
  - `Job`: `id, kind: str` (`ingest|generate_paper|regenerate_slot`), `target_id, config_snapshot_id, status: str = "queued"` (`queued|running|done|failed`), `attempts: int = 0, lease_until|None, error: Text|None, created_at, finished_at|None`
- `tribunal.db.session.make_engine(database_url: str, *, pool_size: int = 5, max_overflow: int = 5) -> Engine`; for SQLite uses `connect_args={"check_same_thread": False, "timeout": 30}` and a connect listener running `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`, `PRAGMA busy_timeout=30000`. `make_session_factory(engine) -> sessionmaker[Session]` with `expire_on_commit=False`. Callers use `with factory.begin() as session:`.
- `tribunal.db.migrate.upgrade(database_url: str) -> None` (alembic `Config` with `script_location="tribunal:migrations"`, URL with `%` escaped as `%%`, `command.upgrade(cfg, "head")`). `env.py` reads `sqlalchemy.url` from the config, uses `Base.metadata`, `render_as_batch=True` on SQLite.
- CLI `db-upgrade`: loads settings, runs `upgrade(settings.database_url)`, prints `database at head`.

- [ ] **Step 1: Write the failing tests**

```python
def test_upgrade_matches_models(tmp_path):
    url = f"sqlite:///{tmp_path}/m.db"; upgrade(url)
    with make_engine(url).connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
def test_upgrade_idempotent(tmp_path): upgrade twice on the same URL raises nothing
def test_document_hash_unique_per_subject(tmp_path):  # same content_hash twice in one subject -> IntegrityError; in another subject -> ok
def test_embedding_roundtrip(tmp_path):  # Passage.embedding = np.arange(4, dtype=float32) -> read back equal
def test_utc_timestamps(tmp_path):  # created_at read back has tzinfo UTC
def test_cli_db_upgrade(test_env, capsys): assert main(["db-upgrade"]) == 0 and "database at head" in capsys.readouterr().out
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && ../.venv/bin/python -m pytest tests/test_db.py -v` -- Expected: import errors.

- [ ] **Step 3: Implement models, session and migrate; generate `versions/0001_initial.py`**

Generate the initial revision with alembic autogenerate against an empty SQLite file (`revision --autogenerate -m initial --rev-id 0001`), then review it by hand: it must import `tribunal.db.types` for the custom types and create every table above.

- [ ] **Step 4: Run tests, ruff; Step 5: Commit** `feat(backend): database models and initial migration`

---

### Task 3: Object storage, PDF reading and the OCR engine

**Files:**
- Create: `backend/src/tribunal/storage/__init__.py`
- Create: `backend/src/tribunal/ingestion/__init__.py`, `backend/src/tribunal/ingestion/pdf.py`, `backend/src/tribunal/ingestion/ocr.py`
- Create: `backend/tests/pdf_fixtures.py` (test helpers, not a test module)
- Test: `backend/tests/test_storage.py`, `backend/tests/test_pdf.py`, `backend/tests/test_ocr.py`

**Interfaces:**
- Produces:
  - `tribunal.storage.LocalStore(root: Path)` with `put(key: str, data: bytes) -> None`, `get(key: str) -> bytes`, `path(key: str) -> Path`; keys that are absolute or contain `..` raise `ValueError`. `make_store(settings: Settings) -> LocalStore`: `local` returns a store rooted at `storage_local_root` (created if absent); `s3` raises `ConfigurationError(["STORAGE_BACKEND 's3' is not available in this build"])`.
  - `tribunal.ingestion.pdf.PageInfo` dataclass `(page_number: int, text: str, image_coverage: float)`.
  - `read_pdf(data: bytes, max_pages: int) -> list[PageInfo]`: raises `ValidationFailed` with messages exactly `"file is not a readable PDF"` (parse error, including encrypted files pypdfium2 cannot open), `"PDF has no pages"`, `"PDF has {n} pages; the limit is {max_pages}"`. Text via pypdfium2 text pages; `image_coverage` = sum of image page-object bounding-box areas divided by page area, clipped to 1.0.
  - `read_toc(data: bytes) -> list[tuple[str, int]]` (bookmark title, 1-based page; entries without a page are skipped).
  - `page_texts(data: bytes, pages: list[int]) -> dict[int, str]`.
  - `tribunal.ingestion.ocr.OcrUnavailable(ServiceUnavailable)` with `code = "ocr_unavailable"`; `OcrFailed(TribunalError)` with `code = "ocr_failed"`.
  - `OcrEngine` Protocol: `ocr_pages(self, data: bytes, pages: list[int], languages: list[str]) -> dict[int, str]`.
  - `OcrmypdfEngine(cfg: OcrConfig, runner=subprocess.run)`; command, in this order: `["ocrmypdf", "--force-ocr", "--pages", "<comma list>", "-l", "<langs joined by +>", "--jobs", str(cfg.jobs), "-q"]`, then `"--rotate-pages"` if `cfg.rotate_pages`, `"--deskew"` if `cfg.deskew`, then input and output paths in a temporary directory; `timeout=cfg.timeout_seconds`. Missing binary (`shutil.which` is None) raises `OcrUnavailable("OCR engine 'ocrmypdf' is not installed; scanned pages cannot be read")`; nonzero exit raises `OcrFailed` with the last 500 characters of stderr; timeout raises `OcrFailed("OCR timed out after {n} seconds")`. Returns `page_texts(output, pages)`. Only the requested pages are taken from the OCR output, so native pages keep their original text.
  - `make_ocr_engine(cfg: OcrConfig) -> OcrEngine`: `engine == "ocrmypdf"` only, else `ConfigurationError([f"ingestion.ocr.engine '{cfg.engine}' is not supported"])`.
  - `tests/pdf_fixtures.py`: `digital_pdf(pages: list[str], bookmarks: list[tuple[str, int]] | None = None) -> bytes` (fpdf2, core Helvetica, one string per page, multi-line text; bookmarks via `start_section` on the given 1-based page); `scanned_pdf(page_count: int) -> bytes` (each page is one full-page Pillow image of drawn text, no text layer); `mixed_pdf() -> bytes` (page 1 has a one-line native header plus a large image); `class FakeOcrEngine` with `__init__(self, texts: dict[int, str])`, `calls: list[tuple[list[int], list[str]]]`, and `ocr_pages` returning `texts` for the requested pages.

- [ ] **Step 1: Write the failing tests**

```python
def test_store_roundtrip(tmp_path): s = LocalStore(tmp_path); s.put("documents/a/b.pdf", b"x"); assert s.get("documents/a/b.pdf") == b"x"
@pytest.mark.parametrize("key", ["../x", "/etc/passwd", "a/../../x"])
def test_store_rejects_escape(tmp_path, key): with pytest.raises(ValueError): LocalStore(tmp_path).put(key, b"")
def test_make_store_s3_not_available(settings): settings.storage_backend = "s3" (model_copy) -> ConfigurationError mentioning "STORAGE_BACKEND 's3'"

def test_read_digital(): pages = read_pdf(digital_pdf(["Alpha beta gamma " * 10, "Second page text " * 10]), 10)
    -> len 2, "Alpha beta gamma" in pages[0].text, pages[0].image_coverage == 0
def test_read_scanned(): pages = read_pdf(scanned_pdf(2), 10) -> all p.text.strip() == "" and p.image_coverage > 0.9
def test_read_mixed(): p = read_pdf(mixed_pdf(), 10)[0] -> p.text.strip() != "" and p.image_coverage >= 0.5
def test_not_pdf(): ValidationFailed, message "file is not a readable PDF" for b"hello"
def test_page_limit(): read_pdf(digital_pdf(["a", "b", "c"]), 2) -> message "PDF has 3 pages; the limit is 2"
def test_toc(): read_toc(digital_pdf(["x", "y"], bookmarks=[("One", 1), ("Two", 2)])) == [("One", 1), ("Two", 2)]

def test_ocrmypdf_command(monkeypatch):  # which -> "/usr/bin/ocrmypdf"; runner records argv and writes a digital_pdf to argv[-1]
    engine = OcrmypdfEngine(cfg(rotate_pages=True, deskew=False, jobs=2), runner=fake)
    out = engine.ocr_pages(scanned_pdf(3), [1, 3], ["eng", "hin"])
    -> argv[:10] == ["ocrmypdf", "--force-ocr", "--pages", "1,3", "-l", "eng+hin", "--jobs", "2", "-q", "--rotate-pages"]
       "--deskew" not in argv; set(out) == {1, 3}
def test_ocrmypdf_missing(monkeypatch): which -> None -> OcrUnavailable with "not installed"
def test_ocrmypdf_nonzero(monkeypatch): runner returns returncode 2, stderr "bad" -> OcrFailed containing "bad"
def test_make_ocr_engine_unknown(): engine "nope" -> ConfigurationError
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): local object store, PDF reading and OCR engine`

---

### Task 4: Chunking, embeddings and the topic outline

**Files:**
- Create: `backend/src/tribunal/ingestion/chunking.py`, `backend/src/tribunal/ingestion/outline.py`
- Create: `backend/src/tribunal/retrieval/__init__.py`, `backend/src/tribunal/retrieval/embedding.py`
- Test: `backend/tests/test_chunking.py`, `backend/tests/test_embedding.py`, `backend/tests/test_outline.py`

**Interfaces:**
- Produces:
  - `tribunal.ingestion.chunking.Chunk` dataclass `(page_number: int, char_start: int, char_end: int, text: str)`; `chunk_pages(pages: dict[int, str], chunk_chars: int, overlap: int) -> list[Chunk]`. Chunks never cross pages; each chunk's `text == pages[page_number][char_start:char_end]`; a cut prefers the last whitespace in the final 20% of the window; whitespace-only chunks are dropped; consecutive chunks on a page overlap by at most `overlap` characters; output ordered by page then offset.
  - `tribunal.retrieval.embedding.terms(text: str) -> set[str]`: lowercase Unicode word tokens (`regex` `\w+`) of length >= 3, with one trailing `s` removed from tokens longer than 3 characters. Shared by retrieval scoring and the hashing embedder.
  - `Embedder` Protocol: attribute `model_id: str`; `embed(self, texts: list[str]) -> np.ndarray` (shape `(n, dims)`, float32, rows L2-normalized; an all-zero row stays zero).
  - `HashingEmbedder(dimensions: int)`: `model_id = f"hashing-{dimensions}"`; feature hashing of `terms()` with `hashlib.blake2b` (stable across processes), bucket = digest mod dims, sign from one digest bit.
  - `SentenceTransformerEmbedder(model: str)`: lazy import of `sentence_transformers`; `ImportError` becomes `ConfigurationError(["retrieval.embedding.provider 'sentence_transformers' needs the embeddings-local extra"])`.
  - `make_embedder(cfg: EmbeddingConfig) -> Embedder`; `openai` raises `ConfigurationError(["retrieval.embedding.provider 'openai' is not available in this build"])`.
  - `tribunal.ingestion.outline.OutlineEntry` dataclass `(title: str, page_start: int, page_end: int, source: str)`; `build_outline(toc: list[tuple[str, int]], pages: dict[int, str], cfg: OutlineConfig, page_count: int) -> list[OutlineEntry]`. If `cfg.use_bookmarks` and `toc` is non-empty, entries come from `toc` with source `"bookmark"`. Otherwise, if `cfg.use_headings`, entries come from stripped lines matching `(?i)^(?:chapter\s+)?\d+(?:\.\d+)*\.?\s+\S.{1,80}$` with source `"heading"`, first occurrence per case-insensitive title. `page_end` = next entry's `page_start - 1`, never below its own `page_start`; the last entry ends at `page_count`. At most `cfg.max_topics` entries. No match returns `[]`. `cfg.use_contents_page` is not implemented in this build (one-line comment saying so).

- [ ] **Step 1: Write the failing tests**

```python
def test_chunks_slice_source():  # pages {1: "word " * 500, 2: "short page"}, chunk 800, overlap 100
    for c in chunks: assert c.text == pages[c.page_number][c.char_start:c.char_end] and len(c.text) <= 800
    assert {c.page_number for c in chunks} == {1, 2}
def test_chunks_drop_blank(): chunk_pages({1: "   \n  "}, 800, 100) == []
def test_chunks_unicode(): text "日本語 " * 300 -> slices still equal the source

def test_terms(): terms("Decision trees and a tree, CAFÉ") == {"decision", "tree", "and", "café"}
def test_hashing_deterministic(): two HashingEmbedder(64) instances give identical vectors; shape (2, 64); norms 1.0
def test_hashing_similarity(): cos("decision tree split entropy", "entropy of a decision tree") > cos(same, "steam engine factory")
def test_zero_text(): HashingEmbedder(8).embed([""])[0] is all zeros
def test_openai_embedding_unavailable(): ConfigurationError

def test_outline_bookmarks(): toc [("Trees", 1), ("Regression", 3)], page_count 4 -> [("Trees",1,2,"bookmark"), ("Regression",3,4,"bookmark")]
def test_outline_headings(): pages {1: "1 Alpha Topic\ntext", 2: "more", 3: "2 Beta Topic\ntext"} -> titles ["1 Alpha Topic", "2 Beta Topic"], ranges (1,2), (3,3), source "heading"
def test_outline_none(): pages without numbered headings and no toc -> []
def test_outline_max_topics(): 30 headings, max_topics 5 -> 5 entries
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): chunking, hashing embeddings and outline extraction`

---

### Task 5: App context, job queue, worker and the ingestion pipeline

**Files:**
- Create: `backend/src/tribunal/context.py`
- Create: `backend/src/tribunal/jobs/__init__.py`, `backend/src/tribunal/jobs/queue.py`, `backend/src/tribunal/jobs/worker.py`
- Create: `backend/src/tribunal/ingestion/pipeline.py`
- Modify: `backend/tests/conftest.py` (add `fake_ocr` and `ctx` fixtures), `backend/src/tribunal/cli.py` (add `worker`)
- Test: `backend/tests/test_queue.py`, `backend/tests/test_ingestion.py`

**Interfaces:**
- Consumes: Tasks 2-4.
- Produces:
  - `tribunal.context.AppContext` dataclass: `settings: Settings`, `snapshot: ConfigSnapshot` (current config), `session_factory: sessionmaker[Session]`, `store: LocalStore`, `ocr_override: OcrEngine | None = None`. Methods: `snapshot_for(snapshot_id: str) -> ConfigSnapshot` (from `config_snapshots`, cached; unknown id raises `NotFound`), `embedder(snapshot) -> Embedder` (cached per embedding config), `ocr(snapshot) -> OcrEngine` (override if set, else `make_ocr_engine`), `policy(snapshot) -> TextPolicy`. Task 8 adds `provider_factory` and `provider()`.
  - `build_context(settings: Settings, *, ocr_override: OcrEngine | None = None, migrate: bool = False) -> AppContext`: loads the runtime config, raises `ConfigurationError` with `hosted_profile_problems(...) + provider_credential_problems(...)` if any, configures logging with `settings.secret_values() + collect_provider_secrets(...)`, runs `upgrade()` when `migrate`, builds engine, factory and store, inserts the current snapshot into `config_snapshots` if absent.
  - conftest: `fake_ocr() -> FakeOcrEngine({})` (tests set `.texts`); `ctx(settings, fake_ocr) -> AppContext` = `build_context(settings, ocr_override=fake_ocr, migrate=True)`; helper `make_user(session, email) -> User` and `make_subject(session, owner, name) -> Subject` (adds the owner `SubjectMember`).
  - `tribunal.jobs.queue`: `ClaimedJob` dataclass `(id, kind, target_id, config_snapshot_id, attempts)`; `enqueue(session, kind: str, target_id: str, snapshot_id: str) -> Job` (add + flush, caller commits); `claim_next(factory, lease_seconds: int, now: datetime | None = None) -> ClaimedJob | None` (oldest job that is `queued`, or `running` with `lease_until < now`; sets `running`, `attempts += 1`, `lease_until = now + lease`; uses `with_for_update(skip_locked=True)`); `finish_job(factory, job_id) -> None` (`done`, `finished_at`); `fail_job(factory, job_id, error: str, max_attempts: int) -> bool` (back to `queued` and returns True while `attempts < max_attempts`, else `failed` and returns False).
  - `tribunal.jobs.worker`: `HANDLERS: dict[str, Callable[[AppContext, ClaimedJob], None]]` (this task registers `"ingest"`; Task 10 adds the others); `FINAL_FAILURE: dict[str, Callable[[AppContext, ClaimedJob, str], None]]` marking the target failed (`ingest`: document `failed` with the error); `run_once(ctx) -> bool` (claims one job with `ctx.snapshot.config.retries.lease_seconds`, runs its handler, `finish_job`; on exception logs it, calls `fail_job` with `retries.max_attempts` and, when that returns False, the `FINAL_FAILURE` entry; returns whether a job was found); `run_forever(ctx, stop: threading.Event | None = None) -> None` (sleeps `retries.poll_interval_seconds` when idle). CLI `worker` builds the context and calls `run_forever`.
  - `tribunal.ingestion.pipeline.extraction_version(cfg: RuntimeConfig, languages: list[str], embedder_id: str) -> str` = `"x-" + canonical_hash({...})[:12]` over the parser name and `pypdfium2` version, OCR engine, languages, `rotate_pages`, `deskew`, `native_min_chars`, `mixed_image_coverage`, `chunk_chars`, `chunk_overlap_chars`, `embedder_id`.
  - `ingest_document(ctx: AppContext, snapshot: ConfigSnapshot, document_id: str) -> None`: sets `processing`; reads the stored original; refuses files over `max_upload_bytes`; `read_pdf`; OCR is needed for pages with fewer than `native_min_chars` stripped characters (method `ocr`, text = OCR text) and for pages with `image_coverage >= mixed_image_coverage` (method `native+ocr`, text = native + `"\n"` + OCR text); calls `ctx.ocr(snapshot).ocr_pages(data, pages, document.ocr_languages)` once, only if any page needs it; replaces the document's existing `Page`, `Passage` and `Topic` (non-faculty) rows; writes pages, chunks (`chunk_pages`), embeddings and outline topics (`build_outline` over the final page texts); sets `page_count`, `extraction_version`, `ready`, `ready_at`. Any `TribunalError` or PDF error sets `failed` with `error` = the exception message and does not re-raise; unexpected exceptions also set `failed` and then re-raise so the job retries. The `"ingest"` handler calls it with `ctx.snapshot_for(job.config_snapshot_id)`.

- [ ] **Step 1: Write the failing tests**

```python
# test_queue.py
def test_claim_order_and_lease(ctx):  # enqueue a then b -> claim_next returns a, then b, then None
def test_expired_lease_reclaimed(ctx):  # claim a; claim_next(now=now + lease + 1s) returns a again with attempts == 2
def test_fail_requeues_then_fails(ctx): fail_job(..., max_attempts=2) -> True first, False second; final status "failed", error kept
def test_run_once_marks_document_failed(ctx):  # register a handler that raises; with max_attempts reached -> document.status == "failed"

# test_ingestion.py (store PDF via ctx.store, create Document row with ocr_languages ["eng"], then ingest_document)
def test_digital_document_ready(ctx):  # digital_pdf with bookmarks [("Decision Trees", 1), ("Linear Regression", 2)]
    -> status "ready", page_count 2, every Page.method == "native", Passage count > 0, passage pages in {1, 2},
       topics titles ["Decision Trees", "Linear Regression"], fake_ocr.calls == []
def test_scanned_document_uses_ocr(ctx, fake_ocr):  # scanned_pdf(2); fake_ocr.texts = {1: "1 The French Revolution\n...", 2: "2 The Industrial Revolution\n..."}
    -> status "ready", methods ["ocr", "ocr"], fake_ocr.calls == [([1, 2], ["eng"])], topic titles from headings
def test_mixed_page_gets_ocr(ctx, fake_ocr): mixed_pdf() -> page 1 method "native+ocr", text contains both header and OCR text
def test_not_a_pdf_fails_visibly(ctx): stored bytes b"not a pdf" -> status "failed", error "file is not a readable PDF"
def test_ocr_unavailable_fails_visibly(ctx): ocr_override raising OcrUnavailable -> status "failed", "not installed" in error
def test_reingest_replaces_rows(ctx): ingest twice -> passage and page counts unchanged
def test_ocr_language_from_config(test_env, monkeypatch, ...):  # TRIBUNAL_OCR_LANGUAGE=hin before build_context; document created with snapshot languages
    -> fake_ocr.calls[0][1] == ["hin"]
def test_extraction_version_changes_with_language(): differs for ["eng"] vs ["hin"]
```

For `test_ocr_language_from_config`: a document's `ocr_languages` defaults to `snapshot.config.ingestion.ocr.languages` when created (Task 12 does this in the upload endpoint; the test creates the row the same way).

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): app context, job queue, worker and ingestion pipeline`

---

### Task 6: Course-scoped retrieval with evidence gaps

**Files:**
- Create: `backend/src/tribunal/retrieval/search.py`
- Test: `backend/tests/test_retrieval.py`

**Interfaces:**
- Consumes: `Passage`, `Document`, `Topic`; `Embedder`, `terms` (Task 4); `canonical_hash`.
- Produces:
  - `ScoredPassage` dataclass `(passage_id: str, document_id: str, page_number: int, text: str, score: float)`; `RetrievalResult` dataclass `(passages: list[ScoredPassage], gap: str | None, version: str)`.
  - `retrieve(session: Session, embedder: Embedder, cfg: RetrievalConfig, *, subject_id: str, document_ids: list[str], query: str, topic: Topic | None = None) -> RetrievalResult`. SQL filters `Passage.subject_id == subject_id`, `Passage.document_id IN document_ids`, joined `Document.status == "ready"` before any scoring. Stage 1: cosine similarity, keep `candidate_k`. Stage 2 score = `(vector_weight * cos + lexical_weight * matched_query_terms / query_terms + topic_page_boost * in_topic_pages) / (vector_weight + lexical_weight + topic_page_boost)`, where `in_topic_pages` is 1 when `topic` has a `document_id` equal to the passage's and the passage page lies in `[page_start, page_end]`. With `require_term_match`, passages sharing no query term are dropped. Keep `score >= min_score`, sort descending, take `top_k`. If fewer than `min_passages` remain: `passages=[]` and `gap = f"The selected books have no passage that supports '{query}'"`. `version = "r-" + canonical_hash({"retrieval": cfg.model_dump(mode="json"), "extractions": sorted(distinct extraction_version of the searched documents)})[:12]`.

- [ ] **Step 1: Write the failing tests** (ingest fixtures through `ingest_document`: subject A "digital" with pages about decision trees / regression; subject B "scanned" via `FakeOcrEngine` about revolutions)

```python
def test_trees_in_subject_a(ctx): r = retrieve(..., subject_id=A, document_ids=[docA], query="trees")
    -> r.gap is None and r.passages[0].page_number == 1 and "tree" in r.passages[0].text.lower()
def test_trees_in_subject_b_is_gap(ctx): same query on subject B -> r.passages == [] and "trees" in r.gap
def test_other_subject_documents_ignored(ctx): subject_id=B, document_ids=[docA] -> gap (subject filter wins)
def test_unready_document_ignored(ctx): docA.status = "processing" -> gap
def test_topic_boost_prefers_topic_pages(ctx): topic on page 2, query shared by both pages -> page 2 ranked first
def test_version_stable(ctx): two calls -> same version; changing cfg.min_score (model_copy) -> different version
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): course-scoped retrieval with evidence gaps`

---

### Task 7: Paper templates and blueprint expansion

**Files:**
- Create: `backend/src/tribunal/generation/__init__.py`, `backend/src/tribunal/generation/blueprint.py`
- Test: `backend/tests/test_blueprint.py`

**Interfaces:**
- Consumes: `RuntimeConfig.question_type_keys()`, `bloom_keys()`, `difficulty_keys()`; `ValidationFailed`.
- Produces:
  - `TemplateSection(BaseModel, extra="forbid")`: `title: str`, `question_type: str`, `count: PositiveInt`, `marks_each: PositiveInt`, `bloom: str`, `difficulty: str`, `answer_any: PositiveInt | None = None`.
  - `TemplateSpec(BaseModel, extra="forbid")`: `sections: list[TemplateSection]` (at least one), `total_marks: PositiveInt`, `instructions: list[str] = []`.
  - `parse_template_spec(data: dict) -> TemplateSpec`: pydantic errors become `ValidationFailed("Template is invalid", details=[f"{loc}: {msg}", ...])`.
  - `SlotSpec` dataclass `(number: int, section: str, question_type: str, marks: int, bloom: str, difficulty: str, topic_id: str)`.
  - `expand_blueprint(spec: TemplateSpec, topic_ids: list[str], cfg: RuntimeConfig) -> list[SlotSpec]`: collects every problem, then raises `ValidationFailed("Template is invalid", details=problems)`. Problem strings: `f"sections[{i}].question_type '{t}' is not enabled"`, `f"sections[{i}].bloom '{b}' is not a configured Bloom level"`, `f"sections[{i}].difficulty '{d}' is not a configured difficulty level"`, `f"sections[{i}].answer_any {a} exceeds count {c}"`, `f"total_marks is {total} but the sections add up to {s}"` (sum of `marks_each * (answer_any or count)`), `"at least one topic is required"`. Slots are numbered from 1 in section order; slot `k` (0-based) gets `topic_ids[k % len(topic_ids)]`.

- [ ] **Step 1: Write the failing tests**

```python
SPEC = {"sections": [{"title": "A", "question_type": "short_answer", "count": 3, "marks_each": 2, "bloom": "understand", "difficulty": "medium"}], "total_marks": 6}
def test_expand_three_slots(cfg): slots = expand_blueprint(parse_template_spec(SPEC), ["t1", "t2"], cfg)
    -> [s.number for s in slots] == [1, 2, 3], [s.topic_id for s in slots] == ["t1", "t2", "t1"], all marks 2
def test_all_problems_reported(cfg): type "essay", bloom "x", difficulty "y", total 7, topics []
    -> details contain all five matching strings above
def test_answer_any_counts_marks(cfg): count 4, answer_any 3, marks_each 5, total 15 -> 4 slots, no error
def test_unknown_field_rejected(): parse_template_spec({**SPEC, "colour": "red"}) -> ValidationFailed
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): template parsing and blueprint expansion`

---

### Task 8: Model providers, retries, token budget and prompt rendering

**Files:**
- Create: `backend/src/tribunal/generation/providers/__init__.py`, `base.py`, `stub.py`, `openai_provider.py`
- Create: `backend/src/tribunal/generation/calls.py`, `backend/src/tribunal/generation/prompts.py`
- Modify: `backend/src/tribunal/context.py` (add `provider_factory` field and `provider()`; `build_context` gains `provider_factory=None`)
- Test: `backend/tests/test_providers.py`, `backend/tests/test_calls.py`, `backend/tests/test_prompts.py`

**Interfaces:**
- Produces:
  - `providers.base`: `ProviderResult` dataclass `(text: str, model: str, input_tokens: int, output_tokens: int)`; `ProviderError(TribunalError)` with `code = "provider_error"`, `status = 502`, `__init__(self, message: str, *, transient: bool)`; `ModelProvider` Protocol `complete(self, role: RoleConfig, system: str, user: str) -> ProviderResult`.
  - `providers.make_provider(name: str, cfg: ProviderConfig) -> ModelProvider`: `stub` -> `StubProvider(cfg)`, `openai` -> `OpenAIProvider(cfg)`, `anthropic` -> `ConfigurationError([f"providers.{name}: kind 'anthropic' is not available in this build"])`. Every provider holds a `threading.BoundedSemaphore(cfg.max_concurrent)` around its call.
  - `StubProvider`: deterministic, no network. Reads from the user prompt the code-rendered markers `Question number: N`, `Marks: M` and every `[P:<id>]` passage header with its text (see `passages_block`). Picks passage `(N - 1) % len(passages)`; returns JSON text `{"question": "Explain the following point from the source material: <first sentence of that passage, max 200 chars>", "answer": "<first 400 chars of that passage>", "rubric": [{"criterion": "Accurate explanation grounded in the source", "marks": M}], "citations": ["<id>"], "verdict": "pass", "reasons": []}`; `model = role.model`; tokens = word counts of user prompt and output. With no passages it returns `{"verdict": "pass", "reasons": []}` only.
  - `OpenAIProvider(cfg: ProviderConfig, client=None)`: default client `openai.OpenAI(api_key=os.environ[cfg.api_key_env], base_url=cfg.base_url, timeout=cfg.timeout_seconds, max_retries=0)`; calls `client.chat.completions.create(model=role.model, messages=[{"role": "system", ...}, {"role": "user", ...}], response_format={"type": "json_object"}, max_completion_tokens=role.max_output_tokens)` plus `temperature` when not None and `reasoning_effort=role.effort` when set; returns `choices[0].message.content`, `usage.prompt_tokens`, `usage.completion_tokens`. `openai.APIConnectionError`, `APITimeoutError`, `RateLimitError`, `InternalServerError` -> `ProviderError(transient=True)`; any other `openai.APIError` -> `ProviderError(transient=False)`.
  - `calls.call_with_retries(fn: Callable[[], ProviderResult], retries: RetryConfig, sleep=time.sleep) -> ProviderResult`: retries only transient `ProviderError`, at most `max_attempts` calls, delay `min(max_delay_seconds, base_delay_seconds * 2 ** (attempt - 1))`.
  - `calls.BudgetExhausted(TribunalError)` with `code = "budget_exhausted"`; `reserve_tokens(session, paper_id: str, amount: int, limit: int) -> None` as one atomic `UPDATE papers SET tokens_used = tokens_used + :amount WHERE id = :id AND tokens_used + :amount <= :limit`, raising `BudgetExhausted(f"paper token budget of {limit} is exhausted")` when no row changes; `settle_tokens(session, paper_id: str, delta: int) -> None` adds `delta` (may be negative).
  - `prompts.render(template: PromptTemplate, **variables) -> tuple[str, str]` (Jinja2 `Environment(undefined=StrictUndefined, autoescape=False)`, returns `(system, user)`).
  - `prompts.slot_block(slot: Slot, cfg: RuntimeConfig) -> str`, exactly these lines: `Question number: {number}`, `Question type: {question_type} ({label})`, `Marks: {marks}`, `Bloom level: {bloom label}`, `Difficulty: {difficulty label}`, `Topic: {topic_title}`.
  - `prompts.passages_block(passages: list[ScoredPassage]) -> str`: per passage `[P:{passage_id}] (page {page_number})` then a newline and the text; passages separated by a blank line.
  - `prompts.draft_block(content: dict) -> str` = `json.dumps(content, indent=2, ensure_ascii=False)`.
  - `AppContext.provider_factory: Callable[[str, ProviderConfig], ModelProvider] | None = None`; `AppContext.provider(snapshot, role_name: str) -> tuple[ModelProvider, RoleConfig]`, provider instances cached per `(snapshot_id, provider name)` so the semaphore is shared; uses `provider_factory` when set, else `make_provider`.

- [ ] **Step 1: Write the failing tests**

```python
def test_stub_cites_supplied_passage(): user = slot_block-style text with "Question number: 2", "Marks: 4" + passages_block of two passages
    -> json: citations == [second id], rubric marks sum 4, model == role.model, verdict "pass"
def test_stub_deterministic(): same input twice -> same text
def test_openai_request_shape(): fake client records kwargs -> model, response_format json_object, max_completion_tokens; no "temperature" key when None
def test_openai_transient_mapping(): fake raises openai.RateLimitError -> ProviderError.transient is True
def test_anthropic_not_available(): ConfigurationError
def test_retries_transient_then_succeeds(): fails twice transient, then ok -> result, sleeps [0.01, 0.02]
def test_no_retry_on_permanent(): transient=False -> raised after 1 call
def test_budget_reserve_and_exhaust(ctx): paper with tokens_used 0; reserve 150 of limit 200 ok; next reserve 100 -> BudgetExhausted; settle -50 -> tokens_used 100
def test_render_strict(): template referencing an undefined variable -> jinja2.UndefinedError
def test_slot_block_lines(cfg): exact six lines for a short_answer slot worth 2 marks
def test_context_provider_cached(ctx): ctx.provider(snap, "writer")[0] is ctx.provider(snap, "writer")[0]
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): model providers, retries, budget and prompt rendering`

---

### Task 9: Draft parsing and checks, with legacy regression cases

**Files:**
- Create: `backend/src/tribunal/generation/draft.py`
- Test: `backend/tests/test_draft.py`, `backend/tests/test_legacy_regressions.py`

**Interfaces:**
- Produces:
  - `RubricItem(BaseModel)`: `criterion: str`, `marks: float`. `Draft(BaseModel, extra="ignore")`: `question: str`, `answer: str`, `rubric: list[RubricItem] = []`, `citations: list[str] = []`, `options: list[str] | None = None`, `correct_option: int | None = None`. `Review(BaseModel, extra="ignore")`: `verdict: Literal["pass", "revise", "faculty_review"]`, `reasons: list[str] = []`.
  - `DraftInvalid(ValidationFailed)` with `code = "draft_invalid"`, `problems: list[str]`.
  - `parse_json_object(text: str) -> dict`: strips one surrounding Markdown code fence (with or without a language tag), `json.loads`; non-JSON raises `DraftInvalid(["model output is not valid JSON"])`, a non-object raises `DraftInvalid(["model output is not a JSON object"])`.
  - `parse_draft(text: str, policy: TextPolicy) -> Draft`: `parse_json_object`, then `policy.clean_value`, then `Draft.model_validate`; pydantic errors become `DraftInvalid([f"{loc}: {msg}", ...])`.
  - `parse_review(text: str, policy: TextPolicy) -> Review` (same pattern).
  - `check_draft(draft: Draft, *, marks: int, question_type: str, passage_ids: set[str], placeholder_patterns: list[str], mcq_option_count: int) -> list[str]`, problem strings: `"question is blank"`, `"answer is blank"`, `"rubric is empty"`, `f"rubric marks total {total:g} but the slot is worth {marks}"`, `"no citations"`, `f"citation '{c}' is not one of the supplied passages"`, `f"question matches placeholder pattern '{p}'"`, `f"answer matches placeholder pattern '{p}'"` (patterns via `re.search`); for `question_type == "mcq"` only: `f"expected {n} options, got {k}"`, `"correct_option is missing or out of range"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_fenced_json_parsed(policy): parse_draft('```json\n{"question": "Q?", "answer": "A"}\n```', policy).question == "Q?"
def test_not_json(policy): DraftInvalid, problems == ["model output is not valid JSON"]
def test_wrong_types(policy): '{"question": 5, "answer": "a", "rubric": "x"}' -> DraftInvalid with a "rubric" problem
def test_emoji_removed_before_validation(policy): question "Why \U0001F333 E = −Σ p?" -> "Why E = −Σ p?"
def test_good_draft_has_no_problems(): marks 2, rubric [("x", 2)], citations ["p1"], passage_ids {"p1"} -> []
def test_problem_strings(): blank question/answer, empty rubric, citation "zz" -> exact strings above
def test_rubric_total(): rubric 1.5 on a 2-mark slot -> "rubric marks total 1.5 but the slot is worth 2"
def test_review_parse(policy): '{"verdict": "revise", "reasons": ["cite page"]}' -> Review(verdict="revise")
def test_review_bad_verdict(policy): verdict "ok" -> DraftInvalid

# test_legacy_regressions.py: LEGACY = Path(__file__).resolve().parents[2] / "data" / "generated_papers"; skip if missing
def test_legacy_blank_and_error_slots_rejected(cfg):
    for every question in every saved paper: d = Draft(question=q["question_text"], answer=q["answer"],
        rubric=[RubricItem(criterion="c", marks=q["marks"])], citations=["p1"])
        problems = check_draft(d, marks=q["marks"], question_type="short_answer", passage_ids={"p1"}, placeholder_patterns=<fixture>, mcq_option_count=4)
        blank question_text -> "question is blank" in problems; blank answer -> "answer is blank" in problems;
        question_text starting "[Error generating question" -> "question matches placeholder pattern '(?i)error generating'" in problems
    assert at least 6 blank-question and 5 error-text cases were seen
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): draft parsing, checks and legacy regression cases`

---

### Task 10: Slot and paper generation

**Files:**
- Create: `backend/src/tribunal/generation/slots.py`, `backend/src/tribunal/generation/papers.py`
- Modify: `backend/src/tribunal/jobs/worker.py` (register `generate_paper` and `regenerate_slot`)
- Test: `backend/tests/test_generation.py`

**Interfaces:**
- Consumes: Tasks 5-9.
- Produces:
  - `generate_slot(ctx: AppContext, snapshot: ConfigSnapshot, slot_id: str) -> None`. Uses short sessions per step (never holds a transaction across a model call). Flow:
    1. Slot `running`. Stop with `failed` and `status_detail = "paper deadline exceeded"` when `now > paper.created_at + paper_deadline_seconds` (checked before every model call).
    2. `retrieve(..., subject_id=paper.subject_id, document_ids=paper.document_ids, query=slot.topic_title, topic=<Topic or None>)`. A gap sets `evidence_gap`, `status_detail = gap`, and returns.
    3. Up to `1 + max_revisions` writer attempts. Each: `reserve_tokens(reserve_per_call)`, `call_with_retries` on the writer, `settle_tokens(actual - reserve)`; prompt `render(writer template, slot_block=..., passages_block=..., type_instructions=template.type_instructions.get(question_type, ""), revision_notes=..., mcq_option_count=...)`, where `revision_notes` is `""` on the first attempt and afterwards the previous problems or reviewer reasons, one per line prefixed `- `. Unparseable output (`DraftInvalid`) counts as an attempt with its problems and stores no revision. Parsed output is stored as a new `Revision` (`author="model"`, `content=draft.model_dump()`, `model`, `prompt_version=f"{template.id}@{template.version}"`, `retrieval_version=result.version`, tokens) and becomes `current_revision_id`. If `check_draft` reports problems, those drive the next attempt.
    4. A clean draft goes to the reviewer when `reviewer_enabled` (reviewer template with `slot_block`, `passages_block`, `draft_block`); `ProviderError` or `DraftInvalid` from the reviewer gives `Review(verdict="faculty_review", reasons=[f"reviewer unavailable: {exc}"])`. The review is saved on the revision as `{"verdict", "reasons", "model"}`. `revise` with attempts left drives the next attempt.
    5. Final status: last draft clean and verdict `pass` (or reviewer disabled) -> `drafted`; verdict `revise` with attempts exhausted -> if `escalate_disputed_reviews`, one call to the `reasoner` role with the reviewer template whose verdict decides (`pass` -> `drafted`, otherwise `needs_review`), else `needs_review`; `faculty_review` -> `needs_review`; problems still present -> `failed` with the problems joined by `"; "` in `status_detail`. `ProviderError` after retries or `BudgetExhausted` -> `failed` with the message. On `drafted` or `needs_review`, set `paper.first_question_at` if it is null.
  - `run_paper(ctx: AppContext, snapshot: ConfigSnapshot, paper_id: str) -> None`: paper `running`, `started_at` if null; runs `generate_slot` for every `pending` slot in a `ThreadPoolExecutor(max_workers=min(concurrency.per_paper, n))`; an unexpected exception in one slot marks that slot `failed` with the message and does not stop the others; afterwards paper `drafted` and `completed_at` set.
  - `regenerate(ctx, snapshot, slot_id) -> None`: slot back to `pending`, `status_detail = None`; then `generate_slot`. Other slots are untouched.
  - Worker handlers: `"generate_paper"` -> `run_paper(ctx, ctx.snapshot_for(job.config_snapshot_id), job.target_id)`; `"regenerate_slot"` -> `regenerate(...)`. `FINAL_FAILURE`: `generate_paper` sets paper `failed` with error; `regenerate_slot` sets slot `failed`.

- [ ] **Step 1: Write the failing tests** (helper `make_paper(ctx, subject, docs, topics, n_slots)` builds Paper and Slots directly from `expand_blueprint`; providers are injected through `build_context(..., provider_factory=...)`)

```python
def test_three_slots_drafted(ctx_with_digital_subject): run_paper -> all slots "drafted", each has a revision whose citations resolve to passages of the paper's documents, paper "drafted", first_question_at and completed_at set, tokens_used > 0
def test_revision_records_versions(...): revision.model == "stub-writer", prompt_version == "writer@1", retrieval_version startswith "r-"
def test_writer_model_from_config(test_env, monkeypatch, ...): TRIBUNAL_WRITER_MODEL=stub-writer-b -> revision.model == "stub-writer-b"
def test_evidence_gap_slot(...): faculty topic "Photosynthesis" -> that slot "evidence_gap", status_detail mentions "Photosynthesis", other slots "drafted"
def test_reviewer_exception_never_passes(...): reviewer provider raises ProviderError(transient=False) -> slot "needs_review", review.verdict == "faculty_review"
def test_invalid_output_fails_slot(...): writer always returns "not json" -> slot "failed", "not valid JSON" in status_detail
def test_revise_then_pass(...): writer returns an uncited draft first, a good one second -> "drafted", 2 revisions, second attempt's prompt contains "- no citations"
def test_disputed_review_escalates(...): reviewer always "revise", reasoner "pass" -> "drafted"; reasoner "revise" -> "needs_review"
def test_budget_exhausted(...): paper_tokens env-tuned low via a copied fixture -> slot "failed", "budget" in status_detail
def test_regenerate_preserves_others(...): run_paper; record revision ids; regenerate slot 2 -> slots 1 and 3 keep their revision ids, slot 2 has a new current revision
def test_concurrent_slots_sqlite(test_env, monkeypatch, ...): TRIBUNAL_PER_PAPER=3, 3 slots -> all "drafted"
def test_worker_runs_generate_job(...): enqueue generate_paper job; run_once -> True; paper "drafted"
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): slot and paper generation with review and revision`

---

### Task 11: Paper and answer-key export

**Files:**
- Create: `backend/src/tribunal/export/__init__.py`, `backend/src/tribunal/export/checks.py`, `backend/src/tribunal/export/render.py`
- Test: `backend/tests/test_export.py`

**Interfaces:**
- Produces:
  - `export_problems(session: Session, paper: Paper, cfg: RuntimeConfig) -> list[str]`, problem strings: `f"question {n} is not approved"` (slot status is not `approved`, or the current revision has no `approved_by`), `f"question {n} has a blank question"`, `f"question {n} has a blank answer"`, `f"question {n} rubric totals {t:g}, expected {m}"`, `f"question {n} cites a passage outside this paper's sources"`, `f"questions {a} and {b} are near duplicates"` (Jaccard similarity of `terms()` of the two question texts `>= generation.duplicate_similarity`).
  - `ExportBlocked(Conflict)` with `code = "export_blocked"`.
  - `build_export(session: Session, paper: Paper, part: Literal["paper", "key"], snapshot: ConfigSnapshot, policy: TextPolicy) -> dict`: raises `ExportBlocked("Paper cannot be exported", details=problems)` when `export_problems` is non-empty. Output keys: `paper_id`, `title`, `subject`, `template` (`{"name", "version"}`), `total_marks`, `instructions`, `sections` (each `{"title", "answer_any", "questions": [{"number", "marks", "question", "options"}]}`; for `part == "key"` each question also has `answer`, `rubric` and, when `export.include_citations_in_answer_key`, `citations: [{"passage_id", "document", "page", "url"}]` with `url = f"/api/documents/{document_id}/file#page={page}"`). `key` adds `approvals: [{"number", "revision_id", "approved_by", "approved_at"}]` (approver email, ISO time). Every string passes through `policy.clean_value` last.
  - `render_pdf(doc: dict, *, font_regular: Path | None, font_bold: Path | None, fallbacks: list[str], cfg: ExportConfig) -> bytes` (fpdf2): with `font_regular` it registers that TTF (bold from `font_bold`, else regular) and `set_fallback_fonts(fallbacks)`; without it, core Helvetica, and any text not encodable as latin-1 raises `ValidationFailed("PDF_FONT_REGULAR is required to export this paper's characters")`. Margins `cfg.pdf_margin_mm`, size `cfg.pdf_font_size`.

- [ ] **Step 1: Write the failing tests** (build a drafted paper with the stub as in Task 10; approve by setting `approved_by`, `approved_at` and slot `approved`)

```python
def test_unapproved_blocked(...): build_export raises ExportBlocked whose details include "question 1 is not approved"
def test_key_has_answers_and_citations(...): all approved -> key questions have answer, rubric, citations[0]["url"] == f"/api/documents/{doc}/file#page={page}"; paper part has no "answer" keys; key["approvals"] lists 3 entries
def test_duplicate_detected(...): two approved slots with identical question text -> "questions 1 and 2 are near duplicates"
def test_export_cleans_emoji(...): approved revision content containing "\U0001F333" and "−Σ log₂ café 日本語" -> no emoji in json.dumps(key), math and scripts intact
def test_pdf_core_font(...): latin-1 only doc -> bytes start with b"%PDF"
def test_pdf_needs_font_for_unicode(...): doc with "日本語", no font -> ValidationFailed
def test_pdf_with_font(...): font /usr/share/fonts/TTF/DejaVuSans.ttf (pytest.skip if absent) -> renders "−Σ log₂" without error
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): paper and answer key export with approval checks`

---

### Task 12: HTTP API with dev and OIDC sign-in

**Files:**
- Create: `backend/src/tribunal/api/__init__.py`, `api/app.py`, `api/deps.py`, `api/auth.py`, `api/routers/__init__.py`, `api/routers/subjects.py`, `api/routers/documents.py`, `api/routers/papers.py`
- Modify: `backend/src/tribunal/cli.py` (add `serve`: `uvicorn.run(create_app(build_context(settings, migrate=False)), host=settings.host, port=settings.port)`)
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `create_app(ctx: AppContext) -> FastAPI`. CORS from `settings.allowed_origins` only. `TribunalError` handler returns `{"error": {"code", "message", "details"}}` with `exc.status`. `app.state.ctx = ctx`.
  - `api.auth.authenticate(request, ctx, session) -> User`: `auth_mode == "dev"`: header `X-Dev-User` holds the email (missing -> `Unauthorized("sign in required")`). `oidc`: `Authorization: Bearer <jwt>` verified with `jwt.PyJWKClient(settings.oidc_jwks_url)`, algorithms `["RS256", "ES256"]`, `audience=oidc_audience`, `issuer=oidc_issuer`; requires an `email` claim. Users are created on first sign-in with a lowercased email; `is_admin` = email in `settings.admin_emails`.
  - `api.deps`: `get_ctx`, `get_session` (yields a session inside `factory.begin()`), `current_user`, `require_member(session, subject_id, user) -> Subject` (unknown subject or non-member who is not admin -> `NotFound("subject not found")`).
  - Endpoints (JSON; ids are hex strings):
    - `GET /healthz` -> `{"status": "ok", "config": snapshot_id}` (no auth)
    - `GET /api/me` -> `{"id", "email", "is_admin"}`
    - `POST /api/subjects` `{"name"}` -> 201 `{"id", "name", "role": "owner"}`; `GET /api/subjects` -> subjects where the user is a member
    - `POST /api/subjects/{sid}/members` `{"email"}` -> 201; non-owner -> 403
    - `POST /api/subjects/{sid}/documents` multipart `file`, optional form `languages` (comma list, default `snapshot.config.ingestion.ocr.languages`) -> 202 `{"document": {...}, "job_id"}`. Reads at most `max_upload_bytes + 1` bytes; larger -> 413 code `upload_too_large`; not starting with `%PDF-` -> 422 `validation_failed`; same sha256 already in this subject -> 409. Stores the original at `documents/{subject_id}/{document_id}.pdf`, enqueues `ingest`.
    - `GET /api/subjects/{sid}/documents` -> `[{"id", "filename", "status", "error", "page_count", "created_at"}]`
    - `GET /api/documents/{did}/file` -> the original PDF (`application/pdf`) after `require_member`
    - `GET /api/subjects/{sid}/topics` -> `[{"id", "title", "document_id", "page_start", "page_end", "source"}]`; `POST /api/subjects/{sid}/topics` `{"title"}` -> 201 (source `faculty`); `PATCH /api/topics/{tid}` `{"title"}` -> renamed topic
    - `GET /api/passages/{pid}` -> `{"id", "document_id", "filename", "page_number", "text", "url"}` with `url = f"/api/documents/{did}/file#page={n}"`
    - `POST /api/templates` `{"name", "spec"}` -> 201 `{"id", "name", "version", "spec"}` (version = previous max for this owner and name + 1; spec validated with `parse_template_spec` and `expand_blueprint(spec, ["probe"], current cfg)` so problems show immediately); `GET /api/templates` -> the user's templates
    - `POST /api/papers` `{"subject_id", "template_id", "topic_ids", "document_ids": null | [...], "title"}` -> 202 `{"paper_id", "job_id"}`. Template must belong to the user; topics and documents must belong to the subject; documents must be `ready` (default: every ready document of the subject); none -> 422 `"the subject has no ready documents"`. Creates Paper (`config_snapshot_id = ctx.snapshot.snapshot_id`) and Slots from `expand_blueprint` (`topic_title` copied from the topic), enqueues `generate_paper`.
    - `GET /api/papers/{pid}` -> `{"id", "title", "status", "error", "subject_id", "template": {"id", "name", "version"}, "config_snapshot_id", "document_ids", "tokens_used", "timings": {"first_question_seconds", "total_seconds"}, "slots": [{"id", "number", "section", "question_type", "marks", "topic_title", "status", "status_detail", "revision": null | {"id", "number", "author", "content", "review", "model", "prompt_version", "approved_by", "approved_at"}, "citations": [{"passage_id", "document_id", "page_number", "url"}]}]}` (timings measured from `created_at`, null until set)
    - `POST /api/papers/{pid}/slots/{n}/regenerate` -> 202 `{"job_id"}` (clears the slot's approval)
    - `PUT /api/papers/{pid}/slots/{n}` `{"content": <Draft fields>}` -> 200 new revision (`author="faculty"`, `created_by`), slot `drafted`; `check_draft` problems -> 422 with details. Any earlier approval no longer counts because the current revision changed.
    - `POST /api/papers/{pid}/slots/{n}/approve` -> 200; slot must be `drafted` or `needs_review` with a current revision, else 409; sets `approved_by`, `approved_at` on the current revision and slot `approved`
    - `GET /api/papers/{pid}/export?part=paper|key&format=json|pdf` -> JSON body or `application/pdf`; blocked -> 409 `export_blocked` with problems
  - All paper, passage, document and topic routes resolve the owning subject and call `require_member`.

- [ ] **Step 1: Write the failing tests** (`TestClient(create_app(ctx))`; `h(email) = {"X-Dev-User": email}`; jobs are processed by calling `run_once(ctx)` until it returns False)

```python
def test_requires_sign_in(client): GET /api/subjects without header -> 401, body["error"]["code"] == "unauthorized"
def test_subject_isolation(client): a creates subject; b GET /api/subjects -> []; b GET /api/subjects/{sid}/documents -> 404
def test_upload_and_ingest(client, ctx): upload digital_pdf -> 202; run jobs; documents list shows "ready", page_count 2; topics listed
def test_upload_rejects_non_pdf(client): b"hello" -> 422
def test_upload_too_large(client, ...): fixture copy with max_upload_bytes 1000 -> 413, code "upload_too_large"
def test_duplicate_upload(client): same bytes twice in one subject -> 409; in a second subject owned by the same user -> 202
def test_file_download_scoped(client): b GET /api/documents/{did}/file -> 404; a -> 200, content starts with b"%PDF"
def test_template_problems_immediate(client): POST /api/templates with question_type "essay" -> 422, details contain "sections[0].question_type 'essay' is not enabled"
def test_paper_flow(client, ctx): template (3 short answers x 2 marks, total 6) -> POST /api/papers 202 -> run jobs -> GET paper: 3 slots "drafted", each citation url opens a page whose stored text contains the passage text, timings non-null
def test_regenerate_one_slot(client, ctx): regenerate slot 2 -> run jobs -> slots 1 and 3 revision ids unchanged
def test_edit_and_approve(client, ctx): PUT slot 1 with blank answer -> 422; valid edit -> 200 author "faculty"; approve all three -> export key JSON 200; export before approval -> 409 "export_blocked"
def test_foreign_paper_hidden(client): b GET /api/papers/{pid} and export -> 404
def test_no_ready_documents(client): POST /api/papers on an empty subject -> 422
```

- [ ] **Step 2: Run to verify failure; Step 3: Implement; Step 4: Run tests and ruff; Step 5: Commit** `feat(backend): HTTP API with subject-scoped access`

---

### Task 13: First-ticket acceptance test, CI and setup guide

**Files:**
- Create: `backend/tests/test_acceptance.py`
- Create: `.github/workflows/backend.yml`
- Create: `backend/README.md`
- Test: `backend/tests/test_acceptance.py`

**Interfaces:**
- Consumes: the API (Task 12), `run_once`, `tests/pdf_fixtures.py`, `FakeOcrEngine`.
- Produces: no new runtime code. Fix defects the acceptance test exposes in the module that owns them.

- [ ] **Step 1: Write the acceptance tests** (all subject and topic content lives in this test file only)

```python
def test_first_ticket(test_env, monkeypatch, capsys):
    # Faculty A: subject "Machine Learning", digital chapter with bookmarks Decision Trees (p1) / Linear Regression (p2) / Neural Networks (p3)
    # Faculty B: subject "Modern History", scanned_pdf(2) with FakeOcrEngine texts headed "1 The French Revolution" / "2 The Industrial Revolution"
    # Both upload through the API, run jobs: documents "ready"; B's pages method "ocr"; topics proposed from bookmarks (A) and headings (B)
    # Each creates a 3-slot short-answer template (2 marks each) and requests a paper over their topics
    # Assert: every slot "drafted"; every citation of A's paper points at A's document only, B's at B's only
    # Assert: GET /api/passages/{id} page_number equals a page whose text contains that passage text, and the url ends with f"#page={page}"
    # Assert: A's topic "trees"-style request in B's subject (faculty topic "Trees") -> slot "evidence_gap"
    # Regenerate slot 2 of A's paper -> slots 1 and 3 keep revision ids
    # Approve A's slots -> export key (json) 200 with 3 answers and citations; export paper (pdf) starts with b"%PDF"
    # B cannot GET A's paper, passage, file or export (404)
    # Print f"timings A={...} B={...}" from GET /api/papers/{id}["timings"]; both first_question_seconds and total_seconds are non-null

def test_new_subject_from_headings_only(...):  # digital_pdf without bookmarks or numbered headings -> zero topics; faculty adds topic "Coral Reefs" whose words appear in the text -> paper slot "drafted"

def test_emoji_response_through_generation_and_export(...):
    # provider_factory wraps StubProvider so the writer JSON's question becomes
    # "Explain \U0001F333 entropy: H = −Σ p log₂ p (café, 日本語) \U0001F44D\U0001F3FD" and the answer gets " \U0001F680" appended
    # GET /api/papers/{id} slot question contains "H = −Σ p log₂ p (café, 日本語)" and no emoji (TextPolicy.findings == [])
    # after approval, export key JSON: TextPolicy.findings over json.dumps(...) == [], and "−Σ", "log₂", "café", "日本語" present

def test_two_deployment_profiles(tmp_path, monkeypatch):
    # profile 1: TRIBUNAL_WRITER_MODEL=stub-writer-a, TRIBUNAL_PER_PAPER=1, TRIBUNAL_OCR_LANGUAGE=eng, own DB/store
    # profile 2: TRIBUNAL_WRITER_MODEL=stub-writer-b, TRIBUNAL_PER_PAPER=3, TRIBUNAL_OCR_LANGUAGE=hin, own DB/store
    # same scanned upload + paper per profile -> revision.model differs as set, FakeOcrEngine saw ["eng"] vs ["hin"],
    # snapshot ids differ, and the template's marks (2 vs 5 per slot) appear in slot marks

def test_missing_values_are_actionable(monkeypatch, capsys):
    # example runtime config without TRIBUNAL_WRITER_MODEL -> main(["check-config"]) == 2 and stderr names TRIBUNAL_WRITER_MODEL

def test_no_sample_course_data_in_product_code():
    # scan backend/src and backend/config: no match for the regex r"\bCO1\b" (case-sensitive) and none of the case-insensitive strings "AIML", "machine learning", "modern history", "gpt-", "claude-"
```

- [ ] **Step 2: Run them**

Run: `cd backend && ../.venv/bin/python -m pytest tests/test_acceptance.py -v -s`
Expected: PASS, and the `timings A=... B=...` line printed. Fix any failure in the owning module.

- [ ] **Step 3: Write `.github/workflows/backend.yml`**

Triggers on `push` and `pull_request` touching `backend/**`, `docs/**` or the workflow file. One `ubuntu-latest` job, Python `3.12`: `pip install -e "backend[dev]"`, then `ruff check backend`, then `cd backend && pytest`, then `tribunal scan-text --config backend/config/runtime.example.yaml backend docs`. Frontend CI is out of scope for this plan.

- [ ] **Step 4: Write `backend/README.md`**

Short plain-text setup guide: install (`pip install -e "backend[dev]"`, optional extras `ocr`, `openai`, `postgres`, `embeddings-local`), the required environment variables (names from `Settings`, with which are required per profile), runtime config (`RUNTIME_CONFIG`, the example file and its `TRIBUNAL_*` variables), commands (`tribunal check-config`, `db-upgrade`, `serve`, `worker`, `scan-text`), how to run tests, and what this build does not yet include (S3 storage, Anthropic provider, OpenAI embeddings, contents-page outlines, layout extraction, progress streaming, legacy import, frontend integration).

- [ ] **Step 5: Run the full suite, ruff and the text scan**

Run: `cd backend && ../.venv/bin/python -m pytest && ../.venv/bin/ruff check . && ../.venv/bin/python -m tribunal.cli scan-text --config config/runtime.example.yaml . ../docs`
Expected: all pass, `All checks passed!`, scan exit 0.

- [ ] **Step 6: Commit** `test(backend): first-ticket acceptance, CI workflow and setup guide`
