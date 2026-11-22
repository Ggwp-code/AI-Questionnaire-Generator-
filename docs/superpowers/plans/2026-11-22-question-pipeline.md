# Question Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-call question generation with the instructor's six-stage pipeline (reader, writer, code gen, verification, review, output), where computational answers are checked by an independently generated Python verifier that runs in an isolated service, failures route back to the stage at fault, and only the revision that passed is released.

**Architecture:** Explicit stage functions in `backend/src/tribunal/generation/pipeline.py` driven by a routing loop with separate writer and code gen retry limits, recording every stage outcome in a new `pipeline_steps` table. Verifier programs run through a `VerifierClient`: an HTTP client for a separate verifier service (no internet, no secrets, strict limits) in hosted use, and the same runner as a subprocess for development and tests. Questions gain parts for every type, MCQs included.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy with Alembic, the stdlib `http.server` for the verifier service, `resource` limits, numpy and sympy inside verifiers, Docker for the verifier image, Cloudflare Containers for hosting.

**Spec:** `docs/superpowers/specs/2026-11-22-question-pipeline-design.md` (approved; its "Settled after the builder check" section is binding).

**Branching:** Tasks 1-7 build on `feat/tribunal-backend` at 4c35d3f (the verification change 252ecc1 included). Task 8 needs the staging-readiness branch (gateway Worker and images) and runs after that plan's Task 6 lands, on a branch that contains both.

## Global Constraints

- The anti-goals of the spec bind every task: verifier code never runs where it can reach the database, secrets, other services or the internet; the proposed answer never reaches code gen; the released revision is the one verified and reviewed; a failing, unparseable or unavailable stage never ends `drafted`.
- "Use plain text without emojis in all project-authored material." Write non-ASCII characters in test sources as escapes.
- Roles, prompts, models, retry limits, tolerances and verifier limits come from the runtime config or `Settings`; no model, course, subject or topic names in `backend/src/`, `backend/config/prompts/` or `deploy/`.
- Model output passes through the text policy before validation; every write in a job goes through `owned_transaction` (fencing); replay protection via `revisions.job_id` stays intact.
- Python `>=3.12`. New dependency allowed: `sympy`, only as a new optional extra `verifier` in `backend/pyproject.toml` and inside the verifier image. CI installs `backend[dev,openai,s3,verifier]`.
- Verifier programs: Python 3.12, standard library plus `numpy` and `sympy`; the last line of stdout is one JSON object as the spec defines.
- Git: never add `Co-Authored-By` or any AI attribution.

## Review Focus

1. A verifier program that reads environment variables, opens network sockets, writes outside its directory, forks repeatedly, allocates without bound, loops forever or prints without bound: the run fails or is stopped within its limits, the worker is unaffected, and the outcome is recorded. Tests in Task 2 (subprocess limits) and Task 3 (image isolation, network off).
2. The proposed answer, rubric or key never appears in code gen's input, for any type, with or without parts. Test in Task 6.
3. Writer and verifier disagree: review routes `revise_question` to the writer and `fix_verifier` to code gen, each loop stops at its limit, and nothing ends `drafted` without a passing verification (computational) and a `pass` review. Tests in Task 6.
4. The verifier service is down, times out or returns garbage: the slot ends `needs_review` with the reason, the job itself succeeds, and no exception escapes. Test in Task 6.
5. Parts that do not add up: part marks that do not sum to the slot's marks, a part rubric that does not total its marks, an MCQ part with zero or two keys, or a missing part. Caught by structural checks and sent back to the writer. Tests in Task 1 and Task 5.

---

### Task 1: Parts in templates, slots and drafts

**Files:**
- Modify: `backend/src/tribunal/generation/blueprint.py`, `backend/src/tribunal/generation/draft.py`, `backend/src/tribunal/generation/prompts.py`, `backend/src/tribunal/db/models.py`, `backend/src/tribunal/api/routers/papers.py`, `backend/src/tribunal/export/render.py`, `backend/config/prompts/writer.yaml`
- Create: `backend/src/tribunal/migrations/versions/0007_slot_parts.py`
- Test: `backend/tests/test_blueprint.py`, `backend/tests/test_draft.py`, `backend/tests/test_export.py`

**Interfaces:**
- `TemplateSection.parts: list[PositiveInt] | None = None` (marks per part); validation problem `f"sections[{i}].parts sum to {s}, expected marks_each {m}"` when the sum differs; `SlotSpec.parts: list[int] | None`; `Slot.parts` nullable JSON column (migration 0007, chained on "0006"); `create_paper` copies it to each slot.
- `DraftPart(BaseModel, extra="ignore")`: `label: str`, `question: str`, `answer: str`, `rubric: list[RubricItem] = []`, `citations: list[str] = []`, `options: list[str] | None = None`, `correct_option: int | None = None`. `Draft.parts: list[DraftPart] | None = None` (the top-level `question` is the shared stem; top-level `answer` may be blank when parts exist).
- `check_draft(..., parts: list[int] | None = None)`: when `parts` is set, problems `f"expected {n} parts, got {k}"`, `f"part {label} rubric totals {t:g}, expected {m}"`, `f"part {label} answer is blank"`, and for MCQ types the existing MCQ rules per part prefixed `f"part {label}: "`; citations of every part must be in the evidence. Without parts, behavior is unchanged.
- `slot_block` states the parts and their marks; the writer prompt explains the parts format; export renders parts under the stem in order, with per-part answers and rubrics in the key.

- [ ] **Step 1: Failing tests:** `test_parts_must_sum_to_marks`; `test_draft_parts_checked` (count, per-part rubric total, blank answer, an MCQ part with an out-of-range key and with duplicate options); `test_draft_without_parts_unchanged`; `test_export_renders_parts` (paper shows part questions in order; key shows part answers and rubrics).
- [ ] **Step 2: Run them**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan (`backend docs`). **Commit** `feat(backend): question parts for every type`.

### Task 2: Verifier runner and client

**Files:**
- Create: `backend/src/tribunal/verifier/__init__.py` (empty), `backend/src/tribunal/verifier/runner.py`, `backend/src/tribunal/verifier/client.py`
- Modify: `backend/src/tribunal/settings.py`, `backend/src/tribunal/runtime_config.py`, `backend/src/tribunal/context.py`, `backend/pyproject.toml`, `backend/config/runtime.example.yaml`, `backend/tests/fixtures/runtime.yaml`, `.github/workflows/backend.yml`
- Test: `backend/tests/test_verifier_runner.py`, `backend/tests/test_settings.py`

**Interfaces:**
- `runner.py` imports only the standard library (the verifier image copies this file alone). `Limits` dataclass: `cpu_seconds`, `wall_seconds`, `memory_mb`, `max_processes`, `max_file_bytes`, `max_output_bytes`. `RunResult` dataclass: `status: Literal["ok", "error", "timeout", "too_large"]`, `last_line: str | None`, `stderr_tail: str`. `run_program(program: str, limits: Limits, python: str = sys.executable) -> RunResult`: writes the program into a fresh temporary directory, runs `python -I` there with an empty environment (only `PATH` and `HOME` set to the temporary directory), applies `resource` limits in the child (`RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_NPROC`, `RLIMIT_FSIZE`), enforces the wall clock, caps captured stdout at `max_output_bytes` (exceeding gives `too_large`), and removes the directory afterwards.
- Runtime config `verifier: {cpu_seconds, wall_seconds, memory_mb, max_processes, max_file_bytes, max_output_bytes}` (example: 10, 20, 512, 16, 1048576, 65536; fixture: smaller values that keep tests fast).
- `Settings.verifier_mode: Literal["service", "subprocess"] = "subprocess"`, `verifier_url: str | None`, `verifier_token: SecretStr | None`. Hosted profiles: `verifier_mode` must be `service` with `verifier_url` set (problem strings name `VERIFIER_MODE` and `VERIFIER_URL`).
- `client.py`: `class VerifierClient(Protocol): def run(self, program: str) -> RunResult`; `SubprocessVerifier(limits)`; `HttpVerifier(url, token, limits, client: httpx.Client | None = None)` posting `{"program", "limits"}` to `<url>/run` with `Authorization: Bearer <token>` when set, mapping transport failures and non-2xx to `RunResult("error", None, "verifier service unavailable: <status or exception type>")`. `AppContext.verifier` built from settings.
- New optional extra `verifier = ["sympy>=1.12"]`; CI installs it.

- [ ] **Step 1: Failing tests:** `test_program_result_last_line`; `test_environment_is_empty` (a program printing `os.environ` sees only PATH and HOME; a secret exported in the parent is absent); `test_cpu_loop_is_stopped`; `test_memory_bomb_is_stopped`; `test_output_flood_is_too_large`; `test_runs_in_its_own_directory_and_cleans_up`; `test_exception_is_error_with_stderr_tail`; `test_http_verifier_maps_failures` (httpx MockTransport: 500 and a raised transport error both give `error`); `test_hosted_profile_requires_service_mode`.
- [ ] **Step 2: Run them**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan. **Commit** `feat(backend): verifier runner with resource limits and a client`.

### Task 3: Verifier service and image

**Files:**
- Create: `backend/src/tribunal/verifier/service.py`, `deploy/verifier.Dockerfile`, `backend/tests/test_verifier_service.py`, `backend/tests/verifier_isolation/test_image_isolation.py`
- Modify: `.github/workflows/backend.yml`, `.dockerignore`, `backend/README.md`

**Interfaces:**
- `service.py` imports only the standard library and `runner.py`: an `http.server` server with one route `POST /run` (body `{"program": str, "limits": {...}}`, program at most 64 KB; limits clamped to the service's own maximums from environment variables `VERIFIER_MAX_*`), optional bearer token from `VERIFIER_TOKEN`, responses `200 {"status", "last_line", "stderr_tail"}`, `400` for a bad body, `401` for a bad token, `404` otherwise. `python -m tribunal.verifier.service` listens on `0.0.0.0:8080`.
- `deploy/verifier.Dockerfile`: `python:3.12-slim`, `pip install numpy sympy` (pinned), copies only `runner.py` and `service.py`, runs as an unprivileged user, no other packages, no secrets.
- CI job `verifier-image`: build the image, then run `backend/tests/verifier_isolation` with the container started as `docker run --network none --read-only --tmpfs /tmp ...` (the tests are skipped unless `TRIBUNAL_VERIFIER_IMAGE_URL` is set).

- [ ] **Step 1: Failing tests:** `test_service_runs_a_program` and `test_service_rejects_bad_token_and_body` (in-process server on a free port); image isolation tests: a program opening a socket to a public address fails; reading `/proc/1/environ` or `os.environ` shows no secret; writing outside `/tmp` fails; a fork loop, a memory bomb and an infinite loop are stopped; `numpy` and `sympy` import.
- [ ] **Step 2: Run** the service tests locally; the image tests need Docker (run them if Docker is usable, otherwise record that CI is the check).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan (`backend docs deploy`), workflow YAML parses. **Commit** `feat(verifier): isolated verifier service and image`.

### Task 4: Comparing verifier results with the proposed answer

**Files:**
- Modify: `backend/src/tribunal/generation/verification.py`
- Test: `backend/tests/test_verification.py`

**Interfaces:**
- Remove the arithmetic-expression evaluator (`_calculate`, `verify_solution`); keep `COMPUTATIONAL_TYPES`.
- `parse_verifier_output(result: RunResult) -> dict`: requires status `ok` and a last line that is one JSON object of the shapes in the spec (`answer`, `option_index` plus `answer`, `parts`, or `unsupported: true`); otherwise raises `VerifierOutputInvalid(reason)`.
- `compare(output: dict, draft: Draft, question_type: str, tolerance: float) -> Verdict`, `Verdict = (status: Literal["pass", "mismatch", "unsupported"], detail: str, per_part: list[str] | None)`. Values compare as numbers with relative tolerance when both parse as plain numbers (`parse_number`), otherwise as strings after whitespace normalisation; `numerical` proposed answers must be plain numbers; computational MCQ (whole question or part) passes only when exactly one option matches the computed value and that option is the key; parts compare position by position.

- [ ] **Step 1: Failing tests:** numeric within and outside tolerance; regex and trace string results with different whitespace; computational MCQ with zero, one and two matching options and a wrong key; parts with one mismatching part (detail names the part); `unsupported`; malformed last lines (not JSON, extra keys, wrong types, status `timeout`) raise `VerifierOutputInvalid`.
- [ ] **Step 2: Run them**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan. **Commit** `feat(backend): compare verifier results with proposed answers`.

### Task 5: Structural check library

**Files:**
- Create: `backend/src/tribunal/generation/checks.py`
- Test: `backend/tests/test_checks.py`

**Interfaces:**
- `CHECKS: dict[str, Check]` where `Check = Callable[[Draft, SlotContext], list[str]]` and `SlotContext` carries marks, parts, question type, evidence passage ids, `mcq_option_count`, placeholder patterns and `words_per_mark` bounds. Mandatory checks (always run): `marks_match`, `parts_complete`, `citations_in_evidence`, `no_placeholders`, and for MCQ types `mcq_option_count` and `mcq_single_key`. Selectable checks (code gen may choose for conceptual types): `answer_not_in_question`, `answer_length_fits_marks`, `rubric_items_distinct`, `options_plausible_length`. Unknown names chosen by code gen are ignored and reported in the result.
- `run_checks(draft, ctx, selected: list[str]) -> CheckReport(passed: bool, problems: list[str], ran: list[str], ignored: list[str])`. `check_draft` keeps its signature and delegates to the mandatory checks, so existing callers and edits are unchanged.
- Runtime config `generation.words_per_mark: {min, max}` (example 5 and 120).

- [ ] **Step 1: Failing tests:** each check with one passing and one failing draft (with and without parts); unknown selected names land in `ignored`; `check_draft` results unchanged for the existing test cases.
- [ ] **Step 2: Run them**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan. **Commit** `feat(backend): structural check library`.

### Task 6: The staged pipeline

**Files:**
- Create: `backend/src/tribunal/generation/pipeline.py`, `backend/src/tribunal/migrations/versions/0008_pipeline_steps.py`
- Modify: `backend/src/tribunal/generation/slots.py`, `backend/src/tribunal/generation/draft.py` (Review verdicts), `backend/src/tribunal/db/models.py`, `backend/config/prompts/codegen.yaml`, `backend/config/prompts/reviewer.yaml`, `backend/config/runtime.example.yaml`, `backend/tests/fixtures/runtime.yaml`, `backend/src/tribunal/runtime_config.py`
- Test: `backend/tests/test_pipeline.py`, `backend/tests/test_generation.py`

**Interfaces:**
- `PipelineStep` model (`pipeline_steps`): `id, slot_id, revision_id|None, job_id|None, stage` (`reader|writer|structure|codegen|verification|review|reasoner|output`)`, attempt: int, outcome: str, feedback: Text|None, model|None, prompt_version|None, input_tokens, output_tokens, created_at`. Migration 0008 chained on "0007". Every write goes through `owned_transaction`.
- `Review.verdict: Literal["pass", "revise_question", "fix_verifier", "faculty_review"]`; the reviewer prompt receives the draft, the structural check report, and for computational types the verifier program, its run result and the comparison verdict, and explains the four verdicts.
- Code gen (`codegen.yaml`, new version): input is the student-facing question only (stem, part questions, options; never answers, rubrics or keys). For computational types it returns `{"program": "<python>"}` or `{"unsupported": true}`; for conceptual types `{"checks": [...]}` from the selectable names (the prompt lists them).
- `run_pipeline(ctx, snapshot, slot_id, deadline, owner) -> tuple[str, str | None]` replaces `_generate` inside `generate_slot` (statuses as today: `drafted`, `needs_review`, `evidence_gap`, `failed`). Routing as the spec states; limits `generation.max_writer_rounds` and `generation.max_codegen_rounds` replace `max_revisions` (example 2 and 2; fixture as tests need). Verifier program, run result, comparison, check report and review are stored in the revision's `review` JSON; Output sets `current_revision_id` to the passing revision and records an `output` step naming it. Replay protection (`job_id` early return only when the slot is not pending or running) is kept.

- [ ] **Step 1: Failing tests** (stub provider scripted per role, `SubprocessVerifier` with fixture limits): computational pass path (steps reader, writer, structure, codegen, verification, review, output in order; released revision id equals the verified one); mismatch then `fix_verifier` then pass; mismatch then `revise_question` then pass; code gen limit reached gives `needs_review` with the reason; verifier service error (fake client) gives `needs_review` and the job succeeds; `unsupported` routed by review; structural failure goes straight back to the writer; conceptual path runs mandatory plus selected checks and always reviews; MCQ with parts verified per part; code gen input never contains the answer, rubric or key (assert on the rendered prompt for every type, with parts); writer-limit dispute escalates to the reasoner, which sees the verifier program; replay of a finished job leaves an approved slot alone.
- [ ] **Step 2: Run them**; confirm failures.
- [ ] **Step 3: Implement.** Update existing generation tests whose expectations change (list each in the report).
- [ ] **Step 4: Run** focused tests, full suite, CI simulation, ruff, scan. **Commit** `feat(backend): staged question pipeline with routed retries`.

### Task 7: Stage history route and documentation

**Files:**
- Modify: `backend/src/tribunal/api/routers/papers.py`, `backend/README.md`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- `GET /api/papers/{pid}/slots/{n}/steps` (member of the paper's subject, else 404): `[{"stage", "attempt", "outcome", "feedback", "revision_id", "model", "created_at"}]` in creation order.
- README section "Question pipeline": the stages, routing and limits, verifier modes and settings (`VERIFIER_MODE`, `VERIFIER_URL`, `VERIFIER_TOKEN`), the verifier program contract, how to run the verifier service locally (`docker build -f deploy/verifier.Dockerfile` and run with `--network none`), and the `verifier` extra.

- [ ] **Step 1: Failing tests:** steps listed in order for a generated slot; 404 for another subject's member.
- [ ] **Step 2: Run**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan. **Commit** `feat(backend): stage history route and pipeline docs`.

### Task 8: Hosted verifier on Cloudflare

Runs after the staging-readiness plan's Task 6 (gateway Worker) and Task 7 (runbook) land, on a branch containing both plans' work.

**Files:**
- Modify: `deploy/worker/wrangler.jsonc`, `deploy/worker/src/containers.ts`, `deploy/worker/src/index.ts`, `deploy/worker/src/dispatch.ts`, `deploy/worker/test/dispatch.test.ts`, `docs/staging-runbook.md`

**Interfaces:**
- A `VerifierContainer` class from `deploy/verifier.Dockerfile`, started with internet disabled and no secrets in its environment (only `VERIFIER_TOKEN` and `VERIFIER_MAX_*`).
- Gateway route `POST /internal/verify/run` (never public: same 404 rule for unauthenticated `/internal/*`), requiring `Authorization: Bearer <INTERNAL_DISPATCH_TOKEN>`, forwarding the body to the verifier container's `/run`. Backend hosted settings: `VERIFIER_MODE=service`, `VERIFIER_URL` pointing at the gateway's internal verify route; the backend sends the dispatch token as `VERIFIER_TOKEN`. Confirm against current Cloudflare Containers documentation how a container is started without internet access and cite it.
- Runbook: the new secret and vars, and a rollout check that a verifier program cannot reach the internet.

- [ ] **Step 1: Failing tests** (vitest): the verify route requires the token and forwards to the verifier container stub; the verifier container's start environment contains no names outside its allow-list; the container is started with internet disabled.
- [ ] **Step 2: Run**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `npx tsc --noEmit`, `npx vitest run`, the wrangler dry run (record output), the Python suite, scan. **Commit** `feat(deploy): hosted verifier container`.
