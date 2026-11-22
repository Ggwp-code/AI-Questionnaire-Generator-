# Staging Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `tribunal` backend deployable to a Cloudflare staging environment: proven on PostgreSQL, hardened, packaged as two container images, and woken by a gateway Worker through Cloudflare Queues and a cron trigger.

**Architecture:** PostgreSQL stays the single source of truth for jobs, leases and fencing (unchanged semantics). A queue message is only a wake-up: after a request commits, the API posts `{job_id, kind}` to the ingestion or generation queue; the consumer Worker calls `POST /internal/drain` on the matching container, which claims and runs jobs until none are claimable. A one-minute cron drain covers delayed retries, reclaimed leases and lost messages. The gateway Worker serves the built frontend as static assets and forwards `/api/*` to the API container; `/internal/*` is never public.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy, psycopg + pgvector, httpx; Docker (linux/amd64); Cloudflare Workers, Containers (Durable Object scheduling), Queues, Cron Triggers, Static Assets; TypeScript with Wrangler and Vitest.

**Spec:** `docs/department-pilot-roadmap.md` (deployment table and "Rollout" paragraph; Milestone 4 "Make jobs and data durable"; Milestone 5 access controls). Frontend integration is a separate plan.

**Not in this plan (deliberate, from the roadmap):** presigned direct-to-R2 uploads (uploads keep going through the API, which is enough at chapter size); dispatching page batches or single slots instead of whole jobs; LangGraph checkpoints; frontend integration.

## Global Constraints

- "Use plain text without emojis in all project-authored material."
- "Every course-specific value, deployment setting and tunable policy must have an explicit source." Deployment values come from `tribunal.settings.Settings` (environment); policy from the runtime config YAML; secrets from the Cloudflare secret store. No account id, hostname, database URL, token or key is committed anywhere.
- "Model credentials stay in the cloud deployment's secret store." "Keep traces free of paper text by default."
- "Report missing required settings by name at startup. Keep secret values out of logs and browser bundles."
- "Make failure visible from provider to browser."
- Hosted profiles keep their existing rules: `AUTH_MODE=oidc`, PostgreSQL `DATABASE_URL`, `STORAGE_BACKEND=s3`, https `PUBLIC_BASE_URL`, no stub provider.
- Python `>=3.12`; Python dependencies only from `backend/pyproject.toml` (extras `postgres`, `s3`, `openai`, `ocr` already exist). The Worker project may add only `wrangler`, `typescript`, `vitest` and `@cloudflare/workers-types`, pinned to exact versions.
- Container images build for `linux/amd64`.
- No course, subject, topic or model names in `backend/src/`, `backend/config/prompts/` or `deploy/`.
- Git: never add `Co-Authored-By` or any AI attribution.

## Review Focus

1. The Cloudflare queue API is down or the token is wrong: the upload or paper request still succeeds (202), the job stays queued, and the next cron drain runs it. Test in Task 4.
2. A drain call with a missing or wrong token is refused with 401 and runs no job; through the gateway, any `/internal/*` path is 404. Tests in Task 3 and Task 6.
3. A queue drain and a cron drain overlap: every job runs exactly once. Test in Task 3.
4. A container is stopped mid-job: after the lease expires, a later drain resumes the job and the stale worker's writes are discarded (existing fencing). Test in Task 3.
5. Two workers claim at the same moment on PostgreSQL: exactly one gets the job (`FOR UPDATE SKIP LOCKED`). The existing `test_simultaneous_claims` runs on PostgreSQL in Task 1.

---

### Task 1: PostgreSQL test target in CI

**Files:**
- Modify: `backend/tests/conftest.py`, `.github/workflows/backend.yml`, `backend/README.md`
- Modify (only as the PostgreSQL run demands): files under `backend/src/tribunal/`
- Test: `backend/tests/test_db.py`

**Interfaces:**
- Produces: env `TRIBUNAL_TEST_POSTGRES_URL` (an admin URL such as `postgresql+psycopg://postgres:postgres@localhost:5432/postgres`). When set, the `test_env` fixture creates a database named `tribunal_test_<12 hex chars>` (AUTOCOMMIT connection), sets `DATABASE_URL` to it, and drops it at teardown (`DROP DATABASE ... WITH (FORCE)`). Tests that build their own SQLite URL stay on SQLite.
- `test_env` also deletes every environment variable that `Settings` reads and every `TRIBUNAL_*` variable before setting its own, so a developer's exported values never leak into tests (deferred finding from the first plan).

- [ ] **Step 1: Failing test** `test_postgres_target_in_use` in `test_db.py`, skipped unless `TRIBUNAL_TEST_POSTGRES_URL` is set: the `settings` fixture's database dialect is `postgresql`, migrations reach head, and `SELECT extname FROM pg_extension` contains `vector`. Plus `test_env_does_not_leak` (always runs): with `TRIBUNAL_WRITER_MODEL` and `PDF_FONT_REGULAR` exported before the fixture, the loaded settings do not see them.
- [ ] **Step 2: Run against a local PostgreSQL** (`docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16`) and the full suite with `TRIBUNAL_TEST_POSTGRES_URL` set; record every failure.
- [ ] **Step 3: Implement** the fixture changes, then fix each PostgreSQL-only failure in `backend/src` at its root (no test skips for PostgreSQL). List each defect and fix in the report.
- [ ] **Step 4: CI** gains a job `test-postgres` with service `pgvector/pgvector:pg16` (`POSTGRES_PASSWORD: postgres`, health check `pg_isready`), install `pip install -e "backend[dev,openai,s3,postgres]"`, env `TRIBUNAL_TEST_POSTGRES_URL`, run `cd backend && pytest`. The existing SQLite job stays.
- [ ] **Step 5: Run** the full suite on SQLite and on PostgreSQL; ruff; text scan. **Commit** `test(backend): run the suite on PostgreSQL in CI`.

### Task 2: Hardening before staging

**Files:**
- Modify: `backend/src/tribunal/logging_setup.py`, `backend/src/tribunal/settings.py`, `backend/src/tribunal/jobs/worker.py`, `backend/src/tribunal/cli.py`, `backend/src/tribunal/context.py`, `backend/README.md`
- Test: `backend/tests/test_logging.py` (create), `backend/tests/test_settings.py`, `backend/tests/test_queue.py`, `backend/tests/test_cli.py`

**Interfaces:**
- `Settings.secret_values()` also returns the password component of `database_url` when the URL has one.
- Redaction covers the formatted traceback and stack text in the JSON log output, not only `record.msg`.
- Worker failure text stored on documents, papers and slots: a `TribunalError` keeps its message; any other exception stores exactly `f"internal error in job {job.id}; see the worker log"` (the traceback is logged).
- `Settings.host` default: `127.0.0.1` for profiles `local` and `test`, `0.0.0.0` for hosted profiles; an explicit `HOST` always wins.
- `tribunal check-config` reports the same problems as startup (hosted-profile and provider-credential checks included). `serve` and `worker` on a database that is not at the migration head exit 2 with `database is not migrated: run tribunal db-upgrade`.

- [ ] **Step 1: Failing tests:** `test_traceback_is_redacted` (log an exception whose message contains the signing secret and the database password; neither appears in the captured JSON line, `[REDACTED]` does); `test_unexpected_error_text_is_generic` (handler raises `RuntimeError("/srv/x.py SELECT secret")`; the document error equals the generic string and contains no path); `test_host_default_by_profile`; `test_check_config_reports_credentials` (example config with `OPENAI_API_KEY` unset gives exit 2 naming it); `test_serve_refuses_unmigrated_database`.
- [ ] **Step 2: Run them**; confirm each fails for the stated reason.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan. **Commit** `fix(backend): redact tracebacks, hide internal errors, safer defaults and checks`.

### Task 3: Job kinds and the internal drain route

**Files:**
- Modify: `backend/src/tribunal/jobs/queue.py`, `backend/src/tribunal/jobs/worker.py`, `backend/src/tribunal/api/app.py`
- Create: `backend/src/tribunal/api/routers/internal.py`
- Test: `backend/tests/test_queue.py`, `backend/tests/test_internal.py` (create)

**Interfaces:**
- `claim_next(factory, lease_seconds, now=None, kinds: Sequence[str] | None = None)`: `kinds` restricts claimable jobs; `None` keeps today's behavior.
- `run_once(ctx, kinds=None) -> bool` (unchanged otherwise); `run_until_idle(ctx, kinds: Sequence[str] | None, max_seconds: float) -> int` returns the number of jobs processed and stops when no job is claimable or `max_seconds` has passed (it never starts a job after the time is up).
- `POST /internal/drain`, body `{"kinds": [...], "max_seconds": <float, 1..840>}` (pydantic, `extra="forbid"`; kinds must be known job kinds). Header `Authorization: Bearer <INTERNAL_DISPATCH_TOKEN>`, compared with `hmac.compare_digest`; 401 `Unauthorized("invalid dispatch token")` otherwise. The router is mounted only when `internal_dispatch_token` is set (404 otherwise). Response 200 `{"processed": n}`. Plain `def` route so FastAPI runs it in its threadpool.
- `run_forever` (the `tribunal worker` command) is unchanged, for single-host installs.

- [ ] **Step 1: Failing tests:** `test_claim_respects_kinds`; `test_drain_requires_token` (missing and wrong token give 401, no job runs; unset token gives 404); `test_drain_runs_only_requested_kinds`; `test_overlapping_drains_run_each_job_once` (two threads drain a queue of 6 stub jobs; each handler ran once, all jobs done); `test_drain_resumes_abandoned_job` (a job left `running` with an expired lease is processed by the next drain); `test_drain_respects_max_seconds`.
- [ ] **Step 2: Run them**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan. **Commit** `feat(backend): internal drain route for queue-woken workers`.

### Task 4: Queue wake-up after commit

**Files:**
- Create: `backend/src/tribunal/jobs/notify.py`
- Modify: `backend/src/tribunal/jobs/queue.py` (`enqueue`), `backend/src/tribunal/context.py`
- Test: `backend/tests/test_notify.py` (create)

**Interfaces:**
- `QUEUE_FOR_KIND = {"ingest": "ingestion", "generate_paper": "generation", "regenerate_slot": "generation"}`.
- `class CloudflareQueueNotifier(settings: Settings, client: httpx.Client | None = None)` with `notify(kind: str, job_id: str) -> None`: POST to the Cloudflare REST "send message to queue" endpoint for `cf_account_id` and the queue id of `QUEUE_FOR_KIND[kind]` (`cf_queue_ingestion_id` / `cf_queue_generation_id`), header `Authorization: Bearer <cf_api_token>`, JSON body carrying `{"job_id": ..., "kind": ...}` as the message body, timeout 5 seconds. Any error or non-2xx response is logged as a warning without the token and never raised. `class NullNotifier` does nothing. Confirm the endpoint path and body shape against Cloudflare's current Queues REST API documentation before coding, and cite the page in the report.
- `AppContext.notifier`: `CloudflareQueueNotifier` when `job_notify == "cloudflare_queue"`, else `NullNotifier`.
- `enqueue(session, kind, target_id, snapshot_id)` records `(kind, job_id)` in `session.info["tribunal_notify"]`; a session `after_commit` listener registered by `build_context` sends them through `ctx.notifier` and clears the list; `after_rollback` clears it without sending. Nothing is sent for a transaction that does not commit.

- [ ] **Step 1: Failing tests** (httpx `MockTransport`): `test_upload_notifies_ingestion_queue_after_commit` (exactly one request, to the ingestion queue URL, bearer header, body with job id and kind `ingest`); `test_paper_notifies_generation_queue`; `test_rolled_back_request_sends_nothing`; `test_notify_failure_keeps_request_and_job` (transport raises; upload still 202, job `queued`, the warning log has no token); `test_null_notifier_by_default`.
- [ ] **Step 2: Run them**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan. **Commit** `feat(backend): wake queue consumers after a job commits`.

### Task 5: Container images

**Files:**
- Create: `deploy/api.Dockerfile`, `deploy/ingest.Dockerfile`, `.dockerignore`
- Modify: `.github/workflows/backend.yml`, `backend/README.md`

**Interfaces:**
- `deploy/api.Dockerfile`: `python:3.12-slim`; installs `backend[openai,s3,postgres]` and a TTF font package; sets `PDF_FONT_REGULAR` and `PDF_FONT_BOLD` to the installed DejaVu Sans files, `HOST=0.0.0.0`, `PORT=8080`; runs as a non-root user; `CMD ["tribunal", "serve"]`. Serves the API and drains generation jobs.
- `deploy/ingest.Dockerfile`: same base plus Debian packages `ocrmypdf`, `tesseract-ocr` and one `tesseract-ocr-<lang>` package per entry of build argument `TESSERACT_LANGS` (default `"eng hin"`); installs `backend[s3,postgres]`; same user, port and command. Drains ingestion jobs.
- `.dockerignore` keeps the build context to `backend/` (excludes the legacy app, `frontend/`, `.venv`, `.superpowers`, `data`, `chroma_db`, caches and databases).
- No secret or environment-specific value is baked into either image.

- [ ] **Step 1: CI job `images`** (write it first; it fails until the Dockerfiles exist): `docker build --platform linux/amd64` both images; then `docker run --rm <api> tribunal --help`, `docker run --rm <ingest> ocrmypdf --version`, and `docker run --rm <ingest> tesseract --list-langs` whose output contains `eng` and `hin`.
- [ ] **Step 2: Write the Dockerfiles and `.dockerignore`**; build locally if Docker is available and record the image sizes in the report.
- [ ] **Step 3: README** section "Container images" (what each image runs, the build argument, that settings and secrets arrive as environment variables at start).
- [ ] **Step 4: Commit** `build: API and ingestion container images`.

### Task 6: Gateway and dispatch Worker

**Files:**
- Create: `deploy/worker/package.json`, `deploy/worker/package-lock.json`, `deploy/worker/tsconfig.json`, `deploy/worker/wrangler.jsonc`, `deploy/worker/src/index.ts`, `deploy/worker/src/containers.ts`, `deploy/worker/src/dispatch.ts`, `deploy/worker/test/dispatch.test.ts`
- Modify: `.github/workflows/backend.yml`

**Interfaces:**
- Follow Cloudflare's current Containers documentation (https://developers.cloudflare.com/containers/get-started/): `containers` entries with `class_name`, `scheduling_policy: "durable_object"` and `images.base.dockerfile`; Durable Object bindings; `exports` entries of type `durable-object` with `storage: "sqlite"`; the Durable Object starts its container with `ctx.container.start({ image, instance, env })`, waits for the port, sets an inactivity timeout and forwards requests through `getTcpPort(8080).fetch`. Re-read the page at implementation time and cite it in the report.
- Container classes `ApiContainer` (image `../api.Dockerfile`, instance `standard-1`) and `IngestContainer` (image `../ingest.Dockerfile`, instance `standard-2`), bindings `API_CONTAINER` and `INGEST_CONTAINER`. Each starts its container with an environment built by `containerEnv(env, names: readonly string[]) -> Record<string, string>` from an explicit allow-list of the `Settings` environment names (plus `RUNTIME_CONFIG`, `TRIBUNAL_*` model and OCR names, `OPENAI_API_KEY`); names not on the list are never passed.
- `dispatch.ts` (pure, unit-tested): `route(pathname: string) -> "api" | "internal" | "assets"`; `drainRequest(kinds: string[], token: string, maxSeconds: number) -> Request` for `POST http://container/internal/drain`; `kindsForQueue(queueName: string) -> string[]` (`tribunal-ingestion` gives `["ingest"]`, `tribunal-generation` gives `["generate_paper", "regenerate_slot"]`).
- `index.ts`: `fetch`: `internal` gives 404; `api` forwards to `API_CONTAINER.getByName("api-" + n)` with `n` random in `[0, API_POOL_SIZE)` (var, default 2); everything else is left to Static Assets. `queue(batch, env)`: one drain call per batch to the container for that queue (`INGEST_CONTAINER.getByName("ingest-0")` or an API instance), `max_seconds` 840 (confirm Cloudflare's current queue consumer wall-clock limit and keep `max_seconds` at least 60 seconds below it; cite the page); ack the batch on 2xx, `retryAll()` otherwise. `scheduled`: drain both kinds once.
- `wrangler.jsonc`: `main: "src/index.ts"`, pinned `compatibility_date`; `assets: {"directory": "../../frontend/dist", "not_found_handling": "single-page-application", "run_worker_first": ["/api/*", "/internal/*"]}`; queue consumers for `tribunal-ingestion` and `tribunal-generation` with `max_batch_size: 10`, `max_retries: 5` and dead-letter queues `tribunal-ingestion-dlq` and `tribunal-generation-dlq`; `triggers.crons: ["* * * * *"]`; an `env.staging` block with non-secret vars only (`APP_PROFILE: "staging"`, `AUTH_MODE: "oidc"`, `STORAGE_BACKEND: "s3"`, `JOB_NOTIFY: "cloudflare_queue"`, `API_POOL_SIZE`). No account id, route, hostname or secret in the file.

- [ ] **Step 1: Failing tests** (`vitest`, fake bindings, no network): `route` sends `/internal/drain` to `internal` and `/api/papers` to `api`; a public `/internal/drain` request returns 404 and never reaches a container stub; `queue` acks on a 200 drain and calls `retryAll` on a 500; the drain request carries the bearer token and the queue's kinds; `scheduled` drains both kind groups; `containerEnv` drops a name that is not on the allow-list.
- [ ] **Step 2: Run** `npm test` in `deploy/worker`; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Verify** `npx tsc --noEmit`, `npx vitest run`, and `npx wrangler deploy --dry-run --env staging` (bundles without deploying; record the output in the report, or why it could not run).
- [ ] **Step 5: CI job `worker`**: `npm ci`, `npx tsc --noEmit`, `npx vitest run` in `deploy/worker`. **Commit** `feat(deploy): gateway and dispatch Worker`.

### Task 7: Staging runbook

**Files:**
- Create: `docs/staging-runbook.md`
- Test: `backend/tests/test_runbook.py` (create)

**Interfaces:**
- The runbook covers, in order: what to provision (Workers Paid plan; managed PostgreSQL with pgvector and a TLS URL; R2 buckets for uploads and exports and an R2 S3 API token; queues `tribunal-ingestion`, `tribunal-generation` and their dead-letter queues; an OIDC application with its issuer, audience and keys URL; the model provider key); a table mapping every hosted setting to its environment name and whether it is a secret (`wrangler secret put <NAME> --env staging`) or a var; running `tribunal db-upgrade` against the staging database from a trusted machine; `npm run build` in `frontend/` then `npx wrangler deploy --env staging`; attaching the staging hostname; the roadmap's rollout checks (sign in from another device, upload a scanned chapter, generate, reconnect during generation, export the approved result, HTTPS with no loopback requests, another course cannot open the source or answer key, a restarted container resumes the job, paper-speed benchmark with cold containers and a concurrent OCR upload); rollback to the previous Worker version and images.

- [ ] **Step 1: Failing test** `test_runbook_names_every_hosted_setting`: every environment name of a `Settings` field that `load_settings` requires or validates for the staging profile, plus `OPENAI_API_KEY` and the `TRIBUNAL_*` names used by `config/runtime.example.yaml`, appears in `docs/staging-runbook.md`.
- [ ] **Step 2: Write the runbook.**
- [ ] **Step 3: Run** the test, the full suite and the text scan (`docs` included). **Commit** `docs: staging runbook`.
