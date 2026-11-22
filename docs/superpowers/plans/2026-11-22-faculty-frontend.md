# Faculty Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy screens with a faculty flow on the new `tribunal` API: choose or create a subject, upload books and watch them become searchable, review topics, build a paper from a template, review, edit, regenerate and approve each question against its cited pages, and export the paper and answer key.

**Architecture:** The React and Vite app in `frontend/` gets a new shell and pages under `frontend/components/faculty/`, a typed client in `frontend/services/tribunalApi.ts`, and hash-based navigation (no router dependency). All requests go to the same origin under `/api` (the Vite dev proxy forwards to the backend; in staging the gateway Worker does). In staging, sign-in happens at the edge through Cloudflare Access, which passes a signed JWT header that the backend verifies, so the browser holds no token; locally, a dev-only email field sends `X-Dev-User`. Legacy modules stay on disk but leave the navigation; deleting them is a later cleanup.

**Tech Stack:** React 18, TypeScript, Vite, Tailwind (as already loaded), Vitest with jsdom and Testing Library; FastAPI for the three small backend additions.

**Spec:** `docs/department-pilot-roadmap.md` (the faculty flow: "A faculty member creates or joins a subject, uploads books, and sees a proposed chapter/topic outline"; "Let instructors open the cited passage, edit, reject, regenerate a single slot or approve it"; "Return real errors throughout the UI").

**Design direction (no Mobbin):** keep the app's existing look (Tailwind as loaded in `index.html`, its accent colours and fonts); a top bar with the signed-in user, a subject header with tabs, tables for documents, cards for questions, and a side panel for passages; plain components, no new UI library.

**Branch:** `feat/faculty-frontend` from `feat/question-pipeline`, which already has question parts. Task 8 waits for the pipeline's stage history route.

**Not in this plan:** progress streaming (pages poll), topic merge, syllabus upload, previous-year papers, deleting the legacy modules, replacing the Tailwind CDN script.

## Global Constraints

- "Use plain text without emojis in all project-authored material, including interface labels, status messages ... tests and documentation." The current header's emoji does not carry over.
- "Return real errors throughout the UI." No mock-data fallback anywhere in the new code; every failed request shows the server's message and each of its `details`.
- Course content (subjects, topics, question types, Bloom and difficulty levels, OCR languages) comes from the API, never from frontend constants. No course, subject, topic or model names in `frontend/components/faculty/` or `frontend/services/tribunal*`.
- Requests use the same origin, path prefix `/api`. No API base URL constant pointing at a host.
- The dev sign-in exists only when `import.meta.env.VITE_AUTH_MODE === "dev"`; any other build never renders it and never sends `X-Dev-User`.
- No new runtime dependencies. Dev dependencies may add only `vitest`, `jsdom`, `@testing-library/react`, `@testing-library/user-event`, pinned to exact versions.
- Every input has a visible label; every action is a button with text.
- Legacy modules (`components/*Module*.tsx`, `services/api.ts`, `services/questionBank.ts`) are not edited and not imported by the new shell.
- Backend constraints from the first plan still bind the backend tasks (Python 3.12, declared dependencies, no Claude or AI attribution in commits).

## Review Focus

1. A request fails with 409 or 422 (export blocked, invalid edit, stale approval): the page shows the message and every detail line and stays usable. Tests in Task 2 and Task 7.
2. Another instructor changed a question between opening it and clicking approve: the 409 is explained, the paper reloads, and the new revision is shown unapproved. Test in Task 7.
3. A page left open during OCR or generation: polling runs every 3 seconds only while something is queued, processing, pending or running, and stops on completion and on unmount. Tests in Task 4 and Task 7.
4. A non-dev build never shows the dev sign-in and never sends `X-Dev-User`. Tests in Task 2 and Task 3.
5. A book with excluded pages, a book with no usable passages, and an evidence-gap slot are shown plainly as states, not as page errors. Tests in Task 4 and Task 7.

---

### Task 1: Backend support for the faculty UI

**Files:**
- Modify: `backend/src/tribunal/api/auth.py`, `backend/src/tribunal/api/routers/papers.py`, `backend/src/tribunal/api/routers/subjects.py`, `backend/README.md`
- Test: `backend/tests/test_api.py`

**Interfaces:**
- Edge sign-in: in `oidc` mode `authenticate` reads the header named by `settings.auth_jwt_header`. When it is `Authorization`, the value must be `Bearer <jwt>` (today's behavior); for any other header name (for example `Cf-Access-Jwt-Assertion`), the whole value is the JWT. Verification is unchanged (issuer, audience, keys URL, required `exp`, `iss`, `aud`, and the `email_verified` rule).
- `GET /api/subjects/{sid}/papers` (member only, else 404): `[{"id", "title", "status", "created_at", "template": {"name", "version"}}]`, newest first.
- `GET /api/options`: from the API's current snapshot, `{"question_types": [{"key", "label"}], "bloom_levels": [{"key", "label"}], "difficulty_levels": [{"key", "label"}], "default_ocr_languages": [...]}`; only enabled question types.
- Docs: in `backend/README.md` (the staging runbook gets the same subsection in the staging-readiness plan's Task 7), a "Cloudflare Access" subsection: `AUTH_JWT_HEADER=Cf-Access-Jwt-Assertion`, `OIDC_ISSUER` the team domain, `OIDC_AUDIENCE` the application AUD tag, `OIDC_JWKS_URL` the team's `/cdn-cgi/access/certs` URL, and the right `OIDC_REQUIRE_EMAIL_VERIFIED` value. Confirm the Access token claims (whether `email_verified` is present) against Cloudflare's current Access JWT documentation and cite the page in the report.

- [ ] **Step 1: Failing tests:** `test_oidc_reads_configured_header` (a valid token in `Cf-Access-Jwt-Assertion` signs in when that header is configured; the same token in `Authorization` is refused then); `test_subject_papers_listed_newest_first_and_scoped` (B gets 404 for A's subject); `test_options_lists_configured_values` (matches the fixture config; a disabled question type is absent).
- [ ] **Step 2: Run them**; confirm failures.
- [ ] **Step 3: Implement**, then update the two documents.
- [ ] **Step 4: Run** focused tests, full suite, ruff, scan. **Commit** `feat(backend): edge sign-in header, subject paper list and UI options`.

### Task 2: Frontend test tooling and typed API client

**Files:**
- Modify: `frontend/package.json`, `frontend/package-lock.json`, `frontend/vite.config.ts` (test config), `frontend/vite-env.d.ts` (`VITE_AUTH_MODE`), `.github/workflows/backend.yml`
- Create: `frontend/services/tribunalTypes.ts`, `frontend/services/tribunalApi.ts`, `frontend/services/tribunalApi.test.ts`

**Interfaces:**
- Scripts: `"test": "vitest run"`, `"typecheck": "tsc --noEmit"`; the Vitest environment is `jsdom`.
- `tribunalTypes.ts`: types that mirror the API JSON exactly: `User`, `Subject`, `DocumentInfo` (`status: "queued" | "processing" | "ready" | "failed"`, `error`, `page_count`, `created_at`, `ready_at`, `searchable_seconds`, `excluded_pages: number[]`), `Topic`, `Passage`, `Options`, `TemplateSection`, `TemplateSpec`, `Template`, `PaperSummary`, `Paper`, `Slot` (status values from the API: `pending`, `running`, `drafted`, `needs_review`, `evidence_gap`, `failed`, `approved`), `Revision`, `Citation`, `ExportDoc`.
- `tribunalApi.ts`: `class ApiError extends Error { status: number; code: string; details: string[] }`. A non-2xx response with the envelope `{"error": {"code", "message", "details"}}` becomes an `ApiError` carrying those fields; a network failure becomes `ApiError` with status 0, code `network_error`, message `The server could not be reached`. In dev mode every request sends `X-Dev-User` from `getDevUser()` (localStorage key `tribunal.devUser`, read inside try/catch); `setDevUser(email)` stores it. One function per route: `me`, `options`, `listSubjects`, `createSubject`, `addMember`, `listDocuments`, `uploadDocument(sid, file, languages?)` (multipart: `file`, optional `languages` comma-joined), `reprocessDocument`, `fetchDocumentPdf(did) -> Promise<Blob>`, `listTopics`, `createTopic`, `renameTopic`, `getPassage`, `listTemplates`, `createTemplate`, `listPapers(sid)`, `createPaper`, `getPaper`, `regenerateSlot`, `editSlot(pid, n, content)`, `approveSlot(pid, n, revisionId)` (body `{"revision_id"}`), `exportJson(pid, part)`, `exportPdf(pid, part) -> Promise<Blob>`.
- CI job `frontend`: `npm ci`, `npm run typecheck`, `npm test`, `npm run build` in `frontend/`; add `frontend/components/faculty` and `frontend/services` to the CI text scan paths.

- [ ] **Step 1: Failing tests** (mocked `fetch`): `approveSlot` sends `POST /api/papers/p1/slots/2/approve` with body `{"revision_id":"r9"}`; a 409 envelope becomes `ApiError` with status, code, message and details; a rejected fetch becomes the `network_error` ApiError; with `VITE_AUTH_MODE` = `dev` and a stored user, requests carry `X-Dev-User`, and without dev mode they never do; `uploadDocument` sends multipart with `file` and `languages`.
- [ ] **Step 2: Run** `npm test`; confirm failures.
- [ ] **Step 3: Implement.** If `npm run typecheck` reports errors in legacy files, record them in the report and leave those files alone; the job must then type-check only the new files (a separate `tsconfig` for the faculty app is acceptable).
- [ ] **Step 4: Run** tests, typecheck, build. **Commit** `feat(frontend): test tooling and typed client for the tribunal API`.

### Task 3: Shell, navigation, dev sign-in and subjects

**Files:**
- Modify: `frontend/App.tsx`
- Create: `frontend/components/faculty/route.ts`, `frontend/components/faculty/ErrorNotice.tsx`, `frontend/components/faculty/DevSignIn.tsx`, `frontend/components/faculty/SubjectsPage.tsx`, `frontend/components/faculty/SubjectPage.tsx`, tests beside them (`*.test.ts(x)`)

**Interfaces:**
- `route.ts`: `type Route = {view: "subjects"} | {view: "subject", sid: string, tab: "library" | "topics" | "papers"} | {view: "paper", pid: string}`; `parseHash(hash: string) -> Route` (unknown hashes give `subjects`); `formatHash(route) -> string` (`#/subjects`, `#/subjects/<sid>/<tab>`, `#/papers/<pid>`); `useRoute() -> [Route, (r: Route) => void]` backed by `hashchange`.
- `ErrorNotice({error: ApiError | null})`: shows the message and each detail as a list; renders nothing for null.
- `App.tsx`: header with the text "Tribunal" and the signed-in email from `me()`; in dev mode with no stored user, `DevSignIn` (labelled email field and "Sign in" button calling `setDevUser`) replaces the content; otherwise the current route's page. No legacy module is imported.
- `SubjectsPage`: lists `listSubjects()`; a labelled "Subject name" field and "Create subject" button; each subject links to `#/subjects/<sid>/library`.
- `SubjectPage`: tabs Library, Topics, Papers (rendering Task 4, 5 and 6 pages; until they exist, a placeholder text is fine) and an "Add colleague" form (labelled email field) calling `addMember`; a refusal shows through `ErrorNotice`.

- [ ] **Step 1: Failing tests:** `parseHash(formatHash(r))` round-trips every route shape and an unknown hash gives `subjects`; `SubjectsPage` lists subjects from a mocked client and creating one adds it; a failing `listSubjects` shows the error message and details (no subject rows, no mock data); `App` in a non-dev build never renders `DevSignIn`.
- [ ] **Step 2: Run**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests, typecheck, build. **Commit** `feat(frontend): faculty shell, navigation and subjects`.

### Task 4: Library page

**Files:**
- Create: `frontend/components/faculty/LibraryPage.tsx`, `frontend/components/faculty/usePolling.ts`, `frontend/components/faculty/openPdf.ts`, tests beside them

**Interfaces:**
- `usePolling(fn: () => Promise<void>, active: boolean, intervalMs = 3000)`: calls `fn` every `intervalMs` only while `active`, never overlapping calls, and stops on unmount.
- `openPdf(blob: Blob, page?: number)`: opens an object URL with `#page=<n>` in a new tab and revokes it later (dev mode needs this, because a plain link cannot send `X-Dev-User`).
- `LibraryPage({sid})`: upload form (labelled PDF file input accepting `application/pdf`; labelled "OCR languages" field, placeholder from `options().default_ocr_languages`, sent only when filled); a table of documents with filename, status, page count, "Searchable after <n> s" when set, the error text, and "Pages kept out of evidence: 3, 7" when `excluded_pages` is non-empty; per document "Open" (via `fetchDocumentPdf` and `openPdf`) and "Reprocess" (disabled while queued or processing). Polls while any document is queued or processing.

- [ ] **Step 1: Failing tests:** statuses, errors and excluded pages render as described; "Reprocess" calls `reprocessDocument` and is disabled for a processing document; with fake timers, polling refreshes while a document is processing and stops after it is ready; `usePolling` stops on unmount.
- [ ] **Step 2: Run**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests, typecheck, build. **Commit** `feat(frontend): library page with upload, status and reprocess`.

### Task 5: Topics page

**Files:**
- Create: `frontend/components/faculty/TopicsPage.tsx`, test beside it

**Interfaces:**
- `TopicsPage({sid})`: lists `listTopics(sid)` with title, a source label ("From bookmarks", "From headings", "Added by faculty" for `bookmark`, `heading`, `faculty`), and "Pages <start>-<end>" with an "Open page" button (document PDF at `page_start`) when the topic has a document and pages; inline rename (labelled field, "Save" and "Cancel") calling `renameTopic`; a labelled "New topic" field and "Add topic" button calling `createTopic`.

- [ ] **Step 1: Failing tests:** source labels and page ranges render; rename calls `renameTopic` and shows the new title; adding a topic calls `createTopic`; a 422 shows its details.
- [ ] **Step 2: Run**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests, typecheck, build. **Commit** `feat(frontend): topics page`.

### Task 6: Templates, paper builder and papers list

**Files:**
- Create: `frontend/components/faculty/PapersPage.tsx`, `frontend/components/faculty/TemplateForm.tsx`, `frontend/components/faculty/PaperBuilder.tsx`, tests beside them

**Interfaces:**
- `PapersPage({sid})`: lists `listPapers(sid)` (title, status, created time, template name and version), each linking to `#/papers/<pid>`, and a "New paper" button opening `PaperBuilder`.
- `TemplateForm({onCreated})`: a template name and one row per section: title, question type, Bloom level and difficulty as selects filled from `options()`, count, marks each, optional "Answer any", and an optional "Split into parts" control that lists part marks (integers that must sum to marks each, sent as `parts`). It shows the computed total (`marks_each * (answer_any ?? count)` summed) and sends `total_marks` equal to it; `createTemplate` errors show their details.
- `PaperBuilder({sid, onCreated})`: choose a template from `listTemplates()` (or create one with `TemplateForm`), choose topics (checkboxes from `listTopics`), choose books (checkboxes of ready documents only, all checked by default), a labelled title; "Generate paper" calls `createPaper` and navigates to the new paper.

- [ ] **Step 1: Failing tests:** the select options come from a mocked `options()` and none are hard-coded; the total updates as rows change and the request carries it; only ready documents are offered; a 422 from `createPaper` shows each detail; success navigates to `#/papers/<pid>`.
- [ ] **Step 2: Run**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests, typecheck, build. **Commit** `feat(frontend): templates, paper builder and paper list`.

### Task 7: Paper review, approval and export

**Files:**
- Create: `frontend/components/faculty/PaperPage.tsx`, `frontend/components/faculty/SlotCard.tsx`, `frontend/components/faculty/SlotEditor.tsx`, `frontend/components/faculty/PassageViewer.tsx`, tests beside them

**Interfaces:**
- `PaperPage({pid})`: header with title, status and timings (`first_question_seconds`, `total_seconds` when set); one `SlotCard` per slot; polls while any slot is pending or running. Export panel: "Paper (PDF)", "Answer key (PDF)" (via `exportPdf` and `openPdf`) and "Answer key (JSON)" (downloads the JSON as a file); a 409 `export_blocked` lists each problem.
- `SlotCard({paper, slot, onChanged})`: status label; `status_detail`; for drafted, needs_review and approved slots, the question, options, answer and rubric; the reviewer verdict when a dispute was escalated (`review.disputed`); each citation with page number and a "View passage" button (`PassageViewer`, which shows the passage text and an "Open page" button); actions "Edit" (`SlotEditor`), "Regenerate" (disabled while pending or running) and "Approve" (only for drafted or needs_review), which sends the shown revision's id. On a 409 from approve, show "This question changed since you opened it. The current version is shown below.", then reload the paper. An evidence-gap slot shows its detail and the hint "Add a book that covers this topic, or choose another topic." and offers only "Regenerate".
- `SlotEditor({paper, slot, onSaved})`: editable question, answer, rubric items, and the citations kept from the current revision; for `mcq` and `computational_mcq` slots also the options (add, remove, edit) and the correct option (a radio per option, sent as `correct_option`); "Save" calls `editSlot` with only the draft fields (never the stored `verification` record); a 422 shows each detail and keeps the edits. After a faculty edit of a computational slot (`numerical`, `trace`, `computational_mcq`), the server sets `needs_review` with a detail saying the revised answer needs an independent check; show it as returned.
- Parts: when the revision content has `parts`, `SlotCard` shows the stem, then each part with its label, question, options, answer, rubric and citations; `SlotEditor` edits every part (question, answer, rubric, options and key for MCQ parts) and keeps their order and labels; a 422 detail naming a part is shown beside that part.
- `SlotCard` also shows the arithmetic check for computational slots when the revision content has `verification`: its status (`pass`, `mismatch`, `unsupported`, `unavailable`, `invalid`), its detail, and the solver's expression when present. Question types and their labels still come from `options()`.

- [ ] **Step 1: Failing tests:** an evidence-gap slot shows its detail, the hint and only "Regenerate"; an MCQ edit sends the edited options and `correct_option` and no `verification` key; a computational slot shows its verification status and expression; approve sends the shown revision id; a 409 on approve shows the changed message and triggers a reload; a blocked export lists each problem; polling stops when every slot has an outcome; a disputed review shows the reviewer's verdict; an edit rejected with 422 keeps the typed text.
- [ ] **Step 2: Run**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests, typecheck, build; run the app against a local backend (`tribunal serve` with `AUTH_MODE=dev` plus `npm run dev` with `VITE_AUTH_MODE=dev`) and walk the flow once: create a subject, upload a PDF, wait for ready, build a 3-question paper, approve, export. Record what you saw in the report.
- [ ] **Step 5: README** for `frontend/`: dev setup (backend command, `VITE_AUTH_MODE=dev`, the proxy), `npm test`, `npm run build` output used by the gateway Worker, and that legacy screens are no longer in the navigation. **Commit** `feat(frontend): paper review, approval and export`.

### Task 8: Stage history panel

Runs after the question-pipeline plan's Task 7 (the `GET /api/papers/{pid}/slots/{n}/steps` route) is merged into this branch.

**Files:**
- Create: `frontend/components/faculty/StageHistory.tsx`, test beside it
- Modify: `frontend/services/tribunalApi.ts`, `frontend/services/tribunalTypes.ts`, `frontend/components/faculty/SlotCard.tsx`

**Interfaces:**
- `listSlotSteps(pid, n) -> Promise<Step[]>` with `Step = {stage, attempt, outcome, feedback, revision_id, model, created_at}`.
- `StageHistory({pid, n})`: a collapsed "How this question was made" section on each `SlotCard`; expanded, one row per step in order: stage name (Reader, Writer, Structure check, Code gen, Verification, Review, Reasoner, Output), attempt, outcome, and feedback when present; a step sending work back (`revise_question`, `fix_verifier`, a structure failure, a mismatch) is marked as a return to the stage it names.

- [ ] **Step 1: Failing tests:** rows render in order with stage names; a returned step is marked; the section loads only when expanded; an error shows through `ErrorNotice`.
- [ ] **Step 2: Run**; confirm failures.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** tests, typecheck, build. **Commit** `feat(frontend): stage history for each question`.

