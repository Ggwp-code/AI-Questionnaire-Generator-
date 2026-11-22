# Question Pipeline Design

Date: 2026-11-22. Status: approved by the instructor (2026-11-22), with MCQ parts added at their request.

## Intent

Every question that reaches a student must be answerable as written, supported by the course's own books, and paired with an answer key that is actually right. Today one writer call produces the draft, and checks are attached around it: a narrow arithmetic check for some types, then a reviewer. The instructor wants a pipeline of distinct stages, each with one job, where a failure goes back to the stage that caused it, and only the exact revision that passed every stage is released.

Computational questions in scope are anything in computer science that can be examined as code: mathematics, AI and ML calculations, algorithm traces, evaluating a regular expression, and similar. Their answers must be checked by a program that solves the question independently.

## The stages (instructor's specification, verbatim)

| Stage | Responsibility |
|---|---|
| Reader | Retrieve the relevant passages and page references. Return an evidence gap if the sources cannot support the requested question. |
| Question writer | Use the evidence and the instructor's exact type, marks, parts, and difficulty to draft the student-facing question, proposed answer, rubric, and citations. |
| Code gen | For numerical, trace, and computational MCQs, produce a verifier from the student-facing question. Keep the proposed answer out of its input so it must solve the problem independently. For conceptual questions, select structural checks and pass the evidence to review. |
| Verification | Run the verifier in isolation and compare its result with the proposed answer. Check marks, parts, and citations for every type. For MCQs, also check the option count and that the key identifies exactly one option. |
| Review | Judge whether the question is answerable as written, the evidence supports the answer, and the rubric fits the marks. Send a flawed question back to the writer and a faulty verifier back to code gen, with specific feedback and a retry limit. |
| Output | Release the exact revision that passed. |

## Goals

- The six stages above, as separate units with one routing table and per-loop retry limits.
- Computational types (`numerical`, `trace`, `computational_mcq`) are verified by a generated Python program that sees only the student-facing question (stem, parts, options), never the proposed answer, and runs in an isolated verifier service.
- Conceptual types (`short_answer`, `long_answer`, `mcq`) get structural checks chosen by code gen from a fixed library, and their evidence goes to review.
- Verification checks marks, parts and citations for every type; for MCQs also the option count and exactly one correct option.
- Review always runs, including when a verifier could not run, so no question skips the evidence check.
- Every stage's outcome is recorded, so an instructor can see why a question was sent back and by which stage.
- Questions can have parts, with marks per part.

## Non-goals

- New question types beyond the six already configured.
- Verifiers in languages other than Python (questions may show code in any language; the verifier is Python).
- LangGraph or another orchestration framework.
- Frontend work (the faculty-frontend plan gets a follow-up amendment for parts and stage history).
- Reusing a verifier across slots or papers.

## Anti-goals (a failure even if it works)

- Verifier code running anywhere that can reach the database, secrets, other services or the internet.
- The proposed answer reaching code gen, directly or through the question text.
- A released revision that differs from the one that was verified and reviewed.
- A failing, unparseable or unavailable stage ending as a passing question. Those end as `needs_review` (a person decides) or `failed`, never `drafted`.

## Constraints (from the existing system)

- Generation runs inside a job with leases, attempt fencing (`owned_transaction`) and replay protection (`revisions.job_id`); every write stays fenced.
- Token budgets reserve an upper bound per model call; the paper deadline still applies to the whole pipeline.
- Roles, prompts, models, retry limits and tolerances come from the runtime config; no model or course names in code.
- Model output passes through the text policy before validation; exports use it too.
- Revisions are immutable once written; approval and export use the slot's current revision.
- Builds on commit 252ecc1 on `feat/tribunal-backend` (codegen role, `verification.py`, more question types, MCQ option checks). Its restricted arithmetic evaluator is replaced by verifier programs.

## Approaches considered

Where verifier code runs:
- **Chosen: a separate verifier service.** A small container image with no internet, no secrets and no database access; strict CPU, memory, time and output limits per run. The generation worker sends it the program and gets the result back. On Cloudflare it is another container with internet switched off; locally it is the same image in Docker. Chosen because it is the only option where a malicious or buggy verifier cannot reach the database password or API keys that the worker containers hold.
- A sandboxed subprocess inside the worker container: simpler, but the code runs next to the secrets and depends on sandboxing tools the container may not allow. Kept only as a development and test mode, refused on hosted profiles.
- Python compiled to WebAssembly in-process: strong isolation, but a heavy runtime with uneven library support and the most setup.

How stages are organised:
- **Chosen: explicit stages in our own code.** One function per stage, a routing table, per-loop retry limits, and a record per stage outcome. Fits the job leases and replay protection already in place.
- LangGraph: already a declared dependency, but its checkpoints would duplicate what the job database guarantees.

The instructor chose both recommendations.

## Design decisions (mine unless marked)

**Stage flow.** Reader -> Writer -> structural check of the draft (parse, marks, parts, citations, MCQ rules; failures go straight back to the writer) -> Code gen -> Verification -> Review -> Output. Code gen and verification apply to every type: computational types get a verifier program; conceptual types get a list of structural checks chosen from the library and executed in-process (they are our code, not generated code).

**Routing.** Review returns one of `pass`, `revise_question` (with feedback, back to the writer), `fix_verifier` (with feedback, back to code gen), or `faculty_review`. A verification mismatch does not decide who is wrong: review sees the question, the proposed answer, the verifier program and its result, and routes. Separate retry limits: `max_writer_rounds` and `max_codegen_rounds` (config). When a limit is reached, the existing reasoner escalation decides between `pass` and `needs_review` (only if `escalate_disputed_reviews`); otherwise `needs_review`. A verifier that cannot run (service down, timeout, crash) after the code gen retries goes to review with that outcome recorded, and the slot can at best end `needs_review`, never `drafted`.

**Verifier program contract.** Plain Python 3.12, standard library plus `numpy` and `sympy` (my call: covers ML calculations and exact math; `re` covers regular expressions). It must print exactly one JSON object as its last line: `{"answer": <value>}`, or for parts `{"parts": [<value>, ...]}`, or for computational MCQs `{"option_index": <n>, "answer": <value>}`, or `{"unsupported": true}` when the question cannot be solved by a program. Comparison: numbers with the configured relative tolerance; strings (trace output, regex results) after whitespace normalisation; MCQs require the computed value to match exactly one option and that option to be the key.

**Verifier service.** A new image `deploy/verifier.Dockerfile` with only Python, numpy and sympy. One HTTP route, `POST /run`, body `{program, timeout_seconds}`, returns `{status: "ok" | "error" | "timeout" | "too_large", last_line, stderr_tail}`. Each run is a fresh subprocess as an unprivileged user in an empty temporary directory, empty environment, isolated mode, with limits on CPU seconds, address space, process count, file size and output bytes (values from config). The container has no internet and holds no secrets. On Cloudflare the generation worker reaches it through the gateway Worker on an internal route that requires `INTERNAL_DISPATCH_TOKEN` and is never public; locally through a configured `VERIFIER_URL`. `VERIFIER_MODE=subprocess` runs the same runner locally for development and tests and is refused on hosted profiles.

**Stage records.** A new table of pipeline steps: slot, revision (when one exists), job, stage, attempt, outcome, short feedback, model, prompt version, tokens, timestamps. The verifier program and its result are stored on the revision they verified. A read-only API route lists a slot's steps for instructors. Stage records hold outcomes and feedback, not prompts.

**Parts.** A template section may list part marks (for example `parts: [2, 3]`, summing to `marks_each`). A draft may carry `parts`, each with its own question text, answer, rubric and citations, under a shared stem. Structural checks enforce the part count and per-part rubric totals. Export renders parts. Approval and edits work per question, as today.

**Output and release.** Output sets the slot's current revision to the revision that passed and records that its verification and review refer to that same revision id. Drafts that fail are kept as revisions for history; a slot ending `needs_review` or `failed` points at its latest revision so an instructor can see and fix it.

**Faculty edits.** An edited computational question is not re-verified automatically (as in commit 252ecc1, it goes to `needs_review`); regenerating runs the full pipeline.

## Settled after the builder check (my calls)

- **Phasing.** The gateway Worker, container images and internal routes are built by the staging-readiness plan (`docs/superpowers/plans/2026-11-22-staging-readiness.md`, branch `feat/staging-readiness`, which holds `deploy/`). This work is planned local-first behind a `VerifierClient` interface (HTTP client for `VERIFIER_URL`, subprocess runner for development and tests), with the hosted wiring (verifier image as a Cloudflare container, internal route in the gateway Worker) as a separate final phase on top of staging-readiness.
- **`unsupported` and runs that fail.** For a computational type, `{"unsupported": true}`, an error, a timeout or an oversized result is a verification outcome, recorded and passed to review like a mismatch. Review routes it: `fix_verifier` when the program is at fault, `revise_question` when the question is not solvable as written, `faculty_review` otherwise. When the code gen limit is reached without a verifier that runs and agrees, the slot ends `needs_review` with the reason recorded; it never ends `drafted`.
- **Value types.** `numerical` answers stay plain numbers. `trace` answers and `computational_mcq` options may be numbers or strings (regular expression results, printed output). Comparison: if both the verifier's value and the proposed value parse as plain numbers, compare with the relative tolerance; otherwise compare strings after whitespace normalisation. A computational MCQ passes only when exactly one option matches the verifier's value and that option is the key.
- **Parts (instructor: include MCQs).** Parts are allowed for every type, MCQs included. An MCQ part (`mcq` or `computational_mcq`) carries its own options and correct option, and the MCQ checks (option count, exactly one key, answer equals the keyed option text) apply per part. A part's answer is a string compared with the same rule as above. For parts, the verifier prints `{"parts": [...]}` with one entry per part in order: a value, or `{"option_index": <n>, "answer": <value>}` for a computational MCQ part. Each computational MCQ part passes only when exactly one of its options matches and that option is its key.
- **Revisions.** As today, every writer draft that parses becomes a revision (history, replay protection via `job_id`). Revision content is immutable; the verifier program, its result, the structural check results and the review are attached to that revision's review record afterwards. Release checks that the verified and reviewed revision id equals the released one.
- **Limits and escalation.** `max_writer_rounds` bounds `revise_question` and structural-check loops; `max_codegen_rounds` bounds `fix_verifier` loops. Reasoner escalation applies only when the writer limit is reached on a `revise_question` dispute, as today; the reasoner sees the verifier program and result. Reaching the code gen limit goes to `needs_review` without escalation.
- **Budget and time.** The paper token budget and the paper deadline stay the only caps; verifier run time counts against the deadline. The job heartbeat already covers the whole handler, so long verifier calls keep the lease.
- **Testing the isolation.** CI builds the verifier image and runs an integration test against it with networking disabled: a program that tries the network, reads environment variables or writes outside its directory must fail or see nothing, and runaway CPU, memory or output must be stopped. Unit tests use the subprocess runner.
- **Structural check results** are stored with the revision and in the stage records, and review reads them.

## Left to the builder

- Exact prompt wording for writer, code gen and reviewer.
- The structural check library's internal shape.
- Default resource limits and retry limits in the example config.
- How the gateway Worker exposes the verifier route on Cloudflare, following current Containers documentation.
- Verifier image library versions.
- Table and column names, migration numbering, and the steps route's JSON shape.
