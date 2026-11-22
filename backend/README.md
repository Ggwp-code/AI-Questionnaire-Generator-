Tribunal backend
================

Tribunal is a multi-subject exam paper generator that writes source-backed questions and routes
them through faculty approval. This package is the backend service and its command line.

Install
-------

    pip install -e "backend[dev,openai,s3]"

The test suite needs the dev, openai and s3 extras (the tests import openai and botocore).
Other optional extras: ocr (ocrmypdf), postgres (psycopg and pgvector), embeddings-local
(sentence-transformers). Python 3.12 or newer.

Environment variables
---------------------

Deployment settings come from the process environment, or from an env file named by
TRIBUNAL_ENV_FILE. Missing or inconsistent values are reported together, by name.

Required on every profile:
  APP_PROFILE          local, test, staging or production
  DATABASE_URL         for example sqlite:///tribunal.db
  RUNTIME_CONFIG       path to the runtime configuration file
  AUTH_MODE            dev or oidc
  STORAGE_BACKEND      local or s3
  SIGNING_SECRET       at least 32 characters

Also required, by profile:
  STORAGE_LOCAL_ROOT   when STORAGE_BACKEND is local
  S3_BUCKET_UPLOADS, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY   when STORAGE_BACKEND is s3; this
                       also needs the s3 extra (pip install -e "backend[s3]")
  S3_ENDPOINT_URL      set for Cloudflare R2 or another S3-compatible service; omit for AWS
  S3_REGION            auto for R2
  S3_BUCKET_EXPORTS    validated but not used yet, because exports are rendered on request
  OIDC_ISSUER, OIDC_AUDIENCE, OIDC_JWKS_URL   when AUTH_MODE is oidc
  OIDC_REQUIRE_EMAIL_VERIFIED   true by default: a token whose email_verified claim is not true is
                       refused; set false only for a provider that does not send the claim
  Hosted profiles (staging, production): AUTH_MODE oidc, ALLOWED_ORIGINS, an https PUBLIC_BASE_URL,
  a PostgreSQL DATABASE_URL, PDF_FONT_REGULAR and STORAGE_BACKEND s3.
  Staging and production also need Postgres with pgvector, an OIDC provider and the s3 extra.

A paper keeps the configuration it was created with: regenerating, editing and exporting it use that
configuration, and only new papers use the current one.

AUTH_MODE dev trusts an X-Dev-User header and is for local use and tests only.

Runtime configuration
---------------------

RUNTIME_CONFIG points to a YAML file holding models, prompts, OCR, retrieval, limits and text
policy. config/runtime.example.yaml is the template. Its values use TRIBUNAL_* variables:
  TRIBUNAL_WRITER_MODEL, TRIBUNAL_REVIEWER_MODEL, TRIBUNAL_REASONER_MODEL   required
  TRIBUNAL_OCR_LANGUAGE   default eng
  TRIBUNAL_WORKER_THREADS default 2
The openai provider also needs OPENAI_API_KEY. Course content and model names never go in code.

With flagged_page_policy: exclude, a page whose text from a source listed in flag_formulas_on or
flag_tables_on contains a formula or a table is stored but kept out of evidence, because a misread
minus sign or exponent could support a wrong answer. The document stays ready and the documents
list reports the excluded page numbers in its excluded_pages field. Changing these settings, or the outline
settings, re-extracts the book on its next ingest; outline topics that the new outline no longer has are hidden and cannot be
picked for new papers. Renaming an outline topic makes it a faculty topic, which rebuilds keep.

Commands
--------

    tribunal check-config   validate settings and runtime configuration (exit 2 names each problem)
    tribunal db-upgrade     apply database migrations
    tribunal serve          run the HTTP API (HOST defaults to 127.0.0.1 on local/test, 0.0.0.0 on hosted profiles)
    tribunal worker         run the background job worker
    tribunal scan-text --config backend/config/runtime.example.yaml PATH...
                            report emoji in text files (exit 1 on findings)

Tests
-----

    pip install -e "backend[dev,openai,s3]"
    cd backend && pytest
    ruff check backend

tests/test_acceptance.py runs the whole first ticket through the HTTP API with an offline stub
provider and a fake OCR engine.

Timers
------

Paper clocks start when the server accepts the paper request (Paper.created_at), so time spent
waiting in the queue counts.
  first_question_seconds  ends when the first slot is drafted or needs review, after review
  total_seconds           ends when every slot has a visible outcome: drafted, needs_review,
                          evidence_gap or failed
Upload to searchable is measured separately: the documents list reports ready_at and
searchable_seconds (ready_at - created_at). ready_at moves forward when a book is re-indexed.

Pilot acceptance
----------------

CI uses simulated OCR and the stub provider. For acceptance, run on a machine with OCRmyPDF and
Tesseract and the deployed model:

    TRIBUNAL_LIVE_SCAN_PDF=/path/to/scanned-chapter.pdf \
    TRIBUNAL_LIVE_RUNTIME_CONFIG=/path/to/runtime.yaml \
    TRIBUNAL_LIVE_REVIEW_SHEET=/path/to/review-sheet.md \
    OPENAI_API_KEY=... pytest tests/test_live_acceptance.py -s

The test skips and names whatever is missing. It uses a temporary database and store, uploads the
scan, requires every page to be OCR text, generates and approves a three-question paper, and
writes a Markdown review sheet (default: under the pytest tmp dir; the path is printed). The sheet
lists each question, its answer and each cited page with the full cited passage. Because the
database is temporary, the reviewer checks each citation against that page of the source scan and
marks the sheet yes or no. A citation on the right page can still support a wrong answer, for
example when OCR misreads a symbol. The sheet holds an unreleased paper and its answer key: send
it only to the reviewer, through the channel used for exam material.

Not in this build
-----------------

The Anthropic provider, OpenAI embeddings, contents-page outlines, layout extraction,
progress streaming, legacy import and frontend integration.
