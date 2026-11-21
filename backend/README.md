Tribunal backend
================

Tribunal is a multi-subject exam paper generator that writes source-backed questions and routes
them through faculty approval. This package is the backend service and its command line.

Install
-------

    pip install -e "backend[dev]"

Optional extras: ocr (ocrmypdf), openai, postgres (psycopg and pgvector), embeddings-local
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
  STORAGE_BACKEND      local or s3; s3 is NOT implemented in this build
  SIGNING_SECRET       at least 32 characters

Also required, by profile:
  STORAGE_LOCAL_ROOT   when STORAGE_BACKEND is local
  OIDC_ISSUER, OIDC_AUDIENCE, OIDC_JWKS_URL   when AUTH_MODE is oidc
  Hosted profiles (staging, production): AUTH_MODE oidc, ALLOWED_ORIGINS, an https PUBLIC_BASE_URL,
  a PostgreSQL DATABASE_URL, PDF_FONT_REGULAR and STORAGE_BACKEND s3.
  Because s3 is not implemented yet, staging and production cannot start in this build; only the
  local and test profiles run.

AUTH_MODE dev trusts an X-Dev-User header and is for local use and tests only.

Runtime configuration
---------------------

RUNTIME_CONFIG points to a YAML file holding models, prompts, OCR, retrieval, limits and text
policy. config/runtime.example.yaml is the template. Its values use TRIBUNAL_* variables:
  TRIBUNAL_WRITER_MODEL, TRIBUNAL_REVIEWER_MODEL, TRIBUNAL_REASONER_MODEL   required
  TRIBUNAL_OCR_LANGUAGE   default eng
  TRIBUNAL_WORKER_THREADS default 2
The openai provider also needs OPENAI_API_KEY. Course content and model names never go in code.

Commands
--------

    tribunal check-config   validate settings and runtime configuration (exit 2 names each problem)
    tribunal db-upgrade     apply database migrations
    tribunal serve          run the HTTP API
    tribunal worker         run the background job worker
    tribunal scan-text --config backend/config/runtime.example.yaml PATH...
                            report emoji in text files (exit 1 on findings)

Tests
-----

    cd backend && pytest
    ruff check backend

tests/test_acceptance.py runs the whole first ticket through the HTTP API with an offline stub
provider and a fake OCR engine.

Not in this build
-----------------

S3 storage, the Anthropic provider, OpenAI embeddings, contents-page outlines, layout extraction,
progress streaming, legacy import and frontend integration.
