"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from tribunal.api.routers import documents, papers, subjects
from tribunal.context import AppContext
from tribunal.errors import TribunalError


def create_app(ctx: AppContext) -> FastAPI:
    # the body is capped before it is parsed or spooled to disk; the exact check stays in the upload handler
    limit = ctx.snapshot.config.ingestion.max_upload_bytes + 1024 * 1024
    app = FastAPI(title=ctx.settings.app_name, max_body_size=limit)
    app.state.ctx = ctx
    app.add_middleware(
        CORSMiddleware, allow_origins=ctx.settings.allowed_origins, allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(TribunalError)
    async def _tribunal_error(_: Request, exc: TribunalError) -> JSONResponse:
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.message, "details": exc.details}}, status_code=exc.status
        )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "config": ctx.snapshot.snapshot_id}

    for module in (subjects, documents, papers):
        app.include_router(module.router)
    return app
