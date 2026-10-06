"""FastAPI application factory. The server binds to 127.0.0.1 and serves the UI and the JSON API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..services.analysis_service import AnalysisError
from ..services.context import AppContext
from .jobs import JobRunner
from .routes_core import router as core_router
from .routes_results import router as results_router
from .routes_review import router as review_router

STATIC_DIR = Path(__file__).resolve().parent.parent / "ui" / "static"


def _plotly_js() -> Path | None:
    try:
        import plotly

        path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        return path if path.exists() else None
    except Exception:  # pragma: no cover - plotly is optional at runtime
        return None


def create_app(ctx: AppContext | None = None, *, run_jobs_inline: bool = False) -> FastAPI:
    ctx = ctx or AppContext.create()
    app = FastAPI(title="Exam Predictor", version=__version__, docs_url="/api/docs", redoc_url=None)
    app.state.ctx = ctx
    app.state.jobs = JobRunner()
    app.state.inline = run_jobs_inline

    @app.exception_handler(KeyError)
    async def not_found(_: Request, exc: KeyError):
        return JSONResponse({"detail": f"Not found: {exc.args[0] if exc.args else ''}"}, status_code=404)

    @app.exception_handler(ValueError)
    async def bad_request(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(AnalysisError)
    async def analysis_error(_: Request, exc: AnalysisError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    app.include_router(core_router, prefix="/api")
    app.include_router(review_router, prefix="/api")
    app.include_router(results_router, prefix="/api")

    @app.get("/vendor/plotly.min.js", include_in_schema=False)
    def plotly_js():
        path = _plotly_js()
        if path is None:
            return Response("/* plotly not installed: charts are shown as tables */", media_type="text/javascript")
        return FileResponse(path, media_type="text/javascript", headers={"Cache-Control": "max-age=86400"})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.on_event("shutdown")
    def _shutdown() -> None:  # pragma: no cover - server lifecycle
        app.state.jobs.shutdown()

    return app
