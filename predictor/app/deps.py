"""Shared helpers for route modules."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import Request

from ..services.context import AppContext


def ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def run_job(request: Request, fn: Callable[..., Any], *args: Any) -> None:
    """Run a job in the background worker (or inline in tests)."""
    if request.app.state.inline:
        try:
            fn(*args)
        except Exception:
            pass
    else:
        request.app.state.jobs.submit(fn, *args)
