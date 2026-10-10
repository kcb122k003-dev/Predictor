"""Syllabus Explorer: course contents, each node's history, source passages and page images."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..services.explorer_service import ExplorerService
from ..services.results_service import ResultsService
from .deps import ctx

router = APIRouter()


@router.get("/explorer/courses")
def explorer_courses(request: Request) -> list[dict[str, Any]]:
    return ExplorerService(ctx(request)).courses()


@router.get("/courses/{course_id}/explorer")
def explorer_overview(request: Request, course_id: int) -> dict[str, Any]:
    return ExplorerService(ctx(request)).overview(course_id)


@router.get("/courses/{course_id}/explorer/nodes/{node_id}")
def explorer_node(request: Request, course_id: int, node_id: int) -> dict[str, Any]:
    return ExplorerService(ctx(request)).node(course_id, node_id)


@router.get("/courses/{course_id}/freshness")
def course_freshness(request: Request, course_id: int) -> dict[str, Any]:
    results = ResultsService(ctx(request))
    return results.freshness(course_id, results.latest_run(course_id))


@router.get("/topics/{topic_id}/source")
def topic_source(request: Request, topic_id: int) -> dict[str, Any]:
    return ExplorerService(ctx(request)).topic_source(topic_id)


@router.get("/questions/{question_id}/source")
def question_source(request: Request, question_id: int) -> dict[str, Any]:
    return ExplorerService(ctx(request)).question_source(question_id)


@router.get("/files/{file_id}/pages/{page_no}/image")
def page_image(request: Request, file_id: int, page_no: int, highlight: str = "") -> Response:
    png = ExplorerService(ctx(request)).page_image(file_id, page_no, highlight)
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})
