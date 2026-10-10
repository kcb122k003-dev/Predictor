"""Analysis runs, results, evidence, search and export."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel

from ..export.service import ExportService
from ..search.service import SearchService
from ..services.analysis_service import AnalysisService
from ..services.results_service import ResultsService
from .deps import ctx, run_job

router = APIRouter()


class PapersIn(BaseModel):
    seed: int = 7
    variants: int = 3


@router.post("/courses/{course_id}/analyze")
def analyze(request: Request, course_id: int) -> dict[str, Any]:
    service = AnalysisService(ctx(request))
    run_id = service.create_run(course_id)
    run_job(request, service.run, run_id)
    return ResultsService(ctx(request)).run(run_id)


@router.get("/runs/{run_id}")
def get_run(request: Request, run_id: int) -> dict[str, Any]:
    return ResultsService(ctx(request)).run(run_id)


@router.get("/courses/{course_id}/runs")
def list_runs(request: Request, course_id: int) -> list[dict[str, Any]]:
    return ResultsService(ctx(request)).runs(course_id)


@router.get("/courses/{course_id}/results")
def latest_results(request: Request, course_id: int) -> dict[str, Any]:
    results = ResultsService(ctx(request))
    run_id = results.latest_run(course_id)
    if run_id is None:
        return {"run": None}
    out = {"run": results.run(run_id), "predictions": results.predictions(run_id, "topic"),
           "freshness": results.freshness(course_id, run_id)}
    for key in ("excluded", "sufficiency", "evidence"):
        try:
            out[key] = results.artifact(run_id, key)
        except KeyError:
            out[key] = None
    return out


@router.get("/runs/{run_id}/predictions")
def predictions(request: Request, run_id: int, layer: str = "topic") -> list[dict[str, Any]]:
    return ResultsService(ctx(request)).predictions(run_id, layer)


@router.get("/runs/{run_id}/topics/{topic_id}")
def topic_detail(request: Request, run_id: int, topic_id: int) -> dict[str, Any]:
    return ResultsService(ctx(request)).topic_detail(run_id, topic_id)


@router.get("/runs/{run_id}/models")
def models(request: Request, run_id: int) -> dict[str, Any]:
    return ResultsService(ctx(request)).models(run_id)


@router.get("/runs/{run_id}/artifacts/{key}")
def artifact(request: Request, run_id: int, key: str) -> dict[str, Any]:
    return ResultsService(ctx(request)).artifact(run_id, key)


@router.post("/runs/{run_id}/papers")
def papers(request: Request, run_id: int, body: PapersIn) -> dict[str, Any]:
    variants = max(1, min(body.variants, 5))
    return {"papers": ResultsService(ctx(request)).regenerate_papers(run_id, body.seed, variants)}


@router.get("/courses/{course_id}/search")
def search(request: Request, course_id: int, q: str = "", year: int | None = None, unit: int | None = None,
           qtype: str | None = None, topic: int | None = None, limit: int = Query(50, le=200)) -> dict[str, Any]:
    return SearchService(ctx(request)).search(course_id, q, year=year, unit_id=unit, qtype=qtype, topic_id=topic,
                                              limit=limit)


@router.get("/runs/{run_id}/export")
def export(request: Request, run_id: int, format: str = "pdf", table: str = "predictions") -> Response:
    service = ExportService(ctx(request))
    fmt = format.lower()
    if fmt == "csv":
        return Response(service.csv(run_id, table), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="run{run_id}_{table}.csv"'})
    if fmt == "xlsx":
        return Response(service.xlsx(run_id),
                        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        headers={"Content-Disposition": f'attachment; filename="run{run_id}_report.xlsx"'})
    if fmt == "json":
        return Response(service.json(run_id), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="run{run_id}.json"'})
    if fmt == "pdf":
        return Response(service.pdf(run_id), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="run{run_id}_report.pdf"'})
    raise ValueError("format must be pdf, xlsx, csv or json")
