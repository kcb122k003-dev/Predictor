"""Health, settings, courses and file uploads."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy import select

from .. import __version__
from ..config.settings import read_user_settings, save_user_settings
from ..database.models import DocumentPage, Exam, SourceFile
from ..embeddings.backends import model_dir, neural_available
from ..ingestion.extract import UnsupportedFileError
from ..services.course_service import CourseService
from ..services.ingest_service import IngestService
from .deps import ctx, run_job

router = APIRouter()


class CourseIn(BaseModel):
    name: str
    code: str = ""
    description: str = ""


class CoursePatch(BaseModel):
    name: str | None = None
    code: str | None = None
    description: str | None = None
    settings: dict[str, Any] | None = None


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    app = ctx(request)
    ocr = app.ocr
    neural_ok, neural_reason = neural_available(app.settings)
    return {
        "version": __version__, "data_dir": str(app.data_dir),
        "ocr": {"available": ocr.available, "version": ocr.version(), "reason": ocr.unavailable_reason},
        "embeddings": {"configured": app.settings.embeddings.backend, "neural_available": neural_ok,
                       "neural_reason": neural_reason, "model": app.settings.embeddings.model_name,
                       "model_dir": str(model_dir(app.settings))},
        "external_services": bool(app.settings.app.allow_external_services),
        "jobs_pending": request.app.state.jobs.pending,
    }


@router.get("/settings")
def get_settings(request: Request) -> dict[str, Any]:
    app = ctx(request)
    return {"effective": app.settings.as_dict(), "overrides": read_user_settings(app.data_dir)}


@router.put("/settings")
def put_settings(request: Request, overrides: dict[str, Any]) -> dict[str, Any]:
    app = ctx(request)
    save_user_settings(app.data_dir, overrides)
    app.reload_settings()
    return {"ok": True, "effective": app.settings.as_dict()}


@router.get("/courses")
def list_courses(request: Request) -> list[dict[str, Any]]:
    return CourseService(ctx(request)).list()


@router.post("/courses")
def create_course(request: Request, body: CourseIn) -> dict[str, Any]:
    cid = CourseService(ctx(request)).create(body.name, body.code, body.description)
    return CourseService(ctx(request)).get(cid)


@router.get("/courses/{course_id}")
def get_course(request: Request, course_id: int) -> dict[str, Any]:
    return CourseService(ctx(request)).get(course_id)


@router.patch("/courses/{course_id}")
def patch_course(request: Request, course_id: int, body: CoursePatch) -> dict[str, Any]:
    CourseService(ctx(request)).update(course_id, name=body.name, code=body.code, description=body.description,
                                       settings=body.settings)
    return CourseService(ctx(request)).get(course_id)


@router.delete("/courses/{course_id}")
def delete_course(request: Request, course_id: int) -> dict[str, Any]:
    CourseService(ctx(request)).delete(course_id)
    return {"ok": True}


@router.post("/courses/{course_id}/files")
async def upload_files(request: Request, course_id: int, files: list[UploadFile] = File(...),
                       kind: str = Form(...), syllabus_version: str = Form("current")) -> list[dict[str, Any]]:
    app = ctx(request)
    CourseService(app).get(course_id)
    service = IngestService(app)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for upload in files:
            name = Path(upload.filename or "upload").name
            path = Path(tmp) / name
            with path.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            try:
                res = service.add_file(course_id, path, name, kind, syllabus_version)
            except (UnsupportedFileError, ValueError) as exc:
                out.append({"filename": name, "error": str(exc)})
                continue
            out.append({"filename": name, "file_id": res.file_id, "duplicate": res.duplicate, "message": res.message})
            if not res.duplicate:
                run_job(request, service.process_file, res.file_id)
    return out


@router.post("/demo")
def create_demo(request: Request) -> dict[str, Any]:
    """Create the synthetic demo course (generated papers, clearly labelled as synthetic)."""
    from ..demo.generator import generate_demo

    app = ctx(request)
    service = IngestService(app)
    cid = CourseService(app).create("Fluid Mechanics (synthetic demo)", "CE 501",
                                    "Generated example data with planted patterns. Not real exams.")
    with tempfile.TemporaryDirectory() as tmp:
        generate_demo(Path(tmp))
        for kind, folder in (("syllabus", "syllabus"), ("exam", "exams")):
            for path in sorted((Path(tmp) / folder).iterdir()):
                res = service.add_file(cid, path, path.name, kind)
                if not res.duplicate:
                    run_job(request, service.process_file, res.file_id)
    return CourseService(app).get(cid)


@router.get("/courses/{course_id}/files")
def list_files(request: Request, course_id: int) -> list[dict[str, Any]]:
    app = ctx(request)
    with app.db.session() as s:
        rows = s.execute(select(SourceFile).where(SourceFile.course_id == course_id)
                         .order_by(SourceFile.kind, SourceFile.filename)).scalars().all()
        out = []
        for f in rows:
            exam = s.execute(select(Exam).where(Exam.source_file_id == f.id)).scalars().first()
            out.append({"id": f.id, "kind": f.kind, "filename": f.filename, "status": f.status, "error": f.error,
                        "pages": f.page_count, "size": f.size_bytes, "summary": f.extraction_summary,
                        "uploaded_at": f.uploaded_at.isoformat() if f.uploaded_at else None,
                        "exam_id": exam.id if exam else None,
                        "exam_label": (exam.structure or {}).get("label") if exam else None,
                        "included": exam.include_in_analysis if exam else None,
                        "exclusion_reason": exam.exclusion_reason if exam else None})
        return out


@router.get("/files/{file_id}/pages")
def file_pages(request: Request, file_id: int) -> list[dict[str, Any]]:
    app = ctx(request)
    with app.db.session() as s:
        rows = s.execute(select(DocumentPage).where(DocumentPage.file_id == file_id)
                         .order_by(DocumentPage.page_no)).scalars().all()
        return [{"page": p.page_no, "method": p.method, "confidence": p.ocr_confidence, "flags": p.quality_flags,
                 "details": p.details, "text": p.text} for p in rows]


@router.post("/files/{file_id}/reprocess")
def reprocess(request: Request, file_id: int) -> dict[str, Any]:
    app = ctx(request)
    with app.db.session() as s:
        f = s.get(SourceFile, file_id)
        if f is None:
            raise KeyError(file_id)
        f.status = "pending"
    run_job(request, IngestService(app).process_file, file_id)
    return {"ok": True}


@router.delete("/files/{file_id}")
def delete_file(request: Request, file_id: int) -> dict[str, Any]:
    app = ctx(request)
    with app.db.session() as s:
        f = s.get(SourceFile, file_id)
        if f is None:
            raise KeyError(file_id)
        for exam in s.execute(select(Exam).where(Exam.source_file_id == file_id)).scalars():
            s.delete(exam)
        path = Path(f.stored_path)
        s.delete(f)
    path.unlink(missing_ok=True)
    return {"ok": True}
