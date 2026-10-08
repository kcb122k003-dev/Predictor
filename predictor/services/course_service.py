"""Course (project) management. Every course is independent: its own files, exams, topics and runs."""

from __future__ import annotations

import shutil
from typing import Any

from sqlalchemy import func, select

from ..config.settings import deep_merge, defaults
from ..database.models import AnalysisRun, Course, CourseTopic, Exam, SourceFile
from .context import AppContext


class CourseService:
    def __init__(self, app: AppContext):
        self.app = app

    def create(self, name: str, code: str = "", description: str = "", *, is_synthetic: bool = False) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("A course needs a name.")
        with self.app.db.session() as s:
            course = Course(name=name[:200], code=code.strip()[:60], description=description.strip(),
                            is_synthetic=bool(is_synthetic))
            s.add(course)
            s.flush()
            return course.id

    def list(self) -> list[dict[str, Any]]:
        with self.app.db.session() as s:
            out = []
            for c in s.execute(select(Course).order_by(Course.created_at.desc())).scalars():
                out.append(self._summary(s, c))
            return out

    def get(self, course_id: int) -> dict[str, Any]:
        with self.app.db.session() as s:
            c = s.get(Course, course_id)
            if c is None:
                raise KeyError(course_id)
            return self._summary(s, c)

    def _summary(self, s, c: Course) -> dict[str, Any]:
        def count(model, *conds):
            return s.execute(select(func.count()).select_from(model).where(*conds)).scalar_one()

        last_run = s.execute(select(AnalysisRun).where(AnalysisRun.course_id == c.id)
                             .order_by(AnalysisRun.id.desc())).scalars().first()
        return {
            "id": c.id, "name": c.name, "code": c.code, "description": c.description,
            "created_at": c.created_at.isoformat() if c.created_at else None, "settings": c.settings or {},
            "is_synthetic": bool(c.is_synthetic),
            "exam_files": count(SourceFile, SourceFile.course_id == c.id, SourceFile.kind == "exam"),
            "syllabus_files": count(SourceFile, SourceFile.course_id == c.id, SourceFile.kind == "syllabus"),
            "exams": count(Exam, Exam.course_id == c.id),
            "exams_included": count(Exam, Exam.course_id == c.id, Exam.include_in_analysis.is_(True)),
            "topics": count(CourseTopic, CourseTopic.course_id == c.id),
            "last_run": None if last_run is None else {
                "id": last_run.id, "status": last_run.status, "progress": last_run.progress,
                "message": last_run.message[:300],
                "finished_at": last_run.finished_at.isoformat() if last_run.finished_at else None},
        }

    def update(self, course_id: int, *, name: str | None = None, code: str | None = None,
               description: str | None = None, settings: dict[str, Any] | None = None) -> None:
        with self.app.db.session() as s:
            c = s.get(Course, course_id)
            if c is None:
                raise KeyError(course_id)
            if name is not None:
                c.name = name.strip()[:200] or c.name
            if code is not None:
                c.code = code.strip()[:60]
            if description is not None:
                c.description = description
            if settings is not None:
                deep_merge(defaults().as_dict(), settings)  # validates keys and types
                c.settings = settings

    def delete(self, course_id: int) -> None:
        with self.app.db.session() as s:
            c = s.get(Course, course_id)
            if c is None:
                raise KeyError(course_id)
            s.delete(c)
        shutil.rmtree(self.app.data_dir / "courses" / str(course_id), ignore_errors=True)
