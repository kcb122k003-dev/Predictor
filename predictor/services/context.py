"""Application context: settings, database, OCR engine and paths for one data folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config.settings import Settings, load_settings
from ..database.models import Course
from ..database.session import Database
from ..ocr.engine import OcrEngine
from ..utils.logging import configure_logging


@dataclass
class AppContext:
    settings: Settings
    db: Database
    data_dir: Path
    _ocr: OcrEngine | None = None

    @classmethod
    def create(cls, data_dir: Path | str | None = None, overrides: Mapping[str, Any] | None = None,
               *, in_memory: bool = False, log: bool = True) -> "AppContext":
        settings = load_settings(data_dir, overrides)
        root = Path(settings.app.data_dir)
        root.mkdir(parents=True, exist_ok=True)
        if log:
            configure_logging(settings.logging.level, root / "logs", json_file=bool(settings.logging.json_file),
                              max_bytes=int(settings.logging.max_bytes), backups=int(settings.logging.backups))
        db = Database(":memory:" if in_memory else root / "predictor.sqlite3")
        return cls(settings=settings, db=db, data_dir=root)

    @property
    def ocr(self) -> OcrEngine:
        if self._ocr is None:
            self._ocr = OcrEngine(self.settings)
        return self._ocr

    def course_settings(self, course: Course | None) -> Settings:
        if course is None or not course.settings:
            return self.settings
        return self.settings.merged(course.settings)

    def course_dir(self, course_id: int) -> Path:
        path = self.data_dir / "courses" / str(course_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def reload_settings(self, overrides: Mapping[str, Any] | None = None) -> None:
        self.settings = load_settings(self.data_dir, overrides)
        self._ocr = None
