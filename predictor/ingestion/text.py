"""Plain text and Markdown files. A form feed (\\f) separates pages."""

from __future__ import annotations

from pathlib import Path

from ..config.settings import Settings
from ..preprocessing.quality import assess_text
from ..preprocessing.textnorm import clean_text
from .types import ExtractionResult, PageText


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            continue
        return text
    return raw.decode("utf-8", errors="replace")  # pragma: no cover - latin-1 always decodes


def extract_text(path: Path, settings: Settings) -> ExtractionResult:
    text = read_text_file(path)
    result = ExtractionResult(pages=[], file_type="text")
    for index, chunk in enumerate(text.split("\f"), start=1):
        cleaned = clean_text(chunk)
        if not cleaned.strip():
            continue
        report = assess_text(cleaned, settings)
        result.pages.append(PageText(index, cleaned, "text", None, report.flags, {}))
    if not result.pages:
        result.warnings.append("The file is empty.")
    return result
