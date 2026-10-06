"""Dispatch a file to the right extractor based on its content and extension."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ..config.settings import Settings
from ..ocr.engine import OcrEngine
from .types import ExtractionResult

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_EXT = {".txt", ".md"}


class UnsupportedFileError(ValueError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sniff_type(path: Path) -> str:
    """Return pdf | docx | image | text from magic bytes, falling back to the extension."""
    with path.open("rb") as fh:
        head = fh.read(16)
    ext = path.suffix.lower()
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK") and ext == ".docx":
        return "docx"
    if head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"II*\x00", b"MM\x00*", b"BM")) or \
            (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return "image"
    if ext in IMAGE_EXT:
        return "image"
    if ext in TEXT_EXT:
        return "text"
    if ext == ".pdf":
        return "pdf"
    if ext == ".doc":
        raise UnsupportedFileError("Old .doc files are not supported. Save the file as .docx or PDF first.")
    raise UnsupportedFileError(f"Unsupported file type '{ext or 'unknown'}'")


def extract_document(path: Path, settings: Settings, ocr: OcrEngine | None = None) -> ExtractionResult:
    kind = sniff_type(path)
    if kind == "pdf":
        from .pdf import extract_pdf

        return extract_pdf(path, settings, ocr)
    if kind == "docx":
        from .docx_reader import extract_docx

        return extract_docx(path)
    if kind == "image":
        from .image import extract_image

        return extract_image(path, settings, ocr)
    from .text import extract_text

    return extract_text(path, settings)
