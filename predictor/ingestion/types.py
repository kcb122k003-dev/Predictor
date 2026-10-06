"""Shared data structures for document extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PageText:
    page_no: int  # 1-based
    text: str
    method: str  # native | ocr | docx | text
    confidence: float | None = None  # OCR mean word confidence (0-100), None for native text
    flags: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    pages: list[PageText]
    file_type: str
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.method == "ocr")

    def summary(self) -> dict[str, Any]:
        confs = [p.confidence for p in self.pages if p.confidence is not None]
        return {
            "file_type": self.file_type,
            "pages": len(self.pages),
            "ocr_pages": self.ocr_pages,
            "mean_ocr_confidence": round(sum(confs) / len(confs), 1) if confs else None,
            "flagged_pages": [p.page_no for p in self.pages if p.flags],
            "warnings": self.warnings,
            "characters": sum(len(p.text) for p in self.pages),
        }
