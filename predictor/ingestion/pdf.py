"""PDF extraction: native text layer when usable, OCR for scanned or garbled pages."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from ..config.settings import Settings
from ..ocr.engine import OcrEngine
from ..preprocessing.quality import assess_text, garbage_ratio
from ..preprocessing.textnorm import clean_text
from ..utils.logging import get_logger, log_event
from .types import ExtractionResult, PageText

log = get_logger("ingestion.pdf")

try:
    import pymupdf  # type: ignore
except Exception:  # pragma: no cover - older PyMuPDF
    import fitz as pymupdf  # type: ignore


def _page_image(page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _native_text(page) -> str:
    # "sort=True" orders blocks top-to-bottom, left-to-right, which matches reading order on
    # single-column papers. Wide horizontal gaps are kept by rebuilding lines from spans.
    lines: list[tuple[float, float, str]] = []
    data = page.get_text("dict", sort=True)
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            spans.sort(key=lambda s: s["bbox"][0])
            parts: list[str] = []
            prev_end = None
            for span in spans:
                size = max(span.get("size", 10.0), 1.0)
                if prev_end is not None:
                    gap = span["bbox"][0] - prev_end
                    if gap > size * 2.5:
                        parts.append("    ")
                    elif gap > size * 0.15 and not parts[-1].endswith(" ") and not span["text"].startswith(" "):
                        parts.append(" ")
                parts.append(span["text"])
                prev_end = span["bbox"][2]
            y0 = line["bbox"][1]
            x0 = line["bbox"][0]
            lines.append((round(y0, 1), x0, "".join(parts).rstrip()))
    # Merge fragments that share a baseline (e.g. marks in a separate text box on the right).
    lines.sort(key=lambda item: (item[0], item[1]))
    merged: list[list] = []
    for y, x, text in lines:
        if merged and abs(merged[-1][0] - y) < 2.5:
            merged[-1][2] = merged[-1][2] + "    " + text.strip()
        else:
            merged.append([y, x, text])
    return "\n".join(item[2] for item in merged)


def extract_pdf(path: Path, settings: Settings, ocr: OcrEngine | None) -> ExtractionResult:
    result = ExtractionResult(pages=[], file_type="pdf")
    cfg = settings.ingestion
    with pymupdf.open(str(path)) as doc:
        if doc.needs_pass:
            raise ValueError("The PDF is password-protected. Remove the password and upload again.")
        for index, page in enumerate(doc, start=1):
            native = clean_text(_native_text(page))
            garbage = garbage_ratio(native) if native else 1.0
            usable = len(native.strip()) >= cfg.native_text_min_chars and garbage <= cfg.native_garbage_max_ratio
            if usable:
                report = assess_text(native, settings)
                result.pages.append(PageText(index, native, "native", None, report.flags,
                                             {"garbage_ratio": round(garbage, 3)}))
                continue
            if ocr is not None and ocr.available:
                image = _page_image(page, int(settings.ocr.dpi))
                ocr_res = ocr.ocr_image(image, label=f"{path.name}#p{index}")
                text = clean_text(ocr_res.text)
                report = assess_text(text, settings, ocr_confidence=ocr_res.confidence)
                details = dict(ocr_res.details)
                details["low_conf_word_ratio"] = round(ocr_res.low_conf_word_ratio, 3)
                details["native_chars"] = len(native)
                result.pages.append(PageText(index, text, "ocr", ocr_res.confidence, report.flags, details))
            else:
                flags = ["needs_ocr_but_unavailable"] if len(native.strip()) < cfg.native_text_min_chars else ["garbled_text_layer"]
                reason = ocr.unavailable_reason if ocr is not None else "OCR engine not configured"
                result.warnings.append(f"Page {index} looks scanned but OCR is unavailable: {reason}")
                result.pages.append(PageText(index, native, "native", None, flags, {"garbage_ratio": round(garbage, 3)}))
    log_event(log, "pdf_extracted", file=path.name, **{k: v for k, v in result.summary().items() if k != "warnings"})
    return result
