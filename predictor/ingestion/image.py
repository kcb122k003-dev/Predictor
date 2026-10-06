"""Image files (JPG, PNG, TIFF, BMP, WEBP): every frame is OCRed."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageSequence

from ..config.settings import Settings
from ..ocr.engine import OcrEngine
from ..preprocessing.quality import assess_text
from ..preprocessing.textnorm import clean_text
from .types import ExtractionResult, PageText


def extract_image(path: Path, settings: Settings, ocr: OcrEngine | None) -> ExtractionResult:
    result = ExtractionResult(pages=[], file_type="image")
    if ocr is None or not ocr.available:
        reason = ocr.unavailable_reason if ocr is not None else "OCR engine not configured"
        raise RuntimeError(f"Cannot read an image without OCR: {reason}")
    with Image.open(path) as img:
        for index, frame in enumerate(ImageSequence.Iterator(img), start=1):
            frame = frame.convert("RGB")
            res = ocr.ocr_image(frame, label=f"{path.name}#p{index}")
            text = clean_text(res.text)
            report = assess_text(text, settings, ocr_confidence=res.confidence)
            details = dict(res.details)
            details["low_conf_word_ratio"] = round(res.low_conf_word_ratio, 3)
            result.pages.append(PageText(index, text, "ocr", res.confidence, report.flags, details))
    return result
