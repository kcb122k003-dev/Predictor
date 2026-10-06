"""Tesseract OCR wrapper with layout-preserving line reconstruction and confidence."""

from __future__ import annotations

import os
import re
import shutil
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from ..config.settings import Settings
from ..utils.logging import get_logger, log_event
from .preprocess import preprocess_image

log = get_logger("ocr")

try:
    import pytesseract  # type: ignore

    HAVE_PYTESSERACT = True
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None
    HAVE_PYTESSERACT = False

_WINDOWS_DEFAULTS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


@dataclass
class OcrResult:
    text: str
    confidence: float | None
    low_conf_word_ratio: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


def find_tesseract(configured: str = "") -> str | None:
    if configured:
        return configured if Path(configured).exists() or shutil.which(configured) else None
    found = shutil.which("tesseract")
    if found:
        return found
    if os.name == "nt":  # pragma: no cover - Windows only
        for candidate in _WINDOWS_DEFAULTS:
            if Path(candidate).exists():
                return candidate
    return None


class OcrEngine:
    """OCR for page images. ``available`` is False when Tesseract or pytesseract is missing."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cmd = find_tesseract(settings.ocr.tesseract_cmd) if settings.ocr.enabled else None
        self.available = bool(HAVE_PYTESSERACT and self.cmd)
        if self.available:
            pytesseract.pytesseract.tesseract_cmd = self.cmd
        self.unavailable_reason = ""
        if not settings.ocr.enabled:
            self.unavailable_reason = "OCR is disabled in settings"
        elif not HAVE_PYTESSERACT:
            self.unavailable_reason = "pytesseract is not installed (pip install pytesseract)"
        elif not self.cmd:
            self.unavailable_reason = "the tesseract program was not found (see docs/USER_GUIDE.md)"

    def version(self) -> str | None:
        if not self.available:
            return None
        try:
            return str(pytesseract.get_tesseract_version())
        except Exception:  # pragma: no cover
            return None

    def _config(self) -> str:
        cfg = self.settings.ocr
        return f"--oem {int(cfg.oem)} --psm {int(cfg.psm)} -c preserve_interword_spaces=1"

    def _detect_rotation(self, img: Image.Image) -> int:
        try:
            osd = pytesseract.image_to_osd(img, config="--psm 0")
        except Exception:
            return 0
        m = re.search(r"Rotate:\s*(\d+)", osd)
        conf = re.search(r"Orientation confidence:\s*([\d.]+)", osd)
        if m and (not conf or float(conf.group(1)) >= 1.5):
            return int(m.group(1)) % 360
        return 0

    def ocr_image(self, img: Image.Image, *, label: str = "") -> OcrResult:
        if not self.available:
            raise RuntimeError(f"OCR unavailable: {self.unavailable_reason}")
        cfg = self.settings.ocr
        details: dict[str, Any] = {}
        if cfg.detect_rotation:
            rotation = self._detect_rotation(img)
            if rotation:
                img = img.rotate(-rotation, expand=True)
            details["rotation"] = rotation
        pre = preprocess_image(img, self.settings)
        details.update(pre.details)
        data = pytesseract.image_to_data(pre.image, lang=cfg.language, config=self._config(),
                                         output_type=pytesseract.Output.DICT)
        text, confidences = _reconstruct_lines(data)
        conf = statistics.fmean(confidences) if confidences else None
        low = sum(1 for c in confidences if c < cfg.low_word_confidence) / len(confidences) if confidences else 0.0
        log_event(log, "ocr_page", label=label, confidence=conf, words=len(confidences),
                  low_conf_ratio=round(low, 3), **{k: v for k, v in details.items() if k != "opencv"})
        return OcrResult(text=text, confidence=conf, low_conf_word_ratio=low, details=details)


def _reconstruct_lines(data: dict[str, list[Any]]) -> tuple[str, list[float]]:
    """Rebuild text lines from Tesseract word boxes, keeping wide gaps as runs of spaces.

    Wide gaps matter: marks printed in the right margin ("... cycle.        [6]") are found
    by the exam parser through these gaps.
    """
    lines: dict[tuple[int, int, int], list[tuple[int, int, str, float]]] = {}
    order: list[tuple[int, int, int]] = []
    confidences: list[float] = []
    n = len(data.get("text", []))
    for i in range(n):
        word = (data["text"][i] or "").strip()
        if not word:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append((int(data["left"][i]), int(data["width"][i]), word, conf))
        if conf >= 0:
            confidences.append(conf)
    out_lines: list[str] = []
    prev_block = None
    for key in order:
        words = sorted(lines[key])
        if prev_block is not None and key[0] != prev_block:
            out_lines.append("")
        prev_block = key[0]
        char_widths = [w / max(len(t), 1) for _, w, t, _ in words]
        median_char = statistics.median(char_widths) if char_widths else 10.0
        parts: list[str] = []
        prev_end = None
        for left, width, word, _ in words:
            if prev_end is not None:
                gap = left - prev_end
                parts.append("    " if gap > 4 * median_char else " ")
            parts.append(word)
            prev_end = left + width
        out_lines.append("".join(parts))
    return "\n".join(out_lines).strip(), confidences
