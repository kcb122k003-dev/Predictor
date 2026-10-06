"""Image preprocessing before OCR: grayscale, upscale, denoise, contrast, deskew, binarise.

OpenCV is used when installed. Without it, a reduced Pillow-only path still converts to
grayscale, upscales and applies autocontrast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..config.settings import Settings

try:
    import cv2  # type: ignore

    HAVE_CV2 = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    HAVE_CV2 = False


@dataclass
class PreprocessResult:
    image: Image.Image
    details: dict[str, Any] = field(default_factory=dict)


def _upscale(img: Image.Image, min_long_side: int) -> tuple[Image.Image, float]:
    long_side = max(img.size)
    if long_side >= min_long_side or long_side == 0:
        return img, 1.0
    scale = min(min_long_side / long_side, 4.0)
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS), scale


def estimate_skew(gray: np.ndarray, max_angle: float = 6.0, step: float = 0.25) -> float:
    """Projection-profile skew estimate in degrees (positive = counter-clockwise text).

    For each candidate angle, rotate a downscaled binary image and measure the variance of
    row sums. Text lines are sharpest (highest variance) at the correct angle.
    """
    if not HAVE_CV2:
        return 0.0
    h, w = gray.shape[:2]
    scale = 1000.0 / max(h, w) if max(h, w) > 1000 else 1.0
    small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale != 1.0 else gray
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if binary.mean() < 1.0:  # blank page
        return 0.0
    center = (binary.shape[1] / 2, binary.shape[0] / 2)
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 1e-9, step):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(binary, matrix, (binary.shape[1], binary.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def rotate_gray(gray: np.ndarray, angle: float) -> np.ndarray:
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def preprocess_image(img: Image.Image, settings: Settings) -> PreprocessResult:
    cfg = settings.ocr
    details: dict[str, Any] = {"opencv": HAVE_CV2}
    img = ImageOps.exif_transpose(img)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    img, scale = _upscale(img, int(cfg.min_long_side_px))
    details["upscale"] = round(scale, 3)

    if not HAVE_CV2:
        gray = ImageOps.grayscale(img)
        if cfg.contrast_normalize:
            gray = ImageOps.autocontrast(gray, cutoff=1)
        return PreprocessResult(gray, details)

    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    if cfg.denoise:
        gray = cv2.medianBlur(gray, 3)
    if cfg.contrast_normalize:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    if cfg.deskew:
        angle = estimate_skew(gray)
        details["deskew_angle"] = angle
        if abs(angle) >= 0.3:
            gray = rotate_gray(gray, angle)
    mode = str(cfg.binarize).lower()
    if mode == "adaptive":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    elif mode == "otsu":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    details["binarize"] = mode
    return PreprocessResult(Image.fromarray(gray), details)
