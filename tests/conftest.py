from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictor.config.settings import defaults, load_settings  # noqa: E402
from predictor.ingestion.types import PageText  # noqa: E402

HAVE_TESSERACT = shutil.which("tesseract") is not None


@pytest.fixture(scope="session")
def settings():
    return defaults()


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def app_settings(data_dir):
    return load_settings(data_dir)


def pages_from(text: str) -> list[PageText]:
    return [PageText(i + 1, t, "text") for i, t in enumerate(text.split("\f"))]


requires_tesseract = pytest.mark.skipif(not HAVE_TESSERACT, reason="tesseract binary not installed")
