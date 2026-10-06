#!/usr/bin/env bash
# One-command install for Linux and macOS.
#   scripts/install.sh            install the app (with OCR support)
#   scripts/install.sh --dev      also install test tools
#   scripts/install.sh --neural   also install the optional neural embedding library
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.10 or newer is required. Install it from https://www.python.org/downloads/ and run this again."
  exit 1
fi
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "Python 3.10 or newer is required (found $("$PY" --version))."
  exit 1
fi

EXTRAS="ocr"
for arg in "$@"; do
  case "$arg" in
    --dev) EXTRAS="$EXTRAS,dev" ;;
    --neural) EXTRAS="$EXTRAS,neural" ;;
  esac
done

echo "Creating a virtual environment in .venv ..."
"$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
echo "Installing Exam Predictor and its dependencies (extras: $EXTRAS) ..."
pip install -e ".[$EXTRAS]"

if ! command -v tesseract >/dev/null 2>&1; then
  echo
  echo "Tesseract OCR was not found. Digital PDFs, Word files and text files work without it,"
  echo "but scanned papers and photos need it:"
  echo "  Ubuntu/Debian:  sudo apt install tesseract-ocr"
  echo "  Fedora:         sudo dnf install tesseract"
  echo "  macOS:          brew install tesseract"
fi

echo
predictor doctor || true
echo
echo "Done. Start the app with:  scripts/start.sh"
echo "Try it with example data:  scripts/start.sh  then click 'Load the synthetic demo course'"
