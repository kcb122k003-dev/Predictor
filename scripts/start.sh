#!/usr/bin/env bash
# Start Exam Predictor and open it in your browser. Extra arguments go to "predictor serve".
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d .venv ]; then
  echo "Not installed yet. Run scripts/install.sh first."
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
exec predictor serve "$@"
