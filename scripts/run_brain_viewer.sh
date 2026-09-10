#!/usr/bin/env bash
set -euo pipefail

PIE_VIEWER_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PIE_VIEWER_ROOT"
PIE_VIEWER_PYTHON="${PIE_VIEWER_PYTHON:-$PIE_VIEWER_ROOT/venv_imaging/bin/python}"

if [[ ! -x "$PIE_VIEWER_PYTHON" ]]; then
  echo "Create venv_imaging first (see documentation/brain_viewer.md), or set PIE_VIEWER_PYTHON."
  exit 1
fi
if ! "$PIE_VIEWER_PYTHON" -c 'import fastapi, uvicorn, nibabel, numpy, scipy, pandas, skimage, SimpleITK' 2>/dev/null; then
  echo "Install viewer dependencies: $PIE_VIEWER_PYTHON -m pip install -r pie/imaging/viewer/requirements.txt"
  exit 1
fi
if [[ ! -d brain-viewer/node_modules ]]; then
  npm --prefix brain-viewer ci
fi
npm --prefix brain-viewer run build
if [[ ! -f Imaging/derived/viewer_sample_plan/plan.json ]]; then
  "$PIE_VIEWER_PYTHON" -m pie.imaging.viewer sample-plan
fi
echo "Open http://127.0.0.1:8765 — images and API are served locally."
exec "$PIE_VIEWER_PYTHON" -m pie.imaging.viewer serve "$@"
