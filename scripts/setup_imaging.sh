#!/usr/bin/env bash
# One-time setup for the PIE imaging layer (pie/imaging): a Python 3.12 venv with FastSurfer's
# dependencies (GPU torch), dcm2niix, pydicom, nibabel, neuroCombat and endgame, plus a
# FastSurfer checkout. Segmentation-only FastSurfer needs no FreeSurfer licence.
#
#   bash scripts/setup_imaging.sh [cu128|cpu]
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${1:-cu128}"
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/"; exit 1; }

[ -d third_party/FastSurfer ] || git clone --depth 1 https://github.com/Deep-MI/FastSurfer.git third_party/FastSurfer
uv venv venv_imaging --python 3.12
uv pip install --python venv_imaging/bin/python --torch-backend="$BACKEND" -r third_party/FastSurfer/requirements.txt
uv pip install --python venv_imaging/bin/python dcm2niix pydicom nibabel pandas numpy neuroCombat scikit-learn \
    catboost lightgbm xgboost interpret shap matplotlib seaborn optuna polars pyarrow pytest
uv pip install --python venv_imaging/bin/python "endgame-ml[tabular]" || true   # or: uv pip install -e ../endgame
venv_imaging/bin/python -c "import torch, dcm2niix, pydicom, nibabel; print('imaging venv ready; cuda:', torch.cuda.is_available())"
