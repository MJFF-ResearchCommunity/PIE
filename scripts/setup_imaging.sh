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

# Pinned: PIE's T1 validation (FS7 agreement, eTIV) ran on FastSurfer commit d0d7d39 (2.6.0-dev0), the same commit
# docker/Dockerfile builds. The v2.6.0 release is different code (Python 3.14, torch 2.14) and is not validated.
# Each IDP row records the version that produced it (fastsurfer_version). Change the commit only with a re-validation.
FASTSURFER_REF="${PIE_FASTSURFER_REF:-d0d7d39369552f8f8b2f14c06d85a3516d06e4cb}"
if [ ! -d third_party/FastSurfer ]; then
  git init -q third_party/FastSurfer
  git -C third_party/FastSurfer fetch -q --depth 1 https://github.com/Deep-MI/FastSurfer.git "$FASTSURFER_REF"
  git -C third_party/FastSurfer checkout -q FETCH_HEAD
fi
uv venv venv_imaging --python 3.12
uv pip install --python venv_imaging/bin/python --torch-backend="$BACKEND" -r third_party/FastSurfer/requirements.txt
uv pip install --python venv_imaging/bin/python dcm2niix pydicom nibabel pandas numpy neuroCombat scikit-learn \
    dipy nilearn antspyx catboost lightgbm xgboost interpret shap matplotlib seaborn optuna polars pyarrow pytest
uv pip install --python venv_imaging/bin/python "endgame-ml[tabular]" || true   # or: uv pip install -e ../endgame
venv_imaging/bin/python -c "import torch, dcm2niix, pydicom, nibabel; print('imaging venv ready; cuda:', torch.cuda.is_available())"
