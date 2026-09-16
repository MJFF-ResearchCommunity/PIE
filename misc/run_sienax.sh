#!/usr/bin/env bash
# FSL SIENAX (cross-sectional brain, grey- and white-matter volumes, normalised for head size) for every
# sub-*/anat/sub-*_T1w.nii.gz under a dataset directory. Not used by pie.imaging.
#
#   bash misc/run_sienax.sh <dataset_dir> <output_dir> [BET options, default "-f 0.2 -g 0.02"]
#
# Writes <output_dir>/<subject>/, where report.sienax holds the volumes. Needs FSL on PATH (sienax).
set -euo pipefail
DATASET_DIR=${1:?usage: run_sienax.sh <dataset_dir> <output_dir> [BET options]}
OUTPUT_DIR=${2:?usage: run_sienax.sh <dataset_dir> <output_dir> [BET options]}
BET_OPTS=${3:-"-f 0.2 -g 0.02"}
command -v sienax >/dev/null || { echo "sienax not found: set up FSL first" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"
shopt -s nullglob
for SUBJECT_DIR in "$DATASET_DIR"/sub-*/; do
    SUBJECT_ID=$(basename "$SUBJECT_DIR")
    T1="${SUBJECT_DIR}anat/${SUBJECT_ID}_T1w.nii.gz"
    if [[ ! -f "$T1" ]]; then echo "no T1 for $SUBJECT_ID, skipped" >&2; continue; fi
    echo "SIENAX: $SUBJECT_ID"
    sienax "$T1" -o "$OUTPUT_DIR/$SUBJECT_ID" -B "$BET_OPTS" || echo "sienax failed for $SUBJECT_ID" >&2
done
echo "reports -> $OUTPUT_DIR/<subject>/report.sienax"
