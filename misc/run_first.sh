#!/usr/bin/env bash
# FSL FIRST subcortical volumes for every sub-*/anat/sub-*_T1w.nii.gz under a dataset directory.
# Not used by pie.imaging (which segments with FastSurfer); an FSL alternative for BIDS-like T1 folders.
#
#   bash misc/run_first.sh <dataset_dir> <output_dir>
#
# Writes <output_dir>/<subject>/<subject>_first* (run_first_all outputs) and <output_dir>/first_volumes.csv
# (Subject,Structure,Label,Volume_mm3). Volumes come from run_first_all's combined segmentation
# <prefix>_all_<method>_firstseg.nii.gz, one label at a time (fslstats -l label-0.5 -u label+0.5 -V), as in the
# FIRST user guide. Needs FSL on PATH (run_first_all, fslstats).
set -euo pipefail
DATASET_DIR=${1:?usage: run_first.sh <dataset_dir> <output_dir>}
OUTPUT_DIR=${2:?usage: run_first.sh <dataset_dir> <output_dir>}
for tool in run_first_all fslstats; do
    command -v "$tool" >/dev/null || { echo "$tool not found: set up FSL first" >&2; exit 1; }
done

# The 15 structures run_first_all segments and their labels in the combined segmentation
STRUCTURES=(L_Thal L_Caud L_Puta L_Pall BrStem L_Hipp L_Amyg L_Accu R_Thal R_Caud R_Puta R_Pall R_Hipp R_Amyg R_Accu)
LABELS=(10 11 12 13 16 17 18 26 49 50 51 52 53 54 58)

mkdir -p "$OUTPUT_DIR"
CSV="$OUTPUT_DIR/first_volumes.csv"
echo "Subject,Structure,Label,Volume_mm3" > "$CSV"
shopt -s nullglob
for SUBJECT_DIR in "$DATASET_DIR"/sub-*/; do
    SUBJECT_ID=$(basename "$SUBJECT_DIR")
    T1="${SUBJECT_DIR}anat/${SUBJECT_ID}_T1w.nii.gz"
    if [[ ! -f "$T1" ]]; then echo "no T1 for $SUBJECT_ID, skipped" >&2; continue; fi
    mkdir -p "$OUTPUT_DIR/$SUBJECT_ID"
    PREFIX="$OUTPUT_DIR/$SUBJECT_ID/${SUBJECT_ID}_first"
    echo "FIRST: $SUBJECT_ID"
    if ! run_first_all -i "$T1" -o "$PREFIX"; then echo "run_first_all failed for $SUBJECT_ID" >&2; continue; fi
    SEGS=("$PREFIX"_all_*_firstseg.nii.gz)
    if (( ${#SEGS[@]} != 1 )); then
        echo "expected one ${PREFIX}_all_*_firstseg.nii.gz for $SUBJECT_ID, found ${#SEGS[@]}" >&2; continue
    fi
    for i in "${!STRUCTURES[@]}"; do
        L=${LABELS[$i]}
        VOLUME=$(fslstats "${SEGS[0]}" -l "$((L - 1)).5" -u "$L.5" -V | awk '{print $2}')
        echo "$SUBJECT_ID,${STRUCTURES[$i]},$L,$VOLUME" >> "$CSV"
    done
done
echo "volumes -> $CSV"
