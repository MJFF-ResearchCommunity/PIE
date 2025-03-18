+++++EXAMPLE BASH SCRIPT FOR SIENAX +++++

## This script:
## 1. Loops through all subject directories.
## 2. Finds the T1-W image in each directory.
## 3. Runs FIRST for each subject.
## 4. Saves the outputs in structured format.


##  NIfTI Images are expected to be organized as follows:

/path/to/dataset/
├── sub-01/
│   └── anat/sub-01_T1w.nii.gz
├── sub-02/
│   └── anat/sub-02_T1w.nii.gz
└── sub-30/
    └── anat/sub-30_T1w.nii.gz


EXAMPLE SCRIPT:

#!/bin/bash

# Path to the dataset containing subjects' directories
DATASET_DIR="/path/to/dataset"
# Output directory to store SIENAX results
OUTPUT_DIR="/path/to/sienax_outputs"
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through each subject folder
for SUBJECT_DIR in "$DATASET_DIR"/sub-*/; do
    # Extract subject ID (e.g., sub-01)
    SUBJECT_ID=$(basename "$SUBJECT_DIR")
    
    # Define input T1-weighted image path
    T1_IMAGE="$SUBJECT_DIR/anat/${SUBJECT_ID}_T1w.nii.gz"
    
    # Check if T1 image exists
    if [[ -f "$T1_IMAGE" ]]; then
        echo "Running SIENAX for $SUBJECT_ID ..."
        
        # Define output directory for the subject
        SUBJECT_OUTPUT="${OUTPUT_DIR}/${SUBJECT_ID}"
        mkdir -p "$SUBJECT_OUTPUT"
        
        # Run SIENAX
        sienax "$T1_IMAGE" -o "$SUBJECT_OUTPUT" -B "-f 0.2 -g 0.02"
        
        echo "Finished SIENAX for $SUBJECT_ID. Results saved in $SUBJECT_OUTPUT"
    else
        echo "T1 image not found for $SUBJECT_ID. Skipping..."
    fi
done

echo "SIENAX processing completed for all subjects."


## HOW TO RUN:
 ## 1. Make script executable:
chmod +x run_sienax.sh

 ## 2. In Terminal: run the following command:
       ./run_sienax.sh

