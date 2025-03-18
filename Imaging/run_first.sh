+++++EXAMPLE BASH SCRIPT FOR FIRST +++++

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
# Output directory to store FIRST segmentation results
OUTPUT_DIR="/path/to/first_outputs"
# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# CSV file to save volumes for all subjects
VOLUME_CSV="${OUTPUT_DIR}/first_volumes.csv"

# Initialize CSV file with header
echo "Subject,Structure,Volume_mm3" > "$VOLUME_CSV"

# Loop through each subject folder
for SUBJECT_DIR in "$DATASET_DIR"/sub-*/; do
    # Extract subject ID (e.g., sub-01)
    SUBJECT_ID=$(basename "$SUBJECT_DIR")
    
    # Define input T1-weighted image path
    T1_IMAGE="$SUBJECT_DIR/anat/${SUBJECT_ID}_T1w.nii.gz"
    
    # Check if T1 image exists
    if [[ -f "$T1_IMAGE" ]]; then
        echo "Running FIRST for $SUBJECT_ID ..."
        
        # Define output prefix for the subject
        SUBJECT_OUTPUT="${OUTPUT_DIR}/${SUBJECT_ID}"
        mkdir -p "$SUBJECT_OUTPUT"
        OUTPUT_PREFIX="${SUBJECT_OUTPUT}/${SUBJECT_ID}_first"
        
        # Run FIRST segmentation
        run_first_all -i "$T1_IMAGE" -o "$OUTPUT_PREFIX"
        
        echo "Finished FIRST for $SUBJECT_ID."
        
        # Extract volumes for all segmented structures
        echo "Extracting volumes for $SUBJECT_ID ..."
        
        # Loop through all segmented structures
        for STRUCTURE in ${OUTPUT_PREFIX}-*firstseg.nii.gz; do
            # Get structure name
            STRUCTURE_NAME=$(basename "$STRUCTURE" | sed 's/.*-//; s/_firstseg.nii.gz//')
            
            # Calculate volume in mm^3
            VOLUME=$(fslstats "$STRUCTURE" -V | awk '{print $2}')
            
            # Append volume to CSV file
            echo "${SUBJECT_ID},${STRUCTURE_NAME},${VOLUME}" >> "$VOLUME_CSV"
        done
        
        echo "Volumes extracted for $SUBJECT_ID. Results saved in $VOLUME_CSV"
    else
        echo "T1 image not found for $SUBJECT_ID. Skipping..."
    fi
done

echo "FIRST processing completed for all subjects."


## HOW TO RUN:
 ## 1. Make script executable:
chmod +x run_first.sh

 ## 2. In Terminal: run the following command:
       ./run_first.sh

