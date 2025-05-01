"""
constants.py

Shared constants and types used across the PIE framework.
"""

# Define constants for modality names to ensure consistency
SUBJECT_CHARACTERISTICS = "subject_characteristics"
MEDICAL_HISTORY = "medical_history"
MOTOR_ASSESSMENTS = "motor_assessments"
NON_MOTOR_ASSESSMENTS = "non_motor_assessments"
BIOSPECIMEN = "biospecimen"

# Define all available modalities
ALL_MODALITIES = [
    SUBJECT_CHARACTERISTICS,
    MEDICAL_HISTORY,
    MOTOR_ASSESSMENTS,
    NON_MOTOR_ASSESSMENTS,
    BIOSPECIMEN
]

# Define folder paths for each modality
FOLDER_PATHS = {
    SUBJECT_CHARACTERISTICS: "_Subject_Characteristics",
    MEDICAL_HISTORY: "Medical_History",
    MOTOR_ASSESSMENTS: "Motor___MDS-UPDRS",
    NON_MOTOR_ASSESSMENTS: "Non-motor_Assessments",
    BIOSPECIMEN: "Biospecimen"
} 