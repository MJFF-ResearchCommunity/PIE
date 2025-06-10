import os
import sys
import logging
from pathlib import Path
from typing import List

# Add the parent directory to the Python path to make the pie module importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pie.classification_report import generate_report
from pie.classifier import Classifier
from pie.feature_selector import FeatureSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PIE.test_classifier")


def test_classification_pipeline(**kwargs):
    """
    Tests the complete classification pipeline by calling the main report generation function.
    """
    logger.info("Starting classification pipeline test...")
    try:
        # Pass all keyword arguments to the report generation function
        generate_report(**kwargs)
        logger.info("Classification pipeline test completed successfully.")
    except Exception as e:
        logger.error(f"Classification pipeline test failed: {e}", exc_info=True)
        # In a real test suite, you might want to re-raise or assert for failure
        # For this script, logging the error is sufficient.
        raise


if __name__ == "__main__":
    # Example features that might cause data leakage
    # We must decide whether we want to:
    # 1. Replicate the clinician – build an algorithm that makes the same baseline diagnosis a movement-disorder specialist would make
    # 2. Find objective, pre-diagnostic biomarkers – discover signals that predict PD without relying on the clinical examination
    # 3. Predict future conversion or progression – within Prodromal or within PD
    leakage_features = [
        "PATNO",
        "EVENT_ID",
        "medical_history_Features_of_Parkinsonism_PSGLVL",
        "subject_characteristics_ENRLSRDC",
        "motor_assessments_PDTRTMNT",
        "motor_assessments_PDTRTMNT_x_orig",
        "motor_assessments_PDTRTMNT_y_orig",
        "subject_characteristics_SCREENEDAM4",
        "motor_assessments_NUPSOURC_x_orig",
        "motor_assessments_DBSYN",
        "motor_assessments_DBSYN_x_orig",
        "motor_assessments_DBSYN_y_orig",
        "subject_characteristics_PISTDY",
        "subject_characteristics_APPRDX",
        "motor_assessments_PDMEDYN",
        "motor_assessments_PDMEDYN_x_orig",
        "motor_assessments_PDMEDYN_y_orig",
        "medical_history_Clinical_Diagnosis_NEWDIAG",
        "subject_characteristics_PATHVAR_COUNT",
        "subject_characteristics_ENRLPRKN",
        "subject_characteristics_ENRLLRRK2",
        "subject_characteristics_RNASEQ_VIS",
        "medical_history_Features_of_Parkinsonism_FEATBRADY",
        "subject_characteristics_AV133STDY",
        "biospecimen_standard_files_PLASPNDR",
        "subject_characteristics_chr12:40340400:G:A_A_LRRK2_G2019S_rs34637584",
        "medical_history_Other_Clinical_Features_FEATCLRLEV",
        "biospecimen_standard_files_BSSPNDR",
        "medical_history_PD_Diagnosis_History_DXRIGID",
        "medical_history_Features_of_Parkinsonism_FEATRIGID",
        "subject_characteristics_ENRLPRKN",
        "subject_characteristics_chr12:40310434:C:G_G_LRRK2_R1441G_rs33939927",
        "subject_characteristics_chr1:155235252:A:G_G_GBA_L444P_rs421016",
        "medical_history_Features_of_Parkinsonism_FEATTREMOR",
        "medical_history_Neurological_Exam_CORDRSP",
        "motor_assessments_MSEADLG",
        "non_motor_assessments_PARKISM",
        "subject_characteristics_GAITSTDY",
        "medical_history_PD_Diagnosis_History_DXOTHSX",
        "subject_characteristics_ENRLGBA",
        "subject_characteristics_SV2ASTDY",
        "motor_assessments_NP4TOT_x_orig",
        "non_motor_assessments_NQCOG01",
        "motor_assessments_NQUEX33",
        "biospecimen_standard_files_PLASPNRT",
        "medical_history_Determination_of_Freezing_and_Falls_FRZGT12M",
        "biospecimen_standard_files_BSSPNRT",
        "motor_assessments_NQUEX28",
        "subject_characteristics_chr4:89828149:C:T_T_SNCA_A53T_rs104893877",
        "subject_characteristics_chr1:155235843:T:C_C_GBA_N370S_rs76763715",
        "medical_history_Features_of_Parkinsonism_FEATPOSINS",
        "medical_history_Other_Clinical_Features_FEATDCRARM",
        "biospecimen_standard_files_PLAALQN",
        "medical_history_Other_Clinical_Features_FEATDELHAL",
        "non_motor_assessments_PTCGBOTH_y_orig1",
        "motor_assessments_NQMOB33",
        "subject_characteristics_chr1:155240660:G:GC_GC_GBA_84GG_rs387906315",
        "subject_characteristics_chr12:40320043:G:C_C_LRRK2_R1628P/H_rs33949390",
        # Add any other features that are too closely related to the target
    ]
    
    # Run the classification pipeline test with pre-split data
    test_classification_pipeline(
        train_csv_path="output/selected_train_data.csv",
        test_csv_path="output/selected_test_data.csv",
        use_feature_selection=False,  # Data is already feature-selected
        target_column="COHORT",
        exclude_features=leakage_features,  # NEW PARAMETER
        tune_best_model=False,
        generate_plots=True,
        budget_time_minutes=30.0
    )
