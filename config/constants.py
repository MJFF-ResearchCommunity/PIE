"""
Central configuration file for constants used across the PIE project.
"""

# This list represents features that are likely to cause data leakage
# and should be excluded from the modeling process.
LEAKAGE_FEATURES = [
    "PATNO", "EVENT_ID", "medical_history_Features_of_Parkinsonism_PSGLVL",
    "subject_characteristics_ENRLSRDC", "motor_assessments_PDTRTMNT",
    "motor_assessments_PDTRTMNT_x_orig", "motor_assessments_PDTRTMNT_y_orig",
    "subject_characteristics_SCREENEDAM4", "motor_assessments_NUPSOURC_x_orig",
    "motor_assessments_DBSYN", "motor_assessments_DBSYN_x_orig",
    "motor_assessments_DBSYN_y_orig", "subject_characteristics_PISTDY",
    "subject_characteristics_APPRDX", "motor_assessments_PDMEDYN",
    "motor_assessments_PDMEDYN_x_orig", "motor_assessments_PDMEDYN_y_orig",
    "medical_history_Clinical_Diagnosis_NEWDIAG", "subject_characteristics_PATHVAR_COUNT",
    "subject_characteristics_ENRLPRKN", "subject_characteristics_ENRLLRRK2",
    "subject_characteristics_RNASEQ_VIS", "medical_history_Features_of_Parkinsonism_FEATBRADY",
    "subject_characteristics_AV133STDY", "biospecimen_standard_files_PLASPNDR",
    "subject_characteristics_chr12:40340400:G:A_A_LRRK2_G2019S_rs34637584",
    "medical_history_Other_Clinical_Features_FEATCLRLEV", "biospecimen_standard_files_BSSPNDR",
    "medical_history_PD_Diagnosis_History_DXRIGID", "medical_history_Features_of_Parkinsonism_FEATRIGID",
    "subject_characteristics_chr12:40310434:C:G_G_LRRK2_R1441G_rs33939927",
    "subject_characteristics_chr1:155235252:A:G_G_GBA_L444P_rs421016",
    "medical_history_Features_of_Parkinsonism_FEATTREMOR", "medical_history_Neurological_Exam_CORDRSP",
    "motor_assessments_MSEADLG", "non_motor_assessments_PARKISM",
    "subject_characteristics_GAITSTDY", "medical_history_PD_Diagnosis_History_DXOTHSX",
    "subject_characteristics_ENRLGBA", "subject_characteristics_SV2ASTDY",
    "motor_assessments_NP4TOT_x_orig", "non_motor_assessments_NQCOG01",
    "motor_assessments_NQUEX33", "biospecimen_standard_files_PLASPNRT",
    "medical_history_Determination_of_Freezing_and_Falls_FRZGT12M", "biospecimen_standard_files_BSSPNRT",
    "motor_assessments_NQUEX28", "subject_characteristics_chr4:89828149:C:T_T_SNCA_A53T_rs104893877",
    "subject_characteristics_chr1:155235843:T:C_C_GBA_N370S_rs76763715",
    "medical_history_Features_of_Parkinsonism_FEATPOSINS",
    "medical_history_Other_Clinical_Features_FEATDCRARM", "biospecimen_standard_files_PLAALQN",
    "medical_history_Other_Clinical_Features_FEATDELHAL", "non_motor_assessments_PTCGBOTH_y_orig1",
    "motor_assessments_NQMOB33", "subject_characteristics_chr1:155240660:G:GC_GC_GBA_84GG_rs387906315",
    "subject_characteristics_chr12:40320043:G:C_C_LRRK2_R1628P/H_rs33949390",
] 