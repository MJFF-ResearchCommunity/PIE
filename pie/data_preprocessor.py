"""
data_preprocessor.py

Methods for cleaning and standardizing data across modalities.
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from pie.data_loader import DataLoader

logger = logging.getLogger(f"PIE.{__name__}")

class DataPreprocessor:
    """
    Handles various data cleaning operations.
    """

    # The screening visit can occur any time between BL and 3 months prior to BL
    EVENT_TIMES = {
            "SC": -3,
            "BL": 0,
            "V01": 3,
            "V02": 6,
            "R01": 6,
            "V03": 9,
            "V04": 12,
            "V05": 18,
            "R04": 18,
            "V06": 24,
            "R06": 30,
            "V07": 30,
            "V08": 36,
            "R08": 42,
            "V09": 42,
            "V10": 48,
            "R10": 54,
            "V11": 54,
            "V12": 60,
            "R12": 66,
            "V13": 72,
            "R13": 78,
            "V14": 84,
            "R14": 90,
            "V15": 96,
            "R15": 102,
            "V16": 108,
            "R16": 114,
            "V17": 120,
            "R17": 126,
            "V18": 132,
            "R18": 138,
            "V19": 144,
            "R19": 150,
            "V20": 156,
            "R20": 162,
            "V21": 168,
    }

    @staticmethod
    def event_id_to_months(eid):
        """
        Convert an EVENT_ID string into number of months into PPMI.
        Applies to scheduled events only, such as BL.
        """
        return DataPreprocessor.EVENT_TIMES.get(eid, np.nan)

    @staticmethod
    def dt_to_datetime(dt_ser):
        """
        Convert a Series of DT strings into a Series of pd.datetimes
        """
        return pd.to_datetime(dt_ser, format="%m/%Y")


    @staticmethod
    def clean(data_dict):
        """
        Clean and standardize the loaded data.

        :param data_dict: Dictionary of data keyed by modality.
        :return: Cleaned data dictionary or combined DataFrame.
        """
        # TODO: Add modalities as actual cleaning logic is implemented.
        data_dict[DataLoader.MEDICAL_HISTORY] = DataPreprocessor.clean_medical_history(data_dict[DataLoader.MEDICAL_HISTORY])
        return data_dict

    @staticmethod
    def clean_medical_history(med_hist_dict):
        # TODO: Add individual column cleaning functions as implemented
        med_hist_dict["Concomitant_Medication"] = DataPreprocessor.clean_concomitant_meds(
                med_hist_dict["Concomitant_Medication"])
        med_hist_dict["Vital_Signs"] = DataPreprocessor.clean_vital_signs(
                med_hist_dict["Vital_Signs"])
        med_hist_dict["Features_of_Parkinsonism"] = DataPreprocessor.clean_features_of_parkinsonism(
                med_hist_dict["Features_of_Parkinsonism"])
        med_hist_dict["General_Physical_Exam"] = DataPreprocessor.clean_gen_physical_exam(
                med_hist_dict["General_Physical_Exam"])
        return med_hist_dict

    @staticmethod
    def clean_vital_signs(vs_df):
        clean_df = vs_df.copy()

        # Convert blood pressures into labelled bands, according to American Heart Association
        # https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings
        def bp(systolic, diastolic):
            if systolic < 120 and diastolic < 80:
                return 0, "Normal"
            if systolic < 130 and diastolic < 80:
                return 1, "Elevated"
            if systolic < 140 or diastolic < 90:
                return 2, "Stage 1 HTN"
            if systolic >= 180 or diastolic >= 120:
                return 4, "Hypertensive crisis"
            return 3, "Stage 2 HTN" # between stage 1 and crisis

        # Supine and standing
        clean_df["Sup BP code"], clean_df["Sup BP label"] = zip(*clean_df.apply(
                lambda row: bp(row["SYSSUP"], row["DIASUP"]), axis=1))
        clean_df["Stnd BP code"], clean_df["Stnd BP label"] = zip(*clean_df.apply(
                lambda row: bp(row["SYSSTND"], row["DIASTND"]), axis=1))

        return clean_df

    @staticmethod
    def clean_features_of_parkinsonism(fop_df, uncertain=0.5):
        clean_df = fop_df.copy()

        # Features are ranked as 0: No, 1: Yes, 2: Uncertain.
        # Convert the Uncertain values to something more conducive for analysis
        for feat in ["FEATBRADY", "FEATPOSINS", "FEATRIGID", "FEATTREMOR"]:
            clean_df[feat] = clean_df[feat].apply(lambda v: v if v != 2 else uncertain)

        return clean_df

    @ staticmethod
    def clean_gen_physical_exam(gpe_df, uncertain=0.5):
        clean_df = gpe_df.copy()

        # Abnormality is ranked as 0: No, 1: Yes, 2: Cannot assess.
        # Convert the uncertain values to something more conducive for analysis
        clean_df["ABNORM"] = clean_df["ABNORM"].apply(lambda v: v if v != 2 else uncertain)

        return clean_df

    @staticmethod
    def clean_concomitant_meds(concom_meds_df):
        clean_df = concom_meds_df.copy()

        # Clean the start and stop dates
        clean_df["STARTDT"] = DataPreprocessor.dt_to_datetime(clean_df["STARTDT"])
        clean_df["STOPDT"] = DataPreprocessor.dt_to_datetime(clean_df["STOPDT"])

        # Not all have start date: assume prior to PPMI enrollment.
        # Not all have stop date: assume still on medication as of date of last visit.
        logger.info(f"There are {clean_df['STARTDT'].isnull().sum()} concomitant medication "
                    f"entries with no start date.")
        logger.info(f"There are {clean_df['STOPDT'].isnull().sum()} concomitant medication "
                    f"entries with no stop date.")

        # All concomitant meds entries have either a code in CMINDC or text in CMIND_TEXT
        # (a handful have neither). The code is useful to indicate PD comorbidity/symptom.
        # Those text entries can be mapped to a code, for a completely coded dataset.
        with open(f"{Path(__file__).parent}/concomitant_meds_indications.json", "r") as f:
            raw = json.load(f)
        indications = dict((int(k), v) for k, v in raw["indications"].items())
        text_map = dict((k, int(v)) for k, v in raw["text_mappings"].items())
        indc, textc = "CMINDC", "CMINDC_TEXT"

        # Algorithm: if a code exists, use it. If not, inspect the text for mappable values.
        def unpack_cm(row):
            # Code exists: simple case
            if not pd.isnull(row[indc]):
                return row[indc]

            # Uh-oh: a few have neither code nor text. Look at the treatment name.
            if pd.isnull(row[textc]):
                if row["CMTRT"] == "ASPIRIN":
                    return 17 # Pain
                elif row["CMTRT"] == "GINKOBIL":
                    return 22 # Supplements
                elif row["CMTRT"] == "HUMULIN NPH":
                    return 11 # Diabetes
                logger.debug(f"Found concomitant med with only CMTRT. Mapping to Other: '{row['CMTRT']}'")
                return 25 # Other

            text = row[textc].strip().lower()

            # Look up the text mappings
            if text in text_map:
                return text_map[text]

            # Shouldn't hit this condition
            return np.nan

        clean_df[indc] = clean_df.apply(unpack_cm, axis=1)
        nulls = clean_df[indc].isnull().sum()
        if nulls > 0:
            logger.warning(f"{nulls} concomitant meds haven't mapped to codes!")
        else:
            clean_df[indc] = clean_df[indc].astype(int)

        # Next, map codes to text
        clean_df[textc] = clean_df[indc].apply(
                lambda code: indications[int(code)] if not pd.isnull(code) else "UNKNOWN")
        return clean_df

    @staticmethod
    def create_concomitant_meds(med_hist_df):
        """
        This is for creating the concomitant meds table in the first place! Only needs to be run
        when there is a new issue of PPMI data which adds entries to the CM table.
        """
        clean_df = med_hist_df.copy()

        indications = {
            1: "Anxiety",
            10: "Depression",
            11: "Diabetes",
            12: "GERD",
            13: "Hyperlipidemia",
            14: "Hypertension",
            15: "Insomnia",
            16: "Nausea",
            17: "Pain",
            18: "REM-Behavior Disorder",
            19: "Restless Leg Syndrome",
            2: "Atrial Fibrillation / Arrhythmias",
            20: "Sexual Dysfunction",
            21: "Sialorrhea / Drooling",
            22: "Supplements / Homeopathic Medication",
            23: "Thyroid Disorder",
            24: "Vitamins / Coenzymes",
            25: "Other",
            3: "Benign Prostatic Hypertrophy / Overactive Bladder",
            4: "Cognitive Dysfunction",
            5: "Congestive Heart Failure",
            6: "Constipation",
            7: "Coronary Artery Disease, Peripheral Artery Disease, Stroke",
            8: "Daytime Sleepiness",
            9: "Delusions, Hallucination, Psychosis"
        }

        text_map = {}
        textc = "CMINDC_TEXT"

        def map_text(text):
            # Sometimes the correct term is actually in the text
            match = [[code, val] for code, val in indications.items() if text in val.lower() and val != "Other"]
            # Set up two mappings of terms: precise matches, and fuzzy matches (in text)
            precise = {
                1: ["anxiousity", "anxious restlessness", "fears", "aniexty", "anciety", "axiety",
                    "anxiety and depression", "anti-anxiety", "mild anxiety", "parkinson's disease, anxiety",
                    "generalized anxiety disorder", "anxiety/depression", "anxiety disorder", "panic attacks",
                    "anxiety/ depression", "anxiety/depression disorder", "anxiety depressive disorder",
                    "anxiety & depression", "anxious depression", "anxiety depression disorder",
                    "anxiety/ depression disorder", "anxiety / depression disorder", "anxiety/depression/pain",
                    "situational anxiety/depression", "anxiety, insomnia", "anxiety.insomnia", "anxiety and pain",
                    "anxiety disorder/insomnia disorder", "panic attack disorder"],
                2: ["bradycardia", "arrythmia", "a-fib", "cardiac arrhythmias", "cardiac arrythmia",
                    "cardiac arrhythmia", "atrial flutter", "atrial fibrillation/hypertension",
                    "anticoagulant to treat episode of atrial fibrillation", "heart arithmia"],
                3: ["urinaty incontinence", "bladder control", "enlarge prostrate", "spastic bladder", "bph",
                    "urinary frquency", "urinary dysfunction", "nocturia", "incontinence", "urge incontinence",
                    "hyperreflexic bladder", "hypereflexic bladder", "benign prostatic hypertrrophy", "urgency",
                    "benign, prostatic, hyperplasia", "incontenence", "hypetrophy of prostate gland",
                    "benign hypertophy of the prostate gland", "bladder dysfunction", "urinary disturbances",
                    "prostats hyperplasia", "prostata hyperplasia", "benighn prostatic hypertrohy", "bladderspasms",
                    "reduce urination frequency", "hyper active bladder", "urge to urinate", "prostatic syndrome",
                    "bladder urgency", "urinary incontinenc e", "urinary tract/incontinence", "urination",
                    "benigne prostatahyperplasia", "urinary flow/frequency", "benign prostae hypertrophy",
                    "benign prostatic hyperplasia", "unstable bladder", "urinary problem", "bph - self diagnosed",
                    "urinary problems"],
                4: ["memory", "short term memory", "short-term memory", "dementia", "cognition",
                    "mental awareness", "pd dementia"],
                5: ["heart insufficiency"],
                6: ["stool softener", "stool softner", "consitpation", "constipaton", "colon regularity",
                    "constiaption", "consipation", "constipation/diverticulosis", "intermittent constipation",
                    "supplement/help digestion", "gi prophylaxis", "stool softener and laxative",
                    "constipation supplement", "bowel movement irregularity", "ibs"],
                7: ["coronar artery disease", "coronery artery disease", "cornary artery disease",
                    "coronory artery disease", "cardiac disease", "atheroscleosis", "carotide thickening",
                    "artery plaque build-up", "artery/plaque build up", "carotid atheroma", "angina",
                    "prizemntal's angina", "chest angina", "myocardial infarction", "mycardial infarction",
                    "mycardial iinfarction", "heartinfarction", "cardiac infarction", "transient ischemic attack",
                    "transient schemic attach", "tia", "transiet ischemic attack", "ischemic caridiopathy",
                    "cerebral artery blockage", "ischemic heart disease", "ischaemic heart disease",
                    "heart disease", "coronary disease", "cardiovascular disease", "ischemic cardiopathy",
                    "schemic cardiopathy", "cardiopathy", "heart attack", "ictus", "cardiac syndrome x",
                    "cardiopathy-high blood pressure", "cholesterol deposit in the carotid artery",
                    "stroke secondary profilaxia", "heart attack (with stent implantation",
                    "ischemic cardiopathy-coronary stent", "ischemic stroke secondary prevention"],
                8: ["excessive daytime fatigue", "increase energy", "daytime somnolence", "somnolence",
                    "wakefulness-promoting", "narcolepsy", "daytime drowsiness", "excessive daytime sleepness",
                    "drowsiness from depression meds", "fatigue secondary to parkinson's disease"],
                9: ["antipsychotic", "anti-psychotic", "psycotic symptoms", "paranoia", "halluzination"],
                10: ["deprsseion", "depressopm", "anitdepressant", "depressive symptoms", "sad", "dystthemia",
                     "depressioon", "deprression", "depresion", "anti- depressant", "fibromyalgia and mood",
                     "depression/insomnia/appetite", "depressive mood", "depression & anxiety", "anti-depressant",
                     "antidepressant", "anti depressant", "insomnia due to depression", "mood", "mood disorder",
                     "depression/anxiety", "depression + anxiety", "mild depression/anxiety",
                     "depression and anxiety", "depression / anxiety state", "depression, anxiety",
                     "depression anxiety disorder", "depression-anxiety disorder", "depression, anxiety disorder",
                     "depression/anxiety disorder", "depression / anxiety disorder", "depressive anxiety disorder",
                     "depressive/anxiety disorder", "depression, headaches"],
                11: ["diabetis mellitus", "dm", "dm type 2", "diabetis", "insulin resistance syndrome",
                     "diabetes type 1.5", "diabetes mellitus", "type 2 diabetes"],
                12: ["eosinophilic esophagitis", "digestion", "inidgestion", "barrette's espohagus",
                     "barrett's esophagus", "barrett's disease", "barrets disease", "barretts esphageous",
                     "barrett's syndrome esophagitis", "barretts esophagus", "heart burn", "heartburn",
                     "acid refulx", "gord", "dyspepsic", "anti acid", "anti-acid", "antacid", "gerd/heartburn",
                     "reflux/hiatal hernia", "hiatal hernia", "stomach protection", "reflux", "hyatal hernia",
                     "hiatus hernia"],
                13: ["cholesterol", "elevated cholesterol", "elevated cholestrol", "high cholesterol",
                     "high cholestrole", "high choloesterol", "high cholersterol", "increased cholesterol",
                     "reduce cholesterol", "hyperchloesterolemia", "hyperchlosterolemia", "hypercholeterolemia",
                     "hpercholesterolemia", "hypercholsterolemia", "hyperchlolesterolemia", "hyper cholestorolemy",
                     "hyper-cholesterolinemia", "cholesterol lowering", "cholesterolemia", "dyslipidemia",
                     "dislipedemia", "dyslipemia", "dislipemia", "dlp", "hyperlipidema", "hyperlipdema",
                     "hyperlipemia", "high tryglicerides", "hypercholesterolemia", "hypercholesteremia",
                     "high cholestrol"],
                14: ["htn", "high blood presure", "hi gh blood pressure", "high blood preasure", "high bp",
                     "high bloodpressure", "high blood preassure", "blood pressure", "blood presure",
                     "hyperstension", "hyperstesion", "hyptertension", "hpertension", "hypertesion",
                     "hypertenison", "hypertensin", "hypertention", "hypertensjon", "hipertension", "hypertenstion",
                     "hypertendion", "blood thinner", "blood pressure control", "blood pressue elevation",
                     "lower renal artery blood pressure", "bp regulation", "hypertension/sparing",
                     "arterial hypertension", "art. hypertension", "high blood pressure", "heart condition"],
                15: ["sleep aid", "sleep aide", "sleeping aid", "sleepin aid", "trouble sleeping", "sleep problems",
                     "sleeping problems", "sleep promotion", "sleep health promotion", "sleep issues",
                     "sleep disturbance", "sleep disturbances", "sleep distrubance", "disturbed sleep",
                     "sleep disorder", "sleeping disorder", "sleep difficulty", "sleepdifficulties",
                     "sleep deficit", "sleep assistance", "hormonal therapy for sleeping", "helps to sleep",
                     "healthy sleep", "for sleep", "sleep better", "regulate sleep", "sleepers", "insomia",
                     "insomina", "insominia", "nighttime wakefulness", "insomnia treatment", "insomnia, anxiety",
                     "intermittent sleep disturbance", "insomnia/anxiety/depression disorder", "sleep/depression",
                     "sleep/ anxiety", "sleep/anxiety", "sleep and anxiety", "insomnia/anxiety", "sleeping",
                     "insomnia and anxiety", "parkinson's, sleep issues, anti-anxiety, aches and pains",
                     "insomnia, depressive symptoms", "sleep disorder and depression", "sleep deficit disorder",
                     "occassional trouble sleeping", "occasional sleep"],
                16: ["gi", "antiemetic", "sickness", "gastroparesis", "diverticulitis",
                     "to calm gastric secondary effect of levodopa", "nausea caused by azacitidine"],
                17: ["low back injury", "muscular discomofort", "neurolagia", "mirgraine", "migrines", "migrane",
                     "migranes", "prolonged migranes", "siactica", "back sorness", "muscle soreness", "arthrosis",
                     "peripheral neuropathy", "right shoulder tendonitis", "fibromyalgia", "neuropathy",
                     "intermittent low back strain", "mirgraines and menorrhagia", "lumbalgia and ciathalgia",
                     "right shoulder injury", "arthiritis", "pulled back muscle", "lt ankle ligament tear",
                     "post lumbar puncture event", "carpal tunnel syndrome", "l5 herniated disc", "oain relief",
                     "tenderness at lp site", "whiplash", "tenderness at lumbar puncture site", "arthrisits",
                     "torn ac, r", "lumbar spinal disc injury", "tendonitis in bilateral hands",
                     "deep brain stimulation post-op", "pinched neck nerve", "right knee meniscus tear",
                     "post lumbar puncture", "partial tear of rotator cuff - bilateral"],
                18: ["rbd", "rapid eye movement", "sleepiniess - rbd", "rem behavour disorder",
                     "rem sleep behavior disorder", "insomnia & rbd", "insomnia+rbd",
                     "insomnia & rem sleep behavior disorder", "insomnia and probable rem sleep behavior disorder"],
                19: ["rls", "suspected rls", "resltess leg syndrome", "night time leg cramps"],
                20: ["sexual enhancement", "hypogonadal", "e.d.", "eredtile dysfunction", "eretile dysfunction",
                     "erectilce dysfunction", "vaginal pain",
                     "erectile dysfunction & benign prostatic hyperplasia"],
                21: [],
                22: ["supplement (eye sight)", "supplment", "iron defficiency", "anemia", "anaemia", "low kalium",
                     "food supplement", "recurrent urinary tract infections", "nutritional supplement",
                     "supplemental", "bone health", "herbal supplement", "wellness", "prophylaxis of cramps",
                     "health", "homeopathic supplement for vitamine d", "health supplement", "prophylaxis",
                     "homeopathic against hypercholesterinemia", "homeopathic against hypertension",
                     "low hdl level", "leg cramps", "high homocysteine", "preventative", "osteoporosis prevention",
                     "muscle cramps", "blood health", "mild anemia"],
                23: ["hypothryroidism", "hypothroidism", "hypothyrodism", "hypothryroid", "hypothryoid",
                     "hypothryoidism", "hyperthyoidism", "hypothiroiism", "hypothyoidism", "hyperthroidsm",
                     "hyperparathyroidism", "throid block", "thyrid block", "thryroid block", "throid blockade",
                     "thryoid blockade", "hyothyreosis", "hypthreosis", "thryoid block"],
                24: ["b-12 defficiency", "b12 defiecency", "b12 defficiency", "bone health, vit d", "tired",
                     "d deficiency", "vitamin b deficiency", "dietary supplement", "general health",
                     "health promotion", "breast health", "osteoporosis", "health prophylaxis", "welllness",
                     "health maintenance", "good health", "health wellness", "sluggishness/general health",
                     "general health maintenance", "vitamin deficeincy", "b12 deficiency", "alimentary supl.",
                     "vitamin supplement", "vitamin d3 supplement", "vit d supplement", "vitmain supplement",
                     "epiretinal membrane", "sluggishness/ general health", "preventative health", "osteopenia",
                     "pd health promotion", "general heatlh", "kidney protection", "prevention of muscle cramps",
                     "mitochondrial support", "sleep / health prophylaxis", "polymyalgia rheumatica",
                     "prophylaxis of osteoporosis", "bone loss", "dopamine promotion", "osteoporosis prophylaxis",
                     "bone density", "bone thinning in hips", "general health promotion", "keep ongoing, fatigue",
                     "fatigue, and good health", "bones health", "transaminitis"],
                25: ["hormonal supplement"],
            }

            fuzzy = {
                1: ["anxiety", "stress", "agitation", "claustrophobia", "anxiolytic", "panic disorder"],
                2: ["atrial", "tachycard", "bradycardia", "arrhythmia", "arrythmia", "dysrhythmia",
                    "fibrillation", "heart rhythm", "heart rythm", "a-fib"],
                3: ["urinary incont", "urinary urgency", "urine frequency", "urinary frequency",
                    "frequent urination", "prostate hyper", "prostatic hyper", "prostatis hyper",
                    "prostrate hypertrophy", "prostate enlargement", "enlarged prostat", # -e, -a
                    "benign prostate", "overactive bladder", "hyperactive bladder", "irritable bladder",
                    "bladder spasm", "hypertrophy of the prostate", "hypertrophy of prostate", "urine leak",
                    "hypertrophy prostate", "neurogenic bladder", "bladder weakness", "urinary leak"],
                4: ["cognitive", "congitive enhancement", "memory", "lewy body"],
                5: ["heart failure"],
                6: ["constipation", "laxative", "obstipation"],
                7: ["coronary vascular", "coronary heart", "coronary artery", "angina pectoris"],
                8: [],
                9: ["hallucination", "psychosis", "psychotic"],
                10: ["depression", "anti-depressant", "antidepressant", "anti depressant", "anti deppressant",
                     "depressive", "dysthymia", "dysthmia", "seasonal affective", "s.a.d.", "depressed mood",
                     "mood swing", "mood -boosts lexapro"],
                11: ["diabetes", "antidiabetic"],
                12: ["reflux", "indigestion", "dyspepsia", "dispepsia", "heartburn"],
                13: ["hypercholest", "hyper-cholest", "high cholest", "hypertriglycer", "hypertriglicer",
                     "high triglycer", "high triglicer", "elevated triglycer", "hyperlipidemia",
                     "cholesterol-lowering"],
                14: ["high blood pressure", "anticoagula", "anti coagula", "hypertensi"],
                15: ["insomnia", "sleep aid"],
                16: ["vomit", "nausea", "anti-emetic", "sea sick", "motion sick"],
                17: ["pain", "discomfort", "migraine", "headache", "sciatic", "radiculopathy", "neuropathy",
                     "back strain", "back spasm", "spasms in back", "back muscle spasm", "back muscle sore",
                     "sore back", "back ache", "backache", "back soreness", "body ache", "intermittent aches",
                     "soreness", "muscular aches", "joint ache", "nerve tingling", "meralgia", "lumbago",
                     "nerve compression", "generalized aches", "muscle spasm", "menstrual cramp", "arthrit",
                     "analgesi", "headahce", "pinched nerve", "fibromyalgia", "back injury", "vulvodemia",
                     "carotidynia", "muscle ache", "pian relief"],
                18: ["rem sleep", "rem-sleep", "rem behavior", "rem behaviour"],
                19: ["restless leg", "restless sleep aid"],
                20: ["peyronie", "erectile", "hypogonadism", "impotence", "vaginal dryness", "vaginal atrophy",
                     "atrophic vagin", "atropic vagin", "decrease vaginal wall", "vaginal driness"],
                21: ["salivation", "chronic drooling"],
                22: ["iron deficiency", "iron substitution", "ferritin deficiency", "lack of ferrum", "low iron",
                     "low in iron", "low ferritin", "magnesium deficiency", "low potassium", "low k+",
                     "hypokalemia", "hypocalcemia", "low calcium", "thiamine deficiency", "folic acid deficiency",
                     "folic acid substitution", "low folic acid", "folacid deficiency", "folate deficiency",
                     "supplement", "suppliment", "suppliement", "supplememt", "supplemement", "suppplement",
                     "supplment", "suplement", "supllement", "alimentary supl", "homeopath", "homoeopatic",
                     "probiotic", "coq10", "homoepatic"],
                23: ["thyroid", "hypothyreos", "hyperthyreos", "basedow", "hashimoto"],
                24: ["vitamin", "b12 deficiency", "b12 defieciency", "low b 12", "low b12", "low vit d",
                     "vit d deficiency", "vit. e deficiency", "lack of vit b12", "bone density",
                     "enhancing intestinal absorption", "antioxidant that protects body", "impact his pd symptoms"],
            }

            ## Now we can try to map the text

            # Sometimes the text is so short, it would accidentally map to another category
            # so do these ones first
            if text in ["ed"]:
                text_map[text] = 20 # Sexual dysfunction
            elif text in ["ert"]:
                text_map[text] = 24 # Vitamins / Coenzymes
            elif text in ["add"]:
                text_map[text] = 25 # Other
            # Now check for text matches to the indication names
            elif len(match) > 0:
                logger.debug(f"Mapping '{text}' to '{match[0][1]}'")
                text_map[text] = match[0][0] # Store code only
            # Now look at the typos and different names
            else:
                for code in sorted(precise):
                    if text in precise[code]:
                        logger.debug(f"Mapping precise '{text}' to {indications[code]}")
                        text_map[text] = code

                    # If we get here, there was no match
                for code in sorted(fuzzy):
                    if text not in text_map:
                        for fragment in fuzzy[code]:
                            if fragment in text:
                                logger.debug(f"Mapping '{text}' to {indications[code]}")
                                text_map[text] = code
                                break # inner for loop: check again for text in text_map


            return text_map.get(text, None)


        ### Look for text that needs to be mapped to an indication
        unmapped = set()
        for i, row in clean_df.iterrows():
            # Code exists: we can ignore the text
            if not pd.isnull(row["CMINDC"]):
                continue

            # A few have neither code nor text. Ignore for now 
            if pd.isnull(row[textc]):
                continue

            # Harmonize the text formatting
            text = row[textc].strip().lower()

            # Might be a duplicate from another row: no need to reanalyze
            if text in text_map:
                continue

            code = map_text(text)

            # Is it still unmapped?
            if not code:
                unmapped.add((text, row["CMTRT"]))

        logger.debug(f"Number of unmapped codes considered as Other: {len(unmapped)}")
        for text, trt in unmapped:
            text_map[text] = 25

        with open("pie/concomitant_meds_indications.json", "w") as f:
            logger.info(f"text_map contains {len(text_map)} entries")
            json.dump({"indications": indications, "text_mappings": text_map}, f)

