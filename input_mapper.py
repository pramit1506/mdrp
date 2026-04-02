"""
input_mapper.py
===============
Converts real-world patient data (basic info + blood test values) into
model-ready feature vectors with SMART defaults for all clinical features.

Extended in v2 to support:
  - HbA1C (Glycated Haemoglobin) — diabetes model + clinical score
  - LDL, HDL, Triglycerides      — heart clinical score + HM model
  - eGFR                         — kidney clinical score
  - MCV (Mean Corpuscular Volume) — health markers anemia detection

Design philosophy
-----------------
1. DERIVE values where a formula exists
   - thalach  = 208 − (0.7 × age)           [Tanaka et al. 2001]
   - fbs      = 1 if glucose > 120 else 0
   - htn      = 1 if systolic_bp ≥ 140 else 0
   - dm       = 1 if any glucose ≥ 200 else 0
   - ane      = 1 if Hb < 12.0 (F) or 13.0 (M)
   - egfr     = CKD-EPI approximation if only creatinine + age + sex available

2. Use dataset-mean values for non-measurable features
   - skin (PIMA mean ≈ 23 mm, sex-adjusted)
   - dpf  (PIMA mean ≈ 0.472)
   - pcv, wbcc, rbcc (UCI CKD means)

3. Use "healthy-normal" values for clinical exam features
   - cp=0, restecg=0, exang=0, oldpeak=0.0, slope=1, ca=0, thal=2
   - rbc=1 (normal), pc=1 (normal), pcc=0, ba=0, cad=0, appet=1, pe=0
"""

import numpy as np
import joblib
import os


# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE LAB-REPORT → MODEL-FEATURE NAME MAP
# All keys should be lowercase for safe matching; the lookup normalizes before use.
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_MAP = {
    # ── Blood Glucose ─────────────────────────────────────────────────────────
    "fbs":                       "glucose",
    "fasting blood sugar":       "glucose",
    "fasting blood glucose":     "glucose",
    "fasting plasma glucose":    "glucose",
    "fpg":                       "glucose",
    "fbg":                       "glucose",
    "fasting sugar":             "glucose",
    "blood glucose":             "glucose",
    "glucose fasting":           "glucose",
    "blood_glucose":             "glucose",

    "rbs":                       "bgr",
    "ppbs":                      "bgr",
    "post prandial blood sugar": "bgr",
    "post prandial glucose":     "bgr",
    "post prandial plasma glucose": "bgr",
    "random blood glucose":      "bgr",
    "random blood sugar":        "bgr",
    "post meal glucose":         "bgr",
    "glucose post prandial":     "bgr",
    "2-hour post prandial":      "bgr",

    # ── HbA1C ────────────────────────────────────────────────────────────────
    "hba1c":                     "hba1c",
    "hba1c (glycated haemoglobin)": "hba1c",
    "hba1c-glycated haemoglobin": "hba1c",
    "hemoglobin a1c":            "hba1c",
    "haemoglobin a1c":           "hba1c",
    "hgba1c":                    "hba1c",
    "a1c":                       "hba1c",
    "glycated haemoglobin":      "hba1c",
    "glycosylated hemoglobin":   "hba1c",
    "glycated hb":               "hba1c",

    # ── Blood Pressure ────────────────────────────────────────────────────────
    "systolic bp":               "trestbps",
    "systolic_bp":               "trestbps",
    "systolic blood pressure":   "trestbps",
    "sbp":                       "trestbps",
    "trestbps":                  "trestbps",

    "diastolic bp":              "bloodpressure",
    "diastolic_bp":              "bloodpressure",
    "diastolic blood pressure":  "bloodpressure",
    "dbp":                       "bloodpressure",
    "blood pressure":            "bloodpressure",
    "bp":                        "bp",  # kidney model uses 'bp'

    # ── Lipid Profile ─────────────────────────────────────────────────────────
    "total cholesterol":         "chol",
    "serum cholesterol":         "chol",
    "cholesterol":               "chol",
    "tc":                        "chol",
    "t. chol":                   "chol",

    "ldl":                       "ldl",
    "ldl-cholesterol":           "ldl",
    "ldl cholesterol":           "ldl",
    "ldl- cholesterol":          "ldl",
    "low-density lipoprotein":   "ldl",
    "ldl-c":                     "ldl",

    "hdl":                       "hdl",
    "hdl-cholesterol":           "hdl",
    "hdl cholesterol":           "hdl",
    "hdl- cholesterol":          "hdl",
    "high-density lipoprotein":  "hdl",
    "hdl-c":                     "hdl",

    "triglycerides":             "triglycerides",
    "tg":                        "triglycerides",
    "trig":                      "triglycerides",
    "serum triglycerides":       "triglycerides",

    # ── Kidney Markers ────────────────────────────────────────────────────────
    "serum creatinine":          "sc",
    "s. creatinine":             "sc",
    "creatinine":                "sc",
    "scr":                       "sc",
    "blood creatinine":          "sc",

    "blood urea":                "bu",
    "blood urea nitrogen":       "bu",
    "bun":                       "bu",
    "urea":                      "bu",
    "serum urea":                "bu",
    "serum urea nitrogen":       "bu",
    "bun (blood urea nitrogen)": "bu",

    "sodium":                    "sod",
    "na":                        "sod",
    "serum sodium":              "sod",

    "potassium":                 "pot",
    "k":                         "pot",
    "serum potassium":           "pot",

    "egfr":                      "egfr",
    "estimated gfr":             "egfr",
    "glomerular filtration rate": "egfr",
    "egfr (estimated glomerular filtration rate)": "egfr",
    "estimated glomerular filtration rate": "egfr",

    # ── CBC / Haematology ─────────────────────────────────────────────────────
    "hemoglobin":                "hemo",
    "haemoglobin":               "hemo",
    "hb":                        "hemo",
    "hgb":                       "hemo",
    "hgb (hemoglobin)":          "hemo",

    "pcv":                       "pcv",
    "haematocrit":               "pcv",
    "hematocrit":                "pcv",
    "hct":                       "pcv",
    "packed cell volume":        "pcv",

    "wbc":                       "wbcc",
    "tlc":                       "wbcc",
    "total leucocyte count":     "wbcc",
    "white blood cell count":    "wbcc",
    "leukocyte count":           "wbcc",
    "wbc count":                 "wbcc",

    "rbc count":                 "rbcc",
    "red blood cell count":      "rbcc",
    "erythrocyte count":         "rbcc",

    "mcv":                       "mcv",
    "mean corpuscular volume":   "mcv",
    "mean cell volume":          "mcv",

    # ── Urine Examination ─────────────────────────────────────────────────────
    "specific gravity":          "sg",
    "urine sg":                  "sg",
    "sp. gr.":                   "sg",

    "urine albumin":             "al",
    "urine protein":             "al",

    "urine sugar":               "su",
    "urine glucose":             "su",

    "urine rbc":                 "rbc",
    "red cells urine":           "rbc",
    "pus cells":                 "pc",
    "wbc urine":                 "pc",
    "pus cell clumps":           "pcc",
    "bacteria":                  "ba",
    "bacteriuria":               "ba",

    # ── Clinical History (categorical) ────────────────────────────────────────
    "hypertension":              "htn",
    "htn":                       "htn",
    "diabetes mellitus":         "dm",
    "dm":                        "dm",
    "coronary artery disease":   "cad",
    "cad":                       "cad",
    "ihd":                       "cad",
    "appetite":                  "appet",
    "pedal edema":               "pe",
    "ankle swelling":            "pe",
    "anemia":                    "ane",
    "anaemia":                   "ane",

    # ── Demographics / Basic ─────────────────────────────────────────────────
    "age":                       "age",
    "bmi":                       "bmi",
    "body mass index":           "bmi",
    "insulin":                   "insulin",
    "skin thickness":            "skin",
    "pregnancies":               "preg",
    "dpf":                       "dpf",
}

# Lowercase version for safe (case-insensitive) lookup
_FEATURE_MAP_LOWER = {k.lower(): v for k, v in FEATURE_MAP.items()}


def normalize_key(raw_key: str) -> str:
    """Normalize a raw lab report key to the canonical model feature name."""
    key_lower = str(raw_key).lower().strip()
    return _FEATURE_MAP_LOWER.get(key_lower, key_lower)


# ─────────────────────────────────────────────────────────────────────────────
# APPROXIMATE eGFR USING CKD-EPI (when only creatinine + age + sex available)
# ─────────────────────────────────────────────────────────────────────────────
def _estimate_egfr(sc: float, age: float, sex_male: bool) -> float:
    """
    Simplified CKD-EPI approximation.
    sc       : serum creatinine mg/dL
    age      : years
    sex_male : True = male
    Returns  : eGFR mL/min/1.73m²
    """
    if sc <= 0 or age <= 0:
        return 100.0  # default = normal kidney function

    kappa = 0.9 if sex_male else 0.7
    alpha = -0.411 if sex_male else -0.329
    sex_factor = 1.0 if sex_male else 1.012

    ratio = sc / kappa
    if ratio < 1.0:
        egfr = 141 * (ratio ** alpha) * (0.993 ** age) * sex_factor
    else:
        egfr = 141 * (ratio ** -1.209) * (0.993 ** age) * sex_factor

    return round(min(max(egfr, 0.0), 200.0), 1)


# ─────────────────────────────────────────────────────────────────────────────
# SMART DEFAULTS — physiologically derived or dataset means
# ─────────────────────────────────────────────────────────────────────────────
def compute_smart_defaults(
    age:         float,
    glucose:     float,
    sex_male:    bool  = True,
    systolic_bp: float = 120.0,
    hemo:        float = None,
    bgr:         float = None,
    sc:          float = None,
) -> dict:
    """
    Returns physiologically DERIVED + dataset-mean defaults for features
    not typically present in a standard blood test.

    Parameters
    ----------
    age         : patient age (years)
    glucose     : fasting blood glucose (mg/dL)
    sex_male    : True = male
    systolic_bp : systolic BP mmHg
    hemo        : hemoglobin g/dL (None = unknown)
    bgr         : random blood glucose mg/dL (None = unknown)
    sc          : serum creatinine mg/dL (None = unknown)

    Returns
    -------
    dict : feature_name → derived/default value
    """
    # Max heart rate — Tanaka, Monahan & Seals 2001
    thalach = max(int(208 - 0.7 * age), 90)

    # Fasting blood sugar flag (for UCI heart model)
    fbs = 1 if glucose > 120 else 0

    # Skin thickness — PIMA dataset means, sex-adjusted
    skin = 29.0 if not sex_male else 20.5

    # Diabetes pedigree function — PIMA dataset mean
    dpf = 0.472

    # Hypertension flag — derived from systolic BP
    htn = 1 if systolic_bp >= 140 else 0

    # Diabetes mellitus flag — from any glucose ≥ 200
    effective_glucose = bgr if (bgr is not None and bgr > 0) else glucose
    dm = 1 if effective_glucose >= 200 else 0

    # Anemia flag — from hemoglobin if available
    if hemo is not None and hemo > 0:
        ane = 1 if (sex_male and hemo < 13.0) or (not sex_male and hemo < 12.0) else 0
    else:
        ane = 0

    # eGFR — derive from creatinine if available, else default to healthy 100
    if sc is not None and sc > 0:
        egfr_derived = _estimate_egfr(sc, age, sex_male)
    else:
        egfr_derived = 100.0

    return {
        # ── Derived (formula-based) ───────────────────────────────────────────
        "fbs":     fbs,
        "thalach": thalach,
        "skin":    skin,
        "dpf":     dpf,
        "htn":     htn,
        "dm":      dm,
        "ane":     ane,
        "egfr":    egfr_derived,

        # ── Clinical exam — healthy normal baseline ───────────────────────────
        "cp":      0,       # asymptomatic chest pain
        "restecg": 0,       # normal ECG
        "exang":   0,       # no exercise-induced angina
        "oldpeak": 0.0,     # no ST depression
        "slope":   1,       # flat ST slope (neutral)
        "ca":      0,       # no calcified vessels
        "thal":    2,       # normal thalassemia

        # ── Urine microscopy — normal ─────────────────────────────────────────
        "rbc":     1,       # normal urine RBC
        "pc":      1,       # normal pus cells
        "pcc":     0,       # pus cell clumps absent
        "ba":      0,       # bacteria absent

        # ── CBC dataset means (UCI CKD) ───────────────────────────────────────
        "pcv":     41.0,
        "wbcc":    7500.0,
        "rbcc":    4.71,
        "mcv":     87.0,    # health markers dataset mean

        # ── Clinical history defaults ─────────────────────────────────────────
        "cad":     0,
        "appet":   1,
        "pe":      0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAFE DEFAULTS — last resort fallback for any remaining gaps
# ─────────────────────────────────────────────────────────────────────────────
SAFE_DEFAULTS = {
    # Basic demographics
    "age":  45,   "sex":  1,

    # Heart model (UCI)
    "cp": 0,  "trestbps": 120,  "chol": 180,
    "fbs": 0, "restecg": 0,     "thalach": 150,
    "exang": 0, "oldpeak": 0.0, "slope": 1,
    "ca": 0,  "thal": 2,

    # Diabetes model (PIMA)
    "preg": 0,  "glucose": 100, "bloodpressure": 80,
    "skin": 23, "insulin": 80,  "bmi": 24.5,
    "dpf": 0.472,

    # Kidney model — numeric (UCI CKD medians)
    "bp": 80, "sg": 1.020, "al": 0,   "su": 0,
    "bgr": 121.0, "bu": 44.0, "sc": 1.2,
    "sod": 138.0, "pot": 4.6, "hemo": 13.5,
    "pcv": 41.0, "wbcc": 7500.0, "rbcc": 4.71,

    # Kidney model — categorical (healthy defaults)
    "rbc": 1, "pc": 1, "pcc": 0, "ba": 0,
    "htn": 0, "dm": 0, "cad": 0, "appet": 1,
    "pe":  0, "ane": 0,

    # Extended markers (health_markers_dataset features)
    "hba1c":        5.0,    # non-diabetic range
    "ldl":          100.0,  # optimal
    "hdl":          55.0,   # healthy
    "triglycerides": 130.0, # borderline
    "egfr":         100.0,  # normal kidney function
    "mcv":          87.0,   # normocytic mean
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN MAPPING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def map_input(user_input: dict, required_features: list) -> np.ndarray:
    """
    Convert a flat patient-data dict into a numpy array in the exact
    feature order expected by a model.

    Steps
    -----
    1. Normalize all lab-report key names → canonical model feature names
    2. Build feature vector in required_features order
    3. Fill missing values with SAFE_DEFAULTS

    Parameters
    ----------
    user_input        : flat dict of patient values (any key format)
    required_features : ordered list of feature names the model expects

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    # Step 1 — normalize all keys
    mapped = {}
    for raw_key, value in user_input.items():
        canonical = normalize_key(raw_key)
        mapped[canonical] = value

    # Step 2 — build ordered vector, filling gaps with safe defaults
    final = []
    for feature in required_features:
        val = mapped.get(feature, SAFE_DEFAULTS.get(feature, 0.0))
        try:
            final.append(float(val))
        except (TypeError, ValueError):
            final.append(float(SAFE_DEFAULTS.get(feature, 0.0)))

    return np.array(final).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
def apply_scaler(X: np.ndarray, scaler_path: str) -> np.ndarray:
    """Apply a saved StandardScaler if the file exists; safe no-op otherwise."""
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler.transform(X)
    return X