"""
input_mapper.py  —  v3
=======================
Converts real-world patient data (basic info + blood test values) into
model-ready feature vectors with MEDICALLY NORMAL defaults for all clinical
features that are absent.

v3 changes
----------
- CBC features removed: hemo, pcv, wbcc, rbcc and their synonyms.
- Urine features removed: sg, al, su, rbc (urine), pc, pcc, ba and synonyms.
- SAFE_DEFAULTS updated to use published medically-normal reference values
  from ADA 2024, ACC/AHA 2019, KDIGO 2022 and WHO guidelines rather than
  dataset means.
- compute_smart_defaults() no longer references hemo (CBC), ane is now
  always 0 (safe default) since Hb is no longer collected.

Design philosophy
-----------------
1. DERIVE values where a formula exists
   - thalach  = 208 − (0.7 × age)           [Tanaka et al. 2001]
   - fbs      = 1 if glucose > 120 else 0
   - htn      = 1 if systolic_bp ≥ 140 else 0
   - dm       = 1 if fasting glucose ≥ 126 OR bgr ≥ 200  [ADA 2024]
   - egfr     = CKD-EPI approximation if only creatinine + age + sex available

2. Use published medically-normal reference values for absent features
   - glucose:  95 mg/dL   (ADA normal <100)
   - hba1c:    5.3 %      (ADA normal <5.7)
   - ldl:      90 mg/dL   (ACC/AHA optimal <100)
   - hdl:      55 mg/dL   (AHA healthy: >40 M / >50 F)
   - triglycerides: 115   (normal <150, AHA)
   - sc:       0.9 mg/dL  (normal 0.6–1.2)
   - bu:       14 mg/dL   (normal 7–20)
   - egfr:     95 mL/min  (KDIGO G1 normal ≥90)
   - sod:     140 mEq/L   (normal 135–145)
   - pot:       4.0 mEq/L (normal 3.5–5.0)

3. Use "healthy-normal" values for clinical exam features (ECG, stress test)
   - cp=0, restecg=0, exang=0, oldpeak=0.0, slope=1, ca=0, thal=2
"""

import numpy as np
import joblib
import os


# ─────────────────────────────────────────────────────────────────────────────
# LAB-REPORT → MODEL FEATURE NAME MAP  (CBC and urine entries removed)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_MAP = {
    # ── Blood Glucose ─────────────────────────────────────────────────────────
    "fbs":                          "glucose",
    "fasting blood sugar":          "glucose",
    "fasting blood glucose":        "glucose",
    "fasting plasma glucose":       "glucose",
    "fpg":                          "glucose",
    "fbg":                          "glucose",
    "fasting sugar":                "glucose",
    "blood glucose":                "glucose",
    "glucose fasting":              "glucose",
    "blood_glucose":                "glucose",

    "rbs":                          "bgr",
    "ppbs":                         "bgr",
    "post prandial blood sugar":    "bgr",
    "post prandial glucose":        "bgr",
    "post prandial plasma glucose": "bgr",
    "random blood glucose":         "bgr",
    "random blood sugar":           "bgr",
    "post meal glucose":            "bgr",
    "glucose post prandial":        "bgr",
    "2-hour post prandial":         "bgr",

    # ── HbA1C ────────────────────────────────────────────────────────────────
    "hba1c":                        "hba1c",
    "hba1c (glycated haemoglobin)": "hba1c",
    "hba1c-glycated haemoglobin":   "hba1c",
    "hemoglobin a1c":               "hba1c",
    "haemoglobin a1c":              "hba1c",
    "hgba1c":                       "hba1c",
    "a1c":                          "hba1c",
    "glycated haemoglobin":         "hba1c",
    "glycosylated hemoglobin":      "hba1c",
    "glycated hb":                  "hba1c",

    # ── Blood Pressure ────────────────────────────────────────────────────────
    "systolic bp":                  "trestbps",
    "systolic_bp":                  "trestbps",
    "systolic blood pressure":      "trestbps",
    "sbp":                          "trestbps",
    "trestbps":                     "trestbps",

    "diastolic bp":                 "bloodpressure",
    "diastolic_bp":                 "bloodpressure",
    "diastolic blood pressure":     "bloodpressure",
    "dbp":                          "bloodpressure",
    "blood pressure":               "bloodpressure",
    "bp":                           "bp",

    # ── Lipid Profile ─────────────────────────────────────────────────────────
    "total cholesterol":            "chol",
    "serum cholesterol":            "chol",
    "cholesterol":                  "chol",
    "tc":                           "chol",
    "t. chol":                      "chol",

    "ldl":                          "ldl",
    "ldl-cholesterol":              "ldl",
    "ldl cholesterol":              "ldl",
    "ldl- cholesterol":             "ldl",
    "low-density lipoprotein":      "ldl",
    "ldl-c":                        "ldl",

    "hdl":                          "hdl",
    "hdl-cholesterol":              "hdl",
    "hdl cholesterol":              "hdl",
    "hdl- cholesterol":             "hdl",
    "high-density lipoprotein":     "hdl",
    "hdl-c":                        "hdl",

    "triglycerides":                "triglycerides",
    "tg":                           "triglycerides",
    "trig":                         "triglycerides",
    "serum triglycerides":          "triglycerides",

    # ── Kidney Markers ────────────────────────────────────────────────────────
    "serum creatinine":             "sc",
    "s. creatinine":                "sc",
    "creatinine":                   "sc",
    "scr":                          "sc",
    "blood creatinine":             "sc",

    "blood urea":                   "bu",
    "blood urea nitrogen":          "bu",
    "bun":                          "bu",
    "urea":                         "bu",
    "serum urea":                   "bu",
    "serum urea nitrogen":          "bu",
    "bun (blood urea nitrogen)":    "bu",

    "sodium":                       "sod",
    "na":                           "sod",
    "serum sodium":                 "sod",

    "potassium":                    "pot",
    "k":                            "pot",
    "serum potassium":              "pot",

    "egfr":                         "egfr",
    "estimated gfr":                "egfr",
    "glomerular filtration rate":   "egfr",
    "egfr (estimated glomerular filtration rate)": "egfr",
    "estimated glomerular filtration rate":        "egfr",

    # ── Clinical History (categorical) ────────────────────────────────────────
    "hypertension":                 "htn",
    "htn":                          "htn",
    "diabetes mellitus":            "dm",
    "dm":                           "dm",
    "coronary artery disease":      "cad",
    "cad":                          "cad",
    "ihd":                          "cad",
    "appetite":                     "appet",
    "pedal edema":                  "pe",
    "ankle swelling":               "pe",
    "anemia":                       "ane",
    "anaemia":                      "ane",

    # ── Demographics / Basic ─────────────────────────────────────────────────
    "age":                          "age",
    "bmi":                          "bmi",
    "body mass index":              "bmi",
    "insulin":                      "insulin",
    "skin thickness":               "skin",
    "pregnancies":                  "preg",
    "dpf":                          "dpf",
}

_FEATURE_MAP_LOWER = {k.lower(): v for k, v in FEATURE_MAP.items()}


def normalize_key(raw_key: str) -> str:
    """Normalize a raw lab report key to the canonical model feature name."""
    return _FEATURE_MAP_LOWER.get(str(raw_key).lower().strip(), str(raw_key).lower().strip())


# ─────────────────────────────────────────────────────────────────────────────
# eGFR — CKD-EPI approximation
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
        return 95.0   # medically normal default

    kappa      = 0.9   if sex_male else 0.7
    alpha      = -0.411 if sex_male else -0.329
    sex_factor = 1.0   if sex_male else 1.012

    ratio = sc / kappa
    if ratio < 1.0:
        egfr = 141 * (ratio ** alpha) * (0.993 ** age) * sex_factor
    else:
        egfr = 141 * (ratio ** -1.209) * (0.993 ** age) * sex_factor

    return round(min(max(egfr, 0.0), 200.0), 1)


# ─────────────────────────────────────────────────────────────────────────────
# SMART DEFAULTS — physiologically derived from available inputs
# ─────────────────────────────────────────────────────────────────────────────
def compute_smart_defaults(
    age:         float,
    glucose:     float,
    sex_male:    bool  = True,
    systolic_bp: float = 118.0,
    bgr:         float = None,
    sc:          float = None,
) -> dict:
    """
    Returns physiologically DERIVED defaults for features not present in a
    standard blood test (CBC and urine features excluded from v3).

    Parameters
    ----------
    age         : patient age (years)
    glucose     : fasting blood glucose (mg/dL)
    sex_male    : True = male
    systolic_bp : systolic BP mmHg
    bgr         : random blood glucose mg/dL (None = unknown)
    sc          : serum creatinine mg/dL (None = unknown)
    """
    # Max heart rate — Tanaka, Monahan & Seals 2001
    thalach = max(int(208 - 0.7 * age), 90)

    # Fasting blood sugar flag (UCI heart dataset feature)
    fbs = 1 if glucose > 120 else 0

    # Skin thickness — PIMA dataset, sex-adjusted approximate normal
    skin = 28.0 if not sex_male else 20.0

    # Diabetes pedigree function — low-risk baseline (no known family history)
    dpf = 0.37

    # Hypertension flag — derived from systolic BP (ACC/AHA 2017: Stage 1 ≥130)
    htn = 1 if systolic_bp >= 140 else 0

    # Diabetes mellitus flag — ADA 2024: FBS ≥126 OR 2-hr PG ≥200
    effective_glucose = bgr if (bgr is not None and bgr > 0) else glucose
    dm = 1 if (glucose >= 126 or effective_glucose >= 200) else 0

    # Anemia: cannot derive without hemoglobin — default healthy (no anemia)
    ane = 0

    # eGFR — derive from creatinine if available; else medically normal 95
    if sc is not None and sc > 0:
        egfr_derived = _estimate_egfr(sc, age, sex_male)
    else:
        egfr_derived = 95.0

    return {
        # Derived (formula-based)
        "fbs":     fbs,
        "thalach": thalach,
        "skin":    skin,
        "dpf":     dpf,
        "htn":     htn,
        "dm":      dm,
        "ane":     ane,
        "egfr":    egfr_derived,

        # Clinical exam — healthy-normal baseline (no ECG/stress test done)
        "cp":      0,        # asymptomatic / typical angina: 0 = asymptomatic
        "restecg": 0,        # normal ECG
        "exang":   0,        # no exercise-induced angina
        "oldpeak": 0.0,      # no ST depression
        "slope":   1,        # flat (normal)
        "ca":      0,        # no calcified vessels visible
        "thal":    2,        # normal thalassemia status

        # Clinical history — healthy defaults
        "cad":     0,
        "appet":   1,        # good appetite
        "pe":      0,        # no pedal edema
    }


# ─────────────────────────────────────────────────────────────────────────────
# SAFE DEFAULTS — medically-normal reference values (v3)
# Sources: ADA 2024, ACC/AHA 2019, KDIGO 2022, WHO
# ─────────────────────────────────────────────────────────────────────────────
SAFE_DEFAULTS = {
    # Demographics
    "age":  45,
    "sex":  1,

    # Heart model (UCI) — clinical exam features set to healthy normal
    "cp":       0,
    "trestbps": 118,      # optimal systolic <120 mmHg
    "chol":     175,      # desirable total cholesterol <200 mg/dL (NCEP)
    "fbs":      0,        # FBS ≤120 → flag = 0
    "restecg":  0,        # normal ECG
    "thalach":  150,      # age-derived (208 − 0.7 × 45 ≈ 177; 150 = conservative normal)
    "exang":    0,
    "oldpeak":  0.0,
    "slope":    1,
    "ca":       0,
    "thal":     2,

    # Diabetes model (PIMA)
    "preg":          0,
    "glucose":       95,    # ADA normal fasting <100 mg/dL
    "bloodpressure": 78,    # normal diastolic <80 mmHg
    "skin":          22,    # normal skinfold thickness
    "insulin":       10,    # normal fasting insulin 2–25 µU/mL
    "bmi":           22.5,  # healthy BMI 18.5–24.9 (WHO)
    "dpf":           0.37,  # low-risk pedigree baseline

    # Kidney model — numeric (medically normal)
    "bp":   78,       # normal diastolic
    "bgr":  115,      # normal random glucose <140 mg/dL
    "bu":   14,       # normal BUN 7–20 mg/dL
    "sc":   0.9,      # normal creatinine 0.6–1.2 mg/dL
    "sod":  140,      # normal sodium 135–145 mEq/L
    "pot":  4.0,      # normal potassium 3.5–5.0 mEq/L

    # Kidney model — categorical (healthy defaults)
    "htn":   0,
    "dm":    0,
    "cad":   0,
    "appet": 1,
    "pe":    0,
    "ane":   0,

    # Extended markers
    "hba1c":         5.3,    # ADA normal <5.7 %
    "ldl":           90,     # ACC/AHA optimal <100 mg/dL
    "hdl":           55,     # AHA healthy: >40 men, >50 women mg/dL
    "triglycerides": 115,    # normal <150 mg/dL (AHA)
    "egfr":          95,     # KDIGO G1 normal ≥90 mL/min/1.73m²
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
    3. Fill missing values with medically-normal SAFE_DEFAULTS

    Parameters
    ----------
    user_input        : flat dict of patient values (any key format)
    required_features : ordered list of feature names the model expects

    Returns
    -------
    np.ndarray of shape (1, n_features)
    """
    mapped = {}
    for raw_key, value in user_input.items():
        canonical = normalize_key(raw_key)
        mapped[canonical] = value

    final = []
    for feature in required_features:
        val = mapped.get(feature, SAFE_DEFAULTS.get(feature, 0.0))
        try:
            final.append(float(val))
        except (TypeError, ValueError):
            final.append(float(SAFE_DEFAULTS.get(feature, 0.0)))

    return np.array(final).reshape(1, -1)


def apply_scaler(X: np.ndarray, scaler_path: str) -> np.ndarray:
    """Apply a saved StandardScaler if the file exists; safe no-op otherwise."""
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler.transform(X)
    return X
