"""
predict.py
==========
Core prediction module — v2 (Blended Prediction Engine).

Architecture
------------
For each disease, the final risk percentage is a weighted combination of:

  1. ML Model (50%)   : Stacking ensemble trained on disease-specific dataset.
                        Learns complex non-linear interactions in clinical features.

  2. Clinical Score (30%): Formula-based domain-knowledge scores derived from
                        established guidelines (ADA, ACC/AHA, KDIGO).
                        Directly interpretable; captures HbA1C→diabetes,
                        LDL/HDL→heart, eGFR→kidney signals precisely.

  3. Health Markers Model (20%): Multi-class ensemble trained on 25 000-record
                        health_markers_dataset. Captures co-occurrence patterns
                        across Blood_glucose, HbA1C, Lipid panel, Hb, MCV.

Blend formula (for each disease d):
    risk_d = clamp(0.50 × ml_prob_d
                 + 0.30 × clinical_score_d / 100
                 + 0.20 × hm_signal_d,  0, 1) × 100

Health markers signal mapping:
    diabetes : P(Diabetes class)
    heart    : max(P(Hypertension), P(High_Cholesterol))
    kidney   : P(Anemia) + 0.3 × max(0, 1 − P(Fit))   [anemia is a CKD marker]

Input contract (what the caller must provide)
----------------------------------------------
Required:
  age           : int   — patient age (years)
  sex           : int   — 1=male, 0=female
  trestbps      : float — systolic BP (mmHg)
  bloodpressure : float — diastolic BP (mmHg)
  bmi           : float — calculated from height / weight

From blood test (provide as many as available):
  glucose       : fasting blood glucose (mg/dL)
  bgr           : post-prandial / random blood glucose (mg/dL)
  hba1c         : glycated haemoglobin (%)
  chol          : total cholesterol (mg/dL)
  ldl           : LDL cholesterol (mg/dL)
  hdl           : HDL cholesterol (mg/dL)
  triglycerides : triglycerides (mg/dL)
  insulin       : fasting insulin (µU/mL)
  hemo          : hemoglobin (g/dL)
  mcv           : mean corpuscular volume (fL)
  sc            : serum creatinine (mg/dL)
  bu            : blood urea / BUN (mg/dL)
  egfr          : eGFR mL/min/1.73m² (or calculated if only sc available)
  sod, pot      : sodium / potassium (mEq/L)
  sg            : urine specific gravity
  al            : urine albumin 0–5 scale
  su            : urine sugar 0–5 scale
  pcv, wbcc, rbcc : packed cell volume, WBC count, RBC count
  rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane : categorical (0/1)
  preg          : pregnancies (default 0)
"""

import os
import joblib
import numpy as np

from input_mapper import map_input, apply_scaler, compute_smart_defaults, SAFE_DEFAULTS
from clinical_risk import calculate_all_risks


# ─────────────────────────────────────────────────────────────────────────────
# BLEND WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
W_ML       = 0.50   # ML model probability
W_CLINICAL = 0.30   # Clinical risk score (normalised to 0–1)
W_HM       = 0.20   # Health markers model signal


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FEATURE ORDERS — must EXACTLY match preprocess.py column orders
# ─────────────────────────────────────────────────────────────────────────────

# UCI Heart Disease — 13 features
HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# PIMA Indians Diabetes — 8 features
DIABETES_FEATURES = [
    "preg", "glucose", "bloodpressure", "skin",
    "insulin", "bmi", "dpf", "age",
]

# UCI Chronic Kidney Disease — 24 features
KIDNEY_FEATURES = [
    # Numeric
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot", "hemo",
    "pcv", "wbcc", "rbcc",
    # Categorical
    "rbc", "pc", "pcc", "ba",
    "htn", "dm", "cad", "appet", "pe", "ane",
]

# Health Markers — 9 features (must match preprocess_health_markers output)
HM_FEATURES = [
    "glucose", "hba1c", "trestbps", "bloodpressure",
    "ldl", "hdl", "triglycerides", "hemo", "mcv",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (lazy; graceful when not yet trained)
# ─────────────────────────────────────────────────────────────────────────────
def _safe_load(path: str):
    """Load a joblib model if the file exists; return None otherwise."""
    return joblib.load(path) if os.path.exists(path) else None


heart_model    = _safe_load("models/heart_model.pkl")
diabetes_model = _safe_load("models/diabetes_model.pkl")
kidney_model   = _safe_load("models/kidney_model.pkl")
hm_model       = _safe_load("models/hm_model.pkl")
hm_classes     = _safe_load("models/hm_classes.pkl")   # list of class names
hm_features    = _safe_load("models/hm_features.pkl")  # canonical column order


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — clamp a blended probability to [0, 100]
# ─────────────────────────────────────────────────────────────────────────────
def _blend(ml_prob: float, clinical_score: float, hm_signal: float) -> float:
    """
    Weighted blend of ML probability, clinical score, and HM signal.

    Parameters
    ----------
    ml_prob        : float in [0, 1]  — ML model probability
    clinical_score : float in [0, 100] — domain-knowledge clinical score
    hm_signal      : float in [0, 1]  — health markers model signal

    Returns
    -------
    float in [0, 100] — final risk percentage, rounded to 2 dp
    """
    blended = (
        W_ML       * ml_prob
        + W_CLINICAL * (clinical_score / 100.0)
        + W_HM       * hm_signal
    )
    return round(float(np.clip(blended, 0.0, 1.0)) * 100.0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH MARKERS SIGNAL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def _hm_signals(full_input: dict) -> dict:
    """
    Run the health markers multi-class model and extract per-disease signals.

    Health markers class mapping (alphabetical LabelEncoder order):
      0 = Anemia
      1 = Diabetes
      2 = Fit
      3 = High_Cholesterol
      4 = Hypertension

    Disease → HM signal:
      diabetes : P(Diabetes)
      heart    : max(P(Hypertension), P(High_Cholesterol))
      kidney   : P(Anemia)   [anemia is the primary HM marker for CKD]

    Returns
    -------
    dict with keys: hm_diabetes, hm_heart, hm_kidney  (all in [0, 1])
    """
    if hm_model is None:
        return {"hm_diabetes": 0.0, "hm_heart": 0.0, "hm_kidney": 0.0}

    features = hm_features if hm_features is not None else HM_FEATURES
    X_hm = map_input(full_input, features)
    X_hm = apply_scaler(X_hm, "models/hm_scaler.pkl")

    try:
        probs = hm_model.predict_proba(X_hm)[0]
    except Exception:
        return {"hm_diabetes": 0.0, "hm_heart": 0.0, "hm_kidney": 0.0}

    # Class indices (alphabetical): Anemia=0, Diabetes=1, Fit=2,
    #                               High_Cholesterol=3, Hypertension=4
    n = len(probs)
    p_anemia    = float(probs[0]) if n > 0 else 0.0
    p_diabetes  = float(probs[1]) if n > 1 else 0.0
    p_high_chol = float(probs[3]) if n > 3 else 0.0
    p_htn       = float(probs[4]) if n > 4 else 0.0

    return {
        "hm_diabetes": p_diabetes,
        "hm_heart":    max(p_htn, p_high_chol),
        "hm_kidney":   p_anemia,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_all(patient_data: dict) -> dict:
    """
    Run all 3 disease risk predictions with blended scoring.

    Parameters
    ----------
    patient_data : dict — flat dict of patient values (see module docstring)

    Returns
    -------
    dict:
        heart           : float (0–100) — blended heart disease risk %
        diabetes        : float (0–100) — blended diabetes risk %
        kidney          : float (0–100) — blended kidney disease risk %
        scores_detail   : dict          — breakdown of all 3 components
        used_defaults   : list[str]     — features that fell back to defaults
        clinical_scores : dict          — raw clinical risk scores (0–100)
    """
    age      = float(patient_data.get("age",      SAFE_DEFAULTS["age"]))
    glucose  = float(patient_data.get("glucose",  SAFE_DEFAULTS["glucose"]))
    sex_male = int(patient_data.get("sex", 1)) == 1

    # ── 1. Compute physiologically derived smart defaults ────────────────────
    smart = compute_smart_defaults(
        age         = age,
        glucose     = glucose,
        sex_male    = sex_male,
        systolic_bp = float(patient_data.get("trestbps", 120.0)),
        hemo        = patient_data.get("hemo"),
        bgr         = patient_data.get("bgr"),
        sc          = patient_data.get("sc"),
    )

    # ── 2. Merge: smart defaults < patient-provided values ───────────────────
    full_input = {**smart, **patient_data}

    # Kidney model uses 'bp' (diastolic); ensure alias exists
    if "bloodpressure" in full_input and "bp" not in full_input:
        full_input["bp"] = full_input["bloodpressure"]

    # ── 3. Track features that fell back to defaults ─────────────────────────
    provided      = set(patient_data.keys())
    all_features  = set(HEART_FEATURES + DIABETES_FEATURES + KIDNEY_FEATURES)
    used_defaults = sorted(
        f for f in all_features
        if f not in provided and f not in smart
    )

    # ── 4. Clinical Risk Scores (domain-knowledge formulas) ──────────────────
    clinical = calculate_all_risks(full_input)

    # ── 5. ML Model Probabilities ─────────────────────────────────────────────
    heart_ml    = _ml_prob(heart_model,    full_input, HEART_FEATURES,    "models/heart_scaler.pkl")
    diabetes_ml = _ml_prob(diabetes_model, full_input, DIABETES_FEATURES, "models/diabetes_scaler.pkl")
    kidney_ml   = _ml_prob(kidney_model,   full_input, KIDNEY_FEATURES,   "models/kidney_scaler.pkl")

    # ── 6. Health Markers Signals ─────────────────────────────────────────────
    hm = _hm_signals(full_input)

    # ── 7. Weighted Blend → Final Risk Scores ────────────────────────────────
    diabetes_risk = _blend(diabetes_ml, clinical["diabetes_clinical"], hm["hm_diabetes"])
    heart_risk    = _blend(heart_ml,    clinical["heart_clinical"],    hm["hm_heart"])
    kidney_risk   = _blend(kidney_ml,   clinical["kidney_clinical"],   hm["hm_kidney"])

    return {
        "heart":         heart_risk,
        "diabetes":      diabetes_risk,
        "kidney":        kidney_risk,
        "clinical_scores": {
            "diabetes_clinical": clinical["diabetes_clinical"],
            "heart_clinical":    clinical["heart_clinical"],
            "kidney_clinical":   clinical["kidney_clinical"],
        },
        "scores_detail": {
            "heart":    {"ml": round(heart_ml * 100, 2),    "clinical": clinical["heart_clinical"],    "hm": round(hm["hm_heart"]    * 100, 2)},
            "diabetes": {"ml": round(diabetes_ml * 100, 2), "clinical": clinical["diabetes_clinical"], "hm": round(hm["hm_diabetes"] * 100, 2)},
            "kidney":   {"ml": round(kidney_ml * 100, 2),   "clinical": clinical["kidney_clinical"],   "hm": round(hm["hm_kidney"]   * 100, 2)},
        },
        "used_defaults": used_defaults,
    }


def _ml_prob(model, full_input: dict, features: list, scaler_path: str) -> float:
    """
    Run a binary ML model and return P(positive class) in [0, 1].
    Returns 0.5 (uncertain) if model is not loaded.
    """
    if model is None:
        return 0.5  # neutral — model not trained yet

    X = map_input(full_input, features)
    X = apply_scaler(X, scaler_path)
    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception:
        prob = 0.5
    return prob


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE-TEST (using blood values from BLOOD_TEST.pdf)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        # Demographics
        "age":           35,
        "sex":           1,          # Male
        "trestbps":      120,        # Systolic BP (normal)
        "bloodpressure": 74,         # Diastolic BP
        "bmi":           24.5,

        # Glucose panel (from blood test PDF)
        "glucose":       102.9,      # FBS — impaired fasting (101–125)
        "bgr":           157.1,      # PPBS — impaired tolerance (140–199)
        "hba1c":         5.6,        # Non-diabetic (4–6 %)

        # Lipid panel (from blood test PDF)
        "chol":          139.3,      # Total cholesterol — desirable (<200)
        "ldl":           75.97,      # LDL — optimal (<100)
        "hdl":           37.2,       # HDL — LOW (risk factor for men <40)
        "triglycerides": 130.65,     # Borderline high (150–199)

        # Kidney panel (from blood test PDF)
        "sc":            0.41,       # Serum creatinine — normal (0.5–1.3)
        "bu":            14.32,      # BUN — normal (3.3–18.7)
        "egfr":          159.0,      # eGFR — excellent (>90)

        # CBC
        "hemo":          13.5,       # Haemoglobin — normal range
        "mcv":           87.0,       # MCV — normocytic

        "preg":          0,
    }

    results = predict_all(sample)
    print("\n── MDRP Prediction Results (Blood Test PDF values) ──────────────")
    print(f"  Heart Disease Risk : {results['heart']}%")
    print(f"  Diabetes Risk      : {results['diabetes']}%")
    print(f"  Kidney Disease Risk: {results['kidney']}%")

    print("\n── Score Breakdown ───────────────────────────────────────────────")
    for disease, detail in results["scores_detail"].items():
        print(f"  {disease.capitalize():<10}: ML={detail['ml']}%  "
              f"Clinical={detail['clinical']}/100  HM={detail['hm']}%")

    print(f"\n── Defaults used for: {results['used_defaults']}")