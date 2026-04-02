"""
predict.py  —  v3 (Unified Weighted Prediction Engine)
=======================================================
Architecture
------------
For each disease the final risk percentage is a weighted combination of:

  1. ML Model (40%)           : Stacking ensemble (XGBoost + RandomForest +
                                LogReg meta-learner) trained on disease-specific
                                dataset.  Captures complex non-linear interactions.

  2. Unified Clinical Score (60%): Evidence-based, tiered scoring from
                                published clinical guidelines (ADA 2024,
                                ACC/AHA 2019, KDIGO 2022).  Feature weights
                                reflect real clinical importance; replaces the
                                old 3-component (clinical 30 % + HM 20 %) split
                                with a single interpretable weighted score.

Blend formula:
    risk = clamp(0.40 × ml_prob + 0.60 × clinical_score/100, 0, 1) × 100

Changes from v2
---------------
- CBC features removed: hemo, pcv, wbcc, rbcc (numeric) + rbc, pc, pcc, ba (categorical).
- Urine features removed: sg, al, su.
- KIDNEY_FEATURES shrunk from 24 → 13.
- HM_FEATURES shrunk from 9 → 7 (no hemo, no mcv).
- HM model kept as standalone health-condition classifier (separate output)
  but no longer included in the blended risk calculation.
- W_ML = 0.40, W_CLINICAL = 0.60 (replaces W_ML 0.50 + W_CLINICAL 0.30 + W_HM 0.20).
- All missing features receive medically-normal defaults (not worst-case assumptions).

Input contract
--------------
Required:
  age           : int   — patient age (years)
  sex           : int   — 1=male, 0=female
  trestbps      : float — systolic BP mmHg
  bloodpressure : float — diastolic BP mmHg
  bmi           : float — computed from height / weight

From blood test (include as many as available):
  glucose, bgr  : fasting / post-prandial glucose (mg/dL)
  hba1c         : glycated haemoglobin (%)
  chol          : total cholesterol (mg/dL)
  ldl, hdl      : LDL / HDL cholesterol (mg/dL)
  triglycerides : triglycerides (mg/dL)
  insulin       : fasting insulin (µU/mL)
  sc            : serum creatinine (mg/dL)
  bu            : blood urea / BUN (mg/dL)
  egfr          : eGFR mL/min/1.73m²
  sod, pot      : sodium / potassium (mEq/L)
  htn, dm, cad, appet, pe, ane : categorical history flags (0/1)
  preg          : pregnancies (default 0)
"""

import os
import joblib
import numpy as np

from input_mapper import map_input, apply_scaler, compute_smart_defaults, SAFE_DEFAULTS
from clinical_risk import calculate_all_risks


# ─────────────────────────────────────────────────────────────────────────────
# BLEND WEIGHTS  (v3: 2-component unified system)
# ─────────────────────────────────────────────────────────────────────────────
W_ML       = 0.40   # ML ensemble probability
W_CLINICAL = 0.60   # Unified weighted clinical score (0–100, normalised to 0–1)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FEATURE ORDERS — must EXACTLY match preprocess.py column orders
# ─────────────────────────────────────────────────────────────────────────────

# UCI Heart Disease — 13 features (unchanged)
HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# PIMA Indians Diabetes — 8 features (unchanged)
DIABETES_FEATURES = [
    "preg", "glucose", "bloodpressure", "skin",
    "insulin", "bmi", "dpf", "age",
]

# UCI Chronic Kidney Disease — 13 features (v3: CBC + urine removed)
KIDNEY_FEATURES = [
    # Numeric
    "age", "bp", "bgr", "bu", "sc", "sod", "pot",
    # Categorical (clinical history only; urine microscopy removed)
    "htn", "dm", "cad", "appet", "pe", "ane",
]

# Health Markers — 7 features (v3: hemo + mcv removed)
HM_FEATURES = [
    "glucose", "hba1c", "trestbps", "bloodpressure",
    "ldl", "hdl", "triglycerides",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS (lazy; graceful when not yet trained)
# ─────────────────────────────────────────────────────────────────────────────
def _safe_load(path: str):
    return joblib.load(path) if os.path.exists(path) else None


heart_model    = _safe_load("models/heart_model.pkl")
diabetes_model = _safe_load("models/diabetes_model.pkl")
kidney_model   = _safe_load("models/kidney_model.pkl")
hm_model       = _safe_load("models/hm_model.pkl")
hm_classes     = _safe_load("models/hm_classes.pkl")
hm_features    = _safe_load("models/hm_features.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# BLEND — 2-component unified blend
# ─────────────────────────────────────────────────────────────────────────────
def _blend(ml_prob: float, clinical_score: float) -> float:
    """
    Weighted blend of ML probability and unified clinical score.

    Parameters
    ----------
    ml_prob        : float in [0, 1]   — ML model probability
    clinical_score : float in [0, 100] — unified weighted clinical score

    Returns
    -------
    float in [0, 100] — final risk percentage
    """
    blended = W_ML * ml_prob + W_CLINICAL * (clinical_score / 100.0)
    return round(float(np.clip(blended, 0.0, 1.0)) * 100.0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# HEALTH MARKERS — standalone multi-class condition classifier (not blended)
# ─────────────────────────────────────────────────────────────────────────────
def _classify_health_condition(full_input: dict) -> dict:
    """
    Run the health-markers multi-class model and return class probabilities.
    This is a bonus output — it is NOT part of the blended risk score.

    Returns dict: {class_name: probability, ...}   or {} if model unavailable.
    """
    if hm_model is None:
        return {}

    features = hm_features if hm_features is not None else HM_FEATURES
    X_hm = map_input(full_input, features)
    X_hm = apply_scaler(X_hm, "models/hm_scaler.pkl")

    try:
        probs = hm_model.predict_proba(X_hm)[0]
        classes = hm_classes if hm_classes is not None else [str(i) for i in range(len(probs))]
        return {cls: round(float(p) * 100, 1) for cls, p in zip(classes, probs)}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
def _ml_prob(model, full_input: dict, features: list, scaler_path: str) -> float:
    """
    Run a binary ML model and return P(positive class) in [0, 1].
    Returns 0.5 (uncertain / neutral) if model is not loaded.
    """
    if model is None:
        return 0.5

    X = map_input(full_input, features)
    X = apply_scaler(X, scaler_path)
    try:
        return float(model.predict_proba(X)[0][1])
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def predict_all(patient_data: dict) -> dict:
    """
    Run all 3 disease risk predictions with unified weighted blending.

    Parameters
    ----------
    patient_data : dict — flat dict of patient values (see module docstring).

    Returns
    -------
    dict:
        heart           : float (0–100) — blended heart disease risk %
        diabetes        : float (0–100) — blended diabetes risk %
        kidney          : float (0–100) — blended kidney disease risk %
        scores_detail   : dict          — ML and clinical score breakdown
        health_condition: dict          — multi-class HM condition probabilities
        used_defaults   : list[str]     — features that used defaults
        clinical_scores : dict          — raw clinical scores (0–100)
    """
    age      = float(patient_data.get("age",     SAFE_DEFAULTS["age"]))
    glucose  = float(patient_data.get("glucose", SAFE_DEFAULTS["glucose"]))
    sex_male = int(patient_data.get("sex", 1)) == 1

    # ── 1. Compute physiologically derived smart defaults ────────────────────
    smart = compute_smart_defaults(
        age         = age,
        glucose     = glucose,
        sex_male    = sex_male,
        systolic_bp = float(patient_data.get("trestbps", 118.0)),
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

    # ── 4. Unified Clinical Risk Scores ──────────────────────────────────────
    clinical = calculate_all_risks(full_input)

    # ── 5. ML Model Probabilities ─────────────────────────────────────────────
    heart_ml    = _ml_prob(heart_model,    full_input, HEART_FEATURES,    "models/heart_scaler.pkl")
    diabetes_ml = _ml_prob(diabetes_model, full_input, DIABETES_FEATURES, "models/diabetes_scaler.pkl")
    kidney_ml   = _ml_prob(kidney_model,   full_input, KIDNEY_FEATURES,   "models/kidney_scaler.pkl")

    # ── 6. Unified 2-component Blend ─────────────────────────────────────────
    heart_risk    = _blend(heart_ml,    clinical["heart_clinical"])
    diabetes_risk = _blend(diabetes_ml, clinical["diabetes_clinical"])
    kidney_risk   = _blend(kidney_ml,   clinical["kidney_clinical"])

    # ── 7. Bonus: Health Condition Classification ─────────────────────────────
    health_condition = _classify_health_condition(full_input)

    return {
        "heart":            heart_risk,
        "diabetes":         diabetes_risk,
        "kidney":           kidney_risk,
        "clinical_scores": {
            "diabetes_clinical": clinical["diabetes_clinical"],
            "heart_clinical":    clinical["heart_clinical"],
            "kidney_clinical":   clinical["kidney_clinical"],
        },
        "scores_detail": {
            "heart":    {
                "ml":       round(heart_ml    * 100, 2),
                "clinical": clinical["heart_clinical"],
            },
            "diabetes": {
                "ml":       round(diabetes_ml * 100, 2),
                "clinical": clinical["diabetes_clinical"],
            },
            "kidney":   {
                "ml":       round(kidney_ml   * 100, 2),
                "clinical": clinical["kidney_clinical"],
            },
        },
        "health_condition": health_condition,
        "used_defaults":    used_defaults,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "age":           35,
        "sex":           1,
        "trestbps":      120,
        "bloodpressure": 74,
        "bmi":           24.5,
        "glucose":       102.9,
        "bgr":           157.1,
        "hba1c":         5.6,
        "chol":          139.3,
        "ldl":           75.97,
        "hdl":           37.2,
        "triglycerides": 130.65,
        "sc":            0.41,
        "bu":            14.32,
        "egfr":          159.0,
        "preg":          0,
    }

    results = predict_all(sample)
    print("\n── MDRP v3 Prediction Results ───────────────────────────────────")
    print(f"  Heart Disease Risk : {results['heart']}%")
    print(f"  Diabetes Risk      : {results['diabetes']}%")
    print(f"  Kidney Disease Risk: {results['kidney']}%")

    print("\n── Score Breakdown (ML 40% + Clinical 60%) ──────────────────────")
    for disease, detail in results["scores_detail"].items():
        print(f"  {disease.capitalize():<10}: ML={detail['ml']}%  "
              f"Clinical Score={detail['clinical']}/100")

    if results["health_condition"]:
        print("\n── Health Condition Classification (HM model) ───────────────────")
        for cond, pct in sorted(results["health_condition"].items(),
                                key=lambda x: -x[1]):
            print(f"  {cond:<20}: {pct:.1f}%")

    print(f"\n── Defaults used: {results['used_defaults']}")
