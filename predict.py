"""
predict.py  —  v3 (Unified Weighted Prediction Engine)
=======================================================
Architecture
------------
For each disease the final risk percentage is a weighted combination of:

  1. ML Model (40%)           : Stacking ensemble trained on disease-specific dataset.
  2. Unified Clinical Score (60%): Evidence-based, tiered scoring from
                                published clinical guidelines (ADA 2024,
                                ACC/AHA 2019, KDIGO 2022).

Blend formula:
    risk = clamp(0.40 × ml_prob + 0.60 × clinical_score/100, 0, 1) × 100

Key fix (v3.1)
--------------
- kidney_features.pkl is now saved by preprocess.py and loaded here so that
  KIDNEY_FEATURES at inference EXACTLY matches the columns the scaler was
  fitted on (both in count and order).  This eliminates the
  "X has N features, but StandardScaler is expecting M features" error.
"""

import os
import joblib
import numpy as np

from input_mapper import map_input, apply_scaler, compute_smart_defaults, SAFE_DEFAULTS
from clinical_risk import calculate_all_risks


# ─────────────────────────────────────────────────────────────────────────────
W_ML       = 0.40
W_CLINICAL = 0.60


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FEATURE ORDERS
# ─────────────────────────────────────────────────────────────────────────────

HEART_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

DIABETES_FEATURES = [
    "preg", "glucose", "bloodpressure", "skin",
    "insulin", "bmi", "dpf", "age",
]

# v3: 13 features — loaded from disk so they match the fitted scaler exactly.
# Falls back to the hardcoded list only when the pkl does not exist yet.
_KIDNEY_FEATURES_DEFAULT = [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot",   # numeric (7)
    "htn", "dm", "cad", "appet", "pe", "ane",         # categorical (6)
]

HM_FEATURES = [
    "glucose", "hba1c", "trestbps", "bloodpressure",
    "ldl", "hdl", "triglycerides",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
def _safe_load(path: str):
    return joblib.load(path) if os.path.exists(path) else None


heart_model    = _safe_load("models/heart_model.pkl")
diabetes_model = _safe_load("models/diabetes_model.pkl")
kidney_model   = _safe_load("models/kidney_model.pkl")
hm_model       = _safe_load("models/hm_model.pkl")
hm_classes     = _safe_load("models/hm_classes.pkl")
hm_features    = _safe_load("models/hm_features.pkl")

# ── Critical: load the feature list that was used when fitting the scaler ────
_kidney_features_pkl = _safe_load("models/kidney_features.pkl")
KIDNEY_FEATURES = _kidney_features_pkl if _kidney_features_pkl is not None \
                  else _KIDNEY_FEATURES_DEFAULT

if _kidney_features_pkl is not None:
    print(f"[predict] Loaded kidney features from pkl: {KIDNEY_FEATURES}")
else:
    print(f"[predict] WARNING — kidney_features.pkl not found; using hardcoded list "
          f"({len(KIDNEY_FEATURES)} features).  Re-run preprocess.py to fix.")


# ─────────────────────────────────────────────────────────────────────────────
def _blend(ml_prob: float, clinical_score: float) -> float:
    blended = W_ML * ml_prob + W_CLINICAL * (clinical_score / 100.0)
    return round(float(np.clip(blended, 0.0, 1.0)) * 100.0, 2)


# ─────────────────────────────────────────────────────────────────────────────
def _classify_health_condition(full_input: dict) -> dict:
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


def _ml_prob(model, full_input: dict, features: list, scaler_path: str) -> float:
    if model is None:
        return 0.5

    X = map_input(full_input, features)
    X = apply_scaler(X, scaler_path)
    try:
        return float(model.predict_proba(X)[0][1])
    except Exception:
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
def predict_all(patient_data: dict) -> dict:
    age      = float(patient_data.get("age",     SAFE_DEFAULTS["age"]))
    glucose  = float(patient_data.get("glucose", SAFE_DEFAULTS["glucose"]))
    sex_male = int(patient_data.get("sex", 1)) == 1

    smart = compute_smart_defaults(
        age         = age,
        glucose     = glucose,
        sex_male    = sex_male,
        systolic_bp = float(patient_data.get("trestbps", 118.0)),
        bgr         = patient_data.get("bgr"),
        sc          = patient_data.get("sc"),
    )

    full_input = {**smart, **patient_data}

    if "bloodpressure" in full_input and "bp" not in full_input:
        full_input["bp"] = full_input["bloodpressure"]

    provided      = set(patient_data.keys())
    all_features  = set(HEART_FEATURES + DIABETES_FEATURES + KIDNEY_FEATURES)
    used_defaults = sorted(
        f for f in all_features
        if f not in provided and f not in smart
    )

    clinical = calculate_all_risks(full_input)

    heart_ml    = _ml_prob(heart_model,    full_input, HEART_FEATURES,    "models/heart_scaler.pkl")
    diabetes_ml = _ml_prob(diabetes_model, full_input, DIABETES_FEATURES, "models/diabetes_scaler.pkl")
    kidney_ml   = _ml_prob(kidney_model,   full_input, KIDNEY_FEATURES,   "models/kidney_scaler.pkl")

    heart_risk    = _blend(heart_ml,    clinical["heart_clinical"])
    diabetes_risk = _blend(diabetes_ml, clinical["diabetes_clinical"])
    kidney_risk   = _blend(kidney_ml,   clinical["kidney_clinical"])

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
            "heart":    {"ml": round(heart_ml    * 100, 2), "clinical": clinical["heart_clinical"]},
            "diabetes": {"ml": round(diabetes_ml * 100, 2), "clinical": clinical["diabetes_clinical"]},
            "kidney":   {"ml": round(kidney_ml   * 100, 2), "clinical": clinical["kidney_clinical"]},
        },
        "health_condition": health_condition,
        "used_defaults":    used_defaults,
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "age": 35, "sex": 1, "trestbps": 120, "bloodpressure": 74,
        "bmi": 24.5, "glucose": 102.9, "bgr": 157.1, "hba1c": 5.6,
        "chol": 139.3, "ldl": 75.97, "hdl": 37.2, "triglycerides": 130.65,
        "sc": 0.41, "bu": 14.32, "egfr": 159.0, "preg": 0,
    }

    results = predict_all(sample)
    print("\n── MDRP v3 Prediction Results ───────────────────────────────────")
    print(f"  Heart Disease Risk : {results['heart']}%")
    print(f"  Diabetes Risk      : {results['diabetes']}%")
    print(f"  Kidney Disease Risk: {results['kidney']}%")
    for disease, detail in results["scores_detail"].items():
        print(f"  {disease.capitalize():<10}: ML={detail['ml']}%  Clinical Score={detail['clinical']}/100")
