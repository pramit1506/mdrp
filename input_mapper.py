import numpy as np
import joblib
import os

FEATURE_MAP = {
    "fbs": "glucose", "fasting blood sugar": "glucose", "fasting blood glucose": "glucose",
    "fasting plasma glucose": "glucose", "fpg": "glucose", "fbg": "glucose",
    "fasting sugar": "glucose", "blood glucose": "glucose", "glucose fasting": "glucose",
    "blood_glucose": "glucose",
    "rbs": "bgr", "ppbs": "bgr", "post prandial blood sugar": "bgr",
    "post prandial glucose": "bgr", "random blood glucose": "bgr",
    "random blood sugar": "bgr", "post meal glucose": "bgr",
    "hba1c": "hba1c", "hemoglobin a1c": "hba1c", "haemoglobin a1c": "hba1c",
    "a1c": "hba1c", "glycated haemoglobin": "hba1c", "glycosylated hemoglobin": "hba1c",
    "systolic bp": "trestbps", "systolic_bp": "trestbps", "systolic blood pressure": "trestbps",
    "sbp": "trestbps", "trestbps": "trestbps",
    "diastolic bp": "bloodpressure", "diastolic_bp": "bloodpressure",
    "diastolic blood pressure": "bloodpressure", "dbp": "bloodpressure",
    "blood pressure": "bloodpressure", "bp": "bp",
    "total cholesterol": "chol", "serum cholesterol": "chol", "cholesterol": "chol",
    "tc": "chol",
    "ldl": "ldl", "ldl-cholesterol": "ldl", "ldl cholesterol": "ldl",
    "low-density lipoprotein": "ldl", "ldl-c": "ldl",
    "hdl": "hdl", "hdl-cholesterol": "hdl", "hdl cholesterol": "hdl",
    "high-density lipoprotein": "hdl", "hdl-c": "hdl",
    "triglycerides": "triglycerides", "tg": "triglycerides", "trig": "triglycerides",
    "serum creatinine": "sc", "s. creatinine": "sc", "creatinine": "sc", "scr": "sc",
    "blood urea": "bu", "blood urea nitrogen": "bu", "bun": "bu", "urea": "bu",
    "sodium": "sod", "na": "sod", "serum sodium": "sod",
    "potassium": "pot", "k": "pot", "serum potassium": "pot",
    "egfr": "egfr", "estimated gfr": "egfr", "glomerular filtration rate": "egfr",
    "hypertension": "htn", "htn": "htn",
    "diabetes mellitus": "dm", "dm": "dm",
    "coronary artery disease": "cad", "cad": "cad", "ihd": "cad",
    "appetite": "appet", "pedal edema": "pe", "ankle swelling": "pe",
    "anemia": "ane", "anaemia": "ane",
    "age": "age", "bmi": "bmi", "body mass index": "bmi",
    "insulin": "insulin", "skin thickness": "skin", "pregnancies": "preg", "dpf": "dpf",
}
_FEATURE_MAP_LOWER = {k.lower(): v for k, v in FEATURE_MAP.items()}

def normalize_key(raw_key):
    return _FEATURE_MAP_LOWER.get(str(raw_key).lower().strip(), str(raw_key).lower().strip())

def _estimate_egfr(sc, age, sex_male):
    if sc <= 0 or age <= 0:
        return 95.0
    kappa      = 0.9    if sex_male else 0.7
    alpha      = -0.411 if sex_male else -0.329
    sex_factor = 1.0    if sex_male else 1.012
    ratio = sc / kappa
    if ratio < 1.0:
        egfr = 141 * (ratio ** alpha) * (0.993 ** age) * sex_factor
    else:
        egfr = 141 * (ratio ** -1.209) * (0.993 ** age) * sex_factor
    return round(min(max(egfr, 0.0), 200.0), 1)

def compute_smart_defaults(age, glucose, sex_male=True, systolic_bp=118.0, bgr=None, sc=None):
    thalach = max(int(208 - 0.7 * age), 90)
    fbs     = 1 if glucose > 120 else 0
    skin    = 28.0 if not sex_male else 20.0
    dpf     = 0.37
    htn     = 1 if systolic_bp >= 140 else 0
    eff_g   = bgr if (bgr is not None and bgr > 0) else glucose
    dm      = 1 if (glucose >= 126 or eff_g >= 200) else 0
    ane     = 0
    egfr_d  = _estimate_egfr(sc, age, sex_male) if (sc is not None and sc > 0) else 95.0
    return {
        "fbs": fbs, "thalach": thalach, "skin": skin, "dpf": dpf,
        "htn": htn, "dm": dm, "ane": ane, "egfr": egfr_d,
        "cp": 0, "restecg": 0, "exang": 0, "oldpeak": 0.0,
        "slope": 1, "ca": 0, "thal": 2,
        "cad": 0, "appet": 1, "pe": 0,
    }

SAFE_DEFAULTS = {
    "age": 45, "sex": 1,
    "cp": 0, "trestbps": 118, "chol": 175, "fbs": 0, "restecg": 0,
    "thalach": 150, "exang": 0, "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 2,
    "preg": 0, "glucose": 95, "bloodpressure": 78, "skin": 22,
    "insulin": 10, "bmi": 22.5, "dpf": 0.37,
    "bp": 78, "bgr": 115, "bu": 14, "sc": 0.9, "sod": 140, "pot": 4.0,
    "htn": 0, "dm": 0, "cad": 0, "appet": 1, "pe": 0, "ane": 0,
    "hba1c": 5.3, "ldl": 90, "hdl": 55, "triglycerides": 115, "egfr": 95,
}

def map_input(user_input, required_features):
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

def apply_scaler(X, scaler_path):
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        return scaler.transform(X)
    return X

# ── pandas-DataFrame version of map_input (silences sklearn feature-name warnings) ──
def map_input_df(user_input, required_features):
    import pandas as pd
    mapped = {}
    for raw_key, value in user_input.items():
        canonical = normalize_key(raw_key)
        mapped[canonical] = value
    row = {}
    for feature in required_features:
        val = mapped.get(feature, SAFE_DEFAULTS.get(feature, 0.0))
        try:
            row[feature] = float(val)
        except (TypeError, ValueError):
            row[feature] = float(SAFE_DEFAULTS.get(feature, 0.0))
    return pd.DataFrame([row], columns=required_features)
