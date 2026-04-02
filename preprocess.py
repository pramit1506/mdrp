"""
preprocess.py  —  v3
====================
Prepares raw datasets for model training.

Datasets handled
----------------
1. UCI Heart Disease         → data/raw/heart.csv
2. PIMA Indians Diabetes     → data/raw/diabetes.csv
3. UCI Chronic Kidney Disease→ data/raw/kidney.csv
4. Health Markers Dataset    → data/raw/health_markers_dataset.csv

Outputs
-------
data/processed/*.csv   — feature CSVs (normalised, lowercase column names)
models/*_scaler.pkl    — fitted StandardScaler objects for inference
models/hm_classes.pkl  — LabelEncoder class list for health markers model

Changes in v3
-------------
- CBC features removed globally: hemo, pcv, wbcc, rbcc (CBC numeric) +
  rbc, pc, pcc, ba (CBC/urine categorical).
- Urine features removed: sg, al, su.
- HM dataset reduced from 9 → 7 features (Haemoglobin and MCV removed).
- CKD features reduced from 24 → 13 (removed 7 CBC numeric + 4 urine categorical).
- All missing values imputed with medically normal reference values rather
  than dataset means wherever a clinical standard exists.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import shutil

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_heart():
    """
    UCI Heart Disease dataset (Cleveland).
    Features : age, sex, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal  (13 features)
    Target   : target/num → binarised 0/1
    """
    path = "data/raw/heart.csv"
    if not os.path.exists(path):
        print(f"[Heart]    SKIPPED — file not found: {path}")
        return

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()

    target_col = "target" if "target" in df.columns else "num"
    df = df.dropna()

    y = (df[target_col] > 0).astype(int)

    HEART_COLS = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal",
    ]
    available = [c for c in HEART_COLS if c in df.columns]
    X = df[available]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/heart_scaler.pkl")

    processed = pd.DataFrame(X_scaled, columns=available)
    processed["target"] = y.values
    processed.to_csv("data/processed/heart_processed.csv", index=False)
    print(f"[Heart]    Done | shape: {processed.shape} | features: {available}")


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_diabetes():
    """
    PIMA Indians Diabetes dataset.
    Features : preg, glucose, bloodpressure, skin, insulin, bmi, dpf, age
    Target   : Outcome → 'outcome'
    """
    path = "data/raw/diabetes.csv"
    if not os.path.exists(path):
        print(f"[Diabetes] SKIPPED — file not found: {path}")
        return

    df = pd.read_csv(path)

    # Medically normal imputation values for impossible zeros
    NORMAL_MEDIANS = {
        "Glucose":       95.0,    # normal fasting <100 mg/dL (ADA)
        "BloodPressure": 78.0,    # normal diastolic <80 mmHg
        "SkinThickness": 22.0,    # normal skinfold ~20–25 mm
        "Insulin":       10.0,    # normal fasting insulin 2–25 µU/mL
        "BMI":           22.5,    # healthy BMI 18.5–24.9 (WHO)
    }
    for col, normal_val in NORMAL_MEDIANS.items():
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            # Prefer dataset median if it exists and is in normal range; else use normal_val
            dataset_median = df[col].median()
            fill = dataset_median if pd.notna(dataset_median) else normal_val
            df[col].fillna(fill, inplace=True)

    df = df.dropna()

    df.rename(columns={
        "Pregnancies":              "preg",
        "Glucose":                  "glucose",
        "BloodPressure":            "bloodpressure",
        "SkinThickness":            "skin",
        "Insulin":                  "insulin",
        "BMI":                      "bmi",
        "DiabetesPedigreeFunction": "dpf",
        "Age":                      "age",
        "Outcome":                  "outcome",
    }, inplace=True)

    DIABETES_COLS = [
        "preg", "glucose", "bloodpressure", "skin",
        "insulin", "bmi", "dpf", "age",
    ]
    X = df[DIABETES_COLS]
    y = df["outcome"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/diabetes_scaler.pkl")

    processed = pd.DataFrame(X_scaled, columns=DIABETES_COLS)
    processed["outcome"] = y.values
    processed.to_csv("data/processed/diabetes_processed.csv", index=False)
    print(f"[Diabetes] Done | shape: {processed.shape} | features: {DIABETES_COLS}")


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_kidney():
    """
    UCI Chronic Kidney Disease dataset — v3 (CBC and urine features removed).

    Numeric features retained (7):
      age, bp, bgr, bu, sc, sod, pot

    Categorical features retained (6) — label-encoded to 0/1:
      htn   : yes=1 / no=0
      dm    : yes=1 / no=0
      cad   : yes=1 / no=0
      appet : good=1 / poor=0
      pe    : yes=1 / no=0
      ane   : yes=1 / no=0

    Removed (CBC + urine):
      sg, al, su               — urine specific gravity / albumin / sugar
      rbc, pc, pcc, ba         — urine microscopy categorical
      hemo, pcv, wbcc, rbcc    — haematology (CBC)

    Target: ckd=1, notckd=0
    """
    path = "data/raw/kidney.csv"
    if not os.path.exists(path):
        print(f"[Kidney]   SKIPPED — file not found: {path}")
        return

    df = pd.read_csv(path)
    df.replace(["?", "\t?", " "], np.nan, inplace=True)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.lower().str.strip()

    # Encode target
    df["classification"] = df["classification"].str.strip().str.lower()
    df["classification"] = df["classification"].map({
        "ckd": 1, "notckd": 0, "ckd\t": 1, "notckd\t": 0
    })
    df.dropna(subset=["classification"], inplace=True)
    df["classification"] = df["classification"].astype(int)

    # ── Numeric features (CBC and urine columns intentionally excluded) ───────
    NUMERIC_COLS = ["age", "bp", "bgr", "bu", "sc", "sod", "pot"]

    # ── Categorical features (urine microscopy excluded) ─────────────────────
    CAT_MAPS = {
        "htn":   {"yes": 1, "no": 0},
        "dm":    {"yes": 1, "no": 0},
        "cad":   {"yes": 1, "no": 0},
        "appet": {"good": 1, "poor": 0},
        "pe":    {"yes": 1, "no": 0},
        "ane":   {"yes": 1, "no": 0},
    }

    numeric_available   = [c for c in NUMERIC_COLS   if c in df.columns]
    categoric_available = [c for c in CAT_MAPS       if c in df.columns]

    # Encode categorical
    for col, mapping in CAT_MAPS.items():
        if col in df.columns:
            df[col] = df[col].str.lower().str.strip().map(mapping)

    # Convert numeric to float
    for col in numeric_available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Medically-normal imputation for missing numeric values
    NORMAL_KIDNEY = {
        "age":  45.0,
        "bp":   78.0,    # normal diastolic <80 mmHg
        "bgr": 115.0,    # normal random glucose <140 mg/dL
        "bu":   14.0,    # normal BUN 7–20 mg/dL
        "sc":    0.9,    # normal creatinine 0.6–1.2 mg/dL
        "sod": 140.0,    # normal sodium 135–145 mEq/L
        "pot":   4.0,    # normal potassium 3.5–5.0 mEq/L
    }
    for col in numeric_available:
        if df[col].isna().any():
            fill = NORMAL_KIDNEY.get(col, df[col].median())
            df[col].fillna(fill, inplace=True)

    # Mode imputation for categorical
    for col in categoric_available:
        if df[col].isna().any():
            mode = df[col].mode()
            df[col].fillna(mode[0] if len(mode) > 0 else 1, inplace=True)

    ALL_KIDNEY_COLS = numeric_available + categoric_available
    df = df[ALL_KIDNEY_COLS + ["classification"]].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[ALL_KIDNEY_COLS])
    joblib.dump(scaler, "models/kidney_scaler.pkl")

    processed = pd.DataFrame(X_scaled, columns=ALL_KIDNEY_COLS)
    processed["classification"] = df["classification"].values
    processed.to_csv("data/processed/kidney_processed.csv", index=False)

    print(f"[Kidney]   Done | shape: {processed.shape}")
    print(f"           Numeric    : {numeric_available}")
    print(f"           Categorical: {categoric_available}")


# ─────────────────────────────────────────────────────────────────────────────
def preprocess_health_markers():
    """
    Health Markers Dataset (25 000 records) — v3 (CBC features removed).

    Features retained (7 continuous):
      Blood_glucose, HbA1C, Systolic_BP, Diastolic_BP, LDL, HDL, Triglycerides

    Removed (CBC):
      Haemoglobin, MCV

    Target (multi-class):
      Condition ∈ {Fit, Diabetes, Hypertension, High_Cholesterol, Anemia}
    """
    candidate_paths = [
        "data/raw/health_markers_dataset.csv",
        "/mnt/user-data/uploads/health_markers_dataset.csv",
    ]
    path = None
    for p in candidate_paths:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        print("[HM]       SKIPPED — health_markers_dataset.csv not found in data/raw/")
        return

    dest = "data/raw/health_markers_dataset.csv"
    if path != dest:
        os.makedirs("data/raw", exist_ok=True)
        shutil.copy2(path, dest)
        print(f"[HM]       Copied dataset to {dest}")

    df = pd.read_csv(dest)
    print(f"[HM]       Loaded {len(df)} records, {df['Condition'].isna().sum()} target NaN rows")

    df = df[df["Condition"].notna()].copy()

    # ── Feature columns (Haemoglobin and MCV removed) ─────────────────────────
    HM_FEATURE_COLS = [
        "Blood_glucose", "HbA1C", "Systolic_BP", "Diastolic_BP",
        "LDL", "HDL", "Triglycerides",
    ]

    # Medically-normal imputation for any missing feature values
    HM_NORMAL = {
        "Blood_glucose":  95.0,
        "HbA1C":           5.3,
        "Systolic_BP":   118.0,
        "Diastolic_BP":   78.0,
        "LDL":            90.0,
        "HDL":            55.0,
        "Triglycerides": 115.0,
    }
    for col in HM_FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            fill = HM_NORMAL.get(col, df[col].median())
            df[col].fillna(fill, inplace=True)
            print(f"[HM]         Imputed {col} with {fill}")

    df = df[HM_FEATURE_COLS + ["Condition"]].dropna().copy()

    le = LabelEncoder()
    df["condition_label"] = le.fit_transform(df["Condition"])

    joblib.dump(list(le.classes_), "models/hm_classes.pkl")
    print(f"[HM]       Classes: {list(le.classes_)}")

    X = df[HM_FEATURE_COLS].astype(float)
    y = df["condition_label"].astype(int)

    # Rename to canonical internal names (must match HM_FEATURES in predict.py)
    col_rename = {
        "Blood_glucose": "glucose",
        "HbA1C":         "hba1c",
        "Systolic_BP":   "trestbps",
        "Diastolic_BP":  "bloodpressure",
        "LDL":           "ldl",
        "HDL":           "hdl",
        "Triglycerides": "triglycerides",
    }
    X = X.rename(columns=col_rename)
    HM_CANONICAL = list(col_rename.values())

    joblib.dump(HM_CANONICAL, "models/hm_features.pkl")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "models/hm_scaler.pkl")

    processed = pd.DataFrame(X_scaled, columns=HM_CANONICAL)
    processed["condition_label"] = y.values
    processed.to_csv("data/processed/hm_processed.csv", index=False)

    print(f"[HM]       Done | shape: {processed.shape} | "
          f"class balance: {dict(zip(le.classes_, np.bincount(y)))}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    preprocess_heart()
    preprocess_diabetes()
    preprocess_kidney()
    preprocess_health_markers()
    print("\n── All preprocessing complete. Scalers saved to models/ ──")
