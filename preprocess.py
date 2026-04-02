"""
preprocess.py
=============
Prepares raw datasets for model training.

Datasets handled
----------------
1. UCI Heart Disease         → data/raw/heart.csv
2. PIMA Indians Diabetes     → data/raw/diabetes.csv
3. UCI Chronic Kidney Disease→ data/raw/kidney.csv
4. Health Markers Dataset    → data/raw/health_markers_dataset.csv  ← NEW

Outputs
-------
data/processed/*.csv   — feature CSVs (normalized, lowercase column names)
models/*_scaler.pkl    — fitted StandardScaler objects for inference
models/hm_classes.pkl  — LabelEncoder class list for health markers model

Key improvements in v2
----------------------
- health_markers_dataset fully integrated (Blood_glucose, HbA1C, LDL, HDL,
  Triglycerides, Haemoglobin, MCV, Systolic_BP, Diastolic_BP → Condition)
- All 24 UCI CKD features retained (was 11)
- Median imputation instead of row-dropping (preserves sample count)
- Consistent lowercase column names across all processed files
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

    Fix: biologically impossible zeros replaced with column median before scaling.
    """
    path = "data/raw/diabetes.csv"
    if not os.path.exists(path):
        print(f"[Diabetes] SKIPPED — file not found: {path}")
        return

    df = pd.read_csv(path)

    # Replace impossible zeros with median for these physiological columns
    zero_replace_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_replace_cols:
        if col in df.columns:
            median_val = df[col].replace(0, np.nan).median()
            df[col] = df[col].replace(0, median_val)

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
    UCI Chronic Kidney Disease dataset.

    Numeric features (14):
      age, bp, sg, al, su, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc

    Categorical features (10) — label-encoded to 0/1:
      rbc   : normal=1 / abnormal=0
      pc    : normal=1 / abnormal=0
      pcc   : present=1 / notpresent=0
      ba    : present=1 / notpresent=0
      htn   : yes=1 / no=0
      dm    : yes=1 / no=0
      cad   : yes=1 / no=0
      appet : good=1 / poor=0
      pe    : yes=1 / no=0
      ane   : yes=1 / no=0

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

    # Handle column name variants
    df.rename(columns={"wc": "wbcc", "rc": "rbcc"}, inplace=True)

    # Encode target
    df["classification"] = df["classification"].str.strip().str.lower()
    df["classification"] = df["classification"].map({
        "ckd": 1, "notckd": 0, "ckd\t": 1, "notckd\t": 0
    })
    df.dropna(subset=["classification"], inplace=True)
    df["classification"] = df["classification"].astype(int)

    NUMERIC_COLS = [
        "age", "bp", "sg", "al", "su",
        "bgr", "bu", "sc", "sod", "pot", "hemo",
        "pcv", "wbcc", "rbcc",
    ]
    CAT_MAPS = {
        "rbc":   {"normal": 1, "abnormal": 0},
        "pc":    {"normal": 1, "abnormal": 0},
        "pcc":   {"present": 1, "notpresent": 0},
        "ba":    {"present": 1, "notpresent": 0},
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

    # Impute with median / mode
    for col in numeric_available:
        df[col].fillna(df[col].median(), inplace=True)

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
    Health Markers Dataset (25 000 records).

    Features (9 continuous):
      Blood_glucose, HbA1C, Systolic_BP, Diastolic_BP, LDL, HDL,
      Triglycerides, Haemoglobin, MCV

    Target (multi-class):
      Condition ∈ {Fit, Diabetes, Hypertension, High_Cholesterol, Anemia}
      Label-encoded → integer (alphabetical: Anemia=0, Diabetes=1, Fit=2,
                                              High_Cholesterol=3, Hypertension=4)

    Processing steps:
    1. Copy raw file to data/raw if it exists in the upload location
    2. Fill 162 missing rows with column median
    3. Label-encode Condition
    4. Standardise features
    5. Save encoder class list for inference-time class → name mapping
    """
    # Support loading from the upload path used during development
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

    # Copy to data/raw if loaded from uploads
    dest = "data/raw/health_markers_dataset.csv"
    if path != dest:
        os.makedirs("data/raw", exist_ok=True)
        shutil.copy2(path, dest)
        print(f"[HM]       Copied dataset to {dest}")

    df = pd.read_csv(dest)
    print(f"[HM]       Loaded {len(df)} records, {df['Condition'].isna().sum()} target NaN rows")

    # Drop rows with missing target
    df = df[df["Condition"].notna()].copy()

    HM_FEATURE_COLS = [
        "Blood_glucose", "HbA1C", "Systolic_BP", "Diastolic_BP",
        "LDL", "HDL", "Triglycerides", "Haemoglobin", "MCV",
    ]

    # Impute missing feature values with column median
    for col in HM_FEATURE_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"[HM]         Imputed {col} with median={median_val:.2f}")

    df = df[HM_FEATURE_COLS + ["Condition"]].dropna().copy()

    # Label-encode target (alphabetical → Anemia=0, Diabetes=1, Fit=2,
    #                                      High_Cholesterol=3, Hypertension=4)
    le = LabelEncoder()
    df["condition_label"] = le.fit_transform(df["Condition"])

    # Save class names so predict.py can map index → condition
    joblib.dump(list(le.classes_), "models/hm_classes.pkl")
    print(f"[HM]       Classes: {list(le.classes_)}")

    X = df[HM_FEATURE_COLS].astype(float)
    y = df["condition_label"].astype(int)

    # Rename columns to canonical internal names (lowercase snake_case)
    col_rename = {
        "Blood_glucose": "glucose",
        "HbA1C":         "hba1c",
        "Systolic_BP":   "trestbps",
        "Diastolic_BP":  "bloodpressure",
        "LDL":           "ldl",
        "HDL":           "hdl",
        "Triglycerides": "triglycerides",
        "Haemoglobin":   "hemo",
        "MCV":           "mcv",
    }
    X = X.rename(columns=col_rename)
    HM_CANONICAL = list(col_rename.values())

    # Save column order for inference
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