"""
train_models.py  —  v3
=======================
Trains stacking ensemble models for all 4 prediction targets:
  1. Heart Disease     (UCI Heart Disease)          → models/heart_model.pkl
  2. Diabetes          (PIMA Indians)               → models/diabetes_model.pkl
  3. Kidney Disease    (UCI CKD — 13 features)      → models/kidney_model.pkl
  4. Health Markers    (health_markers_dataset)     → models/hm_model.pkl

Changes in v3
-------------
- Kidney model now trains on 13 features (CBC + urine removed by preprocess.py).
- HM model now trains on 7 features (hemo + mcv removed).
- Model filename alignment: hm_model.pkl (consistent with evaluate_models.py).
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from ensemble_model import build_ensemble


def train(path: str, target: str, out: str, multiclass: bool = False):
    """
    Load a processed CSV, train the stacking ensemble, and save the model.

    Parameters
    ----------
    path       : path to processed CSV
    target     : target column name
    out        : path to save trained model (.pkl)
    multiclass : True for multi-class classification (health markers)
    """
    df = pd.read_csv(path)
    X  = df.drop(target, axis=1)
    y  = df[target]

    print(f"\nTraining on : {path}")
    print(f"  Samples   : {len(df)}")
    print(f"  Features  : {X.shape[1]}")
    print(f"  Classes   : {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() <= 20 else None,
    )

    model = build_ensemble(multiclass=multiclass)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"  Hold-out accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(model, out)
    print(f"  Model saved → {out}")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    # ── 1. Heart Disease (binary) ──────────────────────────────────────────────
    if os.path.exists("data/processed/heart_processed.csv"):
        train(
            path="data/processed/heart_processed.csv",
            target="target",
            out="models/heart_model.pkl",
            multiclass=False,
        )
    else:
        print("[Heart]    SKIPPED — heart_processed.csv not found")

    # ── 2. Diabetes (binary) ──────────────────────────────────────────────────
    if os.path.exists("data/processed/diabetes_processed.csv"):
        train(
            path="data/processed/diabetes_processed.csv",
            target="outcome",
            out="models/diabetes_model.pkl",
            multiclass=False,
        )
    else:
        print("[Diabetes] SKIPPED — diabetes_processed.csv not found")

    # ── 3. Kidney Disease (binary, 13 features in v3) ─────────────────────────
    if os.path.exists("data/processed/kidney_processed.csv"):
        train(
            path="data/processed/kidney_processed.csv",
            target="classification",
            out="models/kidney_model.pkl",
            multiclass=False,
        )
    else:
        print("[Kidney]   SKIPPED — kidney_processed.csv not found")

    # ── 4. Health Markers Condition (multi-class, 7 features in v3) ───────────
    if os.path.exists("data/processed/hm_processed.csv"):
        train(
            path="data/processed/hm_processed.csv",
            target="condition_label",
            out="models/hm_model.pkl",        # consistent with evaluate_models.py
            multiclass=True,
        )
    else:
        print("[HM]       SKIPPED — hm_processed.csv not found")
        print("           Run preprocess.py first to generate it.")

    print("\n── All models trained and saved to models/ ──")
