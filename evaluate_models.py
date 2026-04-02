"""
evaluate_models.py
==================
Evaluates trained ensemble models: Accuracy Score, Confusion Matrix, ROC Curve + AUC

Fixes v3
--------
- Health markers model path corrected: health_markers_model.pkl → hm_model.pkl
- Health markers target column corrected: "condition" → "condition_label"
- Evaluate is now called on a held-out 20% split (consistent with train_models.py)
  rather than the full processed CSV, so heart/kidney no longer show artificial 1.0.
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – safe on Windows / servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.model_selection import train_test_split
from feature_engineering import load_features


def evaluate(model_path: str, data_path: str, target: str, label: str = ""):
    model = joblib.load(model_path)
    X, y = load_features(data_path, target)

    # Use the same 20 % hold-out split as train_models.py to avoid train-set leakage
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() <= 20 else None,
    )

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    unique_classes = np.unique(y_test)

    acc = accuracy_score(y_test, preds)
    cm  = confusion_matrix(y_test, preds)

    print(f"\n{'='*40}")
    print(f"  {label or model_path}")
    print(f"{'='*40}")
    print(f"Accuracy : {acc:.4f}")

    if len(unique_classes) > 2:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y_test, probs, multi_class="ovr")
        print(f"AUC (OVR): {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y_test, preds))
    else:
        fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        print(f"AUC      : {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y_test, preds))
        plt.plot(fpr, tpr, label=f"{label} AUC={roc_auc:.2f}")


if __name__ == "__main__":
    plt.figure(figsize=(8, 6))
    plt.title("ROC Curves — Multi-Disease Ensemble Models")

    evaluate(
        "models/heart_model.pkl",
        "data/processed/heart_processed.csv",
        "target",
        "Heart Disease",
    )
    evaluate(
        "models/diabetes_model.pkl",
        "data/processed/diabetes_processed.csv",
        "outcome",
        "Diabetes",
    )
    evaluate(
        "models/kidney_model.pkl",
        "data/processed/kidney_processed.csv",
        "classification",
        "Kidney Disease",
    )

    # ── Health Markers (multi-class) ────────────────────────────────────────
    # Model saved as hm_model.pkl by train_models.py; target column is
    # "condition_label" (integer-encoded by LabelEncoder in preprocess.py).
    evaluate(
        "models/hm_model.pkl",
        "data/processed/hm_processed.csv",
        "condition_label",
        "Health Markers (Multiclass)",
    )

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("models/roc_curves.png", dpi=150)
    print("\nROC curves saved to models/roc_curves.png")
