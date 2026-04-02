"""
evaluate_models.py
==================
Evaluates trained ensemble models: Accuracy Score, Confusion Matrix, ROC Curve + AUC
"""
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from feature_engineering import load_features

def evaluate(model_path: str, data_path: str, target: str, label: str = ""):
    model = joblib.load(model_path)
    X, y = load_features(data_path, target)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    unique_classes = np.unique(y)

    acc = accuracy_score(y, preds)
    cm  = confusion_matrix(y, preds)

    print(f"\n{'='*40}")
    print(f"  {label or model_path}")
    print(f"{'='*40}")
    print(f"Accuracy : {acc:.4f}")

    if len(unique_classes) > 2:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(y, probs, multi_class="ovr")
        print(f"AUC (OVR): {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y, preds))
    else:
        fpr, tpr, _ = roc_curve(y, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        print(f"AUC      : {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y, preds))
        plt.plot(fpr, tpr, label=f"{label} AUC={roc_auc:.2f}")

if __name__ == "__main__":
    plt.figure(figsize=(8, 6))
    plt.title("ROC Curves — Multi-Disease Ensemble Models")

    evaluate("models/heart_model.pkl", "data/processed/heart_processed.csv", "target", "Heart Disease")
    evaluate("models/diabetes_model.pkl", "data/processed/diabetes_processed.csv", "outcome", "Diabetes")
    evaluate("models/kidney_model.pkl", "data/processed/kidney_processed.csv", "classification", "Kidney Disease")
    evaluate("models/health_markers_model.pkl", "data/processed/health_markers_processed.csv", "condition", "Health Markers (Multiclass)")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("models/roc_curves.png", dpi=150)
    print("\nROC curves saved to models/roc_curves.png")