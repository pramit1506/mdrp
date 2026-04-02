#!/usr/bin/env python
"""Test imports to diagnose issues"""

print("Starting imports test...")

try:
    print("1. Importing pandas...", end="", flush=True)
    import pandas as pd
    print(" OK")
except Exception as e:
    print(f" ERROR: {e}")
    exit(1)

try:
    print("2. Importing joblib...", end="", flush=True)
    import joblib
    print(" OK")
except Exception as e:
    print(f" ERROR: {e}")
    exit(1)

try:
    print("3. Importing xgboost...", end="", flush=True)
    from xgboost import XGBClassifier
    print(" OK")
except Exception as e:
    print(f" ERROR: {e}")
    exit(1)

try:
    print("4. Importing sklearn...", end="", flush=True)
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    print(" OK")
except Exception as e:
    print(f" ERROR: {e}")
    exit(1)

try:
    print("5. Importing ensemble_model...", end="", flush=True)
    from ensemble_model import build_ensemble
    print(" OK")
except Exception as e:
    print(f" ERROR: {e}")
    exit(1)

try:
    print("6. Loading data/processed/diabetes_processed.csv...", end="", flush=True)
    df = pd.read_csv("data/processed/diabetes_processed.csv")
    print(f" OK ({df.shape})")
except Exception as e:
    print(f" ERROR: {e}")
    exit(1)

print("\nAll imports successful!")
