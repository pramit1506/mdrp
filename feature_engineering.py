"""
feature_engineering.py
=======================
Utility to load features and targets from processed CSVs.
Used by evaluate_models.py and optionally by train_models.py.
"""

import pandas as pd


def load_features(path: str, target: str):
    """
    Load a processed CSV and return X (features) and y (target).

    Parameters
    ----------
    path   : str — path to processed CSV file
    target : str — name of the target column

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}. "
                         f"Available columns: {list(df.columns)}")
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y