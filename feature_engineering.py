import pandas as pd

def load_features(path, target):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}.")
    return df.drop(target, axis=1), df[target]
