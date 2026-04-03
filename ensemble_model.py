from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

def build_ensemble(multiclass=False):
    xgb = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss" if multiclass else "logloss",
        use_label_encoder=False, random_state=42, verbosity=0,
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    meta = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    return StackingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        final_estimator=meta, cv=5,
        stack_method="predict_proba", passthrough=False, n_jobs=-1,
    )
