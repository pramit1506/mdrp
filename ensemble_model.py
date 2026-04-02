"""
ensemble_model.py
=================
Stacking Ensemble: XGBoost (primary) + RandomForest (secondary)
Meta-learner: Logistic Regression

Why this combination?
---------------------
- XGBoost     : Boosting-based; captures non-linear feature interactions.
                High accuracy on tabular medical data. Prone to overfit on small sets.
- RandomForest: Bagging-based; low variance; sees data differently from XGBoost.
                The two models disagree on different hard samples — stacking exploits this.
- LogReg meta : Learns optimal confidence weighting from base model output probabilities.
                Produces well-calibrated risk probabilities (critical for medical use).

Why better than XGBoost alone?
-------------------------------
Stacking uses StratifiedKFold CV to produce out-of-fold base predictions for the
meta-learner, preventing data leakage and reducing overfitting.
"""

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression


def build_ensemble(multiclass: bool = False) -> StackingClassifier:
    """
    Build a stacking ensemble classifier.

    Parameters
    ----------
    multiclass : bool
        If True, configure meta-learner for multi-class output
        (used for the health_markers_dataset which has 5 condition classes).

    Returns
    -------
    StackingClassifier (unfitted)
    """
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss" if multiclass else "logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    solver = "lbfgs" if not multiclass else "lbfgs"
    meta = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver=solver,
        multi_class="auto",
        random_state=42,
    )

    ensemble = StackingClassifier(
        estimators=[("xgb", xgb), ("rf", rf)],
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    return ensemble