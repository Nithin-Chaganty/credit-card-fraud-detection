import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .config import TEST_SIZE, RANDOM_STATE
from .config import MODELS_DIR
from .data import load_raw_data
from .features import build_features_and_target


def evaluate_probs(y_true, y_proba):
    """
    Given true labels and predicted probabilities, search for the best
    threshold based on F1 and return metrics.
    """
    thresholds = np.linspace(0.01, 0.99, 50)
    best_f1 = 0.0
    best_t = 0.5

    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred_best = (y_proba > best_t).astype(int)
    prec = precision_score(y_true, y_pred_best, zero_division=0)
    rec = recall_score(y_true, y_pred_best, zero_division=0)
    roc = roc_auc_score(y_true, y_proba)

    return {
        "best_threshold": best_t,
        "f1": best_f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": roc,
    }


def train_and_compare_models():
    # 1. Load raw data
    df = load_raw_data()

    # 2. Build X and y
    X, y = build_features_and_target(df)

    # 3. Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=False,  # simulate using earlier data for training
    )

    feature_cols = X_train.columns.tolist()

    # Imbalance info
    fraud_rate = y_train.mean()
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    print(f"Fraud rate in train: {fraud_rate:.5f}, pos={pos}, neg={neg}")

    scale_pos_weight = neg / pos

    models = []

    # 4. Logistic Regression (baseline)
    log_reg = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )
    log_reg.fit(X_train, y_train)
    proba_log = log_reg.predict_proba(X_test)[:, 1]
    metrics_log = evaluate_probs(y_test, proba_log)
    models.append(("logistic_regression", log_reg, metrics_log))

    # 5. Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    metrics_rf = evaluate_probs(y_test, proba_rf)
    models.append(("random_forest", rf, metrics_rf))

    # 6. XGBoost
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
    )
    xgb.fit(X_train, y_train)
    proba_xgb = xgb.predict_proba(X_test)[:, 1]
    metrics_xgb = evaluate_probs(y_test, proba_xgb)
    models.append(("xgboost", xgb, metrics_xgb))

    # 7. Print comparison
    print("\nModel comparison on test set:")
    for name, _, m in models:
        print(
            f"{name:18} F1={m['f1']:.3f}, "
            f"Precision={m['precision']:.3f}, "
            f"Recall={m['recall']:.3f}, "
            f"AUC={m['roc_auc']:.3f}, "
            f"threshold={m['best_threshold']:.3f}"
        )

    # 8. Pick best model by F1
    best_model = None
    best_name = None
    best_metrics = None

    best_f1 = -1.0
    for name, model, m in models:
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_model = model
            best_name = name
            best_metrics = m

    print(f"\nBest model: {best_name} with F1={best_metrics['f1']:.3f}")

    # 9. Save best model, feature columns, and best threshold
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
    joblib.dump(best_metrics["best_threshold"], MODELS_DIR / "best_threshold.pkl")

    print("Saved best_model.pkl, feature_cols.pkl, best_threshold.pkl")


if __name__ == "__main__":
    train_and_compare_models()
