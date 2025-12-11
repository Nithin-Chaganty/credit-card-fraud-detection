import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import MODELS_DIR
from .features import build_features_and_target


def predict_file(input_csv: str, output_csv: str | None = None):
    model_path = MODELS_DIR / "best_model.pkl"
    features_path = MODELS_DIR / "feature_cols.pkl"
    threshold_path = MODELS_DIR / "best_threshold.pkl"

    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    best_threshold = float(joblib.load(threshold_path))

    df = pd.read_csv(input_csv)
    X, _ = build_features_and_target(df)

    # Ensure same column order as training
    X = X[feature_cols]

    proba = model.predict_proba(X)[:, 1]
    preds = (proba > best_threshold).astype(int)

    df_out = df.copy()
    df_out["fraud_probability"] = proba
    df_out["fraud_flag"] = preds

    # Default output path
    if not output_csv:
        input_path = Path(input_csv)
        output_csv = str(input_path.with_name(input_path.stem + "_scored.csv"))

    df_out.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score transactions for fraud.")
    parser.add_argument("input_csv", help="Path to input CSV file")
    parser.add_argument(
        "--output_csv",
        help="Optional output CSV path",
        default=None,
    )
    args = parser.parse_args()
    predict_file(args.input_csv, args.output_csv)
