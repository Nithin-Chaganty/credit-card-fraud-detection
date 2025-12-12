Credit Card Fraud Detection — Model Card
Overview

This model is part of an end-to-end machine learning pipeline designed to detect fraudulent credit card transactions in a highly imbalanced dataset where fraud represents less than 0.2 percent of all transactions.

Three models were evaluated:

Logistic Regression
Random Forest
XGBoost

After training and tuning, XGBoost achieved the best performance and was selected as the final model.
The trained model is saved as: models/best_model.pkl
The threshold used to convert predicted probabilities into fraud flags is stored in: models/best_threshold.pkl
The list of features used during training is stored in: models/feature_cols.pkl

Key Metrics (Test Set Performance)
Metric	           XGBoost Result
F1 Score	       0.824
Precision	       0.918
Recall	           0.747
ROC AUC	           0.982
Best Threshold	   0.95

Why this model was selected
- Highest F1 score across all models
- Strong balance of precision (avoiding false alarms) and recall (catching as many frauds as possible)
- Handles class imbalance effectively using scale_pos_weight
- Stable performance on unseen test data

Intended Use
This model is intended for:

- Batch scoring of large transaction datasets
- Downstream fraud monitoring pipelines
- Research, experimentation, and ML engineering demonstrations

It is not intended to be used in production without further calibration, drift monitoring, and fairness evaluation.

Input Features

The model uses the anonymized V1–V28 PCA-transformed features provided in the dataset, along with:

- Amount
- hour (derived from transaction time)

The feature list stored in feature_cols.pkl ensures consistent ordering during prediction.

How to Load the Model

import joblib
import pandas as pd

# Load artifacts
model = joblib.load("models/best_model.pkl")
threshold = joblib.load("models/best_threshold.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")

# Prepare input data
df = pd.read_csv("your_transactions.csv")
X = df[feature_cols]

# Predict probability
probs = model.predict_proba(X)[:, 1]

# Convert to fraud flag
preds = (probs > threshold).astype(int)

Limitations & Future Improvements

Limitations

- Dataset contains only numerical PCA-transformed features, limiting explainability
- Threshold tuning is static; could adjust over time with drift
- Fraud labels may not represent real banking behaviors

Possible Improvements

- Add explainability methods (SHAP values)
- Build real-time detection (streaming pipeline)
- Add hyperparameter optimization (Optuna or GridSearch)

Author

Nithin Chaganty
Credit Card Fraud Detection ML Pipeline
2025