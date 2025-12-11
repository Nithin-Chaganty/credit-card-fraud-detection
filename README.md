Credit Card Fraud Detection: End to End Machine Learning Pipeline

This project implements an end to end machine learning pipeline for credit card fraud detection using the Kaggle credit card dataset. Instead of only training models in a single notebook, the project follows a production style structure where data loading, feature engineering, model training, threshold tuning, and batch scoring are separated into clean Python modules.

The pipeline trains multiple models, compares them on a held out test set, selects the best model based on F1 score, and saves it to disk. A separate prediction script uses that model to score new CSV files by adding a fraud probability and a fraud flag. This structure mirrors how financial institutions build real fraud detection systems and highlights both modeling and engineering skills.

Project Structure

fraud_pipeline/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── raw/
│       └── creditcard.csv
│
├── models/
│   ├── best_model.pkl
│   ├── best_threshold.pkl
│   └── feature_cols.pkl
│
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_modeling_results.ipynb
│   └── 03_threshold_tuning.ipynb
│
└── src/
    ├── config.py
    ├── data.py
    ├── features.py
    ├── train.py
    └── predict.py

Trains multiple models automatically
-------------------------------------
The pipeline trains and compares three different models on the same training data:
- Logistic Regression
- Random Forest
- XGBoost

This allows a fair comparison and prevents over-reliance on a single model type.

Automatic threshold tuning
--------------------------------------
Instead of using the default 0.5 cutoff, the pipeline evaluates many different thresholds for each model and selects the one that gives the highest F1 score. This is critical in imbalanced classification where the optimal threshold is almost never 0.5.

Batch scoring
---------------------------------------
The prediction script loads the saved best model and scores any CSV file, adding:

fraud_probability
fraud_flag

This mirrors how real financial fraud engines generate alerts in production environments.