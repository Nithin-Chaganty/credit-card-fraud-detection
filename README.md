Credit Card Fraud Detection – End-to-End Machine Learning Pipeline

This project implements a full machine learning pipeline for credit card fraud detection using an extremely imbalanced dataset where fraud accounts for less than 0.2 percent of all transactions.
The pipeline automates data processing, feature creation, model training, model selection, threshold tuning, and batch prediction.

The goal is to demonstrate how fraud detection systems are built in real financial environments, where the priority is catching fraudulent transactions while minimizing false alarms.

Project Structure

fraud_pipeline/
│
├── data/
│   └── raw/
│       ├── creditcard.csv
│       └── creditcard_scored.csv       
│
├── models/
│   ├── best_model.pkl                  
│   ├── best_threshold.pkl             
│   └── feature_cols.pkl                 
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling_results.ipynb
│   └── 03_error_analysis.ipynb
│
├── src/
│   ├── config.py
│   ├── data.py                        
│   ├── features.py                      
│   ├── train.py                        
│   └── predict.py                 
│
├── README.md
└── requirements.txt

Key Features
Handles extremely imbalanced data

Fraud makes up less than 0.2 percent of all transactions.
The pipeline applies:

- class weighting
- XGBoost’s scale_pos_weight
- threshold tuning

to ensure the minority class is learned correctly instead of being ignored.


Trains multiple models automatically

The training pipeline evaluates three models on the same dataset:

- Logistic Regression
- Random Forest
- XGBoost

Each model is trained, evaluated, and compared automatically.


Automatic threshold tuning

Instead of relying on a default 0.5 probability cutoff, the pipeline searches over many thresholds to find the one that maximizes F1 score.

This mirrors real industry fraud systems where thresholds are tuned to balance recall (catching fraud) and precision (minimizing false alarms).


Model selection

The model with the best test-set F1 score is selected, and its:

- trained model
- threshold
- feature list

are all saved to the models/ directory.
This enables reproducibility and consistent predictions.

Batch scoring

The prediction script: python -m src.predict data/raw/creditcard.csv

Outputs:

fraud probability
fraud flag (based on tuned threshold)

to a new file: data/raw/creditcard_scored.csv
This reflects how real-world fraud engines score thousands of transactions at a time.

Model Performance

After training and comparison:

Model	                F1 Score	          Precision	          Recall	       ROC AUC
Logistic Regression	     0.589	                0.462	           0.813	        0.987
Random Forest	         0.821	                0.932	           0.733	        0.992
XGBoost (selected)	     0.824	                0.918	           0.747	        0.982

The selected model is saved as best_model.pkl.

A detailed model description is available in: models/MODEL_CARD.md

How to Train the Pipeline
From the project root: python -m src.train

This will:
- load raw data
- prepare features
- train all models
- tune thresholds
- compare performance
- save the best model and artifacts

How to Run Batch Predictions
python -m src.predict data/raw/creditcard.csv

This produces:
data/raw/creditcard_scored.csv

with:
- fraud probability
- fraud flag
for each transaction.

Future Improvements

- Add SHAP explainability
- Deploy as a REST API service
- Add automated hyperparameter tuning
- Use streaming methods for real-time fraud scoring
- Add monitoring for drift and threshold stability

Author

Nithin Chaganty
Machine Learning Engineer – Fraud Detection Pipeline
2025