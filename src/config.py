from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parents[1]

# Data paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"

# Where to save trained models
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Train test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Column names
TARGET_COL = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
