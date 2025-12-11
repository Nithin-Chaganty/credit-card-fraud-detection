import pandas as pd
from .config import RAW_DATA_PATH

def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """Load the original credit card fraud CSV."""
    csv_path = path or RAW_DATA_PATH
    df = pd.read_csv(csv_path)
    return df
