import pandas as pd
from .config import TARGET_COL, TIME_COL

def add_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Time" in df.columns and "hour" not in df.columns:
        df["hour"] = (df["Time"] // 3600) % 24
    return df

def build_features_and_target(df: pd.DataFrame):
    df = add_hour_feature(df)

    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    cols_to_drop = [c for c in ["Time"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    if TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(int)
        X = df.drop(columns=[TARGET_COL])
    else:
        y = None
        X = df

    return X, y
