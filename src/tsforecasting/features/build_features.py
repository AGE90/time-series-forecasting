"""
Build model features: `data/interim/<INTERIM_FILE>` -> `data/processed/<FEATURES_FILE>`.

Run with `make features`. Edit `build_features` with your feature engineering.
"""

import pandas as pd

from tsforecasting.data.data_loader import load_csv
from tsforecasting.data.make_dataset import INTERIM_FILE
from tsforecasting.features.feature_engineering import encode_categorical
from tsforecasting.utils.paths import data_interim_dir, data_processed_dir

FEATURES_FILE = "features.csv"
TARGET = "target"


def build_features(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """
    One-hot encode the categorical columns, leaving the target untouched.

    Parameters
    ----------
    df : pandas.DataFrame
        Clean data.
    target : str, optional
        Name of the target column (default is `TARGET`).

    Returns
    -------
    pandas.DataFrame
        Model-ready features plus the target column.
    """
    categorical = [
        col
        for col in df.select_dtypes(include=["object", "string", "category"]).columns
        if col != target
    ]
    if not categorical:
        return df
    encoded, _ = encode_categorical(df, categorical)
    return encoded


def main() -> None:
    """Read the interim file, build features and write them to `data/processed`."""
    df = build_features(load_csv(data_interim_dir(INTERIM_FILE)))
    df.to_csv(data_processed_dir(FEATURES_FILE), index=False)
    print(f"Wrote {df.shape[1]} columns to {data_processed_dir(FEATURES_FILE)}")


if __name__ == "__main__":
    main()
