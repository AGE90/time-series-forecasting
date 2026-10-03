"""
Data loading utilities.
"""

import pandas as pd

from tsforecasting.utils.paths import data_processed_dir


def load_jena(features_file: str = "jena_features.parquet") -> pd.DataFrame:
    """Load the hourly Jena climate features built by `make data features`."""
    path = data_processed_dir(features_file)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data features` first.")
    return pd.read_parquet(path).asfreq("h")  # parquet drops the index freq


def load_airline() -> pd.Series:
    """Box-Jenkins monthly airline passengers 1949-1960 (classic SARIMA example)."""
    from sktime.datasets import load_airline as _load

    y = _load()
    y.index = y.index.to_timestamp()
    return y.asfreq("MS")


def temporal_split(
    df: pd.DataFrame, fractions: tuple[float, float, float] = (0.7, 0.2, 0.1)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/val/test split (no shuffling: the future must not leak).

    Parameters
    ----------
    df : pandas.DataFrame
        Time-ordered data.
    fractions : tuple of float
        Train, validation and test fractions summing to 1.

    Returns
    -------
    tuple of pandas.DataFrame
        Train, validation and test frames.
    """
    n = len(df)
    i, j = int(n * fractions[0]), int(n * (fractions[0] + fractions[1]))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]
