"""
Clean the raw data: `data/raw/<RAW_FILE>` -> `data/interim/<INTERIM_FILE>`.

Run with `make data`. Edit `make_dataset` with your project's cleaning steps.
"""

import pandas as pd

from tsforecasting.data.data_loader import load_csv
from tsforecasting.utils.paths import data_interim_dir, data_raw_dir

RAW_FILE = "dataset.csv"
INTERIM_FILE = "dataset_clean.csv"


def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a raw DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw data.

    Returns
    -------
    pandas.DataFrame
        Data without duplicate rows or rows with missing values.
    """
    return df.drop_duplicates().dropna().reset_index(drop=True)


def main() -> None:
    """Read the raw file, clean it and write it to `data/interim`."""
    df = make_dataset(load_csv(data_raw_dir(RAW_FILE)))
    df.to_csv(data_interim_dir(INTERIM_FILE), index=False)
    print(f"Wrote {len(df)} rows to {data_interim_dir(INTERIM_FILE)}")


if __name__ == "__main__":
    main()
