"""
Data loading utilities.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """Load data from a CSV file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        Loaded data
    """

    return pd.read_csv(filepath, **kwargs)


def load_excel(
    filepath: str | Path, sheet_name: str | int | None = 0, **kwargs
) -> pd.DataFrame:
    """Load data from an Excel file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the Excel file
    sheet_name : Optional[Union[str, int]], optional
        Name or index of the sheet to load, by default 0

    Returns
    -------
    pd.DataFrame
        Loaded data
    """

    return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)


def load_parquet(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """Load data from a Parquet file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the Parquet file

    Returns
    -------
    pd.DataFrame
        Loaded data
    """

    return pd.read_parquet(filepath, **kwargs)


def load_numpy(filepath: str | Path, **kwargs) -> np.ndarray:
    """Load data from a NumPy file.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the NumPy file

    Returns
    -------
    np.ndarray
        Loaded data
    """

    data: np.ndarray = np.load(filepath, **kwargs)
    return data
