"""
Build model features: `data/interim/<INTERIM_FILE>` -> `data/processed/<FEATURES_FILE>`.

Run with `make features`. Wind direction/speed become a wind vector and the
timestamp becomes sine/cosine "time of day" and "time of year" signals.
"""

import numpy as np
import pandas as pd

from tsforecasting.data.make_dataset import INTERIM_FILE
from tsforecasting.utils.paths import data_interim_dir, data_processed_dir

FEATURES_FILE = "jena_features.parquet"
TARGET = "T (degC)"
HOURS_PER_YEAR = 24 * 365.2425


def wind_vector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace wind speed/direction with Cartesian components.

    $W_x = v\\cos\\theta,\\; W_y = v\\sin\\theta$, so 0 and 360 degrees map to
    the same point and calm wind maps to the origin.
    """
    df = df.copy()
    theta = np.deg2rad(df.pop("wd (deg)"))
    for src, name in [("wv (m/s)", "W"), ("max. wv (m/s)", "maxW")]:
        v = df.pop(src)
        df[f"{name}x"] = v * np.cos(theta)
        df[f"{name}y"] = v * np.sin(theta)
    return df


def cyclical_time_features(
    index: pd.DatetimeIndex,
    periods_h: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Encode time as $\\sin(2\\pi t/P)$ and $\\cos(2\\pi t/P)$ per period $P$ (hours).

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timestamps to encode.
    periods_h : dict of str to float, optional
        Name -> period in hours. Defaults to day (24) and year (8765.82).

    Returns
    -------
    pandas.DataFrame
        One `<name> sin` and `<name> cos` column per period.
    """
    periods_h = periods_h or {"Day": 24.0, "Year": HOURS_PER_YEAR}
    t = (index - pd.Timestamp("1970-01-01")) / pd.Timedelta(hours=1)
    out = {}
    for name, p in periods_h.items():
        out[f"{name} sin"] = np.sin(2 * np.pi * t / p)
        out[f"{name} cos"] = np.cos(2 * np.pi * t / p)
    return pd.DataFrame(out, index=index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wind vector plus cyclical time features."""
    return wind_vector(df).join(cyclical_time_features(df.index))


def main() -> None:
    """Read the interim file, build features and write them to `data/processed`."""
    df = build_features(pd.read_parquet(data_interim_dir(INTERIM_FILE)))
    df.to_parquet(data_processed_dir(FEATURES_FILE))
    print(f"Wrote {df.shape} to {data_processed_dir(FEATURES_FILE)}")


if __name__ == "__main__":
    main()
