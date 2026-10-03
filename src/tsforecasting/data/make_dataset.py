"""
Clean the raw data: `data/raw/<RAW_FILE>` -> `data/interim/<INTERIM_FILE>`.

Run with `make data`. Downloads the Jena climate dataset if it is missing,
sub-samples it from 10-minute to hourly resolution and fixes the `-9999`
wind-speed sentinels.
"""

import io
import urllib.request
import zipfile

import pandas as pd

from tsforecasting.utils.paths import data_interim_dir, data_raw_dir

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
RAW_FILE = "jena_climate_2009_2016.csv"
INTERIM_FILE = "jena_hourly.parquet"


def download_jena() -> None:
    """Download and unzip the Jena climate CSV into `data/raw` (skipped if present)."""
    if data_raw_dir(RAW_FILE).exists():
        return
    with urllib.request.urlopen(URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extract(RAW_FILE, data_raw_dir())


def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw 10-minute Jena table into a clean hourly series.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw Jena climate table with a `Date Time` column.

    Returns
    -------
    pandas.DataFrame
        Hourly data indexed by timestamp, with negative wind speeds set to 0.
        Missing hours are linearly interpolated.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S")
    df.index.name = "time"
    # Keep on-the-hour records. The classic `df[5::6]` slice drifts off the hour
    # after the raw file's missing rows (late 2016), corrupting the last months.
    df = df[(df.index.minute == 0) & ~df.index.duplicated()]
    # ponytail: linear fill; the 2 real outages (77 h total, 2014-09 and 2016-10)
    # become straight lines. Mask them out if a study needs only observed data.
    df = df.asfreq("h").interpolate()
    for col in ["wv (m/s)", "max. wv (m/s)"]:
        df[col] = df[col].clip(lower=0.0)  # -9999 sentinels
    return df


def main() -> None:
    """Download if needed, clean and write to `data/interim`."""
    download_jena()
    df = make_dataset(pd.read_csv(data_raw_dir(RAW_FILE)))
    df.to_parquet(data_interim_dir(INTERIM_FILE))
    print(f"Wrote {len(df)} hourly rows to {data_interim_dir(INTERIM_FILE)}")


if __name__ == "__main__":
    main()
