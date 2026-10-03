"""
Predict with the trained model: `models/<MODEL_FILE>` + features -> predictions.

Run with `make predict`. Point `INPUT_FILE` at new data built the same way.
"""

from typing import Any

import pandas as pd

from tsforecasting.data.data_loader import load_csv
from tsforecasting.features.build_features import FEATURES_FILE, TARGET
from tsforecasting.models.model_utils import load_model
from tsforecasting.models.train_model import MODEL_FILE
from tsforecasting.utils.paths import data_processed_dir, models_dir

INPUT_FILE = FEATURES_FILE
PREDICTIONS_FILE = "predictions.csv"


def predict(model: Any, df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    """
    Add a `prediction` column to the input features.

    Parameters
    ----------
    model : object
        Fitted estimator.
    df : pandas.DataFrame
        Features (a target column, if present, is ignored).
    target : str, optional
        Name of the target column to ignore (default is `TARGET`).

    Returns
    -------
    pandas.DataFrame
        Input data with a `prediction` column.
    """
    features = df.drop(columns=[target], errors="ignore")
    return df.assign(prediction=model.predict(features))


def main() -> None:
    """Load the model, predict on `INPUT_FILE` and write the predictions."""
    model = load_model(models_dir(MODEL_FILE))
    out = predict(model, load_csv(data_processed_dir(INPUT_FILE)))
    out.to_csv(data_processed_dir(PREDICTIONS_FILE), index=False)
    print(f"Wrote predictions to {data_processed_dir(PREDICTIONS_FILE)}")


if __name__ == "__main__":
    main()
