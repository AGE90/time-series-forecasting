"""
Train a model: `data/processed/<FEATURES_FILE>` -> `models/<MODEL_FILE>`.

Run with `make train`. Swap the estimator and metrics for your problem.
"""

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from tsforecasting.data.data_loader import load_csv
from tsforecasting.features.build_features import FEATURES_FILE, TARGET
from tsforecasting.models.model_utils import (
    evaluate_classification,
    log_mlflow_experiment,
    save_model,
    train_test_split_data,
)
from tsforecasting.utils.paths import data_processed_dir, models_dir

MODEL_FILE = "model.joblib"
PARAMS: dict[str, Any] = {"n_estimators": 100, "random_state": 42}


def train(df: pd.DataFrame, target: str = TARGET) -> tuple[Any, dict[str, Any]]:
    """
    Fit a classifier on a train split and evaluate it on the test split.

    Parameters
    ----------
    df : pandas.DataFrame
        Features plus the target column.
    target : str, optional
        Name of the target column (default is `TARGET`).

    Returns
    -------
    model : object
        Fitted estimator.
    metrics : dict of str to float
        Test-set metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split_data(
        df.drop(columns=[target]), df[target]
    )
    model = RandomForestClassifier(**PARAMS).fit(X_train, y_train)
    metrics = evaluate_classification(y_test, model.predict(X_test), average="weighted")
    return model, metrics


def main() -> None:
    """Train on the processed features and save the model to `models/`."""
    model, metrics = train(load_csv(data_processed_dir(FEATURES_FILE)))
    save_model(model, models_dir(MODEL_FILE))
    log_mlflow_experiment(model, PARAMS, metrics, experiment_name="tsforecasting")
    print(f"Saved model to {models_dir(MODEL_FILE)}. Test metrics: {metrics}")


if __name__ == "__main__":
    main()
