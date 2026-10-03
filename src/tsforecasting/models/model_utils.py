"""
Model training and evaluation utilities.
"""

import pickle
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split


def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Parameters
    ----------
    X : pandas.DataFrame
        Features.
    y : pandas.Series or numpy.ndarray
        Target variable.
    test_size : float, optional
        Proportion of data to use for testing (default is 0.2).
    random_state : int, optional
        Random seed for reproducibility (default is 42).

    Returns
    -------
    X_train : numpy.ndarray
        Training features.
    X_test : numpy.ndarray
        Testing features.
    y_train : numpy.ndarray
        Training target.
    y_test : numpy.ndarray
        Testing target.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def evaluate_classification(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    average: Literal["micro", "macro", "samples", "weighted", "binary"]
    | None = "binary",
) -> dict[str, Any]:
    """
    Evaluate classification model performance.

    Parameters
    ----------
    y_true : pandas.Series or numpy.ndarray
        True labels.
    y_pred : pandas.Series or numpy.ndarray
        Predicted labels.
    average : str, optional
        Averaging strategy for multi-class metrics (default is 'weighted').

    Returns
    -------
    metrics : dict of str to float
        Dictionary of metric names and values: accuracy, precision, recall, f1.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
    }


def evaluate_regression(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> dict[str, float]:
    """
    Evaluate regression model performance.

    Parameters
    ----------
    y_true : pandas.Series or numpy.ndarray
        True values.
    y_pred : pandas.Series or numpy.ndarray
        Predicted values.

    Returns
    -------
    metrics : dict of str to float
        Dictionary of metric names and values: mse, rmse, r2.
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
) -> dict[str, Any]:
    """
    Perform cross-validation on a model.

    Parameters
    ----------
    model : object
        Model to evaluate.
    X : pandas.DataFrame
        Features.
    y : pandas.Series or numpy.ndarray
        Target variable.
    cv : int, optional
        Number of cross-validation folds (default is 5).
    scoring : str, optional
        Scoring metric (default is 'accuracy').

    Returns
    -------
    results : dict of str to float
        Dictionary of cross-validation results: mean_score, std_score, scores.
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "scores": scores,
    }


def log_mlflow_experiment(
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    experiment_name: str,
    run_name: str | None = None,
    model_name: str | None = None,
) -> None:
    """
    Log model training results to MLflow.

    Parameters
    ----------
    model : object
        Trained model.
    params : dict
        Model parameters.
    metrics : dict
        Evaluation metrics.
    experiment_name : str
        Name of the MLflow experiment.
    run_name : str, optional
        Optional name for this run.
    model_name : str, optional
        Optional name for the model.
    """
    # Imported lazily so the module works when MLflow is not installed
    import mlflow
    import mlflow.sklearn
    import skops.io as sio

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model. MLflow saves it with skops, which only reloads trusted types;
        # this model was trained in this process, so its own types are trusted.
        trusted = sio.get_untrusted_types(data=sio.dumps(model))
        mlflow.sklearn.log_model(
            model, name=model_name or "model", skops_trusted_types=trusted
        )


def save_model(model: Any, filepath: str | Path, engine: str = "joblib") -> None:
    """
    Save a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model.
    filepath : str or pathlib.Path
        Path to save the model.
    engine : str, optional
        Engine to save the model in ('joblib' or 'pickle'). Default is 'joblib'.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if engine == "joblib":
        joblib.dump(model, filepath)
    elif engine == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
    else:
        raise ValueError(f"Unsupported engine: {engine}")


def load_model(filepath: str | Path, engine: str = "joblib") -> Any:
    """
    Load a trained model from disk.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the saved model.
    engine : str, optional
        Engine the model was saved in ('joblib' or 'pickle'). Default is 'joblib'.

    Returns
    -------
    model : object
        The loaded model.
    """
    filepath = Path(filepath)

    if engine == "joblib":
        return joblib.load(filepath)
    if engine == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported engine: {engine}")
