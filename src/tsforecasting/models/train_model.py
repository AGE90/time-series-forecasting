"""
Benchmark every model family on the same Jena backtest and log to MLflow.

Run with `make train` (after `make data features`). Task: forecast the next
24 h of temperature from weekly origins in the test period (last 10%).
Writes `reports/leaderboard.csv`.
"""

import warnings

import lightgbm as lgb
import mlflow
import pandas as pd
from skforecast.recursive import ForecasterRecursive
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.transformations.series.detrend import Deseasonalizer
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tsforecasting.data.data_loader import load_jena, temporal_split
from tsforecasting.evaluation import backtest, score
from tsforecasting.features.build_features import TARGET
from tsforecasting.models import baselines
from tsforecasting.models.deep import LSTM, WindowDataset, fit, window_forecast_fn
from tsforecasting.models.foundation import (
    get_device,
    load_timesfm,
    timesfm_forecast_fn,
)
from tsforecasting.utils.paths import project_dir, reports_dir

HORIZON = 24
N_FOLDS = 20
SEASON = 24
STAT_TRAIN = 24 * 28  # statistical models refit per fold on the last 4 weeks
INPUT_WIDTH = 72
CALENDAR = ["Day sin", "Day cos", "Year sin", "Year cos"]
# Same store as `make mlflow-ui`, wherever this runs from (e.g. notebooks/)
TRACKING_URI = f"sqlite:///{project_dir('mlflow.db')}"


def ets_fn(y, h, *_):
    """Additive Holt-Winters (damped trend) with daily seasonality."""
    m = ExponentialSmoothing(
        y.to_numpy(),
        trend="add",
        damped_trend=True,
        seasonal="add",
        seasonal_periods=SEASON,
    ).fit()
    return pd.DataFrame({"mean": m.forecast(h)})


def theta_fn(y, h, *_):
    """sktime Theta on an additively deseasonalized series (temperatures go below 0)."""
    f = TransformedTargetForecaster(
        [
            ("deseason", Deseasonalizer(sp=SEASON, model="additive")),
            ("theta", ThetaForecaster(deseasonalize=False)),
        ]
    ).fit(y.reset_index(drop=True))
    return pd.DataFrame({"mean": f.predict(fh=list(range(1, h + 1))).to_numpy()})


def lightgbm_fn(train: pd.DataFrame):
    """skforecast recursive LightGBM on 48 lags + calendar exog, fit once on `train`."""
    f = ForecasterRecursive(lgb.LGBMRegressor(n_estimators=300, verbose=-1), lags=48)
    f.fit(y=train[TARGET], exog=train[CALENDAR])

    def fn(y, h, X_train, X_future):
        return pd.DataFrame(
            {
                "mean": f.predict(
                    h, last_window=y.iloc[-48:], exog=X_future[CALENDAR]
                ).to_numpy()
            }
        )

    return fn


def lstm_fn(train: pd.DataFrame, val: pd.DataFrame, device: str):
    """Single-shot LSTM on all features, fit once on `train` with early stopping."""
    mean, std = train.mean(), train.std()
    norm = lambda d: (d - mean) / std  # noqa: E731
    cols = list(train.columns)
    ds = lambda d: WindowDataset(norm(d), INPUT_WIDTH, HORIZON, HORIZON, [TARGET])  # noqa: E731
    model = LSTM(len(cols), 1, hidden=32, out_steps=HORIZON)
    fit(
        model,
        ds(train).loader(),
        ds(val).loader(shuffle=False),
        epochs=10,
        device=device,
    )
    return window_forecast_fn(model, INPUT_WIDTH, cols, TARGET, mean, std, device)


def main() -> None:
    """Run the benchmark, log one MLflow run per model, write the leaderboard."""
    warnings.filterwarnings("ignore")
    df = load_jena()
    train, val, _ = temporal_split(df)
    y, initial = df[TARGET], len(train) + len(val)
    step = (len(df) - initial - HORIZON) // N_FOLDS
    device = get_device()

    models = {
        "naive": (baselines.naive, None),
        "seasonal_naive": (baselines.seasonal_naive(SEASON), None),
        "ets_holt_winters": (ets_fn, STAT_TRAIN),
        "theta_sktime": (theta_fn, STAT_TRAIN),
        "lightgbm_skforecast": (lightgbm_fn(train), None),
        "lstm_pytorch": (lstm_fn(train, val, device), None),
        "timesfm3_zero_shot": (
            timesfm_forecast_fn(load_timesfm(device), context_len=1024),
            None,
        ),
    }

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("jena-24h-benchmark")
    scores = {}
    for name, (fn, max_train) in models.items():
        res = backtest(
            fn, y, initial, HORIZON, step, X=df, max_train=max_train, m=SEASON
        )
        scores[name] = score(res)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(
                {
                    "horizon": HORIZON,
                    "folds": N_FOLDS,
                    "max_train": max_train,
                    "device": device,
                }
            )
            mlflow.log_metrics(scores[name])
        print(f"{name:22s} {scores[name]}")

    board = pd.DataFrame(scores).T.sort_values("mase")
    board.to_csv(reports_dir("leaderboard.csv"))
    print(board.round(3))


if __name__ == "__main__":
    main()
