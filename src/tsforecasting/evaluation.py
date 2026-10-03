"""
Forecast metrics and rolling-origin backtesting shared by every model family.

Every model is wrapped as a *forecast function*::

    forecast_fn(y_train, h, X_train, X_future) -> pandas.DataFrame

returning `h` rows with a `mean` column and, for probabilistic models, quantile
columns `q10` ... `q90`. That single contract lets baselines, statsmodels,
sktime, skforecast, PyTorch and TimesFM run through the same `backtest`.
"""

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd

ForecastFn = Callable[..., pd.DataFrame]
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def qcol(q: float) -> str:
    """Column name for quantile `q`, e.g. 0.1 -> 'q10'."""
    return f"q{round(q * 100)}"


def mae(y, yhat) -> float:
    """$\\mathrm{MAE}=\\frac{1}{H}\\sum_h |y_h-\\hat y_h|$."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    """$\\mathrm{RMSE}=\\sqrt{\\frac{1}{H}\\sum_h (y_h-\\hat y_h)^2}$."""
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def smape(y, yhat) -> float:
    """
    Symmetric MAPE in %.

    $\\mathrm{sMAPE}=\\frac{200}{H}\\sum_h
    \\frac{|y_h-\\hat y_h|}{|y_h|+|\\hat y_h|}$
    """
    y, yhat = np.asarray(y), np.asarray(yhat)
    denom = np.abs(y) + np.abs(yhat)
    ratio = np.divide(
        np.abs(y - yhat), denom, out=np.zeros_like(denom, dtype=float), where=denom != 0
    )
    return float(200 * np.mean(ratio))


def mase_scale(y_train, m: int = 1) -> float:
    """In-sample seasonal-naive MAE: $\\frac{1}{n-m}\\sum_{t>m}|y_t-y_{t-m}|$."""
    y = np.asarray(y_train, dtype=float)
    return float(np.mean(np.abs(y[m:] - y[:-m])))


def mase(y, yhat, y_train, m: int = 1) -> float:
    """
    $\\mathrm{MASE}=\\mathrm{MAE}/\\text{scale}$.

    Below 1 beats the in-sample seasonal naive.
    """
    return mae(y, yhat) / mase_scale(y_train, m)


def pinball_loss(y, yq, q: float) -> float:
    """
    Mean pinball loss.

    $L_q(y,\\hat y_q)=\\max\\big(q(y-\\hat y_q),\\,(q-1)(y-\\hat y_q)\\big)$
    """
    d = np.asarray(y) - np.asarray(yq)
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def interval_coverage(y, lower, upper) -> float:
    """Fraction of observations inside $[\\text{lower}, \\text{upper}]$."""
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(lower)) & (y <= np.asarray(upper))))


def rolling_origin_splits(
    n: int, initial: int, horizon: int, step: int | None = None
) -> Iterator[tuple[slice, slice]]:
    """
    Expanding-window splits: train on $[0, c)$, test on $[c, c+H)$ for cutoffs
    $c = \\text{initial}, \\text{initial}+\\text{step}, \\dots$

    Yields
    ------
    tuple of slice
        Train and test positional slices.
    """
    step = step or horizon
    for c in range(initial, n - horizon + 1, step):
        yield slice(0, c), slice(c, c + horizon)


def backtest(
    forecast_fn: ForecastFn,
    y: pd.Series,
    initial: int,
    horizon: int,
    step: int | None = None,
    X: pd.DataFrame | None = None,
    max_train: int | None = None,
    m: int = 1,
) -> pd.DataFrame:
    """
    Run `forecast_fn` over rolling origins and collect forecasts next to actuals.

    Parameters
    ----------
    forecast_fn : callable
        `(y_train, h, X_train, X_future) -> DataFrame` with `h` rows.
    y : pandas.Series
        Target series.
    initial, horizon, step : int
        See `rolling_origin_splits`.
    X : pandas.DataFrame, optional
        Exogenous variables aligned with `y` (future values must be known).
    max_train : int, optional
        Keep only the last `max_train` points of each training window
        (sliding instead of expanding window; also caps cost).
    m : int
        Seasonal period used for the MASE scale.

    Returns
    -------
    pandas.DataFrame
        Indexed by time with `fold`, `step`, `y`, `scale` and the forecast columns.
    """
    folds = []
    for k, (tr, te) in enumerate(rolling_origin_splits(len(y), initial, horizon, step)):
        if max_train:
            tr = slice(max(0, tr.stop - max_train), tr.stop)
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        X_tr = X.iloc[tr] if X is not None else None
        X_te = X.iloc[te] if X is not None else None
        fc = forecast_fn(y_tr, horizon, X_tr, X_te)
        fc = pd.DataFrame(np.asarray(fc), columns=fc.columns, index=y_te.index)
        fc.insert(0, "y", y_te.to_numpy())
        fc.insert(0, "step", np.arange(1, horizon + 1))
        fc.insert(0, "fold", k)
        fc["scale"] = mase_scale(y_tr, m)
        folds.append(fc)
    return pd.concat(folds)


def score(results: pd.DataFrame) -> dict[str, float]:
    """
    Aggregate metrics over all folds of a `backtest` result.

    Point metrics use `mean`; pinball loss and 80% coverage are added when the
    `q10` ... `q90` columns are present.
    """
    y, yhat = results["y"], results["mean"]
    out = {
        "mae": mae(y, yhat),
        "rmse": rmse(y, yhat),
        "smape": smape(y, yhat),
        "mase": float(np.mean(np.abs(y - yhat) / results["scale"])),
    }
    if all(qcol(q) in results for q in QUANTILES):
        out["pinball"] = float(
            np.mean([pinball_loss(y, results[qcol(q)], q) for q in QUANTILES])
        )
        out["coverage80"] = interval_coverage(y, results["q10"], results["q90"])
    return out
