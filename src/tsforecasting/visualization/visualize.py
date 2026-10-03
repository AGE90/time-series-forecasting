"""
Plotting helpers for the forecasting notebooks. Each returns the matplotlib
Axes/Figure so notebooks can keep customizing.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from tsforecasting.evaluation import rolling_origin_splits


def plot_forecast(
    history: pd.Series,
    forecast: pd.DataFrame,
    actual: pd.Series | None = None,
    title: str = "",
    ax=None,
):
    """
    History, optional actuals, the `mean` forecast and, if present, the
    $[q_{10}, q_{90}]$ and $[q_{30}, q_{70}]$ bands.
    """
    ax = ax or plt.subplots(figsize=(12, 4))[1]
    ax.plot(history.index, history, color="0.3", lw=1, label="history")
    if actual is not None:
        ax.plot(actual.index, actual, color="black", lw=1.5, label="actual")
    idx = (
        actual.index
        if actual is not None and len(actual) == len(forecast)
        else forecast.index
    )
    for lo, hi, alpha in [("q10", "q90", 0.2), ("q30", "q70", 0.35)]:
        if lo in forecast and hi in forecast:
            ax.fill_between(
                idx,
                forecast[lo],
                forecast[hi],
                color="C0",
                alpha=alpha,
                label=f"{lo}-{hi}",
                lw=0,
            )
    ax.plot(idx, forecast["mean"], color="C0", lw=2, label="forecast")
    ax.axvline(history.index[-1], color="0.6", ls="--", lw=1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_backtest_folds(
    n: int, initial: int, horizon: int, step: int | None = None, ax=None
):
    """Diagram of rolling-origin splits: one row per fold, train vs test spans."""
    splits = list(rolling_origin_splits(n, initial, horizon, step))
    ax = ax or plt.subplots(figsize=(10, 0.35 * len(splits) + 1))[1]
    for k, (tr, te) in enumerate(splits):
        ax.add_patch(
            Rectangle((tr.start, k), tr.stop - tr.start, 0.8, color="C0", alpha=0.6)
        )
        ax.add_patch(Rectangle((te.start, k), te.stop - te.start, 0.8, color="C1"))
    ax.set_xlim(0, n)
    ax.set_ylim(len(splits), -0.2)
    ax.set_xlabel("time index")
    ax.set_ylabel("fold")
    ax.legend(
        handles=[
            Rectangle((0, 0), 1, 1, color="C0", alpha=0.6),
            Rectangle((0, 0), 1, 1, color="C1"),
        ],
        labels=["train", "test"],
    )
    return ax


def plot_backtest(results: pd.DataFrame, y: pd.Series, title: str = "", ax=None):
    """Actual series with each fold's forecast overlaid."""
    ax = ax or plt.subplots(figsize=(12, 4))[1]
    span = y.loc[results.index.min() : results.index.max()]
    ax.plot(span.index, span, color="black", lw=1, label="actual")
    for k, fold in results.groupby("fold"):
        if "q10" in fold:
            ax.fill_between(
                fold.index, fold["q10"], fold["q90"], color="C0", alpha=0.2, lw=0
            )
        ax.plot(
            fold.index,
            fold["mean"],
            color="C0",
            lw=1.5,
            label="forecast" if k == 0 else None,
        )
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_acf_pacf(y: pd.Series, lags: int = 48):
    """ACF and PACF side by side (statsmodels)."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    plot_acf(y, lags=lags, ax=axes[0])
    plot_pacf(y, lags=lags, ax=axes[1], method="ywm")
    fig.tight_layout()
    return fig


def plot_decomposition(y: pd.Series, period: int, robust: bool = True):
    """STL decomposition $y_t = T_t + S_t + R_t$."""
    from statsmodels.tsa.seasonal import STL

    res = STL(y, period=period, robust=robust).fit()
    fig = res.plot()
    fig.set_size_inches(12, 7)
    return fig


def plot_periodogram(
    y: pd.Series, samples_per_unit: float, unit: str = "year", ax=None
):
    """Amplitude spectrum $|\\mathcal{F}(y)|$ against frequency in cycles per `unit`."""
    ax = ax or plt.subplots(figsize=(10, 3.5))[1]
    amp = np.abs(np.fft.rfft(np.asarray(y) - np.mean(y)))
    freq = np.fft.rfftfreq(len(y), d=1 / samples_per_unit)
    ax.step(freq[1:], amp[1:])
    ax.set_xscale("log")
    ax.set_xlabel(f"frequency [cycles / {unit}]")
    ax.set_ylabel("amplitude")
    return ax


def plot_leaderboard(scores: pd.DataFrame, metric: str = "mase", ax=None):
    """Horizontal bar chart of `scores[metric]` (rows = models), best on top."""
    s = scores[metric].sort_values(ascending=False)
    ax = ax or plt.subplots(figsize=(8, 0.4 * len(s) + 1))[1]
    ax.barh(s.index, s.to_numpy(), color=["C1" if v == s.min() else "C0" for v in s])
    if metric == "mase":
        ax.axvline(1.0, color="0.4", ls="--", lw=1)
    ax.set_xlabel(metric)
    return ax
