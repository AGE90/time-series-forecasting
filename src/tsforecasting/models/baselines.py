"""
Naive benchmarks every serious model must beat. All follow the `forecast_fn`
contract from `tsforecasting.evaluation`.
"""

import numpy as np
import pandas as pd


def naive(y, h, *_):
    """$\\hat y_{T+h}=y_T$ (random-walk forecast)."""
    return pd.DataFrame({"mean": np.repeat(np.asarray(y)[-1], h)})


def seasonal_naive(m: int):
    """
    Repeat the last season of length $m$: $\\hat y_{T+h}=y_{T+h-m\\lceil h/m\\rceil}$.
    """

    def fn(y, h, *_):
        last = np.asarray(y)[-m:]
        return pd.DataFrame({"mean": np.resize(last, h)})

    return fn


def mean(y, h, *_):
    """$\\hat y_{T+h}=\\bar y$."""
    return pd.DataFrame({"mean": np.repeat(np.mean(y), h)})


def drift(y, h, *_):
    """
    Line through first and last point: $\\hat y_{T+h}=y_T + h\\,\\frac{y_T-y_1}{T-1}$.
    """
    y = np.asarray(y, dtype=float)
    slope = (y[-1] - y[0]) / (len(y) - 1)
    return pd.DataFrame({"mean": y[-1] + slope * np.arange(1, h + 1)})
