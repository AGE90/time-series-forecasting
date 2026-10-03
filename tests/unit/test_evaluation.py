import numpy as np
import pandas as pd

from tsforecasting.evaluation import (
    backtest,
    mase,
    pinball_loss,
    rolling_origin_splits,
    score,
    smape,
)
from tsforecasting.models.baselines import naive, seasonal_naive


def test_metrics_on_hand_computed_values() -> None:
    assert smape([1, 2], [1, 2]) == 0
    assert np.isclose(pinball_loss([10], [8], 0.9), 0.9 * 2)
    assert np.isclose(pinball_loss([10], [12], 0.9), 0.1 * 2)
    # naive in-sample MAE of [1, 2, 4] is 1.5
    assert np.isclose(mase([5], [2], [1, 2, 4]), 3 / 1.5)


def test_rolling_origin_splits_never_leak() -> None:
    splits = list(rolling_origin_splits(n=10, initial=4, horizon=2, step=2))
    assert [(te.start, te.stop) for _, te in splits] == [(4, 6), (6, 8), (8, 10)]
    assert all(tr.stop == te.start for tr, te in splits)


def test_seasonal_naive_is_perfect_on_a_pure_season() -> None:
    y = pd.Series(
        np.tile([1.0, 5.0, 3.0], 10), index=pd.date_range("2020", periods=30, freq="h")
    )
    res = backtest(seasonal_naive(3), y, initial=12, horizon=3, m=3)
    assert score(res)["mae"] == 0
    assert score(backtest(naive, y, initial=12, horizon=3, m=3))["mase"] > 0
