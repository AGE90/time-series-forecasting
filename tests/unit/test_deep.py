import numpy as np
import pandas as pd
import torch

from tsforecasting.models.deep import (
    CNN,
    LSTM,
    Dense,
    FeedBack,
    MultiStepDense,
    WindowDataset,
)


def test_window_dataset_geometry() -> None:
    df = pd.DataFrame({"a": np.arange(50.0), "b": -np.arange(50.0)})
    ds = WindowDataset(df, input_width=6, label_width=1, shift=1, label_columns=["a"])
    x, y = ds[0]
    assert x.shape == (6, 2) and y.shape == (1, 1)
    assert y.item() == 6.0  # label is the step right after the inputs
    assert len(ds) == 50 - 7 + 1


def test_model_output_shapes() -> None:
    x = torch.randn(4, 24, 3)
    assert Dense(3, 1, hidden=(8,))(x).shape == (4, 24, 1)
    assert MultiStepDense(3, 24, 1, out_steps=12)(x).shape == (4, 12, 1)
    assert CNN(3, 1, kernel_size=3)(x).shape == (4, 22, 1)
    assert CNN(3, 1, kernel_size=3, out_steps=12)(x).shape == (4, 12, 1)
    assert LSTM(3, 1)(x).shape == (4, 24, 1)
    assert LSTM(3, 1, out_steps=12)(x).shape == (4, 12, 1)
    assert FeedBack(3, out_steps=12)(x).shape == (4, 12, 3)
