"""
Deep-learning forecasters in PyTorch (port of the TensorFlow time-series tutorial).

Tensor convention everywhere: inputs `(batch, input_width, n_features)`,
outputs `(batch, out_steps, n_outputs)`.
"""

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    """
    Sliding windows over a DataFrame (replaces the Keras `WindowGenerator`).

    A window spans `input_width + shift` steps: the first `input_width` are
    inputs, the last `label_width` are labels::

        input_width=6, label_width=1, shift=1   ->  x: t-5..t, y: t+1
        input_width=24, label_width=24, shift=24 -> x: t-23..t, y: t+1..t+24

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized features (time-ordered).
    input_width, label_width, shift : int
        Window geometry.
    label_columns : list of str, optional
        Columns to predict. All columns when None.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        input_width: int,
        label_width: int,
        shift: int,
        label_columns: list[str] | None = None,
    ) -> None:
        self.data = torch.tensor(df.to_numpy(np.float32))
        self.input_width, self.label_width, self.shift = input_width, label_width, shift
        self.total = input_width + shift
        cols = list(df.columns)
        self.label_columns = label_columns or cols
        self.label_idx = [cols.index(c) for c in self.label_columns]

    def __len__(self) -> int:
        return len(self.data) - self.total + 1

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        w = self.data[i : i + self.total]
        return w[: self.input_width], w[self.total - self.label_width :, self.label_idx]

    def __repr__(self) -> str:
        return (
            f"WindowDataset(total={self.total}, inputs=[0..{self.input_width - 1}], "
            f"labels=[{self.total - self.label_width}..{self.total - 1}], "
            f"label_columns={self.label_columns})"
        )

    def loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Mini-batch iterator over the windows."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


def _mlp(n_in: int, hidden: tuple[int, ...], n_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(n_in, h), nn.ReLU()]
        n_in = h
    return nn.Sequential(*layers, nn.Linear(n_in, n_out))


class Dense(nn.Module):
    """Per-time-step MLP $\\hat y_t = f(x_t)$; `hidden=()` gives the linear model."""

    def __init__(self, n_in: int, n_out: int, hidden: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.net = _mlp(n_in, hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiStepDense(nn.Module):
    """Flatten the whole input window and emit all `out_steps` at once (single-shot)."""

    def __init__(
        self,
        n_in: int,
        input_width: int,
        n_out: int,
        out_steps: int = 1,
        hidden: tuple[int, ...] = (32, 32),
    ) -> None:
        super().__init__()
        self.out_steps, self.n_out = out_steps, n_out
        self.net = _mlp(n_in * input_width, hidden, out_steps * n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1)).view(-1, self.out_steps, self.n_out)


class CNN(nn.Module):
    """
    1-D convolution over time with kernel size $k$.

    `out_steps=None`: one prediction per position (output length $T-k+1$).
    `out_steps=H`: use only the last $k$ steps and emit $H$ steps at once.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        kernel_size: int,
        out_steps: int | None = None,
        channels: int = 32,
    ) -> None:
        super().__init__()
        self.k, self.out_steps, self.n_out = kernel_size, out_steps, n_out
        self.conv = nn.Sequential(nn.Conv1d(n_in, channels, kernel_size), nn.ReLU())
        self.head = nn.Linear(channels, (out_steps or 1) * n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_steps:
            x = x[:, -self.k :]
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T-k+1, C)
        if self.out_steps:
            return self.head(z[:, -1]).view(-1, self.out_steps, self.n_out)
        return self.head(z)


class LSTM(nn.Module):
    """
    LSTM encoder.

    `out_steps=None`: a prediction at every input step (sequence-to-sequence).
    `out_steps=H`: single-shot $H$-step forecast from the final hidden state.
    """

    def __init__(
        self, n_in: int, n_out: int, hidden: int = 32, out_steps: int | None = None
    ) -> None:
        super().__init__()
        self.out_steps, self.n_out = out_steps, n_out
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, (out_steps or 1) * n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.lstm(x)
        if self.out_steps:
            return self.head(z[:, -1]).view(-1, self.out_steps, self.n_out)
        return self.head(z)


class FeedBack(nn.Module):
    """
    Autoregressive LSTM: warm up on the inputs, then feed each prediction back
    as the next input:
    $\\hat x_{t+1} = g(h_t)$, $h_{t+1} = \\mathrm{LSTM}(\\hat x_{t+1}, h_t)$.
    Predicts all features, so labels must be all columns.
    """

    def __init__(self, n_features: int, out_steps: int, hidden: int = 32) -> None:
        super().__init__()
        self.out_steps = out_steps
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.cell = nn.LSTMCell(n_features, hidden)
        self.head = nn.Linear(hidden, n_features)
        # The cell shares weights with the warm-up LSTM, like Keras' RNN(LSTMCell).
        self.cell.weight_ih, self.cell.weight_hh = (
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
        )
        self.cell.bias_ih, self.cell.bias_hh = (
            self.lstm.bias_ih_l0,
            self.lstm.bias_hh_l0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, (h, c) = self.lstm(x)
        state = (h[0], c[0])
        pred = self.head(z[:, -1])
        preds = [pred]
        for _ in range(1, self.out_steps):
            state = self.cell(pred, state)
            pred = self.head(state[0])
            preds.append(pred)
        return torch.stack(preds, dim=1)


class Residual(nn.Module):
    """$\\hat y_t = x_t + f(x_t)$: the model learns the change, not the level."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


def evaluate(
    model: nn.Module, loader: DataLoader, device: str = "cpu"
) -> dict[str, float]:
    """Mean MSE and MAE of `model` over a loader."""
    model.eval()
    se = ae = n = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            d = model(x) - y
            se += float((d**2).sum())
            ae += float(d.abs().sum())
            n += d.numel()
    return {"mse": se / n, "mae": ae / n}


def fit(
    model: nn.Module,
    train: DataLoader,
    val: DataLoader,
    epochs: int = 20,
    patience: int = 2,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """
    Adam + MSE training with early stopping on validation MSE.

    The best weights (lowest validation loss) are restored at the end.

    Returns
    -------
    dict
        `train_mse` and `val_mse` per epoch.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {"train_mse": [], "val_mse": []}
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            loss = nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, count = total + float(loss) * len(x), count + len(x)
        val_mse = evaluate(model, val, device)["mse"]
        history["train_mse"].append(total / count)
        history["val_mse"].append(val_mse)
        if val_mse < best:
            best, best_state, bad = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def window_forecast_fn(
    model: nn.Module,
    input_width: int,
    columns: list[str],
    target: str,
    mean: pd.Series,
    std: pd.Series,
    device: str = "cpu",
):
    """
    Adapt a trained single-shot model to the `forecast_fn` contract.

    The model is *not* refit per fold: it reads the last `input_width` rows of
    `X_train` (which must contain every column in `columns`, target included),
    and its output for `target` is de-normalized. `out_steps` must equal `h`.
    """
    t = columns.index(target)

    def fn(y_train, h, X_train, X_future=None):
        x = (
            (X_train[columns].iloc[-input_width:] - mean[columns]) / std[columns]
        ).to_numpy(np.float32)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(x[None], device=device))[0].cpu().numpy()
        col = out[:, t] if out.shape[1] > 1 else out[:, 0]
        return pd.DataFrame({"mean": col[:h] * std[target] + mean[target]})

    return fn
