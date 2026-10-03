"""
Zero-shot foundation models: TimesFM 3 (google/timesfm-3.0-pytorch, 330M params).

TimesFM 3 splits each series into patches of $P=32$ points, runs a decoder-only
transformer that alternates causal attention over time with full attention
across variates, and emits 9 quantiles (10th..90th percentile) for the whole
horizon in a single forward pass.
"""

import warnings

import numpy as np
import pandas as pd
import torch

from tsforecasting.evaluation import QUANTILES, qcol


def get_device() -> str:
    """
    Return `"cuda"` only if a kernel actually runs on the GPU, else `"cpu"`.

    `torch.cuda.is_available()` alone is not enough on old GPUs: a wheel built
    without sm_50 (Maxwell, e.g. GTX 960M) or a driver older than the wheel's
    CUDA reports available/unavailable inconsistently or fails at the first op.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if not torch.cuda.is_available():
                return "cpu"
            (torch.ones(2, device="cuda") * 2).sum().item()
        return "cuda"
    except Exception:
        return "cpu"


def load_timesfm(device: str | None = None):
    """Load TimesFM 3 from the Hugging Face Hub (~1.3 GB, cached after first call)."""
    from timesfm3 import TimesFM3Forecaster

    return TimesFM3Forecaster.from_pretrained(device=device or get_device())


def timesfm_forecast_fn(
    model, context_len: int = 512, covariates: list[str] | None = None
):
    """
    Adapt TimesFM 3 to the `forecast_fn` contract (zero-shot, no fitting).

    Parameters
    ----------
    model : TimesFM3Forecaster
        Loaded model.
    context_len : int
        Number of most recent points fed as context (max 15360).
    covariates : list of str, optional
        Columns of `X` passed as past-and-future-known covariates
        (e.g. calendar features), using `X_train` and `X_future`.
    """

    def fn(y_train, h, X_train=None, X_future=None):
        ctx = np.asarray(y_train, dtype=np.float32)[-context_len:]
        pf = None
        if covariates:
            past = X_train[covariates].iloc[-len(ctx) :]
            pf = pd.concat([past, X_future[covariates].iloc[:h]]).to_numpy(np.float32).T
        out = model.predict(
            ctx, horizon=h, past_future_covariates=pf, return_quantiles=True
        )
        df = pd.DataFrame(out.quantiles, columns=[qcol(q) for q in QUANTILES])
        df.insert(0, "mean", out.forecast)
        return df

    return fn
