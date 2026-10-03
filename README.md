# Time Series Forecasting Lab

A laboratory for studying time-series forecasting, from classical statistics to zero-shot foundation models. Every model family runs through the **same rolling-origin backtest and metrics**, so the numbers are comparable. The notebooks explain each method with math (`$…$`) and visualizations. Reusable logic lives in `src/tsforecasting/`.

| # | Notebook | Topics | Libraries |
|---|---|---|---|
| 01 | `data_exploration` | STL decomposition, ACF/PACF, ADF/KPSS, periodogram, cyclical features | statsmodels |
| 02 | `baselines_and_evaluation` | naive/seasonal naive, MAE/RMSE/sMAPE/MASE, pinball loss, rolling-origin backtesting | `tsforecasting.evaluation` |
| 03 | `statistical_models` | Holt-Winters/ETS, AutoETS, SARIMA (airline model), residual diagnostics, Theta | statsmodels, sktime |
| 04 | `ml_forecasting_skforecast` | reduction to regression, lags/window features/exog, recursive vs direct, bootstrap and conformal intervals | skforecast, LightGBM |
| 05 | `sktime_unified_interface` | `fh`, reduction, transform pipelines, ensembles, `evaluate` | sktime |
| 06 | `deep_learning_pytorch` | windowing, linear/dense/CNN/LSTM, single-shot vs autoregressive, residual nets | PyTorch |
| 07 | `timesfm3_zero_shot` | patching, variate attention, quantile head, context length, calibration, multivariate and covariates, CPU vs GPU | TimesFM 3 |
| 08 | `benchmark` | every family on the same 24 h task, error by lead time, per-fold spread, MLflow | all |

**Datasets:** Jena climate (hourly, 14 variables, 2009–2016, downloaded by `make data`) and Box-Jenkins airline passengers (monthly, from sktime).

## Benchmark (`make train`)

This table shows 24 h-ahead temperature forecasts from 20 origins over the last 10% of the Jena data. MASE below 1 beats the in-sample seasonal naive.

| model | MASE | RMSE [°C] | 80% coverage |
|---|---|---|---|
| LSTM (PyTorch, single-shot) | 0.682 | 2.35 | – |
| LightGBM (skforecast, recursive) | 0.699 | 2.39 | – |
| **TimesFM 3, zero-shot** | 0.714 | **2.29** | 0.81 |
| seasonal naive | 0.889 | 2.90 | – |
| Theta (sktime) | 1.043 | 3.15 | – |
| Holt-Winters (statsmodels) | 1.210 | 3.80 | – |
| naive | 1.247 | 4.58 | – |

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.11 is installed automatically.

```bash
make install          # all dependency groups (torch, sktime, skforecast, timesfm, …)
make data features    # download Jena climate -> data/interim -> data/processed
make train            # benchmark, logged to MLflow; `make mlflow-ui` to browse
make notebook         # Jupyter Lab
make help             # every task
```

The first TimesFM call downloads `google/timesfm-3.0-pytorch` (~1.3 GB) to the Hugging Face cache.

## Package layout

```text
src/tsforecasting/
├── evaluation.py            metrics, rolling_origin_splits, backtest, score
├── data/make_dataset.py     download + hourly cleaning (make data)
├── data/data_loader.py      load_jena, load_airline, temporal_split
├── features/build_features.py  wind vector, sin/cos calendar features (make features)
├── models/baselines.py      naive, seasonal_naive, mean, drift
├── models/deep.py           WindowDataset, Dense, MultiStepDense, CNN, LSTM, FeedBack, Residual, fit
├── models/foundation.py     get_device, load_timesfm, timesfm_forecast_fn
├── models/train_model.py    the benchmark (make train)
└── visualization/visualize.py  forecast/backtest/fold/ACF/STL/periodogram/leaderboard plots
```

Every model is wrapped as `forecast_fn(y_train, h, X_train, X_future) -> DataFrame[mean, q10…q90]`, which plugs into `evaluation.backtest`.

## GPU notes (GTX 960M and other old cards)

The default install is **CPU**. TimesFM 3 runs fine there, at about 0.6 s per series with a 512-step context. `get_device()` returns `"cuda"` only if a kernel actually runs, so code never crashes on an unusable GPU.

A GTX 960M is Maxwell (sm_50) with 2 GB of VRAM. It can't use the default torch wheel (CUDA 13), and CUDA 12.8+ wheels dropped sm_50. To experiment anyway (driver must support CUDA ≥ 12.6):

```bash
uv pip install "torch==2.7.*" --index-url https://download.pytorch.org/whl/cu126
uv run python -c "from tsforecasting.models.foundation import get_device; print(get_device())"
```

Then run the timing cell at the end of notebook 07. TimesFM's fp32 weights (~1.3 GB) barely fit, so expect a modest speed-up at best. `uv sync` restores the CPU setup.

## Development

```bash
make check   # ruff format + ruff check + mypy
make test    # pytest
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
