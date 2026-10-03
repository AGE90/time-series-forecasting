# Time Series Forecasting

A laboratory for classical, ML, deep learning and foundation-model time-series forecasting

## Conventions

- Python package: `src/tsforecasting/` (data, features, models, visualization, utils).
- Environment: uv. Run everything with `uv run <cmd>`; add deps with `uv add <pkg>` (or `uv add --group <dev|test|notebook|data-science|viz> <pkg>`). Never use pip directly.
- Paths: never hardcode. Use the helpers in `tsforecasting.utils.paths` (`data_raw_dir("file.csv")`, `data_processed_dir(...)`, `models_dir(...)`, `reports_figures_dir(...)`, ...).
- Data flow: `data/raw` is immutable input -> `data/interim` -> `data/processed` (model-ready). Third-party data goes in `data/external`. Data files are not committed to git.
- Notebooks in `notebooks/` are for exploration only; move reusable code into `src/`.
- Trained models go in `models/`, figures in `reports/figures/`.
- Secrets go in `.env` (git-ignored), loaded via `tsforecasting.credentials`.
- Docstrings: numpy style. Type hints on public functions.

## Commands

- `make install`: install all dependency groups
- `make data` / `make features` / `make train`: download+clean Jena, build features, run the 24 h benchmark of every model family (MLflow + `reports/leaderboard.csv`)
- Models follow the `forecast_fn(y_train, h, X_train, X_future) -> DataFrame[mean, q10..q90]` contract in `tsforecasting.evaluation` so they share `backtest`/`score`
- Notebooks are generated with outputs; keep math in `$...$` and lead with plots
- `make check`: ruff format + ruff check + mypy
- `make test`: pytest with coverage (tests live in `tests/unit` and `tests/e2e`)
- `make mlflow-ui`: MLflow tracking UI
- `make help`: list every target
