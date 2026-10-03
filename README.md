# Time Series Forecasting

A laboratory for classical, ML, deep learning and foundation-model time-series forecasting

<!-- Add a brief overview of the project here. -->

---

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.11 for you if it is missing.

```bash
git clone <repository-url>
cd time-series-forecasting
make install                  # uv sync --all-groups: creates .venv with every dependency group
uv run pre-commit install     # optional: lint and format on every commit
```

Run any command inside the environment with `uv run <command>`, or activate it with `source .venv/bin/activate`. Add dependencies with `uv add <package>` (or `uv add --group <dev|test|notebook|data-science|viz> <package>`) and update them with `uv lock --upgrade && uv sync`.

---

## Usage

Run `make help` to list every task.

### Pipeline

The pipeline is a set of small stubs to edit for your problem. Each step reads the previous step's output:

| Command | Module | Reads | Writes |
|---|---|---|---|
| `make data` | `data/make_dataset.py` | `data/raw/dataset.csv` | `data/interim/dataset_clean.csv` |
| `make features` | `features/build_features.py` | `data/interim/dataset_clean.csv` | `data/processed/features.csv` |
| `make train` | `models/train_model.py` | `data/processed/features.csv` | `models/model.joblib` |
| `make predict` | `models/predict_model.py` | `models/model.joblib` + features | `data/processed/predictions.csv` |

`make pipeline` runs `data`, `features` and `train` in order. File names and the target column (`TARGET = "target"`) are constants at the top of each module.

### Code quality and tests

```bash
make check    # ruff format + ruff check + mypy
make test     # pytest with coverage
```

### Paths and data

Never hardcode paths. The helpers in `utils/paths.py` resolve from the project root, so they work the same in scripts, notebooks and tests:

```python
from tsforecasting.data.data_loader import load_csv
from tsforecasting.utils.paths import data_raw_dir, reports_figures_dir
from tsforecasting.visualization.visualize import plot_distribution

df = load_csv(data_raw_dir("dataset.csv"))
plot_distribution(
    df["size"],
    title="Size distribution",
    xlabel="size",
    save_path=reports_figures_dir("size.png"),
)
```

### Notebooks

Start Jupyter Lab with `make notebook`. Put reusable code in `src/` and import it; add this at the top of a notebook to pick up code changes without restarting the kernel:

```python
%load_ext autoreload
%autoreload 2
```

### Experiment tracking (MLflow)

`make train` logs parameters, metrics and the model to MLflow with `log_mlflow_experiment` (in `models/model_utils.py`). Runs are stored in `mlflow.db` and `mlruns/` at the project root (both git-ignored). Browse them with:

```bash
make mlflow-ui    # http://127.0.0.1:5000
```

---

## Project Structure

```text
├── CLAUDE.md               <- Project conventions for Claude Code
├── Makefile                <- Tasks: `make help`
├── pyproject.toml          <- Metadata, dependency groups and tool configuration
├── uv.lock                 <- Locked dependency versions
├── app/                    <- Application entry point (if applicable)
├── config/                 <- Configuration files
├── data/
│   ├── raw/                <- Original, immutable data
│   ├── interim/            <- Intermediate, cleaned data
│   ├── processed/          <- Final, model-ready data
│   └── external/           <- Data from third-party sources
├── docs/                   <- Developer guide and code of conduct
├── logs/                   <- Log files
├── models/                 <- Trained models
├── notebooks/              <- Exploration notebooks, named e.g. `01-abc-initial-eda.ipynb`
├── references/             <- Data dictionaries, manuals, papers
├── reports/figures/        <- Generated figures
├── scripts/                <- Helper shell scripts
├── src/tsforecasting/
│   ├── credentials.py      <- Loads secrets from `.env`
│   ├── data/               <- Loading (`data_loader.py`) and cleaning (`make_dataset.py`)
│   ├── features/           <- Feature engineering helpers and `build_features.py`
│   ├── models/             <- Model helpers, `train_model.py`, `predict_model.py`
│   ├── utils/paths.py      <- Project-relative path helpers
│   └── visualization/      <- Plotting helpers
└── tests/                  <- `unit/` and `e2e/` tests
```

---

## Documentation

- [Developer Guide](docs/developer_guide.md): code style, testing, Git workflow and contributing
- [Code of Conduct](docs/code_of_conduct.md)

---

## License

This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for details.
