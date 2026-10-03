"""
Project-relative path helpers.

The project root is found with `pyprojroot.here()` (it looks for `.git/`,
`pyproject.toml`, etc.), so paths resolve the same way from scripts, notebooks
and tests, wherever they run from.

Each helper takes optional extra path parts:

>>> data_raw_dir("dataset.csv")  # <project root>/data/raw/dataset.csv
"""

from collections.abc import Callable
from pathlib import Path

from pyprojroot import here


def make_dir_function(*parts: str) -> Callable[..., Path]:
    """
    Build a function returning paths under `<project root>/<parts>`.

    Parameters
    ----------
    *parts : str
        Subdirectories of the project root, e.g. `"data", "raw"`.

    Returns
    -------
    Callable[..., Path]
        Function that joins any extra arguments onto that directory.
    """

    def dir_path(*args: str) -> Path:
        return here().joinpath(*parts, *args)

    return dir_path


project_dir = make_dir_function()
app_dir = make_dir_function("app")
config_dir = make_dir_function("config")
data_dir = make_dir_function("data")
data_raw_dir = make_dir_function("data", "raw")  # original, immutable data
data_interim_dir = make_dir_function("data", "interim")  # intermediate data
data_processed_dir = make_dir_function("data", "processed")  # model-ready data
data_external_dir = make_dir_function("data", "external")  # third-party data
docs_dir = make_dir_function("docs")
logs_dir = make_dir_function("logs")
models_dir = make_dir_function("models")
notebooks_dir = make_dir_function("notebooks")
references_dir = make_dir_function("references")
reports_dir = make_dir_function("reports")
reports_figures_dir = make_dir_function("reports", "figures")
scripts_dir = make_dir_function("scripts")
tests_dir = make_dir_function("tests")
