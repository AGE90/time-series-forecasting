from tsforecasting.utils.paths import data_raw_dir, project_dir


def test_dir_helpers_resolve_from_project_root() -> None:
    assert data_raw_dir("x.csv") == project_dir("data", "raw", "x.csv")
    assert (project_dir() / "pyproject.toml").is_file()
