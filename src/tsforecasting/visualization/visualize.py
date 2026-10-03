"""
Visualization utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_distribution(
    data: pd.Series | np.ndarray,
    title: str,
    xlabel: str,
    bins: int = 30,
    figsize: tuple[int, int] = (10, 6),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot the distribution of a numerical variable.

    Parameters
    ----------
    data : pandas.Series or numpy.ndarray
        Input data.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    bins : int, optional
        Number of histogram bins (default is 30).
    figsize : tuple, optional
        Figure size (default is (10, 6)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    sns.histplot(data=data, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a correlation matrix heatmap.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    title : str, optional
        Plot title (default is 'Correlation Matrix').
    figsize : tuple, optional
        Figure size (default is (12, 8)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True
    )
    plt.title(title)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_time_series(
    df: pd.DataFrame,
    time_column: str,
    value_column: str,
    title: str,
    figsize: tuple[int, int] = (12, 6),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    time_column : str
        Name of the time column.
    value_column : str
        Name of the value column.
    title : str
        Plot title.
    figsize : tuple, optional
        Figure size (default is (12, 6)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    plt.plot(df[time_column], df[value_column])
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Box Plots",
    figsize: tuple[int, int] = (12, 6),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot box plots for multiple columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        List of columns to plot.
    title : str, optional
        Plot title (default is 'Box Plots').
    figsize : tuple, optional
        Figure size (default is (12, 6)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    df[columns].boxplot()
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_scatter_matrix(
    df: pd.DataFrame,
    columns: list[str],
    title: str = "Scatter Matrix",
    figsize: tuple[int, int] = (12, 12),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a scatter matrix for multiple columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        List of columns to plot.
    title : str, optional
        Plot title (default is 'Scatter Matrix').
    figsize : tuple, optional
        Figure size (default is (12, 12)).
    save_path : str or pathlib.Path, optional
        Path to save the plot (default is None).

    Returns
    -------
    None
    """
    plt.figure(figsize=figsize)
    pd.plotting.scatter_matrix(df[columns], diagonal="kde", figsize=figsize)
    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
