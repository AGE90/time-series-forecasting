"""
Feature engineering utilities for the project.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def scale_features(
    df: pd.DataFrame, columns: list[str], scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        List of columns to scale.
    scaler : sklearn.preprocessing.StandardScaler, optional
        Optional pre-fitted scaler. If None, a new scaler is fitted.

    Returns
    -------
    df_scaled : pandas.DataFrame
        Scaled DataFrame.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler.
    """
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[columns])

    df_scaled = df.copy()
    df_scaled[columns] = scaler.transform(df[columns])
    return df_scaled, scaler


def encode_categorical(
    df: pd.DataFrame,
    columns: list[str],
    encoder: OneHotEncoder | None = None,
    drop: str = "first",
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """
    One-hot encode categorical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        List of columns to encode.
    encoder : sklearn.preprocessing.OneHotEncoder, optional
        Optional pre-fitted encoder. If None, a new encoder is fitted.
    drop : str, optional
        Strategy for dropping categories (default is 'first').

    Returns
    -------
    df_encoded : pandas.DataFrame
        Encoded DataFrame.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder.
    """
    if encoder is None:
        encoder = OneHotEncoder(drop=drop, sparse_output=False)
        encoder.fit(df[columns])

    encoded = encoder.transform(df[columns])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(columns), index=df.index
    )

    df_encoded = df.drop(columns=columns).join(encoded_df)
    return df_encoded, encoder


def create_time_features(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Create time-based features from a datetime column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    datetime_column : str
        Name of the datetime column.

    Returns
    -------
    df_time : pandas.DataFrame
        DataFrame with added time features (year, month, day, hour, dayofweek).
    """
    df_time = df.copy()
    df_time[datetime_column] = pd.to_datetime(df_time[datetime_column])

    # Extract time components
    df_time[f"{datetime_column}_year"] = df_time[datetime_column].dt.year
    df_time[f"{datetime_column}_month"] = df_time[datetime_column].dt.month
    df_time[f"{datetime_column}_day"] = df_time[datetime_column].dt.day
    df_time[f"{datetime_column}_hour"] = df_time[datetime_column].dt.hour
    df_time[f"{datetime_column}_dayofweek"] = df_time[datetime_column].dt.dayofweek

    return df_time


def create_interaction_features(
    df: pd.DataFrame, columns: list[str], operation: str = "multiply"
) -> pd.DataFrame:
    """
    Create interaction features between columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list of str
        List of columns to create interactions from.
    operation : str, optional
        Operation to perform ('multiply', 'add', 'subtract', 'divide').
        Default is 'multiply'.

    Returns
    -------
    df_interact : pandas.DataFrame
        DataFrame with added interaction features.
    """
    df_interact = df.copy()

    for i, col1 in enumerate(columns):
        for col2 in columns[i + 1 :]:
            if operation == "multiply":
                df_interact[f"{col1}_{col2}_product"] = df[col1] * df[col2]
            elif operation == "add":
                df_interact[f"{col1}_{col2}_sum"] = df[col1] + df[col2]
            elif operation == "subtract":
                df_interact[f"{col1}_{col2}_diff"] = df[col1] - df[col2]
            elif operation == "divide":
                df_interact[f"{col1}_{col2}_ratio"] = df[col1] / df[col2]

    return df_interact
