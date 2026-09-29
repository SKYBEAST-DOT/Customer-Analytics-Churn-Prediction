"""Data preprocessing utilities for customer churn datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

TARGET_COLUMN = "Churn"


def load_data(file_source: Union[str, Path, object]) -> pd.DataFrame:
    """Load churn data from a CSV file path or file-like object."""
    data = pd.read_csv(file_source)
    if data.empty:
        raise ValueError("The provided dataset is empty.")
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean churn dataset by handling blanks, duplicates, missing values and numeric casting."""
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]

    object_columns = cleaned.select_dtypes(include=["object"]).columns
    for col in object_columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()

    cleaned.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    cleaned.drop_duplicates(inplace=True)

    for numeric_candidate in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if numeric_candidate in cleaned.columns:
            cleaned[numeric_candidate] = pd.to_numeric(cleaned[numeric_candidate], errors="coerce")

    numeric_columns = cleaned.select_dtypes(include=[np.number]).columns
    categorical_columns = [col for col in cleaned.columns if col not in numeric_columns]

    for col in numeric_columns:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in categorical_columns:
        mode_series = cleaned[col].mode(dropna=True)
        fallback_value = mode_series.iloc[0] if not mode_series.empty else "Unknown"
        cleaned[col] = cleaned[col].fillna(fallback_value)

    return cleaned


def encode_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Encode churn target to binary values (1 for churn, 0 for non-churn)."""
    if target_column not in df.columns:
        raise ValueError(f"The dataset must contain a '{target_column}' column.")

    encoded = df.copy()
    target = encoded[target_column]

    if pd.api.types.is_bool_dtype(target):
        encoded[target_column] = target.astype(int)
        return encoded

    if pd.api.types.is_numeric_dtype(target):
        unique_vals = set(pd.Series(target).dropna().astype(int).unique().tolist())
        if unique_vals.issubset({0, 1}):
            encoded[target_column] = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
            return encoded

    target_str = target.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "churn": 1,
        "not churn": 0,
    }

    encoded[target_column] = target_str.map(mapping)
    if encoded[target_column].isna().any():
        invalid_values = sorted(target_str[encoded[target_column].isna()].unique().tolist())
        raise ValueError(
            "Unsupported target labels detected in 'Churn': "
            + ", ".join(map(str, invalid_values))
        )

    encoded[target_column] = encoded[target_column].astype(int)
    return encoded


def prepare_data(
    file_source: Union[str, Path, object],
    save_path: Union[str, Path, None] = None,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Load, clean and encode churn dataset. Optionally save cleaned output to CSV."""
    data = load_data(file_source)
    cleaned = clean_data(data)
    prepared = encode_target(cleaned, target_column=target_column)

    if save_path:
        save_target = Path(save_path)
        save_target.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_csv(save_target, index=False)

    return prepared
