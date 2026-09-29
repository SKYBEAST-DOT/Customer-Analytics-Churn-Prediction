"""Customer analytics helper functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def customer_summary(df: pd.DataFrame, target_column: str = "Churn") -> dict:
    """Return key customer churn KPIs as a dictionary."""
    total_customers = int(len(df))
    churned_customers = int(df[target_column].sum()) if target_column in df.columns else 0
    churn_rate = (churned_customers / total_customers * 100) if total_customers else 0.0

    avg_monthly_charges = (
        float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else np.nan
    )
    avg_tenure = float(df["tenure"].mean()) if "tenure" in df.columns else np.nan
    total_revenue = float(df["TotalCharges"].sum()) if "TotalCharges" in df.columns else np.nan

    return {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "churn_rate": churn_rate,
        "avg_monthly_charges": avg_monthly_charges,
        "avg_tenure": avg_tenure,
        "total_revenue": total_revenue,
    }


def _group_churn_rate(df: pd.DataFrame, group_col: str, target_column: str = "Churn") -> pd.DataFrame:
    if group_col not in df.columns or target_column not in df.columns:
        return pd.DataFrame(columns=[group_col, "customers", "churn_rate"])

    grouped = (
        df.groupby(group_col, dropna=False)[target_column]
        .agg(customers="count", churn_rate="mean")
        .reset_index()
    )
    grouped["churn_rate"] = grouped["churn_rate"] * 100
    return grouped.sort_values("churn_rate", ascending=False).reset_index(drop=True)


def churn_by_contract(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """Return churn rate grouped by contract type."""
    return _group_churn_rate(df, "Contract", target_column=target_column)


def churn_by_payment_method(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """Return churn rate grouped by payment method."""
    return _group_churn_rate(df, "PaymentMethod", target_column=target_column)


def churn_by_tenure(df: pd.DataFrame, target_column: str = "Churn") -> pd.DataFrame:
    """Return churn rate grouped by tenure buckets."""
    if "tenure" not in df.columns or target_column not in df.columns:
        return pd.DataFrame(columns=["tenure_group", "customers", "churn_rate"])

    tenure_df = df.copy()
    bins = [0, 12, 24, 48, 72, np.inf]
    labels = ["0-12", "13-24", "25-48", "49-72", "72+"]
    tenure_df["tenure_group"] = pd.cut(
        tenure_df["tenure"], bins=bins, labels=labels, include_lowest=True
    )

    grouped = (
        tenure_df.groupby("tenure_group", observed=False)[target_column]
        .agg(customers="count", churn_rate="mean")
        .reset_index()
    )
    grouped["churn_rate"] = grouped["churn_rate"] * 100
    return grouped


def high_value_customers(
    df: pd.DataFrame,
    percentile: float = 0.75,
) -> pd.DataFrame:
    """Identify high-value customers using charge-based threshold."""
    charge_col = "TotalCharges" if "TotalCharges" in df.columns else "MonthlyCharges"
    if charge_col not in df.columns:
        return pd.DataFrame()

    threshold = df[charge_col].quantile(percentile)
    high_value = df[df[charge_col] >= threshold].copy()

    preferred_columns = [
        col
        for col in ["customerID", "CustomerID", "customer_id", "Contract", "PaymentMethod", charge_col, "Churn"]
        if col in high_value.columns
    ]

    if preferred_columns:
        high_value = high_value[preferred_columns]

    return high_value.sort_values(charge_col, ascending=False).reset_index(drop=True)
