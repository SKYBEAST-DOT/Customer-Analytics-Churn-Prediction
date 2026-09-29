"""Visualization helpers for churn analytics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def plot_churn_distribution(df: pd.DataFrame, target_column: str = "Churn"):
    """Plot churn distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    if target_column not in df.columns:
        ax.text(0.5, 0.5, f"'{target_column}' column not found", ha="center", va="center")
    else:
        churn_labels = df[target_column].map({1: "Churn", 0: "No Churn"}).fillna(df[target_column])
        sns.countplot(x=churn_labels, ax=ax)
        ax.set_title("Churn Distribution")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Customer Count")
    plt.tight_layout()
    return fig


def plot_churn_by_contract(df: pd.DataFrame, target_column: str = "Churn"):
    """Plot churn rate by contract type."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if "Contract" not in df.columns or target_column not in df.columns:
        ax.text(0.5, 0.5, "Required columns missing: Contract/Churn", ha="center", va="center")
    else:
        grouped = (
            df.groupby("Contract", dropna=False)[target_column].mean().mul(100).sort_values(ascending=False)
        )
        sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
        ax.set_title("Churn Rate by Contract")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Churn Rate (%)")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    return fig


def plot_churn_by_payment_method(df: pd.DataFrame, target_column: str = "Churn"):
    """Plot churn rate by payment method."""
    fig, ax = plt.subplots(figsize=(10, 5))
    if "PaymentMethod" not in df.columns or target_column not in df.columns:
        ax.text(0.5, 0.5, "Required columns missing: PaymentMethod/Churn", ha="center", va="center")
    else:
        grouped = (
            df.groupby("PaymentMethod", dropna=False)[target_column]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
        ax.set_title("Churn Rate by Payment Method")
        ax.set_xlabel("Payment Method")
        ax.set_ylabel("Churn Rate (%)")
        ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    return fig


def plot_monthly_charges(df: pd.DataFrame, target_column: str = "Churn"):
    """Plot monthly charges distribution by churn status."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if "MonthlyCharges" not in df.columns:
        ax.text(0.5, 0.5, "'MonthlyCharges' column not found", ha="center", va="center")
    elif target_column in df.columns:
        churn_labels = df[target_column].map({1: "Churn", 0: "No Churn"}).fillna(df[target_column])
        sns.boxplot(x=churn_labels, y=df["MonthlyCharges"], ax=ax)
        ax.set_title("Monthly Charges vs Churn")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Monthly Charges")
    else:
        sns.histplot(df["MonthlyCharges"], kde=True, ax=ax)
        ax.set_title("Monthly Charges Distribution")
        ax.set_xlabel("Monthly Charges")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def plot_tenure_distribution(df: pd.DataFrame):
    """Plot customer tenure distribution."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if "tenure" not in df.columns:
        ax.text(0.5, 0.5, "'tenure' column not found", ha="center", va="center")
    else:
        sns.histplot(df["tenure"], bins=30, kde=True, ax=ax)
        ax.set_title("Customer Tenure Distribution")
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Customer Count")
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation heatmap for numeric variables if available."""
    fig, ax = plt.subplots(figsize=(9, 6))
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        ax.text(0.5, 0.5, "No numeric columns available for correlation analysis", ha="center", va="center")
    else:
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Correlation Heatmap")
        ax.set_xlabel("Features")
        ax.set_ylabel("Features")
    plt.tight_layout()
    return fig
