"""Machine learning pipeline utilities for churn prediction."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "Churn"
ID_COLUMNS = {"customerid", "customer_id"}


def _coerce_target(y: pd.Series) -> pd.Series:
    """Coerce supported churn label formats into binary values."""
    if pd.api.types.is_bool_dtype(y):
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        y_num = pd.to_numeric(y, errors="coerce")
        unique_values = set(y_num.dropna().astype(int).unique().tolist())
        if unique_values.issubset({0, 1}):
            return y_num.fillna(0).astype(int)

    y_str = y.astype(str).str.strip().str.lower()
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
    y_mapped = y_str.map(mapping)

    if y_mapped.isna().any():
        invalid = sorted(y_str[y_mapped.isna()].unique().tolist())
        raise ValueError(f"Unsupported churn labels for training: {', '.join(invalid)}")

    return y_mapped.astype(int)


def prepare_features(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> Tuple[pd.DataFrame, pd.Series, list, list]:
    """Prepare feature matrix and target vector."""
    if target_column not in df.columns:
        raise ValueError(f"The dataset must contain a '{target_column}' column.")

    X = df.drop(columns=[target_column]).copy()
    y = _coerce_target(df[target_column])

    removable = [col for col in X.columns if col.lower() in ID_COLUMNS]
    if removable:
        X = X.drop(columns=removable)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return X, y, numeric_features, categorical_features


def build_model(numeric_features: list, categorical_features: list) -> Pipeline:
    """Build a preprocessing + Random Forest pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, object]:
    """Calculate model evaluation metrics."""
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, zero_division=0)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def train_model(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Train churn model and return trained pipeline with metrics."""
    X, y, numeric_features, categorical_features = prepare_features(df, target_column=target_column)
    model = build_model(numeric_features, categorical_features)

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_test))

    metrics = evaluate_model(y_test, y_pred)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "metrics": metrics,
        "feature_columns": X.columns.tolist(),
    }


def predict_churn(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Predict churn class and churn probability."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(X))

    return pd.DataFrame(
        {
            "churn_prediction": preds,
            "churn_probability": probs,
        }
    )
