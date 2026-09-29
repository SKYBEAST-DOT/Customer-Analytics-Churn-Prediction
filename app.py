"""Streamlit app for customer analytics and churn prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.churn_prediction import prepare_features, predict_churn, train_model
from src.customer_analysis import (
    churn_by_contract,
    churn_by_payment_method,
    churn_by_tenure,
    customer_summary,
    high_value_customers,
)
from src.data_preprocessing import prepare_data
from src.visualization import (
    plot_churn_by_contract,
    plot_churn_by_payment_method,
    plot_churn_distribution,
    plot_correlation_heatmap,
    plot_monthly_charges,
    plot_tenure_distribution,
)


st.set_page_config(
    page_title="Customer Analytics & Churn Prediction",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Analytics & Churn Prediction")
st.caption(
    "Analyze customer behavior, identify churn patterns, and predict customers at risk of leaving."
)

repo_root = Path(__file__).resolve().parent
raw_data_path = repo_root / "data" / "raw" / "customer_data.csv"
processed_data_path = repo_root / "data" / "processed" / "cleaned_customer_data.csv"

st.sidebar.header("Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
model_test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

section = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Analytics",
        "Visuals",
        "Prediction",
    ],
)

if uploaded_file is not None:
    data_source = uploaded_file
elif raw_data_path.exists():
    data_source = raw_data_path
else:
    data_source = None
    st.info("Upload a CSV file to get started.")

if data_source is None:
    st.stop()

try:
    with st.spinner("Preparing dataset..."):
        df = prepare_data(data_source, save_path=processed_data_path)
    st.success("Dataset loaded and preprocessed successfully.")
except pd.errors.EmptyDataError:
    st.error("The uploaded file is empty. Please upload a valid CSV file.")
    st.stop()
except Exception as exc:
    st.error(f"Failed to process dataset: {exc}")
    st.stop()

if section == "Overview":
    st.subheader("Section 1 — Dataset Overview")
    c1, c2 = st.columns(2)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Missing Value Summary", expanded=True):
        missing_df = df.isna().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df, use_container_width=True)

    st.subheader("Section 2 — Customer KPIs")
    kpi = customer_summary(df)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Customers", f"{kpi['total_customers']:,}")
    k2.metric("Churned Customers", f"{kpi['churned_customers']:,}")
    k3.metric("Churn Rate", f"{kpi['churn_rate']:.2f}%")
    k4.metric("Avg Monthly Charges", f"{kpi['avg_monthly_charges']:.2f}" if pd.notna(kpi["avg_monthly_charges"]) else "N/A")
    k5.metric("Avg Tenure", f"{kpi['avg_tenure']:.2f}" if pd.notna(kpi["avg_tenure"]) else "N/A")

elif section == "Analytics":
    st.subheader("Section 3 — Customer Analytics")
    st.write("Observed customer churn patterns by key business dimensions.")

    contract_df = churn_by_contract(df)
    payment_df = churn_by_payment_method(df)
    tenure_df = churn_by_tenure(df)

    tabs = st.tabs(["Contract", "Payment Method", "Tenure", "High-Value Customers"])
    with tabs[0]:
        st.dataframe(contract_df, use_container_width=True)
    with tabs[1]:
        st.dataframe(payment_df, use_container_width=True)
    with tabs[2]:
        st.dataframe(tenure_df, use_container_width=True)
    with tabs[3]:
        st.dataframe(high_value_customers(df), use_container_width=True)

    st.subheader("Section 7 — Business Insights")
    summary = customer_summary(df)
    insights = [f"Observed overall churn rate: **{summary['churn_rate']:.2f}%**."]

    if not contract_df.empty:
        top_contract = contract_df.iloc[0]
        insights.append(
            f"Highest observed contract churn: **{top_contract['Contract']}** ({top_contract['churn_rate']:.2f}%)."
        )
    if not payment_df.empty:
        top_payment = payment_df.iloc[0]
        insights.append(
            f"Highest observed payment-method churn: **{top_payment['PaymentMethod']}** ({top_payment['churn_rate']:.2f}%)."
        )
    if not tenure_df.empty:
        top_tenure = tenure_df.sort_values("churn_rate", ascending=False).iloc[0]
        insights.append(
            f"Highest observed tenure-group churn: **{top_tenure['tenure_group']}** ({top_tenure['churn_rate']:.2f}%)."
        )

    for insight in insights:
        st.markdown(f"- {insight}")
    st.caption("These insights are observed patterns in the dataset and are not causal conclusions.")

elif section == "Visuals":
    st.subheader("Section 4 — Visual Analytics")
    chart_tabs = st.tabs([
        "Churn Distribution",
        "Contract Churn",
        "Payment Churn",
        "Monthly Charges",
        "Tenure",
        "Correlation",
    ])

    with chart_tabs[0]:
        st.pyplot(plot_churn_distribution(df))
    with chart_tabs[1]:
        st.pyplot(plot_churn_by_contract(df))
    with chart_tabs[2]:
        st.pyplot(plot_churn_by_payment_method(df))
    with chart_tabs[3]:
        st.pyplot(plot_monthly_charges(df))
    with chart_tabs[4]:
        st.pyplot(plot_tenure_distribution(df))
    with chart_tabs[5]:
        st.pyplot(plot_correlation_heatmap(df))

else:
    st.subheader("Section 5 — Churn Prediction")
    if "Churn" not in df.columns:
        st.error("The uploaded dataset must contain a 'Churn' column.")
        st.stop()

    if st.button("Train Churn Prediction Model", type="primary"):
        try:
            with st.spinner("Training churn model..."):
                model_output = train_model(df, test_size=model_test_size, random_state=42)
            metrics = model_output["metrics"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            m2.metric("Precision", f"{metrics['precision']:.3f}")
            m3.metric("Recall", f"{metrics['recall']:.3f}")
            m4.metric("F1 Score", f"{metrics['f1_score']:.3f}")

            st.write("Confusion Matrix")
            cm_df = pd.DataFrame(metrics["confusion_matrix"], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm_df, use_container_width=True)

            st.write("Classification Report")
            st.dataframe(pd.DataFrame(metrics["classification_report"]).transpose(), use_container_width=True)

            st.subheader("Section 6 — Risk Analysis")
            X_full, _, _, _ = prepare_features(df)
            risk_preds = predict_churn(model_output["model"], X_full)

            id_col = next((col for col in ["customerID", "CustomerID", "customer_id"] if col in df.columns), None)
            risk_table = pd.DataFrame(
                {
                    "Customer ID": df[id_col].astype(str) if id_col else df.index.astype(str),
                    "Churn Probability": risk_preds["churn_probability"],
                }
            )

            risk_table["Risk Level"] = pd.cut(
                risk_table["Churn Probability"],
                bins=[-0.01, 0.33, 0.66, 1.0],
                labels=["Low", "Medium", "High"],
            )

            st.dataframe(
                risk_table.sort_values("Churn Probability", ascending=False).reset_index(drop=True),
                use_container_width=True,
            )
            st.success("Model training and risk scoring completed.")
        except ValueError as exc:
            st.error(f"Model training error: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error while training model: {exc}")
