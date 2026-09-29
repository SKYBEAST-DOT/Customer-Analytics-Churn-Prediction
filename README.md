# Customer Analytics & Churn Prediction

An end-to-end customer analytics and churn prediction project that cleans customer data, analyzes behavior and churn patterns, trains a machine learning model, and presents insights through an interactive Streamlit dashboard.

## Features

- Data preprocessing
- Customer analytics
- Churn analysis
- Exploratory data analysis
- Machine learning
- Churn prediction
- Model evaluation
- Interactive Streamlit dashboard
- Business insights

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

## Project Structure

```text
Customer-Analytics-Churn-Prediction/
│
├── data/
│   ├── raw/
│   │   └── customer_data.csv
│   │
│   └── processed/
│       └── cleaned_customer_data.csv
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── customer_analysis.py
│   ├── churn_prediction.py
│   └── visualization.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Workflow

```text
Customer Data
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Customer Analytics
      ↓
Churn Analysis
      ↓
Feature Engineering
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Business Insights
      ↓
Streamlit Dashboard
```

## How to Run

```bash
git clone <repository-url>
cd Customer-Analytics-Churn-Prediction
pip install -r requirements.txt
streamlit run app.py
```

Place your dataset in:

```text
data/raw/customer_data.csv
```

Then upload it in the dashboard (or use the default file if present).

## Business Use Cases

- Customer retention
- Churn monitoring
- Customer segmentation
- Risk identification
- Revenue protection
- Data-driven decision making
