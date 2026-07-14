import streamlit as st
from utils import load_data

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Sales Forecasting & Analytics Dashboard")

st.markdown(
    """
Welcome! This dashboard shows the sales forecasting analysis. Use the sidebar to navigate between pages:

- **1 — Sales Overview**: total sales by year, monthly trend, and interactive
  filters by region and category
- **2 — Forecast Explorer**: SARIMA-based forecasts (the best-performing model
  from the notebook) for any category or region, with MAE / RMSE
- **3 — Anomaly Report**: Isolation Forest and Z-score anomaly detection on
  weekly sales
- **4 — Product Demand Segments**: K-Means clustering of sub-categories into
  demand groups

👈 Pick a page from the sidebar to get started.
"""
)

with st.spinner("Loading data..."):
    df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
col2.metric("Orders", f"{df['Order ID'].nunique():,}")
col3.metric("Date Range", f"{df['Order Date'].min():%b %Y} – {df['Order Date'].max():%b %Y}")
col4.metric("Categories", df["Category"].nunique())

st.dataframe(df.head(20), use_container_width=True)
