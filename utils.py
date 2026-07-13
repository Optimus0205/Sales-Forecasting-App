"""
Shared utilities for the Sales Forecasting Dashboard.
Handles data loading, SARIMA forecasting, anomaly detection, and clustering,
mirroring the methodology from the source analysis notebook.
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "train.csv")

# Best model determined in the notebook (Task 3): SARIMA beat Prophet & XGBoost
# on MAE / MAPE / RMSE for the last 3 historical months.
SARIMA_ORDER = (2, 1, 0)
SARIMA_SEASONAL_ORDER = (1, 0, 0, 12)


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Order Quarter"] = df["Order Date"].dt.quarter

    if "Postal Code" in df.columns:
        df["Postal Code"] = df["Postal Code"].fillna("05401")

    return df


@st.cache_data
def get_monthly_sales(df: pd.DataFrame, category=None, region=None, sub_category=None):
    """Aggregate to monthly sales, optionally filtered."""
    sub = df.copy()
    if category and category != "All":
        sub = sub[sub["Category"] == category]
    if region and region != "All":
        sub = sub[sub["Region"] == region]
    if sub_category and sub_category != "All":
        sub = sub[sub["Sub-Category"] == sub_category]

    monthly = (
        sub.set_index("Order Date")
        .resample("ME")["Sales"]
        .sum()
        .rename("Monthly Sales")
    )
    return monthly


@st.cache_data
def get_weekly_sales(df: pd.DataFrame):
    weekly = (
        df.set_index("Order Date")
        .resample("W")["Sales"]
        .sum()
        .rename("Weekly Sales")
        .reset_index()
        .rename(columns={"Order Date": "Week Start Date"})
    )
    return weekly


# ----------------------------------------------------------------------
# Forecasting (SARIMA — best model per Task 3)
# ----------------------------------------------------------------------
def _fit_sarima(series: pd.Series):
    model = SARIMAX(
        series,
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


@st.cache_data(show_spinner=False)
def sarima_forecast(monthly_sales: pd.Series, steps: int = 3):
    """Fit SARIMA on the full series and forecast `steps` months ahead with 95% CI."""
    series = monthly_sales.dropna()
    if len(series) < 24:
        return None

    result = _fit_sarima(series)
    forecast_obj = result.get_forecast(steps=steps)
    forecast_values = forecast_obj.predicted_mean.rename("Predicted Sales")
    ci = forecast_obj.conf_int(alpha=0.05)
    ci.columns = ["Lower Bound (95% CI)", "Upper Bound (95% CI)"]

    forecast_df = pd.concat([forecast_values, ci], axis=1)
    return forecast_df


@st.cache_data(show_spinner=False)
def sarima_backtest(monthly_sales: pd.Series, steps: int = 3):
    """
    Hold out the last `steps` actual months, refit SARIMA on the remaining
    history, forecast those months, and compare against the true values.
    Returns (mae, rmse, backtest_df) or None if insufficient data.
    """
    series = monthly_sales.dropna()
    if len(series) < 24 + steps:
        return None

    train = series.iloc[:-steps]
    test = series.iloc[-steps:]

    result = _fit_sarima(train)
    forecast_obj = result.get_forecast(steps=steps)
    preds = forecast_obj.predicted_mean.values

    mae = mean_absolute_error(test.values, preds)
    rmse = root_mean_squared_error(test.values, preds)

    backtest_df = pd.DataFrame(
        {"Actual": test.values, "Predicted": preds}, index=test.index
    )
    return mae, rmse, backtest_df


# ----------------------------------------------------------------------
# Anomaly detection (Task 5)
# ----------------------------------------------------------------------
@st.cache_data
def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05, window_size: int = 4, std_threshold: float = 2.0):
    weekly = get_weekly_sales(df).set_index("Week Start Date")
    weekly_values = weekly["Weekly Sales"].values.reshape(-1, 1)

    # Isolation Forest
    iso = IsolationForest(random_state=42, contamination=contamination)
    iso.fit(weekly_values)
    weekly["isolation_forest_anomaly"] = iso.predict(weekly_values)

    # Z-score (rolling mean / std)
    weekly["rolling_mean"] = weekly["Weekly Sales"].rolling(window=window_size).mean()
    weekly["rolling_std"] = weekly["Weekly Sales"].rolling(window=window_size).std()
    is_anomaly = (
        weekly["Weekly Sales"] > weekly["rolling_mean"] + std_threshold * weekly["rolling_std"]
    ) | (
        weekly["Weekly Sales"] < weekly["rolling_mean"] - std_threshold * weekly["rolling_std"]
    )
    weekly["z_score_anomaly"] = np.where(is_anomaly, -1, 1)

    return weekly.reset_index()


# ----------------------------------------------------------------------
# Clustering (Task 6)
# ----------------------------------------------------------------------
@st.cache_data
def cluster_sub_categories(df: pd.DataFrame, k: int = 4):
    d = df.copy().sort_values(by=["Sub-Category", "Order Date"])

    agg = d.groupby("Sub-Category").agg(
        total_sales_volume=("Sales", "sum"),
        total_orders=("Order ID", "nunique"),
    ).reset_index()

    monthly_by_subcat = (
        d.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="ME")])["Sales"]
        .sum()
        .unstack(fill_value=0)
    )
    agg = agg.set_index("Sub-Category")
    agg["sales_volatility"] = monthly_by_subcat.std(axis=1)

    def approx_growth(sales: pd.Series, years: pd.Series):
        by_year = sales.groupby(years).sum()
        if len(by_year) < 2:
            return np.nan
        first, last = by_year.iloc[0], by_year.iloc[-1]
        if first == 0:
            return 0.0
        return (last - first) / first

    growth = (
        d.assign(_year=d["Order Date"].dt.year)
        .groupby("Sub-Category")
        .apply(lambda g: approx_growth(g["Sales"], g["_year"]))
    )
    agg["sales_growth_rate"] = growth
    agg["average_order_value"] = agg["total_sales_volume"] / agg["total_orders"]
    agg = agg.reset_index()

    features = agg[["total_sales_volume", "sales_growth_rate", "sales_volatility", "average_order_value"]].fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    k = min(k, len(agg) - 1) if len(agg) > 1 else 1
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    agg["cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(scaled)
    agg["PC1"] = pcs[:, 0]
    agg["PC2"] = pcs[:, 1]

    # Label clusters based on unscaled cluster-center characteristics
    centers = agg.groupby("cluster")[
        ["total_sales_volume", "sales_growth_rate", "sales_volatility", "average_order_value"]
    ].mean()

    vol_median = centers["total_sales_volume"].median()
    volat_median = centers["sales_volatility"].median()

    labels = {}
    for c, row in centers.iterrows():
        vol_tag = "High Volume" if row["total_sales_volume"] >= vol_median else "Low Volume"
        stab_tag = "High Volatility" if row["sales_volatility"] >= volat_median else "Stable Demand"
        labels[c] = f"{vol_tag}, {stab_tag}"

    agg["cluster_label"] = agg["cluster"].map(labels)

    return agg, centers, pca
