import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import load_data, get_monthly_sales, sarima_forecast, sarima_backtest

st.set_page_config(page_title="Forecast Explorer", page_icon="🔮", layout="wide")
st.title("🔮 Forecast Explorer")
st.caption("Forecasts generated with SARIMA(2,1,0)(1,0,0,12) — the best-performing model")
           #"model identified in the notebook's model comparison (Task 3).")

df = load_data()

# ---------------- Controls ----------------
c1, c2 = st.columns(2)
with c1:
    dim_type = st.selectbox("Select dimension", ["Category", "Region"])
with c2:
    if dim_type == "Category":
        options = sorted(df["Category"].unique().tolist())
    else:
        options = sorted(df["Region"].unique().tolist())
    dim_value = st.selectbox(dim_type, options)

horizon = st.select_slider(
    "Forecast horizon (months ahead)",
    options=[1, 2, 3],
    value=3,
)

# ---------------- Build series & forecast ----------------
if dim_type == "Category":
    monthly = get_monthly_sales(df, category=dim_value)
else:
    monthly = get_monthly_sales(df, region=dim_value)

with st.spinner(f"Fitting SARIMA for {dim_value}..."):
    forecast_df = sarima_forecast(monthly, steps=horizon)
    backtest_result = sarima_backtest(monthly, steps=horizon)

if forecast_df is None:
    st.error("Not enough historical data (need at least 24 months) to fit SARIMA for this selection.")
    st.stop()

# ---------------- Chart ----------------
st.subheader(f"{dim_value} — {horizon}-Month Forecast")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly.index, y=monthly.values, mode="lines+markers",
    name="Actual Sales", line=dict(color="#1f77b4"),
))

# bridge point so the forecast line connects visually to history
bridge_x = [monthly.index[-1]] + list(forecast_df.index)
bridge_y = [monthly.values[-1]] + list(forecast_df["Predicted Sales"].values)
fig.add_trace(go.Scatter(
    x=bridge_x, y=bridge_y, mode="lines+markers",
    name="SARIMA Forecast", line=dict(color="#ff7f0e", dash="dash"),
))

fig.add_trace(go.Scatter(
    x=list(forecast_df.index) + list(forecast_df.index[::-1]),
    y=list(forecast_df["Upper Bound (95% CI)"]) + list(forecast_df["Lower Bound (95% CI)"][::-1]),
    fill="toself", fillcolor="rgba(255,127,14,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="95% Confidence Interval", showlegend=True,
))

fig.update_layout(
    xaxis_title="Date", yaxis_title="Sales",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Values")
st.dataframe(
    forecast_df.style.format("{:.2f}"),
    use_container_width=True,
)

# ---------------- Model performance ----------------
st.subheader("Model Performance")
if backtest_result is None:
    st.info("Not enough data to backtest at this horizon.")
else:
    mae, rmse, backtest_df = backtest_result
    m1, m2 = st.columns(2)
    m1.metric("MAE", f"{mae:,.2f}")
    m2.metric("RMSE", f"{rmse:,.2f}")
    st.caption(
        f"Backtest: model trained on all data except the last {horizon} month(s), "
        "then compared its forecast for those months against the true actuals."
    )
    with st.expander("Backtest detail"):
        st.dataframe(backtest_df.style.format("{:.2f}"), use_container_width=True)
