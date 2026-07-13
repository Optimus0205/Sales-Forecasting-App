import plotly.graph_objects as go
import streamlit as st
from utils import load_data, detect_anomalies

st.set_page_config(page_title="Anomaly Report", page_icon="🚨", layout="wide")
st.title("🚨 Anomaly Report")
st.caption("Weekly sales anomalies detected with Isolation Forest and a rolling "
           "Z-score method.")

df = load_data()

with st.sidebar:
    st.header("Detection Settings")
    contamination = st.slider("Isolation Forest contamination", 0.01, 0.20, 0.05, 0.01)
    window_size = st.slider("Z-score rolling window (weeks)", 2, 12, 4)
    std_threshold = st.slider("Z-score std deviation threshold", 1.0, 4.0, 2.0, 0.5)

weekly = detect_anomalies(df, contamination=contamination, window_size=window_size, std_threshold=std_threshold)

anomalies_if = weekly[weekly["isolation_forest_anomaly"] == -1]
anomalies_zs = weekly[weekly["z_score_anomaly"] == -1]

c1, c2 = st.columns(2)
c1.metric("Isolation Forest Anomalies", len(anomalies_if))
c2.metric("Z-Score Anomalies", len(anomalies_zs))

# ---------------- Chart ----------------
st.subheader("Weekly Sales with Detected Anomalies")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=weekly["Week Start Date"], y=weekly["Weekly Sales"],
    mode="lines", name="Weekly Sales", line=dict(color="#1f77b4"),
))
fig.add_trace(go.Scatter(
    x=anomalies_if["Week Start Date"], y=anomalies_if["Weekly Sales"],
    mode="markers", name="Isolation Forest Anomaly",
    marker=dict(color="red", size=11, symbol="circle"),
))
fig.add_trace(go.Scatter(
    x=anomalies_zs["Week Start Date"], y=anomalies_zs["Weekly Sales"],
    mode="markers", name="Z-Score Anomaly",
    marker=dict(color="green", size=11, symbol="x"),
))
fig.update_layout(xaxis_title="Week", yaxis_title="Weekly Sales",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig, use_container_width=True)

# ---------------- Tables ----------------
st.subheader("Detected Anomaly Dates")

tab1, tab2 = st.tabs(["Isolation Forest", "Z-Score"])
with tab1:
    if anomalies_if.empty:
        st.info("No anomalies detected by Isolation Forest at this setting.")
    else:
        st.dataframe(
            anomalies_if[["Week Start Date", "Weekly Sales"]]
            .sort_values("Week Start Date")
            .style.format({"Weekly Sales": "{:,.2f}"}),
            use_container_width=True,
        )
with tab2:
    if anomalies_zs.empty:
        st.info("No anomalies detected by Z-Score at this setting.")
    else:
        st.dataframe(
            anomalies_zs[["Week Start Date", "Weekly Sales", "rolling_mean", "rolling_std"]]
            .sort_values("Week Start Date")
            .style.format({"Weekly Sales": "{:,.2f}", "rolling_mean": "{:,.2f}", "rolling_std": "{:,.2f}"}),
            use_container_width=True,
        )
