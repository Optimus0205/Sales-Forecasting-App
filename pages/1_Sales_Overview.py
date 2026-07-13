import plotly.express as px
import streamlit as st
from utils import load_data

st.set_page_config(page_title="Sales Overview", page_icon="📈", layout="wide")
st.title("📈 Sales Overview Dashboard")

df = load_data()

# ---------------- Filters ----------------
st.sidebar.header("Filters")
regions = ["All"] + sorted(df["Region"].unique().tolist())
categories = ["All"] + sorted(df["Category"].unique().tolist())

sel_region = st.sidebar.selectbox("Region", regions)
sel_category = st.sidebar.selectbox("Category", categories)

filtered = df.copy()
if sel_region != "All":
    filtered = filtered[filtered["Region"] == sel_region]
if sel_category != "All":
    filtered = filtered[filtered["Category"] == sel_category]

if filtered.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------- KPIs ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
c2.metric("Orders", f"{filtered['Order ID'].nunique():,}")
c3.metric("Avg Order Value", f"${filtered['Sales'].sum() / max(filtered['Order ID'].nunique(),1):,.2f}")

st.divider()

# ---------------- Total sales by year ----------------
st.subheader("Total Sales by Year")
yearly = filtered.groupby("Order Year")["Sales"].sum().reset_index()
fig_year = px.bar(
    yearly, x="Order Year", y="Sales", text_auto=".2s",
    labels={"Order Year": "Year", "Sales": "Total Sales"},
)
fig_year.update_xaxes(type="category")
st.plotly_chart(fig_year, use_container_width=True)

# ---------------- Monthly sales trend ----------------
st.subheader("Monthly Sales Trend")
monthly = filtered.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
fig_month = px.line(
    monthly, x="Order Date", y="Sales", markers=True,
    labels={"Order Date": "Month", "Sales": "Monthly Sales"},
)
st.plotly_chart(fig_month, use_container_width=True)

# ---------------- Sales by region and category ----------------
st.subheader("Sales by Region and Category")
grouped = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
fig_grouped = px.bar(
    grouped, x="Region", y="Sales", color="Category", barmode="group",
    labels={"Sales": "Total Sales"},
)
st.plotly_chart(fig_grouped, use_container_width=True)

with st.expander("View underlying data"):
    st.dataframe(filtered, use_container_width=True)
