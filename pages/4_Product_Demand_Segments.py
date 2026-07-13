import plotly.express as px
import streamlit as st
from utils import load_data, cluster_sub_categories

st.set_page_config(page_title="Product Demand Segments", page_icon="🧩", layout="wide")
st.title("🧩 Product Demand Segments")
st.caption("K-Means clustering of product sub-categories by total sales volume, "
           "growth rate, volatility, and average order value.")

df = load_data()

k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)

agg, centers, pca = cluster_sub_categories(df, k=k)

# ---------------- Cluster scatter (PCA) ----------------
st.subheader("Sub-Category Clusters (PCA-Reduced)")
fig = px.scatter(
    agg, x="PC1", y="PC2", color="cluster_label", text="Sub-Category",
    hover_data=["total_sales_volume", "sales_growth_rate", "sales_volatility", "average_order_value"],
    labels={
        "PC1": f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
        "PC2": f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
    },
)
fig.update_traces(marker=dict(size=14), textposition="top center")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Cluster table ----------------
st.subheader("Sub-Categories by Demand Cluster")
display_df = agg[[
    "Sub-Category", "cluster", "cluster_label", "total_sales_volume",
    "sales_growth_rate", "sales_volatility", "average_order_value",
]].sort_values(["cluster", "Sub-Category"]).rename(columns={
    "cluster": "Cluster #",
    "cluster_label": "Cluster Label",
    "total_sales_volume": "Total Sales Volume",
    "sales_growth_rate": "Sales Growth Rate",
    "sales_volatility": "Sales Volatility",
    "average_order_value": "Avg Order Value",
})
st.dataframe(
    display_df.style.format({
        "Total Sales Volume": "{:,.0f}",
        "Sales Growth Rate": "{:.1%}",
        "Sales Volatility": "{:,.1f}",
        "Avg Order Value": "{:,.2f}",
    }),
    use_container_width=True,
)

# ---------------- Cluster centers ----------------
with st.expander("Cluster centers (mean feature values)"):
    st.dataframe(centers.style.format("{:.2f}"), use_container_width=True)
