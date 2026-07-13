# Sales Forecasting Dashboard (Streamlit)

Interactive multi-page dashboard built from the sales forecasting analysis notebook.

## Pages
1. **Sales Overview** — total sales by year, monthly trend, filterable by region & category
2. **Forecast Explorer** — SARIMA(2,1,0)(1,0,0,12) forecasts (best model per the notebook's
   comparison) for any category or region, 1–3 month horizon, with MAE/RMSE
3. **Anomaly Report** — Isolation Forest + Z-score anomaly detection on weekly sales
4. **Product Demand Segments** — K-Means clustering of sub-categories into demand groups

## Run locally

```bash
pip install -r requirements.txt
streamlit run Sales Forecasting App.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Folder structure

```
sales_app/
├── app.py                  # Landing page
├── utils.py                 # Shared data loading, forecasting, anomaly & clustering logic
├── requirements.txt
├── data/
│   └── train.csv             # Sales dataset (swap this file to use your own data)
└── pages/
    ├── 1_Sales_Overview.py
    ├── 2_Forecast_Explorer.py
    ├── 3_Anomaly_Report.py
    └── 4_Product_Demand_Segments.py
```

## Deploying

- **Streamlit Community Cloud**: push this folder to a GitHub repo, then connect it at
  https://share.streamlit.io — point it at `app.py`.
- **Docker / other hosts**: install `requirements.txt` and run
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

## Notes

- All forecasting/anomaly/clustering results are computed live from `data/train.csv`
  each time filters change, so they'll stay correct if you replace the CSV with updated data
  (same column schema: `Order Date`, `Ship Date`, `Sales`, `Category`, `Region`,
  `Sub-Category`, `Order ID`, etc.).
- SARIMA needs at least ~24 months of history per segment; very small/sparse segments
  may show a warning instead of a forecast.
