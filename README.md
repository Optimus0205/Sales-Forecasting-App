<div align="center">

# 📊 Sales Forecasting Dashboard

**An interactive multi-page Streamlit app for sales analytics, forecasting, anomaly detection, and product demand segmentation.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](#)

[Live Demo](https://salesforecasting-dashboard.streamlit.app/) · [Report Bug](../../issues) · [Request Feature](../../issues)

</div>

---

## 🧭 Overview

This dashboard turns a sales forecasting analysis notebook into a fully interactive, deployable web app. It's built for exploring historical sales trends, generating short-term forecasts with SARIMA, flagging unusual sales weeks, and segmenting product sub-categories by demand behavior — all without touching a line of code.

<div align="center">

| 📈 Sales Overview | 🔮 Forecast Explorer | 🚨 Anomaly Report | 🧩 Demand Segments |
|:---:|:---:|:---:|:---:|
| Yearly & monthly trends | SARIMA forecasts + MAE/RMSE | Isolation Forest + Z-score | K-Means clustering |

</div>

---

## ✨ Features

| Page | What it does |
|---|---|
| 📈 **Sales Overview** | Total sales by year (bar chart), monthly sales trend (line chart), and interactive filters by region & category |
| 🔮 **Forecast Explorer** | Pick a Category or Region, choose a 1–3 month horizon, and get live **SARIMA(2,1,0)(1,0,0,12)** forecasts — the best-performing model from the notebook's comparison — complete with 95% confidence intervals, MAE, and RMSE |
| 🚨 **Anomaly Report** | Weekly sales anomalies flagged by **Isolation Forest** and a rolling **Z-score** method, visualized on a trend chart with a searchable table of anomaly dates |
| 🧩 **Product Demand Segments** | **K-Means clustering** of product sub-categories by sales volume, growth rate, volatility, and average order value, with a PCA scatter plot and cluster membership table |

---

## 🖥️ Tech Stack

- **Frontend/App:** [Streamlit](https://streamlit.io/)
- **Data Processing:** [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Forecasting:** [statsmodels](https://www.statsmodels.org/) (SARIMAX)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (Isolation Forest, K-Means, PCA)
- **Visualization:** [Plotly](https://plotly.com/python/)

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Optimus0205/Sales-Forecasting-App.git
cd Sales-Forecasting-App
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open automatically, or visit **http://localhost:8501** in your browser.

---

## 📁 Folder Structure

```
sales_app/
├── app.py                     # Landing page
├── utils.py                   # Shared data loading, forecasting, anomaly & clustering logic
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
├── .gitignore                 # Files/folders excluded from git
├── data/
│   └── train.csv               # Sales dataset (swap this to use your own data)
└── pages/
    ├── 1_Sales_Overview.py
    ├── 2_Forecast_Explorer.py
    ├── 3_Anomaly_Report.py
    └── 4_Product_Demand_Segments.py
```

---

## ☁️ Deployment

<details>
<summary><strong>Streamlit Community Cloud</strong> (recommended, free)</summary>
<br>

1. Push this repo to GitHub (already done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **Create app** → select this repo, branch `main`, and main file `app.py`
4. Click **Deploy**

</details>

<details>
<summary><strong>Docker / other hosts</strong></summary>
<br>

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

</details>

---

## 📝 Notes

- All forecasting, anomaly, and clustering results are computed **live** from `data/train.csv` on every filter change — swap in your own data (same schema: `Order Date`, `Ship Date`, `Sales`, `Category`, `Region`, `Sub-Category`, `Order ID`, etc.) and everything recalculates automatically.
- SARIMA requires at least **~24 months** of history per segment. Very small or sparse segments will show a warning instead of a forecast.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ using Streamlit

## 🙋‍♂️ Author

# **Ashutosh Singh**
**[Optimus0205](https://github.com/Optimus0205)**

⭐ If you found this project useful, consider giving it a star!

</div>
