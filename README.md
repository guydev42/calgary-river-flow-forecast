<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=River%20Flow%20Forecasting&fontSize=36&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Multi-model%20Bow%20River%20forecasting%20on%209.5M%2B%20observations&descSize=16&descAlignY=55&descColor=c8e0ff" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/XGBoost-0.89_R²-blue?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Prophet-Forecasting-3b5998?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Calgary_Open_Data-Socrata_API-orange?style=for-the-badge" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

> **Problem** -- The 2013 Calgary floods caused over $6 billion in damage and displaced 100,000+ residents. Accurate river-flow forecasting is critical for early-warning systems that protect lives and infrastructure.
>
> **Solution** -- This project builds a multi-model forecasting tool using 9.5M+ five-minute Bow River observations from Calgary Open Data, comparing ARIMA/SARIMA, Random Forest, XGBoost, LSTM, and Prophet approaches.
>
> **Impact** -- Supports flood early-warning systems with accurate short-term river flow predictions, giving emergency services and residents advance notice of dangerous water levels.

---

## Results

| Model | R-squared | MAE (m3/s) | RMSE (m3/s) |
|-------|-----------|------------|-------------|
| **XGBoost** | **~0.89** | ~10 | ~17 |
| Random Forest | ~0.85 | ~12 | ~20 |
| ARIMA/SARIMA | ~0.72 | ~18 | ~28 |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Calgary Open   │────>│  9.5M+ 5-min     │────>│  Daily resample  │────>│  Model suite   │────>│  Streamlit      │
│  Data (Socrata) │     │  observations    │     │  Rolling avgs    │     │  ARIMA/SARIMA  │     │  dashboard      │
│  Bow River      │     │  Level & flow    │     │  7d / 30d lags   │     │  XGBoost / RF  │     │  Flow viz       │
│  sensors        │     │  cleaning        │     │  Calendar feats  │     │  LSTM/Prophet  │     │  Forecasting    │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘     └─────────────────┘
```

---

## Project structure

<details>
<summary>Click to expand</summary>

```
project_04_river_flow_forecasting/
├── app.py                          # Streamlit dashboard
├── index.html                      # Static landing page
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   └── river_flow_raw.csv          # Cached river observation data
├── models/                         # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py              # Data fetching & feature engineering
    └── model.py                    # Model training, evaluation & forecasting
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/river-flow-forecasting.git
cd river-flow-forecasting

# Install dependencies
pip install -r requirements.txt

# Fetch river data from Calgary Open Data
python src/data_loader.py

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Calgary Open Data -- River Flow Observations](https://data.calgary.ca/) |
| Records | 9,500,000+ five-minute observations |
| Access method | Socrata API (sodapy) |
| Key fields | Timestamp, water level (m), flow rate (m3/s), station ID |
| Target variable | Daily mean flow rate (m3/s) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/statsmodels-ARIMA/SARIMA-4B8BBE?style=flat-square" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Prophet-3b5998?style=flat-square" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-Socrata_API-orange?style=flat-square" />
</p>

---

## Methodology

### Data ingestion and resampling

- Fetched 9.5M+ five-minute river level and flow observations via the Socrata API
- Resampled to daily means to reduce noise and create a tractable time series
- Handled missing values with forward-fill and interpolation

### Feature engineering

- Engineered rolling averages (7-day, 30-day) and lag features (1, 3, 7, 14, 30 days)
- Created calendar features: day-of-year, month, season, and year
- Built sequences for LSTM input with sliding window approach

### Classical time-series models

- Trained ARIMA/SARIMA for classical time-series modeling with confidence intervals
- Used auto-ARIMA for automated order selection
- SARIMA achieved R-squared of ~0.72 with interpretable seasonal decomposition

### Machine learning models

- Trained Random Forest and XGBoost regressors on engineered lag and calendar features
- XGBoost achieved the best performance with R-squared of ~0.89
- Feature importance analysis revealed 7-day lag and rolling averages as top predictors

### Interactive dashboard

- Built a Streamlit dashboard for interactive flow visualization and forecasting
- Supports model comparison, forecast horizon selection, and historical trend exploration

---

## Acknowledgements

- [City of Calgary Open Data Portal](https://data.calgary.ca/) for providing river flow observation data
- [Socrata Open Data API](https://dev.socrata.com/) for programmatic data access

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>
