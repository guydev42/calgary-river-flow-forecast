# River flow forecasting

## Problem statement
The 2013 Calgary floods caused over $6 billion in damage and displaced 100,000+ residents. Accurate river-flow forecasting is critical for early-warning systems that protect lives and infrastructure. This project builds a multi-model forecasting tool using 9.5M+ five-minute Bow River observations from Calgary Open Data.

## Approach
- Fetched 9.5M+ five-minute river level and flow observations via the Socrata API
- Resampled to daily means and engineered rolling averages (7-day, 30-day) and lag features
- Trained ARIMA/SARIMA for classical time-series modeling with confidence intervals
- Trained Random Forest and XGBoost regressors on engineered lag and calendar features
- Built a Streamlit dashboard for interactive flow visualization and forecasting

## Key results

| Model | R-squared | MAE (m3/s) | RMSE (m3/s) |
|-------|-----------|------------|-------------|
| **XGBoost** | **~0.89** | ~10 | ~17 |
| Random Forest | ~0.85 | ~12 | ~20 |
| ARIMA/SARIMA | ~0.72 | ~18 | ~28 |

## How to run
```bash
pip install -r requirements.txt
python src/data_loader.py    # fetch river data
streamlit run app.py         # launch dashboard
```

## Project structure
```
project_04_river_flow_forecasting/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching & feature engineering
    └── model.py            # Model training, evaluation & forecasting
```

## Technical stack
pandas, NumPy, statsmodels (ARIMA/SARIMA), scikit-learn, XGBoost, Plotly, Streamlit, sodapy
