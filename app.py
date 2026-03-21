"""
Bow River Flow Forecasting & Flood Risk Monitor
=================================================

A Streamlit dashboard for exploring historical Bow River flow data,
generating forecasts, analysing seasonal patterns, and comparing
model performance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project's src/ package is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_prepare, DATA_DIR, PROCESSED_FILE
from src.model import (
    temporal_train_test_split,
    fit_arima,
    arima_forecast,
    train_random_forest,
    train_xgboost,
    evaluate,
    ml_multi_step_forecast,
    save_model,
    load_model,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Bow River Flow Forecasting & Flood Risk Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=13),
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)


def styled_figure(fig: go.Figure) -> go.Figure:
    """Apply consistent professional styling to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading river data ...")
def get_data() -> pd.DataFrame:
    """Load processed daily river data, fetching from API if needed."""
    if PROCESSED_FILE.exists():
        df = pd.read_csv(PROCESSED_FILE, parse_dates=["date"])
    else:
        df = load_and_prepare()
        df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource(show_spinner="Training models ...")
def get_trained_models(df: pd.DataFrame):
    """Train all models once and cache in session.

    Returns
    -------
    dict with keys: rf_model, xgb_model, arima_result, metrics, feature_cols, target
    """
    target = "flow_rate" if "flow_rate" in df.columns else "level"
    feature_cols = [
        c for c in df.columns
        if any(tag in c for tag in ("lag_", "rolling_", "day_of_week", "month", "year"))
    ]
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        df, target_col=target, feature_cols=feature_cols
    )

    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_metrics = evaluate(y_test, rf_preds)

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = evaluate(y_test, xgb_preds)

    # ARIMA
    series = df.set_index("date")[target]
    arima_result = fit_arima(series, order=(5, 1, 0))
    arima_metrics = {}
    arima_preds = None
    if arima_result is not None:
        try:
            arima_in_sample = arima_result.predict(
                start=X_test.index[0], end=X_test.index[-1]
            )
            arima_metrics = evaluate(y_test, arima_in_sample.values[: len(y_test)])
            arima_preds = arima_in_sample.values[: len(y_test)]
        except Exception:
            arima_metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

    return {
        "rf_model": rf_model,
        "xgb_model": xgb_model,
        "arima_result": arima_result,
        "rf_metrics": rf_metrics,
        "xgb_metrics": xgb_metrics,
        "arima_metrics": arima_metrics,
        "rf_preds": rf_preds,
        "xgb_preds": xgb_preds,
        "arima_preds": arima_preds,
        "y_test": y_test,
        "X_test": X_test,
        "feature_cols": feature_cols,
        "target": target,
    }


# =========================================================================
# Sidebar navigation
# =========================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "River Dashboard",
        "Flow Forecasting",
        "Seasonal Analysis",
        "Model Performance",
        "About",
    ],
)

# Load data
df = get_data()

# =========================================================================
# 1. River Dashboard
# =========================================================================
if page == "River Dashboard":
    st.title("Bow River Dashboard")
    st.markdown("Real-time overview of river levels and flow rates from Calgary Open Data.")

    target = "flow_rate" if "flow_rate" in df.columns else "level"
    level_col = "level" if "level" in df.columns else target

    # --- Key metrics ---------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        latest = df[level_col].iloc[-1] if level_col in df.columns else np.nan
        st.metric("Latest Level (m)", f"{latest:.2f}")
    with col2:
        avg_flow = df[target].mean()
        st.metric("Avg Daily Flow (m\u00b3/s)", f"{avg_flow:.1f}")
    with col3:
        max_flow = df[target].max()
        st.metric("Max Recorded Flow", f"{max_flow:.1f}")
    with col4:
        n_days = len(df)
        st.metric("Days of Data", f"{n_days:,}")

    st.markdown("---")

    # --- Date range selector -------------------------------------------------
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
    mask = (df["date"].dt.date >= date_range[0]) & (df["date"].dt.date <= date_range[1])
    filtered = df.loc[mask]

    # --- Level chart ---------------------------------------------------------
    if level_col in filtered.columns:
        fig_level = px.line(
            filtered,
            x="date",
            y=level_col,
            title="Daily Average River Level",
            labels={"date": "Date", level_col: "Level (m)"},
        )
        st.plotly_chart(styled_figure(fig_level), use_container_width=True)

    # --- Flow chart ----------------------------------------------------------
    if target in filtered.columns:
        fig_flow = px.line(
            filtered,
            x="date",
            y=target,
            title="Daily Average Flow Rate",
            labels={"date": "Date", target: "Flow Rate (m\u00b3/s)"},
        )
        fig_flow.update_traces(line_color="#1f77b4")
        st.plotly_chart(styled_figure(fig_flow), use_container_width=True)

    # --- Rolling average overlay ---------------------------------------------
    rolling_cols = [c for c in filtered.columns if "rolling" in c and target in c]
    if rolling_cols:
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=filtered["date"], y=filtered[target],
            mode="lines", name="Daily", opacity=0.4,
        ))
        colors = ["#ff7f0e", "#2ca02c"]
        for idx, rc in enumerate(rolling_cols):
            fig_roll.add_trace(go.Scatter(
                x=filtered["date"], y=filtered[rc],
                mode="lines", name=rc.replace("_", " ").title(),
                line=dict(color=colors[idx % len(colors)], width=2),
            ))
        fig_roll.update_layout(title="Flow Rate with Rolling Averages")
        st.plotly_chart(styled_figure(fig_roll), use_container_width=True)


# =========================================================================
# 2. Flow Forecasting
# =========================================================================
elif page == "Flow Forecasting":
    st.title("Flow Forecasting")
    st.markdown("Generate multi-step river flow forecasts and assess flood risk.")

    models_info = get_trained_models(df)
    target = models_info["target"]

    col_a, col_b = st.columns(2)
    with col_a:
        horizon = st.selectbox("Forecast horizon (days)", [7, 14, 30], index=1)
    with col_b:
        flood_threshold = st.number_input(
            "Flood threshold (m\u00b3/s)",
            min_value=0.0,
            value=float(df[target].quantile(0.95)),
            step=10.0,
            help="Horizontal line showing the user-defined flood-risk level.",
        )

    forecast_model = st.radio(
        "Select model for forecast",
        ["XGBoost", "Random Forest", "ARIMA"],
        horizontal=True,
    )

    # --- Generate forecast ---------------------------------------------------
    last_date = pd.to_datetime(df["date"].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon)

    if forecast_model == "ARIMA":
        arima_result = models_info["arima_result"]
        if arima_result is not None:
            forecast_vals, lower, upper = arima_forecast(arima_result, steps=horizon)
        else:
            st.warning("ARIMA model could not be fitted. Try an ML model instead.")
            forecast_vals, lower, upper = None, None, None
    else:
        chosen_model = (
            models_info["xgb_model"] if forecast_model == "XGBoost"
            else models_info["rf_model"]
        )
        last_row = df[models_info["feature_cols"]].iloc[[-1]]
        forecast_vals = ml_multi_step_forecast(
            chosen_model, last_row, steps=horizon, target_col=target
        )
        # Simple confidence band approximation for ML models
        residual_std = np.std(
            models_info["y_test"].values
            - (models_info["xgb_preds"] if forecast_model == "XGBoost"
               else models_info["rf_preds"])
        )
        lower = forecast_vals - 1.96 * residual_std
        upper = forecast_vals + 1.96 * residual_std

    if forecast_vals is not None:
        forecast_df = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast_vals,
            "lower_95": lower,
            "upper_95": upper,
        })

        fig_fc = go.Figure()

        # Historical tail
        tail = df.tail(90)
        fig_fc.add_trace(go.Scatter(
            x=tail["date"], y=tail[target],
            mode="lines", name="Historical",
            line=dict(color="#1f77b4"),
        ))

        # Confidence interval
        fig_fc.add_trace(go.Scatter(
            x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
            y=pd.concat([forecast_df["upper_95"], forecast_df["lower_95"][::-1]]),
            fill="toself",
            fillcolor="rgba(255, 127, 14, 0.15)",
            line=dict(color="rgba(255,127,14,0)"),
            name="95% Confidence",
        ))

        # Point forecast
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df["forecast"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#ff7f0e", width=2),
        ))

        # Flood threshold
        fig_fc.add_hline(
            y=flood_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Flood Threshold",
            annotation_position="top left",
        )

        fig_fc.update_layout(
            title=f"{forecast_model} {horizon}-Day Forecast",
            xaxis_title="Date",
            yaxis_title=f"{target.replace('_', ' ').title()} (m\u00b3/s)",
        )
        st.plotly_chart(styled_figure(fig_fc), use_container_width=True)

        # Flood risk alert
        if np.any(forecast_vals > flood_threshold):
            days_above = int(np.sum(forecast_vals > flood_threshold))
            st.error(
                f"Warning: Forecast exceeds flood threshold on "
                f"{days_above} of {horizon} days."
            )
        else:
            st.success("Forecast remains below the flood threshold for the selected horizon.")

        with st.expander("Forecast data table"):
            st.dataframe(forecast_df.style.format({"forecast": "{:.2f}", "lower_95": "{:.2f}", "upper_95": "{:.2f}"}))


# =========================================================================
# 3. Seasonal Analysis
# =========================================================================
elif page == "Seasonal Analysis":
    st.title("Seasonal Analysis")
    st.markdown("Explore monthly patterns and year-over-year trends in river flow.")

    target = "flow_rate" if "flow_rate" in df.columns else "level"

    # --- Monthly average flow ------------------------------------------------
    monthly_avg = df.groupby("month")[target].mean().reset_index()
    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    monthly_avg["month_name"] = monthly_avg["month"].apply(lambda m: month_names[int(m) - 1])

    fig_monthly = px.bar(
        monthly_avg,
        x="month_name",
        y=target,
        title="Average Monthly Flow Rate",
        labels={"month_name": "Month", target: "Avg Flow (m\u00b3/s)"},
        color=target,
        color_continuous_scale="Blues",
    )
    fig_monthly.update_layout(coloraxis_showscale=False)
    st.plotly_chart(styled_figure(fig_monthly), use_container_width=True)

    # --- Year-over-year comparison -------------------------------------------
    available_years = sorted(df["year"].dropna().unique().astype(int))
    selected_years = st.multiselect(
        "Select years to compare",
        available_years,
        default=available_years[-3:] if len(available_years) >= 3 else available_years,
    )

    if selected_years:
        yoy = df[df["year"].isin(selected_years)].copy()
        yoy["day_of_year"] = yoy["date"].dt.dayofyear

        fig_yoy = px.line(
            yoy,
            x="day_of_year",
            y=target,
            color=yoy["year"].astype(str),
            title="Year-over-Year Flow Comparison",
            labels={"day_of_year": "Day of Year", target: "Flow (m\u00b3/s)", "color": "Year"},
        )
        st.plotly_chart(styled_figure(fig_yoy), use_container_width=True)

    # --- Rolling average visualization ---------------------------------------
    rolling_cols = [c for c in df.columns if "rolling" in c]
    if rolling_cols:
        chosen_rolling = st.selectbox("Rolling average", rolling_cols)
        fig_ra = px.line(
            df, x="date", y=chosen_rolling,
            title=chosen_rolling.replace("_", " ").title(),
            labels={"date": "Date", chosen_rolling: "Value"},
        )
        st.plotly_chart(styled_figure(fig_ra), use_container_width=True)

    # --- Monthly box plot ----------------------------------------------------
    df_box = df.copy()
    df_box["month_name"] = df_box["month"].apply(lambda m: month_names[int(m) - 1])
    fig_box = px.box(
        df_box,
        x="month_name",
        y=target,
        title="Monthly Flow Distribution",
        labels={"month_name": "Month", target: "Flow (m\u00b3/s)"},
    )
    st.plotly_chart(styled_figure(fig_box), use_container_width=True)


# =========================================================================
# 4. Model Performance
# =========================================================================
elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.markdown("Side-by-side evaluation of ARIMA, Random Forest, and XGBoost forecasters.")

    models_info = get_trained_models(df)
    target = models_info["target"]
    y_test = models_info["y_test"]
    test_dates = df.iloc[y_test.index]["date"]

    # --- Metrics table -------------------------------------------------------
    metrics_df = pd.DataFrame({
        "Random Forest": models_info["rf_metrics"],
        "XGBoost": models_info["xgb_metrics"],
        "ARIMA": models_info["arima_metrics"],
    }).T

    st.subheader("Metric Summary")
    st.dataframe(
        metrics_df.style.format("{:.4f}").highlight_min(axis=0, subset=["MAE", "RMSE", "MAPE"])
                       .highlight_max(axis=0, subset=["R2"]),
        use_container_width=True,
    )

    # --- Actual vs predicted plot --------------------------------------------
    st.subheader("Actual vs Predicted")

    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(
        x=test_dates, y=y_test,
        mode="lines", name="Actual",
        line=dict(color="#1f77b4", width=2),
    ))
    fig_avp.add_trace(go.Scatter(
        x=test_dates, y=models_info["rf_preds"],
        mode="lines", name="Random Forest",
        line=dict(color="#ff7f0e", dash="dot"),
    ))
    fig_avp.add_trace(go.Scatter(
        x=test_dates, y=models_info["xgb_preds"],
        mode="lines", name="XGBoost",
        line=dict(color="#2ca02c", dash="dash"),
    ))
    if models_info["arima_preds"] is not None:
        fig_avp.add_trace(go.Scatter(
            x=test_dates, y=models_info["arima_preds"],
            mode="lines", name="ARIMA",
            line=dict(color="#d62728", dash="dashdot"),
        ))

    fig_avp.update_layout(
        title="Test-Set Predictions vs Actual Values",
        xaxis_title="Date",
        yaxis_title=f"{target.replace('_', ' ').title()} (m\u00b3/s)",
    )
    st.plotly_chart(styled_figure(fig_avp), use_container_width=True)

    # --- Residual plot -------------------------------------------------------
    st.subheader("Residuals (Actual - Predicted)")

    residual_rf = y_test.values - models_info["rf_preds"]
    residual_xgb = y_test.values - models_info["xgb_preds"]

    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=test_dates, y=residual_rf,
        mode="markers", name="RF Residuals",
        marker=dict(size=4, opacity=0.5, color="#ff7f0e"),
    ))
    fig_res.add_trace(go.Scatter(
        x=test_dates, y=residual_xgb,
        mode="markers", name="XGB Residuals",
        marker=dict(size=4, opacity=0.5, color="#2ca02c"),
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_res.update_layout(
        title="Residual Distribution Over Time",
        xaxis_title="Date",
        yaxis_title="Residual",
    )
    st.plotly_chart(styled_figure(fig_res), use_container_width=True)

    # --- Feature importance (RF) ---------------------------------------------
    st.subheader("Feature Importance (Random Forest)")
    importances = models_info["rf_model"].feature_importances_
    feat_names = models_info["feature_cols"]
    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(15)
    )
    fig_imp = px.bar(
        imp_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Top 15 Features by Importance",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    st.plotly_chart(styled_figure(fig_imp), use_container_width=True)


# =========================================================================
# 5. About
# =========================================================================
elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Bow River Flow Forecasting & Flood Risk Monitor

    ### Problem Statement

    The devastating **2013 Calgary floods** caused over \\$6 billion in damage,
    displaced more than 100,000 residents, and underscored the urgent need for
    better river-flow monitoring and early-warning systems.  This project
    builds a data-driven forecasting tool that uses historical river
    level and flow data to predict future conditions and flag potential flood
    risk.

    ### Dataset

    | Attribute | Detail |
    |-----------|--------|
    | **Source** | [Calgary Open Data Portal](https://data.calgary.ca/) |
    | **Dataset ID** | `5fdg-ifgr` (River Levels and Flows) |
    | **Records** | ~9.5 million (5-minute intervals) |
    | **Key Fields** | timestamp, level (m), flow_rate (m\u00b3/s), station |

    For development purposes, a subset of 50,000 recent records is fetched
    by default.

    ### Methodology

    1. **Data Engineering** -- Raw 5-minute observations are resampled to
       daily means.  Rolling averages (7-day, 30-day) and lag features
       (1, 7, 14, 30 days) are computed to capture temporal dependencies.

    2. **ARIMA / SARIMA** -- Classical time-series model fitted on the
       univariate flow series.  Provides well-calibrated confidence
       intervals but may struggle with non-linear dynamics.

    3. **Random Forest Regression** -- Ensemble of 200 decision trees
       trained on the engineered lag and calendar features.

    4. **XGBoost Regression** -- Gradient-boosted trees (300 rounds) for
       potentially higher accuracy on structured tabular features.

    5. **Evaluation** -- All models are compared using MAE, RMSE, MAPE,
       and R\u00b2 on a temporally held-out test set (last 20 % of data).

    ### Key Findings

    * River flow exhibits strong **seasonal patterns**, peaking in June
      due to snowmelt and declining through winter.
    * Lag-based ML models (especially XGBoost) tend to outperform ARIMA
      on this dataset, achieving lower MAE and higher R\u00b2.
    * The 7-day rolling average is the most important predictor,
      highlighting the value of short-term momentum features.

    ### How to Run

    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```

    ### Data Source & Licence

    Data provided by the **City of Calgary** under the
    [Open Data Licence](https://data.calgary.ca/stories/s/Open-Calgary-Terms-of-Use/u45n-7awa).
    """)

    st.info(
        "This project is part of a Calgary Open Data portfolio demonstrating "
        "applied data science skills across civic datasets."
    )
