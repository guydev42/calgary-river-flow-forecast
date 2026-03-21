"""
Forecasting models for Bow River flow prediction.

Provides both classical time-series (ARIMA / SARIMA) and machine-learning
(Random Forest, XGBoost) approaches.  All models are evaluated with the
same metric suite and can be persisted via joblib.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Suppress convergence warnings from statsmodels during grid search
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =========================================================================
# Evaluation helpers
# =========================================================================

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error (MAPE).

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE expressed as a percentage (0-100+).
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute a standard suite of regression metrics.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Dictionary with keys ``MAE``, ``RMSE``, ``MAPE``, ``R2``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
    }


# =========================================================================
# Temporal train / test split
# =========================================================================

def temporal_train_test_split(
    df: pd.DataFrame,
    target_col: str = "flow_rate",
    feature_cols: Optional[List[str]] = None,
    test_fraction: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data respecting temporal order (last *test_fraction* as test).

    Parameters
    ----------
    df : pd.DataFrame
        Daily dataframe with features and target.
    target_col : str
        Name of the target column.
    feature_cols : list of str or None
        Feature columns to use.  If *None*, all lag, rolling, and
        calendar columns are selected automatically.
    test_fraction : float
        Proportion of data reserved for testing (default 20 %).

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if any(tag in c for tag in ("lag_", "rolling_", "day_of_week", "month", "year"))
        ]
        if not feature_cols:
            raise ValueError("No feature columns detected. Provide feature_cols explicitly.")

    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols]
    X_test = test[feature_cols]
    y_train = train[target_col]
    y_test = test[target_col]

    logger.info("Train: %d rows | Test: %d rows", len(train), len(test))
    return X_train, X_test, y_train, y_test


# =========================================================================
# ARIMA / SARIMA wrapper
# =========================================================================

def fit_arima(
    series: pd.Series,
    order: Tuple[int, int, int] = (5, 1, 0),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> Any:
    """Fit an ARIMA or SARIMA model, handling common errors gracefully.

    Parameters
    ----------
    series : pd.Series
        Univariate time series to fit.
    order : tuple
        (p, d, q) order for ARIMA.
    seasonal_order : tuple or None
        (P, D, Q, s) for seasonal component.  Pass *None* for plain ARIMA.

    Returns
    -------
    statsmodels results object or None if fitting fails.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order or (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=200)
        logger.info("ARIMA%s fit complete. AIC=%.2f", order, results.aic)
        return results

    except Exception as exc:
        logger.warning("ARIMA fitting failed: %s", exc)
        return None


def arima_forecast(
    fitted_model: Any,
    steps: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a multi-step forecast from a fitted ARIMA / SARIMA model.

    Parameters
    ----------
    fitted_model : statsmodels results object
        A fitted SARIMAX results object.
    steps : int
        Number of steps (days) to forecast.

    Returns
    -------
    forecast : np.ndarray
        Point predictions.
    lower : np.ndarray
        Lower bound of 95 % confidence interval.
    upper : np.ndarray
        Upper bound of 95 % confidence interval.
    """
    forecast_obj = fitted_model.get_forecast(steps=steps)
    predicted = forecast_obj.predicted_mean.values
    conf_int = forecast_obj.conf_int()
    lower = conf_int.iloc[:, 0].values
    upper = conf_int.iloc[:, 1].values
    return predicted, lower, upper


# =========================================================================
# ML models (Random Forest & XGBoost)
# =========================================================================

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: Optional[int] = 15,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a Random Forest regressor on lag-based features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_estimators : int
        Number of trees.
    max_depth : int or None
        Maximum tree depth.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    RandomForestRegressor
        Fitted model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest trained with %d estimators.", n_estimators)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> Any:
    """Train an XGBoost regressor on lag-based features.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    n_estimators : int
        Boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    xgboost.XGBRegressor
        Fitted model.
    """
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        logger.error("xgboost is not installed: %s", exc)
        raise

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost trained with %d rounds.", n_estimators)
    return model


# =========================================================================
# Multi-step ML forecast
# =========================================================================

def ml_multi_step_forecast(
    model: Any,
    last_known: pd.DataFrame,
    steps: int = 30,
    target_col: str = "flow_rate",
) -> np.ndarray:
    """Generate a recursive multi-step forecast from an ML model.

    At each step the latest prediction is fed back as a lag feature
    for the next step.

    Parameters
    ----------
    model : fitted sklearn / xgboost estimator
        Must implement ``.predict()``.
    last_known : pd.DataFrame
        Single-row dataframe with the most recent feature values.
    steps : int
        Number of days to forecast.
    target_col : str
        Name of the target variable (used for lag column naming).

    Returns
    -------
    np.ndarray
        Array of predicted values with length *steps*.
    """
    predictions = []
    current_features = last_known.copy()

    for _ in range(steps):
        pred = model.predict(current_features)[0]
        predictions.append(pred)

        # Shift lag features forward by one step
        lag_cols = sorted(
            [c for c in current_features.columns if f"{target_col}_lag_" in c],
            key=lambda c: int(c.split("_lag_")[1].replace("d", "")),
        )
        for i in range(len(lag_cols) - 1, 0, -1):
            current_features[lag_cols[i]] = current_features[lag_cols[i - 1]].values
        if lag_cols:
            current_features[lag_cols[0]] = pred

    return np.array(predictions)


# =========================================================================
# Persistence
# =========================================================================

def save_model(model: Any, filename: str) -> Path:
    """Save a model to the models/ directory using joblib.

    Parameters
    ----------
    model : any picklable object
        Fitted model or results object.
    filename : str
        File name (e.g. ``"rf_model.joblib"``).

    Returns
    -------
    pathlib.Path
        Full path to the saved file.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)
    return filepath


def load_model(filename: str) -> Any:
    """Load a model from the models/ directory.

    Parameters
    ----------
    filename : str
        File name to load (e.g. ``"rf_model.joblib"``).

    Returns
    -------
    The deserialized model object.
    """
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    model = joblib.load(filepath)
    logger.info("Model loaded from %s", filepath)
    return model


# =========================================================================
# Script entry point -- quick sanity check
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from data_loader import load_and_prepare

    daily_df = load_and_prepare()
    target = "flow_rate" if "flow_rate" in daily_df.columns else "level"

    X_train, X_test, y_train, y_test = temporal_train_test_split(
        daily_df, target_col=target
    )

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_metrics = evaluate(y_test, rf_preds)
    print("Random Forest metrics:", rf_metrics)
    save_model(rf, "rf_river_flow.joblib")

    # XGBoost
    xgb = train_xgboost(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_metrics = evaluate(y_test, xgb_preds)
    print("XGBoost metrics:", xgb_metrics)
    save_model(xgb, "xgb_river_flow.joblib")

    # ARIMA
    series = daily_df.set_index("date")[target]
    arima_result = fit_arima(series, order=(5, 1, 0))
    if arima_result is not None:
        forecast, lower, upper = arima_forecast(arima_result, steps=30)
        print(f"ARIMA 30-day forecast (first 5): {forecast[:5]}")
