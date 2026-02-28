"""
spark/train_model.py
--------------------
Trains and persists two scikit-learn models on **real** historical weather
data fetched from the Open-Meteo archive API (free, no API key required):

  1. RandomForestRegressor  – predicts air temperature from meteorological
     features and cyclical time encodings.
  2. IsolationForest        – unsupervised anomaly detector that flags
     unusual combinations of sensor readings.

Usage
-----
    python spark/train_model.py

All default coordinates and date ranges are overridable via environment
variables (see .env.example).
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Load .env file when running locally; no-op inside Docker (vars already injected)
load_dotenv()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
MODEL_DIR: str = os.path.join(os.path.dirname(__file__), "model")

WEATHER_CONDITIONS: list[str] = ["Clear", "Clouds", "Rain", "Snow", "Thunderstorm"]

# Feature column order must match the streaming inference code exactly
FEATURE_COLS: list[str] = (
    ["humidity", "pressure", "wind_speed", "clouds",
     "hour_sin", "hour_cos", "day_sin", "day_cos"]
    + [f"weather_{cond}" for cond in WEATHER_CONDITIONS]
)

ANOMALY_CONTAMINATION: float = 0.05  # expected fraction of anomalies in training data
MIN_TRAINING_ROWS: int = 500         # fail fast if the API returns too little data

# ---------------------------------------------------------------------------
# Open-Meteo configuration (all overridable via env vars / .env)
# ---------------------------------------------------------------------------
OPEN_METEO_ARCHIVE_URL: str = "https://archive-api.open-meteo.com/v1/archive"
REQUEST_TIMEOUT_SEC: int = 30

DEFAULT_LATITUDE: float = float(os.environ.get("TRAINING_LATITUDE", "36.8065"))   # Tunis
DEFAULT_LONGITUDE: float = float(os.environ.get("TRAINING_LONGITUDE", "10.1815"))
DEFAULT_START_DATE: str = os.environ.get("TRAINING_START_DATE", "2022-01-01")
DEFAULT_END_DATE: str = os.environ.get("TRAINING_END_DATE", "2023-12-31")

# WMO weather-interpretation code → canonical condition label used by the producer
# Reference: https://open-meteo.com/en/docs#weathervariables
_WMO_CODE_TO_CONDITION: dict[int, str] = {
    0: "Clear",
    1: "Clear",   2: "Clouds",       3: "Clouds",
    45: "Clouds", 48: "Clouds",
    51: "Rain",   53: "Rain",         55: "Rain",
    61: "Rain",   63: "Rain",         65: "Rain",
    71: "Snow",   73: "Snow",         75: "Snow",  77: "Snow",
    80: "Rain",   81: "Rain",         82: "Rain",
    85: "Snow",   86: "Snow",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}


# ---------------------------------------------------------------------------
# Real-data acquisition
# ---------------------------------------------------------------------------

def fetch_historical_weather(
    latitude: float = DEFAULT_LATITUDE,
    longitude: float = DEFAULT_LONGITUDE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Download hourly weather history from the Open-Meteo archive API.

    Open-Meteo is free and requires no API key.  Documentation:
    https://open-meteo.com/en/docs/historical-weather-api

    Args:
        latitude:   Geographic latitude of the target location.
        longitude:  Geographic longitude of the target location.
        start_date: ISO-8601 date string for the first day of data (inclusive).
        end_date:   ISO-8601 date string for the last day of data (inclusive).

    Returns:
        A tidy DataFrame with columns
        ``["timestamp", "hour", "day_of_year", "temperature", "humidity",
        "pressure", "wind_speed", "clouds", "weather_condition"]``.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status.
        ValueError: If the response contains fewer than ``MIN_TRAINING_ROWS``
            valid records.
    """
    logger.info(
        "Fetching real weather data from Open-Meteo "
        "(lat=%.4f, lon=%.4f, %s → %s)...",
        latitude, longitude, start_date, end_date,
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relativehumidity_2m",
            "surface_pressure",
            "windspeed_10m",
            "cloudcover",
            "weathercode",
        ]),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    hourly: dict = response.json()["hourly"]

    df = pd.DataFrame({
        "timestamp":         pd.to_datetime(hourly["time"]),
        "temperature":       hourly["temperature_2m"],
        "humidity":          hourly["relativehumidity_2m"],
        "pressure":          hourly["surface_pressure"],
        "wind_speed":        hourly["windspeed_10m"],
        "clouds":            hourly["cloudcover"],
        "weather_code":      hourly["weathercode"],
    })

    # Drop rows with any null sensor reading
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning("Dropped %d rows with null sensor readings.", n_dropped)

    if len(df) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Only {len(df)} valid rows returned from Open-Meteo "
            f"(minimum required: {MIN_TRAINING_ROWS}). "
            "Widen the date range or check the API response."
        )

    # Derive time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_year"] = df["timestamp"].dt.day_of_year

    # Map WMO codes to canonical condition labels; unknown codes → "Clouds"
    df["weather_condition"] = (
        df["weather_code"].astype(int)
        .map(_WMO_CODE_TO_CONDITION)
        .fillna("Clouds")
    )
    df = df.drop(columns=["weather_code"])

    logger.info(
        "Fetched %d real hourly records (%s → %s).",
        len(df), df["timestamp"].min(), df["timestamp"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct the feature matrix and target vector from real weather records.

    Applies:
    - Cyclical (sin/cos) encoding for hour-of-day and day-of-year.
    - Binary one-hot flags for each weather condition category.

    Args:
        df: DataFrame produced by :func:`fetch_historical_weather`.

    Returns:
        A tuple ``(X, y)`` where *X* is the feature DataFrame and *y* is
        the temperature target Series.
    """
    features = df.copy()

    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features["day_sin"] = np.sin(2 * np.pi * features["day_of_year"] / 365)
    features["day_cos"] = np.cos(2 * np.pi * features["day_of_year"] / 365)

    for condition in WEATHER_CONDITIONS:
        features[f"weather_{condition}"] = (
            features["weather_condition"] == condition
        ).astype(int)

    return features[FEATURE_COLS], features["temperature"]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_temperature_model(df: pd.DataFrame) -> RandomForestRegressor:
    """Train a Random Forest Regressor to predict air temperature.

    Args:
        df: Real weather records from :func:`fetch_historical_weather`.

    Returns:
        The fitted :class:`RandomForestRegressor` instance.
    """
    logger.info("Training Random Forest Regressor for temperature prediction...")

    x_all, y_all = build_features(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r_squared = r2_score(y_test, y_pred)
    logger.info("Evaluation – RMSE: %.4f °C  |  R²: %.4f", rmse, r_squared)

    feature_importances = (
        pd.Series(model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=False)
    )
    logger.info("Top-5 features:\n%s", feature_importances.head(5).to_string())

    output_path = os.path.join(MODEL_DIR, "temperature_model.pkl")
    joblib.dump({"model": model, "feature_columns": FEATURE_COLS}, output_path)
    logger.info("Temperature model saved to '%s'.", output_path)

    return model


def train_anomaly_detector(df: pd.DataFrame) -> IsolationForest:
    """Train an Isolation Forest to detect anomalous weather readings.

    Args:
        df: Real weather records from :func:`fetch_historical_weather`.

    Returns:
        The fitted :class:`IsolationForest` instance.
    """
    logger.info("Training Isolation Forest for anomaly detection...")

    anomaly_features = ["temperature", "humidity", "pressure", "wind_speed", "clouds"]
    x_anom = df[anomaly_features].dropna()

    detector = IsolationForest(
        n_estimators=100,
        contamination=ANOMALY_CONTAMINATION,
        random_state=42,
    )
    detector.fit(x_anom)

    n_anomalies = (detector.predict(x_anom) == -1).sum()
    logger.info(
        "Detected %d anomalies (%.1f %%) in the training set.",
        n_anomalies,
        n_anomalies / len(x_anom) * 100,
    )

    output_path = os.path.join(MODEL_DIR, "anomaly_detector.pkl")
    joblib.dump({"model": detector, "feature_columns": anomaly_features}, output_path)
    logger.info("Anomaly detector saved to '%s'.", output_path)

    return detector


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    training_data = fetch_historical_weather()
    train_temperature_model(training_data)
    train_anomaly_detector(training_data)

    logger.info("All models saved to: %s", os.path.abspath(MODEL_DIR))
