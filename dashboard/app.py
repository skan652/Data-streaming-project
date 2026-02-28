"""
dashboard/app.py
----------------
Streamlit dashboard that auto-refreshes to display real-time weather data,
ML temperature predictions and windowed aggregations written by the Spark job.

Data is read from two rolling JSON-lines files:
  - ``{OUTPUT_PATH}/predictions.json``  – per-event ML predictions
  - ``{OUTPUT_PATH}/aggregations.json`` – 5-minute windowed statistics
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

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
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR: str = os.environ.get("OUTPUT_PATH", "/opt/spark-apps/output")
PREDICTIONS_FILE: str = os.path.join(OUTPUT_DIR, "predictions.json")
AGGREGATIONS_FILE: str = os.path.join(OUTPUT_DIR, "aggregations.json")

TIMESTAMP_COL: str = "timestamp_readable"
WINDOW_END_COL: str = "window_end"

# Colour palette (consistent across charts)
COLOR_ACTUAL: str = "#e74c3c"
COLOR_PREDICTED: str = "#3498db"
COLOR_HUMIDITY: str = "#2ecc71"
COLOR_PRESSURE: str = "#9b59b6"
COLOR_ERROR: str = "#e67e22"
COLOR_WIND: str = "#3498db"

CHART_HEIGHT: int = 380
AGG_CHART_HEIGHT: int = 350

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Weather Streaming Dashboard",
    page_icon="🌤",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5)
def load_predictions() -> pd.DataFrame:
    """Read the latest prediction records from the JSON-lines file.

    Returns:
        Sorted DataFrame of weather predictions, or an empty DataFrame
        if the file does not exist or cannot be parsed.
    """
    if not os.path.exists(PREDICTIONS_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_json(PREDICTIONS_FILE, lines=True)
        if TIMESTAMP_COL in df.columns:
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
            df = df.sort_values(TIMESTAMP_COL)
        return df
    except ValueError as exc:
        logger.error("Failed to parse predictions file: %s", exc)
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_aggregations() -> pd.DataFrame:
    """Read the latest windowed aggregations from the JSON-lines file.

    Returns:
        Sorted DataFrame of aggregated statistics, or an empty DataFrame
        if the file does not exist or cannot be parsed.
    """
    if not os.path.exists(AGGREGATIONS_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_json(AGGREGATIONS_FILE, lines=True)
        for col_name in ("window_start", WINDOW_END_COL):
            if col_name in df.columns:
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
        if WINDOW_END_COL in df.columns:
            df = df.sort_values(WINDOW_END_COL)
        return df
    except ValueError as exc:
        logger.error("Failed to parse aggregations file: %s", exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_temperature_chart(df: pd.DataFrame) -> go.Figure:
    """Build a dual line chart: actual vs. ML-predicted temperature.

    Args:
        df: Predictions DataFrame with ``temperature`` and
            ``predicted_temperature`` columns.

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[TIMESTAMP_COL],
            y=df["temperature"],
            mode="lines+markers",
            name="Actual",
            line=dict(color=COLOR_ACTUAL, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[TIMESTAMP_COL],
            y=df["predicted_temperature"],
            mode="lines+markers",
            name="RF Predicted",
            line=dict(color=COLOR_PREDICTED, width=2, dash="dash"),
        )
    )
    # Overlay anomaly markers (IsolationForest detections)
    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies[TIMESTAMP_COL],
                    y=anomalies["temperature"],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(
                        color="#e74c3c",
                        symbol="x",
                        size=14,
                        line=dict(color="black", width=1),
                    ),
                )
            )
    fig.update_layout(
        title="🌡 Temperature – Actual vs ML Prediction",
        xaxis_title="Time",
        yaxis_title="°C",
        legend=dict(orientation="h"),
        height=CHART_HEIGHT,
        hovermode="x unified",
    )
    return fig


def build_error_chart(df: pd.DataFrame) -> go.Figure:
    """Build a bar chart of absolute temperature prediction error.

    Args:
        df: Predictions DataFrame with a ``temperature_error`` column.

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[TIMESTAMP_COL],
            y=df["temperature_error"],
            name="Prediction Error",
            marker_color=COLOR_ERROR,
            opacity=0.75,
        )
    )
    fig.update_layout(
        title="📉 Prediction Error (|Actual − Predicted|)",
        xaxis_title="Time",
        yaxis_title="°C",
        height=CHART_HEIGHT,
    )
    return fig


def build_humidity_pressure_chart(df: pd.DataFrame) -> go.Figure:
    """Build a dual-axis chart for humidity (left) and pressure (right).

    Args:
        df: Predictions DataFrame.

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()
    if "humidity" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[TIMESTAMP_COL],
                y=df["humidity"],
                mode="lines+markers",
                name="Humidity (%)",
                line=dict(color=COLOR_HUMIDITY),
            )
        )
    if "pressure" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[TIMESTAMP_COL],
                y=df["pressure"],
                mode="lines+markers",
                name="Pressure (hPa)",
                line=dict(color=COLOR_PRESSURE),
                yaxis="y2",
            )
        )
    fig.update_layout(
        title="💧 Humidity & Pressure Over Time",
        yaxis=dict(title="Humidity (%)"),
        yaxis2=dict(title="Pressure (hPa)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=CHART_HEIGHT,
    )
    return fig


def build_windowed_temp_chart(agg_df: pd.DataFrame) -> go.Figure:
    """Build a line chart showing windowed avg/min/max temperature.

    Args:
        agg_df: Aggregations DataFrame.

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg_df[WINDOW_END_COL],
            y=agg_df["avg_temperature"],
            mode="lines+markers",
            name="Avg Temp",
            line=dict(color=COLOR_ACTUAL),
        )
    )
    if "max_temperature" in agg_df.columns:
        fig.add_trace(
            go.Scatter(
                x=agg_df[WINDOW_END_COL],
                y=agg_df["max_temperature"],
                mode="lines",
                name="Max Temp",
                line=dict(color=COLOR_ACTUAL, dash="dot"),
            )
        )
    if "min_temperature" in agg_df.columns:
        fig.add_trace(
            go.Scatter(
                x=agg_df[WINDOW_END_COL],
                y=agg_df["min_temperature"],
                mode="lines",
                name="Min Temp",
                line=dict(color=COLOR_PREDICTED, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(52,152,219,0.1)",
            )
        )
    fig.update_layout(
        title="🌡 Windowed Avg / Min / Max Temperature",
        xaxis_title="Window End",
        yaxis_title="°C",
        legend=dict(orientation="h"),
        height=AGG_CHART_HEIGHT,
    )
    return fig


def build_windowed_humidity_wind_chart(agg_df: pd.DataFrame) -> go.Figure:
    """Build a grouped bar chart for windowed humidity and wind speed.

    Args:
        agg_df: Aggregations DataFrame.

    Returns:
        A configured Plotly Figure.
    """
    fig = go.Figure()
    if "avg_humidity" in agg_df.columns:
        fig.add_trace(
            go.Bar(
                x=agg_df[WINDOW_END_COL],
                y=agg_df["avg_humidity"],
                name="Avg Humidity (%)",
                marker_color=COLOR_HUMIDITY,
            )
        )
    if "avg_wind_speed" in agg_df.columns:
        fig.add_trace(
            go.Bar(
                x=agg_df[WINDOW_END_COL],
                y=agg_df["avg_wind_speed"],
                name="Avg Wind (m/s)",
                marker_color=COLOR_WIND,
            )
        )
    fig.update_layout(
        title="💨 Windowed Avg Humidity & Wind Speed",
        xaxis_title="Window End",
        barmode="group",
        height=AGG_CHART_HEIGHT,
    )
    return fig


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

def render_kpi_row(latest: pd.Series) -> None:
    """Render the top-row KPI metric cards.

    Args:
        latest: A single row from the predictions DataFrame (the most recent).
    """
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    # Use None-safe retrieval so N/A is shown when data is missing
    temperature: Optional[float] = latest.get("temperature")
    predicted: Optional[float] = latest.get("predicted_temperature")
    error: Optional[float] = latest.get("temperature_error")
    humidity: Optional[float] = latest.get("humidity")
    wind_speed: Optional[float] = latest.get("wind_speed")
    cloud_cover: Optional[float] = latest.get("clouds")
    is_anomaly: Optional[int] = latest.get("is_anomaly")

    col1.metric(
        "🌡 Temperature",
        f"{temperature:.1f} °C" if temperature is not None else "N/A",
    )
    col2.metric(
        "🔮 Predicted Temp",
        f"{predicted:.1f} °C" if predicted is not None else "N/A",
        delta=f"err {error:.1f} °C" if error is not None else None,
        delta_color="inverse",
    )
    col3.metric(
        "💧 Humidity",
        f"{humidity:.0f} %" if humidity is not None else "N/A",
    )
    col4.metric(
        "🌬 Wind Speed",
        f"{wind_speed:.1f} m/s" if wind_speed is not None else "N/A",
    )
    col5.metric(
        "☁ Cloud Cover",
        f"{cloud_cover:.0f} %" if cloud_cover is not None else "N/A",
    )
    col6.metric(
        "🔍 Anomaly",
        "⚠ Detected" if is_anomaly == 1 else ("✅ Normal" if is_anomaly == 0 else "N/A"),
    )


def render_main_dashboard() -> None:
    """Render the full dashboard layout."""
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🌤 Real-Time Weather Streaming Dashboard")
    st.caption(
        "OpenWeather API  |  Kafka  |  "
        "Spark Structured Streaming  |  Random Forest + Isolation Forest"
    )
    st.markdown("---")

    # ── Sidebar controls ──────────────────────────────────────────────────────
    st.sidebar.header("Controls")
    refresh_rate: int = st.sidebar.slider("Auto-refresh (seconds)", 5, 60, 10)
    n_points: int = st.sidebar.slider("Points to display", 10, 100, 50)
    show_raw: bool = st.sidebar.checkbox("Show raw data table", value=False)
    st.sidebar.markdown(f"**Output dir:** `{OUTPUT_DIR}`")

    # ── Load data ─────────────────────────────────────────────────────────────
    predictions_df = load_predictions()
    aggregations_df = load_aggregations()

    if predictions_df.empty:
        st.warning(
            f"⏳ Waiting for data from `{PREDICTIONS_FILE}`.  \n"
            "Make sure the Kafka producer and Spark streaming job are running."
        )
        time.sleep(refresh_rate)
        st.rerun()

    predictions_df = predictions_df.tail(n_points)
    latest_row: pd.Series = predictions_df.iloc[-1]

    # ── KPI metrics ───────────────────────────────────────────────────────────
    render_kpi_row(latest_row)
    st.markdown("---")

    # ── Row 1: Temperature actual vs predicted  |  Prediction error ───────────
    left_col, right_col = st.columns(2)
    with left_col:
        st.plotly_chart(
            build_temperature_chart(predictions_df), use_container_width=True
        )
    with right_col:
        st.plotly_chart(
            build_error_chart(predictions_df), use_container_width=True
        )

    # ── Row 2: Humidity & Pressure  |  Weather distribution ───────────────────
    left_col, right_col = st.columns(2)
    with left_col:
        st.plotly_chart(
            build_humidity_pressure_chart(predictions_df), use_container_width=True
        )
    with right_col:
        if "weather" in predictions_df.columns and not predictions_df["weather"].dropna().empty:
            condition_counts = predictions_df["weather"].value_counts()
            weather_pie = px.pie(
                values=condition_counts.values,
                names=condition_counts.index,
                title="☁ Weather Conditions Distribution",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.35,
            )
            weather_pie.update_layout(height=CHART_HEIGHT)
            st.plotly_chart(weather_pie, use_container_width=True)

    # ── Row 3: Windowed aggregations ──────────────────────────────────────────
    if not aggregations_df.empty:
        st.markdown("---")
        st.subheader("📊 5-Minute Windowed Aggregations (Spark Structured Streaming)")
        left_col, right_col = st.columns(2)
        with left_col:
            st.plotly_chart(
                build_windowed_temp_chart(aggregations_df), use_container_width=True
            )
        with right_col:
            st.plotly_chart(
                build_windowed_humidity_wind_chart(aggregations_df),
                use_container_width=True,
            )

    # ── Raw data table ────────────────────────────────────────────────────────
    if show_raw:
        st.markdown("---")
        st.subheader("📋 Raw Data")
        display_columns = [
            TIMESTAMP_COL,
            "temperature",
            "predicted_temperature",
            "temperature_error",
            "humidity",
            "pressure",
            "weather",
            "wind_speed",
            "clouds",
        ]
        display_df = predictions_df[
            [c for c in display_columns if c in predictions_df.columns]
        ].copy()
        display_df[TIMESTAMP_COL] = display_df[TIMESTAMP_COL].astype(str)
        st.dataframe(display_df, use_container_width=True, height=300)

    # ── Footer & auto-refresh ─────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
        f"Records: {len(predictions_df)}  |  "
        f"Auto-refresh: every {refresh_rate}s"
    )

    time.sleep(refresh_rate)
    st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

render_main_dashboard()
