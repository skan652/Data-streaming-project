"""
spark/streaming_ml.py
---------------------
Spark Structured Streaming job that:
  1. Consumes weather events from a Kafka topic.
  2. Applies data cleaning (null filtering) and feature engineering.
  3. Runs scikit-learn ML inference (temperature prediction) per micro-batch.
  4. Computes 5-minute windowed aggregations.
  5. Persists results as rolling JSON-lines files for the Streamlit dashboard.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    from_json,
    max,
    min,
    to_timestamp,
    window,
)
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Load .env file when running locally; no-op inside Docker (vars already injected)
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration – all values overridable via environment variables
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.environ.get(
    "MODEL_PATH", "/opt/spark-apps/model/temperature_model.pkl"
)
ANOMALY_MODEL_PATH: str = os.environ.get(
    "ANOMALY_MODEL_PATH", "/opt/spark-apps/model/anomaly_detector.pkl"
)
OUTPUT_PATH: str = os.environ.get("OUTPUT_PATH", "/opt/spark-apps/output")
KAFKA_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")

KAFKA_TOPIC: str = "weather-data"
TIMESTAMP_FORMAT: str = "yyyy-MM-dd HH:mm:ss"

# Streaming trigger intervals
PREDICTIONS_TRIGGER: str = "10 seconds"
AGGREGATIONS_TRIGGER: str = "30 seconds"
CONSOLE_TRIGGER: str = "10 seconds"

# Windowing parameters
WINDOW_DURATION: str = "5 minutes"
WATERMARK_DELAY: str = "10 minutes"

# Rolling file limits
MAX_PREDICTION_ROWS: int = 200
MAX_AGGREGATION_ROWS: int = 100

# Fallback heuristic when no trained model is available
FALLBACK_TEMP_SCALE: float = 0.9
FALLBACK_TEMP_OFFSET: float = 2.0

WEATHER_CONDITIONS: list[str] = ["Clear", "Clouds", "Rain", "Snow", "Thunderstorm"]

SPARK_KAFKA_PACKAGES: str = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
    "org.apache.kafka:kafka-clients:3.4.0"
)

# Module-level model cache – populated in __main__ before Spark starts
_model_data: Optional[dict] = None
_anomaly_data: Optional[dict] = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_sklearn_model() -> Optional[dict]:
    """Load the pre-trained scikit-learn model bundle from disk.

    Returns:
        A dict with keys ``"model"`` and ``"feature_columns"``, or ``None``
        when the model file does not yet exist (a heuristic fallback is used).
    """
    if not os.path.exists(MODEL_PATH):
        logger.warning(
            "Model file not found at '%s'. Using linear heuristic as fallback.", MODEL_PATH
        )
        return None

    bundle: dict = joblib.load(MODEL_PATH)
    logger.info("ML model loaded from '%s'.", MODEL_PATH)
    return bundle


def load_anomaly_model() -> Optional[dict]:
    """Load the pre-trained IsolationForest anomaly detector from disk.

    Returns:
        A dict with keys ``"model"`` and ``"feature_columns"``, or ``None``
        when the model file does not yet exist (anomaly detection is disabled).
    """
    if not os.path.exists(ANOMALY_MODEL_PATH):
        logger.warning(
            "Anomaly model not found at '%s'. Anomaly detection disabled.",
            ANOMALY_MODEL_PATH,
        )
        return None

    bundle: dict = joblib.load(ANOMALY_MODEL_PATH)
    logger.info("Anomaly detector loaded from '%s'.", ANOMALY_MODEL_PATH)
    return bundle


# ---------------------------------------------------------------------------
# Feature engineering (pure pandas – runs on the driver per micro-batch)
# ---------------------------------------------------------------------------

def engineer_features(batch: pd.DataFrame) -> pd.DataFrame:
    """Enrich a batch with cyclical time features and one-hot weather flags.

    Cyclical encoding (sin/cos) correctly represents the circular nature of
    hours-of-day and days-of-year without imposing a false ordinal distance.

    Args:
        batch: Raw weather records as a pandas DataFrame.

    Returns:
        A copy of *batch* with additional feature columns appended.
    """
    result = batch.copy()

    result["dt"] = pd.to_datetime(result["timestamp_readable"], errors="coerce")
    result["hour"] = result["dt"].dt.hour.fillna(0).astype(int)
    result["day_of_year"] = result["dt"].dt.day_of_year.fillna(1).astype(int)

    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["day_sin"] = np.sin(2 * np.pi * result["day_of_year"] / 365)
    result["day_cos"] = np.cos(2 * np.pi * result["day_of_year"] / 365)

    for condition in WEATHER_CONDITIONS:
        result[f"weather_{condition}"] = (result["weather"] == condition).astype(int)

    return result


def predict_temperature(batch: pd.DataFrame) -> pd.DataFrame:
    """Run temperature prediction on a batch of weather records.

    Uses the trained Random Forest when available; falls back to a linear
    heuristic otherwise so the pipeline never halts.

    Args:
        batch: Weather records; receives feature columns from
               :func:`engineer_features`.

    Returns:
        *batch* with ``predicted_temperature`` and ``temperature_error`` added.
    """
    batch = engineer_features(batch)

    if _model_data is not None:
        model = _model_data["model"]
        feature_cols: list[str] = _model_data["feature_columns"]

        # Guarantee every expected column exists (fill absent ones with 0)
        for feature in feature_cols:
            if feature not in batch.columns:
                batch[feature] = 0

        batch["predicted_temperature"] = model.predict(batch[feature_cols])
    else:
        batch["predicted_temperature"] = (
            batch["temperature"] * FALLBACK_TEMP_SCALE + FALLBACK_TEMP_OFFSET
        )

    batch["temperature_error"] = (
        batch["temperature"] - batch["predicted_temperature"]
    ).abs()

    # Anomaly detection – IsolationForest: -1 = anomaly, 1 = normal
    if _anomaly_data is not None:
        anomaly_model = _anomaly_data["model"]
        anomaly_feature_cols: list[str] = _anomaly_data["feature_columns"]
        for feat in anomaly_feature_cols:
            if feat not in batch.columns:
                batch[feat] = 0
        batch["is_anomaly"] = (
            anomaly_model.predict(batch[anomaly_feature_cols]) == -1
        ).astype(int)
    else:
        batch["is_anomaly"] = 0

    return batch


# ---------------------------------------------------------------------------
# Rolling JSON-lines file writer
# ---------------------------------------------------------------------------

def _append_to_json_file(
    new_rows: pd.DataFrame,
    file_path: str,
    max_rows: int,
) -> None:
    """Append *new_rows* to a rolling JSON-lines file, capped at *max_rows*.

    Args:
        new_rows:  DataFrame rows to append.
        file_path: Absolute path to the JSON-lines output file.
        max_rows:  Maximum number of rows retained after appending.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        try:
            existing = pd.read_json(file_path, lines=True)
            new_rows = pd.concat([existing, new_rows], ignore_index=True).tail(max_rows)
        except ValueError as exc:
            logger.warning(
                "Could not parse existing file '%s' (%s). Overwriting.", file_path, exc
            )

    new_rows.to_json(file_path, orient="records", lines=True, date_format="iso")


# ---------------------------------------------------------------------------
# foreachBatch sink callbacks
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: list[str] = [
    "timestamp_readable",
    "city",
    "temperature",
    "predicted_temperature",
    "temperature_error",
    "is_anomaly",
    "humidity",
    "pressure",
    "weather",
    "weather_description",
    "wind_speed",
    "clouds",
]


def write_predictions(batch_df: DataFrame, epoch_id: int) -> None:
    """foreachBatch callback: apply ML inference and write predictions to JSON.

    Args:
        batch_df:  Spark DataFrame for the current micro-batch.
        epoch_id:  Monotonically increasing batch identifier provided by Spark.
    """
    if batch_df.rdd.isEmpty():
        logger.debug("Prediction batch %d is empty – skipping.", epoch_id)
        return

    batch = batch_df.toPandas()
    batch = predict_temperature(batch)
    batch = batch[[c for c in OUTPUT_COLUMNS if c in batch.columns]]

    output_file = os.path.join(OUTPUT_PATH, "predictions.json")
    _append_to_json_file(batch, output_file, MAX_PREDICTION_ROWS)
    logger.info("Batch %d: %d prediction records → %s", epoch_id, len(batch), output_file)


def write_aggregations(batch_df: DataFrame, epoch_id: int) -> None:
    """foreachBatch callback: flatten window struct and write aggregations to JSON.

    Args:
        batch_df:  Spark DataFrame for the current micro-batch.
        epoch_id:  Monotonically increasing batch identifier provided by Spark.
    """
    if batch_df.rdd.isEmpty():
        logger.debug("Aggregation batch %d is empty – skipping.", epoch_id)
        return

    batch = batch_df.toPandas()

    # The ``window`` column is a struct {start, end} – flatten to plain strings
    if "window" in batch.columns:
        try:
            batch["window_start"] = batch["window"].apply(
                lambda w: str(w["start"]) if isinstance(w, dict) else str(w.start)
            )
            batch["window_end"] = batch["window"].apply(
                lambda w: str(w["end"]) if isinstance(w, dict) else str(w.end)
            )
        except (KeyError, AttributeError) as exc:
            logger.warning("Could not unpack window struct: %s – using empty strings.", exc)
            batch["window_start"] = ""
            batch["window_end"] = ""
        batch = batch.drop(columns=["window"])

    output_file = os.path.join(OUTPUT_PATH, "aggregations.json")
    _append_to_json_file(batch, output_file, MAX_AGGREGATION_ROWS)
    logger.info(
        "Aggregation batch %d: %d window rows → %s", epoch_id, len(batch), output_file
    )


# ---------------------------------------------------------------------------
# Spark Structured Streaming application
# ---------------------------------------------------------------------------

class WeatherStreamingApp:
    """Spark Structured Streaming application for real-time weather analytics.

    Consumes events from Kafka, cleans data, predicts temperatures with a
    scikit-learn model and writes rolling JSON files for the dashboard.
    """

    #: Kafka message schema – must match the producer's output exactly
    SCHEMA: StructType = StructType(
        [
            StructField("timestamp", LongType()),
            StructField("city", StringType()),
            StructField("temperature", DoubleType()),
            StructField("feels_like", DoubleType()),
            StructField("humidity", IntegerType()),
            StructField("pressure", IntegerType()),
            StructField("weather", StringType()),
            StructField("weather_description", StringType()),
            StructField("wind_speed", DoubleType()),
            StructField("wind_deg", IntegerType()),
            StructField("clouds", IntegerType()),
            StructField("timestamp_readable", StringType()),
        ]
    )

    def __init__(self) -> None:
        self._spark: SparkSession = self._build_spark_session()
        self._spark.sparkContext.setLogLevel("WARN")
        logger.info("SparkSession initialised.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spark_session() -> SparkSession:
        """Create and return a configured SparkSession."""
        return (
            SparkSession.builder.appName("WeatherStreamingML")
            .config("spark.jars.packages", SPARK_KAFKA_PACKAGES)
            .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint_weather")
            .config("spark.sql.adaptive.enabled", "false")
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
            .config("spark.driver.host", "localhost")
            .config("spark.driver.bindAddress", "0.0.0.0")
            .getOrCreate()
        )

    def _read_kafka(self) -> DataFrame:
        """Return an unbounded streaming DataFrame backed by the Kafka topic."""
        return (
            self._spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", KAFKA_SERVERS)
            .option("subscribe", KAFKA_TOPIC)
            .option("startingOffsets", "latest")
            .option("failOnDataLoss", "false")
            .option("maxOffsetsPerTrigger", "50")
            .load()
        )

    def _parse_and_clean(self, kafka_df: DataFrame) -> DataFrame:
        """Deserialise JSON payloads, enforce schema and drop null temperatures.

        Args:
            kafka_df: Raw Kafka streaming DataFrame (binary value column).

        Returns:
            Typed streaming DataFrame with an ``event_time`` TimestampType
            column for watermarking and windowing.
        """
        parsed = (
            kafka_df
            .select(from_json(col("value").cast("string"), self.SCHEMA).alias("data"))
            .select("data.*")
            .filter(col("temperature").isNotNull())
        )

        return parsed.withColumn(
            "event_time",
            to_timestamp(col("timestamp_readable"), TIMESTAMP_FORMAT),
        )

    def _windowed_aggregations(self, parsed_df: DataFrame) -> DataFrame:
        """Compute per-city statistics over 5-minute tumbling windows.

        A 10-minute watermark is applied to handle late-arriving events
        without holding state indefinitely.

        Args:
            parsed_df: Cleaned streaming DataFrame with ``event_time`` column.

        Returns:
            Aggregated streaming DataFrame (one row per window per city).
        """
        return (
            parsed_df
            .withWatermark("event_time", WATERMARK_DELAY)
            .groupBy(window(col("event_time"), WINDOW_DURATION), col("city"))
            .agg(
                avg("temperature").alias("avg_temperature"),
                max("temperature").alias("max_temperature"),
                min("temperature").alias("min_temperature"),
                avg("humidity").alias("avg_humidity"),
                avg("wind_speed").alias("avg_wind_speed"),
                avg("pressure").alias("avg_pressure"),
                count("*").alias("readings_count"),
            )
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start all streaming queries and block until one terminates or fails."""
        kafka_df = self._read_kafka()
        parsed_df = self._parse_and_clean(kafka_df)
        agg_df = self._windowed_aggregations(parsed_df)

        # Query 1 – ML predictions → JSON (consumed by the dashboard)
        (
            parsed_df.writeStream
            .outputMode("append")
            .foreachBatch(write_predictions)
            .trigger(processingTime=PREDICTIONS_TRIGGER)
            .option("checkpointLocation", "/tmp/checkpoint_predictions")
            .queryName("predictions")
            .start()
        )

        # Query 2 – windowed aggregations → JSON (consumed by the dashboard)
        (
            agg_df.writeStream
            .outputMode("append")
            .foreachBatch(write_aggregations)
            .trigger(processingTime=AGGREGATIONS_TRIGGER)
            .option("checkpointLocation", "/tmp/checkpoint_aggregations")
            .queryName("aggregations")
            .start()
        )

        # Query 3 – console sink for development / debugging
        (
            parsed_df.writeStream
            .outputMode("append")
            .format("console")
            .option("truncate", "false")
            .trigger(processingTime=CONSOLE_TRIGGER)
            .queryName("console")
            .start()
        )

        logger.info("All streaming queries started. Output directory: %s", OUTPUT_PATH)
        self._spark.streams.awaitAnyTermination()

    def stop(self) -> None:
        """Gracefully stop the SparkSession and release resources."""
        self._spark.stop()
        logger.info("SparkSession stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load both ML models on the driver *before* Spark starts so they are
    # available inside the foreachBatch closures without being re-loaded
    # on every micro-batch.
    _model_data = load_sklearn_model()
    _anomaly_data = load_anomaly_model()

    streaming_app = WeatherStreamingApp()
    try:
        streaming_app.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt – shutting down gracefully.")
        streaming_app.stop()
    except Exception as exc:  # noqa: BLE001
        logger.error("Fatal error: %s", exc, exc_info=True)
        streaming_app.stop()
        raise
