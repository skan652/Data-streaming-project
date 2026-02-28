"""
producer/weather_producer.py
----------------------------
Kafka producer that polls the OpenWeather current-conditions API at a
configurable interval and publishes structured JSON messages to the
``weather-data`` topic.
"""

from __future__ import annotations

import json
import logging
import os
import time

import requests
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Load .env file when running locally; no-op inside Docker (vars already injected)
load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """Return the value of a mandatory environment variable.

    Args:
        name: Name of the environment variable.

    Returns:
        The non-empty string value.

    Raises:
        EnvironmentError: If the variable is unset or empty.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set.\n"
            f"Copy '.env.example' to '.env' and fill in the values."
        )
    return value


# ---------------------------------------------------------------------------
# Configuration – all values overridable via environment variables
# ---------------------------------------------------------------------------
API_KEY: str = _require_env("OPENWEATHER_API_KEY")  # mandatory – no default
CITY: str = os.environ.get("WEATHER_CITY", "Tunis")
KAFKA_SERVERS: list[str] = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS", "localhost:29092"
).split(",")
KAFKA_TOPIC: str = "weather-data"
FETCH_INTERVAL_SEC: int = int(os.environ.get("FETCH_INTERVAL_SEC", "60"))

OPENWEATHER_URL: str = "http://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT_SEC: int = 10
KAFKA_SEND_TIMEOUT_SEC: int = 10
KAFKA_MAX_RETRIES: int = 10
KAFKA_RETRY_DELAY_SEC: int = 5
KAFKA_PRODUCER_RETRIES: int = 5


# ---------------------------------------------------------------------------
# Weather API
# ---------------------------------------------------------------------------

def fetch_weather_data() -> dict:
    """Fetch the current weather observation for the configured city.

    Returns:
        Parsed JSON response from the OpenWeather API.

    Raises:
        requests.HTTPError: If the API returns a non-2xx status code.
        requests.RequestException: On any network-level failure.
    """
    response = requests.get(
        OPENWEATHER_URL,
        params={"q": CITY, "appid": API_KEY, "units": "metric"},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return response.json()


def parse_weather_record(raw: dict) -> dict:
    """Extract and normalise relevant fields from a raw API response.

    Args:
        raw: Full JSON response dict from the OpenWeather API.

    Returns:
        A flat dict with the fields consumed by the Spark streaming schema.
    """
    return {
        "timestamp": raw["dt"],
        "city": raw["name"],
        "temperature": raw["main"]["temp"],
        "feels_like": raw["main"]["feels_like"],
        "humidity": raw["main"]["humidity"],
        "pressure": raw["main"]["pressure"],
        "weather": raw["weather"][0]["main"],
        "weather_description": raw["weather"][0]["description"],
        "wind_speed": raw["wind"]["speed"],
        "wind_deg": raw["wind"].get("deg", 0),
        "clouds": raw["clouds"]["all"],
        "timestamp_readable": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.gmtime(raw["dt"])
        ),
    }


# ---------------------------------------------------------------------------
# Kafka producer
# ---------------------------------------------------------------------------

def create_kafka_producer() -> KafkaProducer:
    """Create a KafkaProducer, retrying until Kafka is ready.

    Retries up to ``KAFKA_MAX_RETRIES`` times with a fixed delay so the
    producer container can start alongside the Kafka broker in Docker Compose.

    Returns:
        A connected :class:`KafkaProducer` instance.

    Raises:
        RuntimeError: If the broker remains unreachable after all retries.
    """
    for attempt in range(1, KAFKA_MAX_RETRIES + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                retries=KAFKA_PRODUCER_RETRIES,
            )
            logger.info("Kafka producer connected to %s.", KAFKA_SERVERS)
            return producer
        except KafkaError as exc:
            logger.warning(
                "Kafka not ready (attempt %d/%d): %s. Retrying in %ds...",
                attempt,
                KAFKA_MAX_RETRIES,
                exc,
                KAFKA_RETRY_DELAY_SEC,
            )
            time.sleep(KAFKA_RETRY_DELAY_SEC)

    raise RuntimeError(
        f"Could not connect to Kafka at {KAFKA_SERVERS} after {KAFKA_MAX_RETRIES} attempts."
    )


def publish_record(producer: KafkaProducer, record: dict) -> None:
    """Serialise and publish one weather record to the Kafka topic.

    Args:
        producer: An active :class:`KafkaProducer` instance.
        record:   Weather record dict from :func:`parse_weather_record`.
    """
    message_key: str = str(record["timestamp"])
    metadata = producer.send(
        KAFKA_TOPIC, key=message_key, value=record
    ).get(timeout=KAFKA_SEND_TIMEOUT_SEC)

    logger.info(
        "Published | city=%-10s | temp=%5.1f °C | condition=%-12s | "
        "partition=%d | offset=%d",
        record["city"],
        record["temperature"],
        record["weather"],
        metadata.partition,
        metadata.offset,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Kafka producer loop indefinitely."""
    producer = create_kafka_producer()
    logger.info(
        "Starting weather stream for '%s' (interval: %ds).", CITY, FETCH_INTERVAL_SEC
    )

    try:
        while True:
            try:
                raw_data = fetch_weather_data()
                record = parse_weather_record(raw_data)
                publish_record(producer, record)
            except requests.RequestException as exc:
                logger.error("API request failed: %s", exc)
            except KafkaError as exc:
                logger.error("Kafka publish error: %s", exc)
            except (KeyError, IndexError) as exc:
                logger.error("Unexpected API response structure: %s", exc)

            time.sleep(FETCH_INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("Interrupt received – stopping producer.")
    finally:
        producer.close()
        logger.info("Kafka producer closed.")


if __name__ == "__main__":
    main()
