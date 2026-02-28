# Real-Time Weather Streaming Application

A complete data-streaming pipeline that ingests live weather data, processes it
with Apache Spark Structured Streaming, applies Machine Learning, and visualises
the results in a Streamlit dashboard — all connected via Apache Kafka.

---

## Architecture

```text
┌─────────────────────────────────────────┐
│          OpenWeather API                │
│       (weather_producer.py)             │
└──────────────────┬──────────────────────┘
                   │  Kafka: weather-data
                   ▼
┌─────────────────────────────────────────┐
│      Spark Structured Streaming         │
│         (streaming_ml.py)               │
│                                         │
│  • data cleaning   • 5-min windows      │
│  • feature eng.    • foreachBatch       │
│  • Random Forest   (temperature pred.)  │
│  • Isolation Forest (anomaly detect.)   │
└──────────────────┬──────────────────────┘
                   │  spark/output/ (JSON)
                   ▼
┌─────────────────────────────────────────┐
│         Streamlit Dashboard             │
│              (app.py)                   │
│                                         │
│  • actual vs predicted temperature      │
│  • anomaly detection markers            │
│  • humidity & pressure charts           │
│  • windowed aggregations                │
└─────────────────────────────────────────┘
```

### Components

| Component | Technology | Role |
| --------- | ---------- | ---- |
| **Producer** | kafka-python | Polls OpenWeather API → Kafka |
| **Message broker** | Kafka 3.5 + Zookeeper | Durable event streaming |
| **Stream processor** | PySpark 3.5 | Consumes Kafka, ML, JSON output |
| **ML model** | scikit-learn | RF regression + anomaly detection |
| **Dashboard** | Streamlit + Plotly | Auto-refreshing charts |

---

## Machine Learning

Two models are trained in `spark/train_model.py` on real historical data
fetched from the Open-Meteo archive API (free, no API key required):

1. **Random Forest Regressor** — predicts air temperature from
   humidity, pressure, wind speed, cloud cover and cyclical time
   features (sin/cos encoded hour and day-of-year).
   Evaluated with RMSE and R².
2. **Isolation Forest** — unsupervised anomaly detector; flags readings with
   unusual combinations of weather variables.

Models are serialised with `joblib` to `spark/model/` and loaded
at Spark startup.
The training date range and city coordinates are configurable via environment
variables (`TRAINING_START_DATE`, `TRAINING_END_DATE`, `TRAINING_LATITUDE`,
`TRAINING_LONGITUDE`) in your `.env` file.

---

## Project Structure

```text
.
├── docker-compose.yaml          # Full stack (Kafka + Spark + Producer + Dashboard)
├── requirements.txt             # Python dependencies
├── producer/
│   ├── Dockerfile
│   └── weather_producer.py      # Kafka producer (OpenWeather API)
├── spark/
│   ├── Dockerfile
│   ├── streaming_ml.py          # Spark Structured Streaming + ML inference
│   ├── train_model.py           # Model training (run once before streaming)
│   └── model/                   # Saved models (created after training)
│       ├── temperature_model.pkl
│       └── anomaly_detector.pkl
└── dashboard/
    ├── Dockerfile
    └── app.py                   # Streamlit dashboard
```

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
# 1. Clone repo
git clone https://github.com/skan652/Data-streaming-project.git
cd Data-streaming-project

# 2. Start the full stack
docker-compose up --build
```

The Spark container automatically runs `train_model.py`
before starting the streaming job.

| Service | URL |
| ------- | --- |
| Streamlit dashboard | <http://localhost:8501> |
| Spark UI | <http://localhost:4040> |
| Kafka broker (external) | localhost:29092 |

### Option B — Run locally (step by step)

**Prerequisites:** Python ≥ 3.9, Java ≥ 11, Docker (for Kafka)

```bash
# 1. Start Kafka + Zookeeper only
docker-compose up -d zookeeper kafka

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Train the ML models (one-time step)
python spark/train_model.py

# 4. Start the Kafka producer
python producer/weather_producer.py

# 5. Start the Spark streaming job (new terminal)
python spark/streaming_ml.py
# OR with spark-submit:
# spark-submit \
#   --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
#   spark/streaming_ml.py

# 6. Start the Streamlit dashboard (new terminal)
OUTPUT_PATH=./spark/output streamlit run dashboard/app.py
```

Open <http://localhost:8501> in your browser.

---

## Configuration

Environment variables (set in `docker-compose.yaml` or your shell):

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `OPENWEATHER_API_KEY` | *(required)* | OpenWeather API key |
| `WEATHER_CITY` | `Tunis` | City to monitor |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:29092` | Kafka broker address(es) |
| `FETCH_INTERVAL_SEC` | `60` | Producer polling interval in seconds |
| `MODEL_PATH` | see docker-compose | Trained RF temperature model path |
| `ANOMALY_MODEL_PATH` | see docker-compose | Trained anomaly detector path |
| `OUTPUT_PATH` | `/opt/spark-apps/output` | Directory for JSON output files |
| `TRAINING_LATITUDE` | `36.8065` | Training data latitude |
| `TRAINING_LONGITUDE` | `10.1815` | Training data longitude |
| `TRAINING_START_DATE` | `2022-01-01` | Training start date (ISO-8601) |
| `TRAINING_END_DATE` | `2023-12-31` | Training end date (ISO-8601) |

---

## Technology Stack

- **Apache Kafka 3.5** — distributed event streaming
- **Apache Spark 3.5** Structured Streaming — micro-batch stream processing
- **scikit-learn 1.3** — Random Forest regression +
  Isolation Forest anomaly detection
- **Streamlit 1.28** + **Plotly 5.17** — interactive real-time dashboard
- **OpenWeather API** — free-tier weather data source
- **Docker / Docker Compose** — containerised deployment
