FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    kafka-python==2.0.2 \
    python-dotenv==1.0.0 \
    requests==2.31.0
COPY producer/weather_producer.py .
USER nobody
CMD ["python", "weather_producer.py"]
