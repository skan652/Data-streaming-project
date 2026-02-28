FROM apache/spark:3.5.0
USER root
RUN pip install --no-cache-dir \
    joblib==1.3.2 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    scikit-learn==1.3.0
USER spark
WORKDIR /opt/spark-apps
CMD ["bash", "-c", \
    "python /opt/spark-apps/train_model.py && \
    spark-submit \
      --master local[*] \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.apache.kafka:kafka-clients:3.4.0 \
      /opt/spark-apps/streaming_ml.py"]
