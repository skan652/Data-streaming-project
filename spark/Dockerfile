FROM apache/spark:3.5.0
USER root
RUN apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip install --no-cache-dir \
    joblib==1.3.2 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    scikit-learn==1.3.0
# Create .ivy2 directory with proper permissions
RUN mkdir -p /home/spark/.ivy2 && chown -R spark:spark /home/spark/.ivy2
USER spark
WORKDIR /opt/spark-apps
CMD ["bash", "-c", \
    "python3 /opt/spark-apps/train_model.py && \
    /opt/spark/bin/spark-submit \
      --master local[*] \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.apache.kafka:kafka-clients:3.4.0 \
      /opt/spark-apps/streaming_ml.py"]
