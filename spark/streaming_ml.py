# spark/streaming_ml_pyspark.py
from itertools import count

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp, window, to_timestamp, when, hour, dayofyear, avg, max, min, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.regression import RandomForestRegressionModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PySparkStreamingML:
    def __init__(self):
        self.setup_spark()
        self.load_models()
        
    def setup_spark(self):
        """Initialize Spark Session with Kafka support"""
        self.spark = SparkSession.builder \
            .appName("WeatherStreamingML") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
            .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoint") \
            .config("spark.sql.adaptive.enabled", "false") \
            .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "localhost") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info("Spark session created")
    
    def load_models(self):
        """Load pre-trained ML models"""
        try:
            self.model = RandomForestRegressionModel.load("/opt/spark-apps/model")
            self.scaler = StandardScalerModel.load("/opt/spark-apps/scaler")
            logger.info("ML models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.warning("Using simple heuristic (no ML)")
            self.model = None
            self.scaler = None
    
    def get_schema(self):
        """Define the schema for weather data"""
        return StructType([
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
            StructField("timestamp_readable", StringType())
        ])
    
    def read_from_kafka(self):
        """Read streaming data from Kafka"""
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", "localhost:29092") \
            .option("subscribe", "weather-data") \
            .option("startingOffsets", "latest") \
            .option("failOnDataLoss", "false") \
            .option("maxOffsetsPerTrigger", "100") \
            .load()
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        
        # Parse timestamp and extract time features
        df = df.withColumn("dt", to_timestamp("timestamp_readable"))
        df = df.withColumn("hour", hour("dt"))
        df = df.withColumn("day_of_year", dayofyear("dt"))
        
        # Create cyclical features
        from pyspark.sql.functions import sin, cos
        import numpy as np
        
        df = df.withColumn("hour_sin", sin(2 * np.pi * col("hour") / 24))
        df = df.withColumn("hour_cos", cos(2 * np.pi * col("hour") / 24))
        df = df.withColumn("day_sin", sin(2 * np.pi * col("day_of_year") / 365))
        df = df.withColumn("day_cos", cos(2 * np.pi * col("day_of_year") / 365))
        
        # One-hot encode weather condition
        weather_conditions = ['Clear', 'Clouds', 'Rain', 'Snow', 'Thunderstorm']
        for condition in weather_conditions:
            df = df.withColumn(f"weather_{condition}", 
                               when(col("weather") == condition, 1).otherwise(0))
        
        return df
    
    def apply_ml_model(self, df):
        """Apply ML model to make predictions"""
        
        # Prepare features
        df_prepared = self.prepare_features(df)
        
        if self.model and self.scaler:
            # Assemble features
            feature_cols = ["humidity", "pressure", "wind_speed", "clouds",
                           "hour_sin", "hour_cos", "day_sin", "day_cos",
                           "weather_Clear", "weather_Clouds", "weather_Rain"]
            
            # Filter to only available columns
            available_cols = [c for c in feature_cols if c in df_prepared.columns]
            
            assembler = VectorAssembler(
                inputCols=available_cols,
                outputCol="features_raw"
            )
            
            df_assembled = assembler.transform(df_prepared)
            
            # Scale features
            df_scaled = self.scaler.transform(df_assembled)
            
            # Make predictions
            predictions = self.model.transform(df_scaled)
            
            # Select relevant columns
            result = predictions.select(
                "timestamp_readable",
                "city",
                "temperature",
                col("prediction").alias("predicted_temperature"),
                "humidity",
                "pressure",
                "weather",
                "wind_speed",
                "clouds"
            ).withColumn(
                "temperature_error", 
                abs(col("temperature") - col("predicted_temperature"))
            )
            
        else:
            # Fallback: simple heuristic
            result = df_prepared.select(
                "timestamp_readable",
                "city",
                "temperature",
                "humidity",
                "pressure",
                "weather",
                "wind_speed",
                "clouds"
            ).withColumn(
                "predicted_temperature", 
                col("temperature") * 0.9 + 2
            ).withColumn(
                "temperature_error",
                abs(col("temperature") - col("predicted_temperature"))
            )
        
        return result
    
    def aggregate_windowing(self, df):
        """Apply windowed aggregations"""
        return df \
            .withWatermark("timestamp_readable", "10 minutes") \
            .groupBy(
                window(col("timestamp_readable"), "5 minutes"),
                col("city")
            ) \
            .agg(
                avg("temperature").alias("avg_temperature"),
                avg("predicted_temperature").alias("avg_predicted_temperature"),
                avg("temperature_error").alias("avg_prediction_error"),
                max("temperature").alias("max_temperature"),
                min("temperature").alias("min_temperature"),
                avg("humidity").alias("avg_humidity"),
                avg("wind_speed").alias("avg_wind_speed"),
                count("*").alias("readings_count")
            )
    
    def start_streaming(self):
        """Start the streaming application"""
        
        try:
            # Read from Kafka
            kafka_df = self.read_from_kafka()
            
            # Parse JSON
            schema = self.get_schema()
            parsed_df = kafka_df.select(
                from_json(col("value").cast("string"), schema).alias("data")
            ).select("data.*")
            
            # Filter out nulls
            parsed_df = parsed_df.filter(col("temperature").isNotNull())
            
            # Apply ML model
            predictions_df = self.apply_ml_model(parsed_df)
            
            # Apply windowed aggregations
            aggregated_df = self.aggregate_windowing(predictions_df)
            
            # Write predictions to console (for debugging)
            console_query = predictions_df \
                .writeStream \
                .outputMode("append") \
                .format("console") \
                .option("truncate", "false") \
                .trigger(processingTime="10 seconds") \
                .queryName("console_output") \
                .start()
            
            # Write to memory for dashboard
            memory_query = predictions_df \
                .writeStream \
                .outputMode("append") \
                .format("memory") \
                .queryName("current_weather") \
                .trigger(processingTime="5 seconds") \
                .start()
            
            # Write aggregations to memory
            agg_memory_query = aggregated_df \
                .writeStream \
                .outputMode("append") \
                .format("memory") \
                .queryName("weather_aggregates") \
                .trigger(processingTime="10 seconds") \
                .start()
            
            logger.info("✅ Streaming queries started successfully")
            logger.info("📊 Console output will appear below:")
            logger.info("=" * 60)
            
            # Wait for termination
            self.spark.streams.awaitAnyTermination()
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")

if __name__ == "__main__":
    streaming_app = PySparkStreamingML()
    try:
        streaming_app.start_streaming()
    except KeyboardInterrupt:
        logger.info("🛑 Stopping streaming application...")
        streaming_app.cleanup()
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        streaming_app.cleanup()