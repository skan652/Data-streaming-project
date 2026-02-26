import requests
import json
import time
from kafka import KafkaProducer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenWeather API configuration
API_KEY = "3987243f4c1f1a2d36b312e9b47bee3a"  # Your API key
CITY = "Tunis"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:29092']
TOPIC = 'weather-data'

def fetch_weather_data():
    """Fetch current weather data from OpenWeather API"""
    params = {
        'q': CITY,
        'appid': API_KEY,
        'units': 'metric'  # For Celsius
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather data: {e}")
        return None

def create_kafka_producer():
    """Create and return a Kafka producer"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda v: v.encode('utf-8') if v else None
        )
        logger.info("Kafka producer created successfully")
        return producer
    except Exception as e:
        logger.error(f"Error creating Kafka producer: {e}")
        return None

def main():
    producer = create_kafka_producer()
    if not producer:
        return
    
    logger.info(f"Starting weather data streaming for {CITY}")
    
    while True:
        try:
            # Fetch weather data
            weather_data = fetch_weather_data()
            
            if weather_data:
                # Extract relevant fields
                processed_data = {
                    'timestamp': weather_data['dt'],
                    'city': weather_data['name'],
                    'temperature': weather_data['main']['temp'],
                    'feels_like': weather_data['main']['feels_like'],
                    'humidity': weather_data['main']['humidity'],
                    'pressure': weather_data['main']['pressure'],
                    'weather': weather_data['weather'][0]['main'],
                    'weather_description': weather_data['weather'][0]['description'],
                    'wind_speed': weather_data['wind']['speed'],
                    'wind_deg': weather_data['wind'].get('deg', 0),
                    'clouds': weather_data['clouds']['all'],
                    'timestamp_readable': time.strftime('%Y-%m-%d %H:%M:%S', 
                                                       time.gmtime(weather_data['dt']))
                }
                
                # Send to Kafka
                future = producer.send(
                    TOPIC,
                    key=str(weather_data['dt']),
                    value=processed_data
                )
                
                # Wait for send confirmation
                record_metadata = future.get(timeout=10)
                logger.info(f"Sent weather data to topic {record_metadata.topic} "
                          f"partition {record_metadata.partition} "
                          f"offset {record_metadata.offset}")
                
                logger.info(f"Weather in {CITY}: {processed_data['temperature']}°C, "
                          f"{processed_data['weather']}")
            
            # Wait before next fetch (OpenWeather free tier: 60 calls/minute max)
            time.sleep(60)  # Fetch every 60 seconds
            
        except KeyboardInterrupt:
            logger.info("Stopping weather producer...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(10)
    
    producer.close()

if __name__ == "__main__":
    main()