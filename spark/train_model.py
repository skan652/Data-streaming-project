# spark/train_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import json
from datetime import datetime, timedelta
import random

# Create model directory if it doesn't exist
os.makedirs('spark/model', exist_ok=True)

def generate_synthetic_weather_data(n_samples=10000):
    """
    Generate synthetic weather data for training
    In production, you'd use historical data from OpenWeather API
    """
    print("Generating synthetic weather data...")
    
    data = []
    start_time = datetime.now() - timedelta(days=365)
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(minutes=10*i)
        
        # Features with realistic correlations
        hour_of_day = timestamp.hour
        day_of_year = timestamp.timetuple().tm_yday
        
        # Temperature varies by season and time of day
        base_temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365)  # Seasonal variation
        daily_variation = 5 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)  # Daily variation
        temperature = base_temp + daily_variation + np.random.normal(0, 2)
        
        # Humidity inversely correlated with temperature (simplified)
        humidity = max(30, min(95, 70 - 0.5 * (temperature - 15) + np.random.normal(0, 10)))
        
        # Pressure slightly correlated with weather
        pressure = 1013 + np.random.normal(0, 5) - 0.1 * (temperature - 15)
        
        # Wind speed random but slightly higher in extreme temperatures
        wind_speed = max(0, 5 + 0.2 * abs(temperature - 15) + np.random.normal(0, 2))
        
        # Cloud cover random
        clouds = np.random.randint(0, 101)
        
        # Weather condition based on features
        if humidity > 80 and clouds > 70:
            weather = "Rain"
        elif clouds > 70:
            weather = "Clouds"
        else:
            weather = "Clear"
        
        data.append({
            'timestamp': timestamp.timestamp(),
            'hour': hour_of_day,
            'day_of_year': day_of_year,
            'month': timestamp.month,
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'wind_speed': wind_speed,
            'clouds': clouds,
            'weather_condition': weather
        })
    
    return pd.DataFrame(data)

def prepare_features(df):
    """Prepare features for ML model"""
    # One-hot encode weather condition
    weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather')
    
    # Cyclical encoding for hour and day_of_year
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Select features
    feature_cols = ['humidity', 'pressure', 'wind_speed', 'clouds',
                    'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    
    features = pd.concat([df[feature_cols], weather_dummies], axis=1)
    
    return features, df['temperature']

def train_temperature_model():
    """Train a model to predict temperature"""
    print("Training temperature prediction model...")
    
    # Generate synthetic data
    df = generate_synthetic_weather_data(20000)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R2 Score: {r2:.2f}")
    print(f"  RMSE: {np.sqrt(mse):.2f}°C")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Important Features:")
    print(feature_importance.head())
    
    # Save model and feature columns
    model_path = 'spark/model/temperature_model.pkl'
    joblib.dump({
        'model': model,
        'feature_columns': list(X.columns)
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    return model, X.columns

if __name__ == "__main__":
    model, feature_columns = train_temperature_model()