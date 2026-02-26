# dashboard/app_pyspark.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pyspark.sql import SparkSession
import time
from datetime import datetime
import threading

# Page config
st.set_page_config(
    page_title="Weather Streaming Dashboard",
    page_icon="🌤️",
    layout="wide"
)

# Initialize Spark session
@st.cache_resource
def get_spark_session():
    return SparkSession.builder \
        .appName("WeatherDashboard") \
        .config("spark.sql.streaming.checkpointLocation", "/tmp/dashboard_checkpoint") \
        .config("spark.driver.host", "localhost") \
        .getOrCreate()

spark = get_spark_session()

# Title
st.title("🌤️ Real-Time Weather Streaming Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Controls")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)
show_raw_data = st.sidebar.checkbox("Show Raw Data", False)

# Main layout
col1, col2, col3, col4 = st.columns(4)

# Placeholders for metrics
metric_placeholders = {
    'temp': col1.empty(),
    'pred_temp': col2.empty(),
    'humidity': col3.empty(),
    'wind': col4.empty()
}

# Charts
chart_col1, chart_col2 = st.columns(2)
temp_chart = chart_col1.empty()
error_chart = chart_col2.empty()

agg_chart1, agg_chart2 = st.columns(2)
hourly_temp_chart = agg_chart1.empty()
weather_dist_chart = agg_chart2.empty()

# Raw data table
raw_data_table = st.empty()

# Status
status_text = st.empty()

def update_dashboard():
    """Update dashboard with latest data"""
    last_update = time.time()
    
    while True:
        try:
            # Check if tables exist
            tables = [row.tableName for row in spark.catalog.listTables()]
            
            if 'current_weather' in tables:
                # Query current weather data
                current_df = spark.sql("""
                    SELECT 
                        timestamp_readable,
                        city,
                        temperature,
                        predicted_temperature,
                        temperature_error,
                        humidity,
                        weather,
                        wind_speed
                    FROM current_weather 
                    ORDER BY timestamp_readable DESC 
                    LIMIT 1
                """).toPandas()
                
                # Query all current data
                all_current_df = spark.sql("""
                    SELECT * FROM current_weather 
                    ORDER BY timestamp_readable DESC 
                    LIMIT 50
                """).toPandas()
                
                # Update metrics
                if not current_df.empty:
                    row = current_df.iloc[0]
                    
                    metric_placeholders['temp'].metric(
                        "Actual Temperature",
                        f"{row['temperature']:.1f}°C"
                    )
                    
                    metric_placeholders['pred_temp'].metric(
                        "Predicted Temperature",
                        f"{row['predicted_temperature']:.1f}°C",
                        f"Error: {row['temperature_error']:.1f}°C"
                    )
                    
                    metric_placeholders['humidity'].metric(
                        "Humidity",
                        f"{row['humidity']:.0f}%"
                    )
                    
                    metric_placeholders['wind'].metric(
                        "Wind Speed",
                        f"{row['wind_speed']:.1f} m/s"
                    )
                
                # Update temperature comparison chart
                if not all_current_df.empty:
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(
                        x=all_current_df['timestamp_readable'],
                        y=all_current_df['temperature'],
                        mode='lines+markers',
                        name='Actual Temperature',
                        line=dict(color='red', width=2)
                    ))
                    fig_temp.add_trace(go.Scatter(
                        x=all_current_df['timestamp_readable'],
                        y=all_current_df['predicted_temperature'],
                        mode='lines+markers',
                        name='Predicted Temperature',
                        line=dict(color='blue', width=2, dash='dash')
                    ))
                    fig_temp.update_layout(
                        title="Temperature: Actual vs Predicted",
                        xaxis_title="Time",
                        yaxis_title="Temperature (°C)",
                        height=400,
                        hovermode='x unified'
                    )
                    temp_chart.plotly_chart(fig_temp, use_container_width=True)
                    
                    # Prediction error chart
                    fig_error = go.Figure()
                    fig_error.add_trace(go.Bar(
                        x=all_current_df['timestamp_readable'],
                        y=all_current_df['temperature_error'],
                        name='Prediction Error',
                        marker_color='orange',
                        opacity=0.7
                    ))
                    fig_error.update_layout(
                        title="Temperature Prediction Error",
                        xaxis_title="Time",
                        yaxis_title="Error (°C)",
                        height=400,
                        yaxis_range=[0, all_current_df['temperature_error'].max() * 1.1]
                    )
                    error_chart.plotly_chart(fig_error, use_container_width=True)
                    
                    # Weather distribution
                    weather_counts = all_current_df['weather'].value_counts()
                    if not weather_counts.empty:
                        fig_pie = px.pie(
                            values=weather_counts.values,
                            names=weather_counts.index,
                            title="Weather Conditions Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        weather_dist_chart.plotly_chart(fig_pie, use_container_width=True)
                
                # Query aggregates if available
                if 'weather_aggregates' in tables:
                    agg_df = spark.sql("""
                        SELECT 
                            window.end as window_end,
                            avg_temperature,
                            avg_predicted_temperature,
                            avg_prediction_error,
                            readings_count
                        FROM weather_aggregates 
                        ORDER BY window_end DESC 
                        LIMIT 10
                    """).toPandas()
                    
                    if not agg_df.empty:
                        fig_agg = go.Figure()
                        fig_agg.add_trace(go.Scatter(
                            x=agg_df['window_end'],
                            y=agg_df['avg_temperature'],
                            mode='lines+markers',
                            name='Avg Temperature',
                            line=dict(color='red')
                        ))
                        fig_agg.add_trace(go.Scatter(
                            x=agg_df['window_end'],
                            y=agg_df['avg_predicted_temperature'],
                            mode='lines+markers',
                            name='Avg Predicted',
                            line=dict(color='blue', dash='dash')
                        ))
                        fig_agg.update_layout(
                            title="5-Minute Average Temperature",
                            xaxis_title="Time",
                            yaxis_title="Temperature (°C)",
                            height=400
                        )
                        hourly_temp_chart.plotly_chart(fig_agg, use_container_width=True)
                
                # Show raw data
                if show_raw_data and not all_current_df.empty:
                    display_df = all_current_df[['timestamp_readable', 'temperature', 
                                                'predicted_temperature', 'temperature_error',
                                                'humidity', 'weather', 'wind_speed']]
                    raw_data_table.dataframe(
                        display_df,
                        use_container_width=True,
                        height=300
                    )
                
                # Update status
                status_text.success(f"✅ Last update: {datetime.now().strftime('%H:%M:%S')} | "
                                   f"Records: {len(all_current_df)}")
            
            time.sleep(refresh_rate)
            
        except Exception as e:
            status_text.error(f"❌ Error: {e}")
            time.sleep(refresh_rate)

# Start dashboard update thread
if 'dashboard_running' not in st.session_state:
    st.session_state.dashboard_running = True
    thread = threading.Thread(target=update_dashboard, daemon=True)
    thread.start()

# Instructions
st.markdown("---")
st.markdown("""
### 📊 Dashboard Information
- **Data Source**: Real-time weather data from OpenWeather API
- **ML Model**: Random Forest Regressor (PySpark ML)
- **Features**: Humidity, Pressure, Wind Speed, Clouds, Time features, Weather condition
- **Update Frequency**: Every {} seconds
- **Streaming Engine**: Apache Spark Structured Streaming

*Waiting for data... The dashboard will update automatically when data arrives.*
""".format(refresh_rate))