FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    pandas==2.0.3 \
    plotly==5.17.0 \
    python-dotenv==1.0.0 \
    streamlit==1.28.0
COPY dashboard/app.py .
EXPOSE 8501
USER nobody
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
