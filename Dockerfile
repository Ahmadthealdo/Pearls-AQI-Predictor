# Production Dockerfile for Pearls AQI Predictor Dashboard
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source and pre-computed artifacts
COPY .streamlit/ .streamlit/
COPY data/ data/
COPY models/ models/
COPY 1_feature_pipeline.py 2_training_pipeline.py 3_app.py api.py audit_submission.py generate_final_report.py verify_system.py main.py ./

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck definition
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint to run the Streamlit dashboard
ENTRYPOINT ["streamlit", "run", "3_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
