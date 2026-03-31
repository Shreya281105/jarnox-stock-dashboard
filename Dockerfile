# ─────────────────────────────────────────────────────────────
# JarNox Stock Intelligence Platform — Dockerfile
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="Internship Candidate"
LABEL description="JarNox Stock Intelligence API"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY backend/   ./backend/
COPY data/      ./data/
COPY frontend/  ./frontend/
COPY data_ingestion.py .
COPY ml_predictions.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run FastAPI via uvicorn
CMD uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 2
