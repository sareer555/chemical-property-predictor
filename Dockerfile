# =============================================================================
# Chemical Property Prediction - Production Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
# Suitable for: MSc/PhD research, cheminformatics, AI for chemistry
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Base Dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim as base

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for RDKit and scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libboost-all-dev \
    libeigen3-dev \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Stage 2: Python Dependencies
# ---------------------------------------------------------------------------
FROM base as python-deps

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3: Application
# ---------------------------------------------------------------------------
FROM python-deps as app

WORKDIR /app

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models/saved logs

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: run both API and dashboard using a startup script
CMD ["bash", "scripts/start.sh"]
