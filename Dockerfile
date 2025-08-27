# ================================
# ScrollIntel Backend Dockerfile
# Multi-stage build for production optimization
# ================================

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r scrollintel && useradd -r -g scrollintel scrollintel

# Stage 2: Dependencies installation
FROM base as dependencies

# Set work directory
WORKDIR /app

# Copy requirements
COPY requirements_docker.txt .

# Install Python dependencies with TensorFlow/Keras compatibility fix
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir tf-keras && \
    pip install --no-cache-dir -r requirements_docker.txt

# Stage 3: Development environment
FROM dependencies as development

# Copy source code
COPY . .

# Change ownership to app user
RUN chown -R scrollintel:scrollintel /app

# Switch to app user
USER scrollintel

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Development command
CMD ["uvicorn", "scrollintel.api.gateway:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 4: Production environment
FROM dependencies as production

# Copy only necessary files
COPY scrollintel/ ./scrollintel/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY init_database.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/uploads

# Change ownership to app user
RUN chown -R scrollintel:scrollintel /app

# Switch to app user
USER scrollintel

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command
CMD ["uvicorn", "scrollintel.api.gateway:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]