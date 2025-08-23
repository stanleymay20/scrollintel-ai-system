# Multi-stage build for Intelligence Engine
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL org.opencontainers.image.title="Agent Steering Intelligence Engine"
LABEL org.opencontainers.image.description="Enterprise-grade AI intelligence and decision engine"
LABEL org.opencontainers.image.version=$VERSION
LABEL org.opencontainers.image.created=$BUILD_DATE
LABEL org.opencontainers.image.revision=$VCS_REF
LABEL org.opencontainers.image.vendor="ScrollIntel"
LABEL org.opencontainers.image.source="https://github.com/scrollintel/agent-steering-system"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-intelligence.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-intelligence.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r intelligence && useradd -r -g intelligence intelligence

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    dumb-init \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY scrollintel/ ./scrollintel/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Copy intelligence-specific files
COPY scrollintel/engines/intelligence/ ./scrollintel/engines/intelligence/
COPY scrollintel/models/ ./scrollintel/models/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /app/config /app/models && \
    chown -R intelligence:intelligence /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_MODULE=intelligence
ENV PORT=8081
ENV WORKERS=2
ENV LOG_LEVEL=INFO
ENV MODEL_CACHE_DIR=/app/models

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8081/health/live || exit 1

# Switch to non-root user
USER intelligence

# Expose port
EXPOSE 8081

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the intelligence engine
CMD ["python", "-m", "scrollintel.engines.intelligence.main"]