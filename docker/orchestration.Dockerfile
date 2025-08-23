# Multi-stage build for Orchestration Engine
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Add metadata
LABEL org.opencontainers.image.title="Agent Steering Orchestration Engine"
LABEL org.opencontainers.image.description="Enterprise-grade AI agent orchestration engine"
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-orchestration.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-orchestration.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r orchestration && useradd -r -g orchestration orchestration

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    dumb-init \
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

# Copy orchestration-specific files
COPY scrollintel/engines/orchestration/ ./scrollintel/engines/orchestration/
COPY scrollintel/core/agent_steering/ ./scrollintel/core/agent_steering/

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /app/config && \
    chown -R orchestration:orchestration /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_MODULE=orchestration
ENV PORT=8080
ENV WORKERS=4
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Switch to non-root user
USER orchestration

# Expose port
EXPOSE 8080 8090

# Use dumb-init to handle signals properly
ENTRYPOINT ["dumb-init", "--"]

# Start the orchestration engine
CMD ["python", "-m", "scrollintel.engines.orchestration.main"]