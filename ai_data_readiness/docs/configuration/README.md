# Configuration Guide

This guide provides comprehensive information about configuring the AI Data Readiness Platform.

## Table of Contents

1. [Environment Variables](#environment-variables)
2. [Database Configuration](#database-configuration)
3. [Security Settings](#security-settings)
4. [Performance Tuning](#performance-tuning)
5. [Feature Flags](#feature-flags)
6. [Integration Settings](#integration-settings)
7. [Monitoring Configuration](#monitoring-configuration)
8. [Logging Configuration](#logging-configuration)

## Environment Variables

### Core Application Settings

```bash
# Application Environment
ENVIRONMENT=production|development|testing
DEBUG=false
LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
APP_NAME=AI Data Readiness Platform
VERSION=1.0.0

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4
WORKER_TIMEOUT=300
RELOAD=false

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key
JWT_EXPIRATION_HOURS=24
JWT_REFRESH_EXPIRATION_DAYS=30
CORS_ORIGINS=https://your-domain.com,https://admin.your-domain.com
```

### Database Configuration

```bash
# Primary Database
DATABASE_URL=postgresql://user:password@host:port/database
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Read Replica (Optional)
READ_DATABASE_URL=postgresql://user:password@replica-host:port/database
ENABLE_READ_REPLICA=true

# Connection Settings
DATABASE_ECHO=false
DATABASE_POOL_PRE_PING=true
DATABASE_CONNECT_TIMEOUT=10
```

### Cache and Message Queue

```bash
# Redis Configuration
REDIS_URL=redis://host:port/db
REDIS_POOL_SIZE=10
REDIS_SOCKET_TIMEOUT=5
REDIS_SOCKET_CONNECT_TIMEOUT=5
REDIS_RETRY_ON_TIMEOUT=true

# Celery Configuration
CELERY_BROKER_URL=redis://host:port/1
CELERY_RESULT_BACKEND=redis://host:port/2
CELERY_TASK_SERIALIZER=json
CELERY_RESULT_SERIALIZER=json
CELERY_ACCEPT_CONTENT=json
CELERY_TIMEZONE=UTC
```

### File Storage

```bash
# Local Storage
STORAGE_BACKEND=local|s3|gcs|azure
UPLOAD_PATH=/app/uploads
MAX_FILE_SIZE=5368709120  # 5GB
ALLOWED_FILE_TYPES=csv,json,parquet,avro,xlsx

# AWS S3
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_S3_BUCKET=your-bucket-name
AWS_S3_REGION=us-west-2
AWS_S3_ENDPOINT_URL=https://s3.amazonaws.com

# Google Cloud Storage
GCS_BUCKET=your-bucket-name
GCS_CREDENTIALS_PATH=/path/to/credentials.json

# Azure Blob Storage
AZURE_STORAGE_ACCOUNT=your-account
AZURE_STORAGE_KEY=your-key
AZURE_CONTAINER=your-container
```

## Database Configuration

### PostgreSQL Settings

**postgresql.conf:**
```ini
# Memory Settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 4MB

# Checkpoint Settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
min_wal_size = 1GB
max_wal_size = 4GB

# Query Planner
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# Parallel Processing
max_worker_processes = 8
max_parallel_workers_per_gather = 2
max_parallel_workers = 8
max_parallel_maintenance_workers = 2

# Logging
log_statement = 'mod'
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
```

### Connection Pooling

**PgBouncer Configuration:**
```ini
[databases]
ai_data_readiness = host=localhost port=5432 dbname=ai_data_readiness

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
reserve_pool_size = 5
server_lifetime = 3600
server_idle_timeout = 600
```

## Security Settings

### Authentication Configuration

```bash
# JWT Settings
JWT_ALGORITHM=HS256
JWT_AUDIENCE=ai-data-readiness
JWT_ISSUER=your-organization

# Session Settings
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=Strict
SESSION_TIMEOUT=3600

# Password Policy
PASSWORD_MIN_LENGTH=12
PASSWORD_REQUIRE_UPPERCASE=true
PASSWORD_REQUIRE_LOWERCASE=true
PASSWORD_REQUIRE_NUMBERS=true
PASSWORD_REQUIRE_SYMBOLS=true
PASSWORD_HISTORY_COUNT=5
```

### SSL/TLS Configuration

```bash
# HTTPS Settings
FORCE_HTTPS=true
SECURE_SSL_REDIRECT=true
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=true
SECURE_HSTS_PRELOAD=true

# SSL Certificate Paths
SSL_CERT_PATH=/etc/ssl/certs/ai-data-readiness.crt
SSL_KEY_PATH=/etc/ssl/private/ai-data-readiness.key
SSL_CA_PATH=/etc/ssl/certs/ca-bundle.crt
```

### Access Control

```bash
# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST=20
RATE_LIMIT_STORAGE=redis

# IP Whitelisting
IP_WHITELIST_ENABLED=false
IP_WHITELIST=10.0.0.0/8,172.16.0.0/12,192.168.0.0/16

# API Key Settings
API_KEY_HEADER=X-API-Key
API_KEY_EXPIRATION_DAYS=365
```

## Performance Tuning

### Application Performance

```bash
# Worker Configuration
WORKER_PROCESSES=4
WORKER_THREADS=2
WORKER_CONNECTIONS=1000
WORKER_MAX_REQUESTS=1000
WORKER_MAX_REQUESTS_JITTER=100

# Async Settings
ASYNC_POOL_SIZE=100
ASYNC_TIMEOUT=30

# Caching
CACHE_TTL=3600
CACHE_MAX_SIZE=1000
ENABLE_QUERY_CACHE=true
ENABLE_RESULT_CACHE=true
```

### Data Processing

```bash
# Processing Limits
MAX_CONCURRENT_JOBS=10
MAX_PROCESSING_TIME=3600
CHUNK_SIZE=10000
BATCH_SIZE=1000

# Memory Management
MAX_MEMORY_USAGE=8589934592  # 8GB
MEMORY_LIMIT_PER_JOB=2147483648  # 2GB
ENABLE_MEMORY_MONITORING=true

# Parallel Processing
ENABLE_PARALLEL_PROCESSING=true
MAX_PARALLEL_WORKERS=4
PARALLEL_CHUNK_SIZE=1000
```

## Feature Flags

### Core Features

```bash
# Quality Assessment
ENABLE_QUALITY_ASSESSMENT=true
ENABLE_ADVANCED_QUALITY_METRICS=true
QUALITY_ASSESSMENT_TIMEOUT=1800

# Bias Detection
ENABLE_BIAS_DETECTION=true
ENABLE_INTERSECTIONAL_BIAS_ANALYSIS=true
BIAS_DETECTION_TIMEOUT=900

# Feature Engineering
ENABLE_FEATURE_ENGINEERING=true
ENABLE_AUTOMATED_FEATURE_SELECTION=true
FEATURE_ENGINEERING_TIMEOUT=1200

# Compliance Validation
ENABLE_COMPLIANCE_VALIDATION=true
ENABLE_GDPR_VALIDATION=true
ENABLE_CCPA_VALIDATION=true
ENABLE_HIPAA_VALIDATION=false
```

### Advanced Features

```bash
# Drift Monitoring
ENABLE_DRIFT_MONITORING=true
DRIFT_MONITORING_INTERVAL=3600
DRIFT_THRESHOLD=0.1

# Lineage Tracking
ENABLE_LINEAGE_TRACKING=true
LINEAGE_RETENTION_DAYS=365

# Advanced Analytics
ENABLE_ADVANCED_ANALYTICS=true
ENABLE_PREDICTIVE_ANALYTICS=true
ENABLE_ANOMALY_DETECTION=true
```

## Integration Settings

### External APIs

```bash
# ML Platform Integrations
ENABLE_MLFLOW_INTEGRATION=false
MLFLOW_TRACKING_URI=http://mlflow:5000

ENABLE_KUBEFLOW_INTEGRATION=false
KUBEFLOW_ENDPOINT=http://kubeflow:8080

# BI Tool Integrations
ENABLE_TABLEAU_INTEGRATION=false
TABLEAU_SERVER_URL=https://tableau.example.com
TABLEAU_USERNAME=service-account
TABLEAU_PASSWORD=password

ENABLE_POWERBI_INTEGRATION=false
POWERBI_CLIENT_ID=your-client-id
POWERBI_CLIENT_SECRET=your-client-secret
```

### Notification Services

```bash
# Email Configuration
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USERNAME=notifications@example.com
SMTP_PASSWORD=password
SMTP_USE_TLS=true
EMAIL_FROM=AI Data Readiness <notifications@example.com>

# Slack Integration
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_CHANNEL=#data-alerts
SLACK_USERNAME=AI Data Readiness Bot

# Microsoft Teams
TEAMS_WEBHOOK_URL=https://outlook.office.com/webhook/...
```

## Monitoring Configuration

### Metrics Collection

```bash
# Prometheus Settings
ENABLE_METRICS=true
METRICS_PORT=9090
METRICS_PATH=/metrics
METRICS_NAMESPACE=ai_data_readiness

# Custom Metrics
ENABLE_CUSTOM_METRICS=true
METRICS_COLLECTION_INTERVAL=30
ENABLE_DETAILED_METRICS=false
```

### Health Checks

```bash
# Health Check Configuration
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_PATH=/health
HEALTH_CHECK_TIMEOUT=10
HEALTH_CHECK_INTERVAL=30

# Dependency Checks
CHECK_DATABASE_HEALTH=true
CHECK_REDIS_HEALTH=true
CHECK_EXTERNAL_APIS=false
```

### Alerting

```bash
# Alert Configuration
ENABLE_ALERTING=true
ALERT_COOLDOWN_PERIOD=300
ALERT_ESCALATION_TIMEOUT=1800

# Alert Thresholds
ERROR_RATE_THRESHOLD=0.05
RESPONSE_TIME_THRESHOLD=5000
MEMORY_USAGE_THRESHOLD=0.8
DISK_USAGE_THRESHOLD=0.9
```

## Logging Configuration

### Log Levels and Formats

```bash
# Logging Settings
LOG_LEVEL=INFO
LOG_FORMAT=json|text
LOG_TIMESTAMP_FORMAT=%Y-%m-%d %H:%M:%S
ENABLE_STRUCTURED_LOGGING=true

# Log Destinations
LOG_TO_CONSOLE=true
LOG_TO_FILE=true
LOG_FILE_PATH=/var/log/ai-data-readiness/app.log
LOG_FILE_MAX_SIZE=100MB
LOG_FILE_BACKUP_COUNT=5

# Syslog Configuration
ENABLE_SYSLOG=false
SYSLOG_HOST=localhost
SYSLOG_PORT=514
SYSLOG_FACILITY=local0
```

### Log Categories

```bash
# Application Logs
ENABLE_ACCESS_LOGS=true
ENABLE_ERROR_LOGS=true
ENABLE_AUDIT_LOGS=true
ENABLE_PERFORMANCE_LOGS=true

# Security Logs
ENABLE_SECURITY_LOGS=true
LOG_FAILED_LOGINS=true
LOG_PERMISSION_DENIALS=true
LOG_SUSPICIOUS_ACTIVITY=true

# Data Processing Logs
ENABLE_PROCESSING_LOGS=true
LOG_DATA_TRANSFORMATIONS=true
LOG_QUALITY_ASSESSMENTS=true
LOG_BIAS_DETECTIONS=true
```

### External Log Aggregation

```bash
# ELK Stack
ENABLE_ELASTICSEARCH_LOGGING=false
ELASTICSEARCH_HOST=elasticsearch:9200
ELASTICSEARCH_INDEX=ai-data-readiness

# Splunk
ENABLE_SPLUNK_LOGGING=false
SPLUNK_HOST=splunk.example.com
SPLUNK_PORT=8088
SPLUNK_TOKEN=your-token

# Fluentd
ENABLE_FLUENTD_LOGGING=false
FLUENTD_HOST=fluentd:24224
FLUENTD_TAG=ai.data.readiness
```

## Configuration Validation

### Environment Validation Script

```python
#!/usr/bin/env python3
"""Configuration validation script."""

import os
import sys
from urllib.parse import urlparse

def validate_database_url():
    """Validate database URL format."""
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        return False, "DATABASE_URL not set"
    
    try:
        parsed = urlparse(db_url)
        if parsed.scheme != 'postgresql':
            return False, "Database URL must use postgresql scheme"
        if not all([parsed.hostname, parsed.username, parsed.password]):
            return False, "Database URL missing required components"
        return True, "Database URL valid"
    except Exception as e:
        return False, f"Invalid database URL: {e}"

def validate_redis_url():
    """Validate Redis URL format."""
    redis_url = os.getenv('REDIS_URL')
    if not redis_url:
        return False, "REDIS_URL not set"
    
    try:
        parsed = urlparse(redis_url)
        if parsed.scheme != 'redis':
            return False, "Redis URL must use redis scheme"
        return True, "Redis URL valid"
    except Exception as e:
        return False, f"Invalid Redis URL: {e}"

def validate_secret_keys():
    """Validate secret keys are set and secure."""
    secret_key = os.getenv('SECRET_KEY')
    jwt_secret = os.getenv('JWT_SECRET_KEY')
    
    if not secret_key:
        return False, "SECRET_KEY not set"
    if len(secret_key) < 32:
        return False, "SECRET_KEY too short (minimum 32 characters)"
    
    if not jwt_secret:
        return False, "JWT_SECRET_KEY not set"
    if len(jwt_secret) < 32:
        return False, "JWT_SECRET_KEY too short (minimum 32 characters)"
    
    return True, "Secret keys valid"

def main():
    """Run all validation checks."""
    checks = [
        validate_database_url,
        validate_redis_url,
        validate_secret_keys,
    ]
    
    all_passed = True
    for check in checks:
        passed, message = check()
        status = "✓" if passed else "✗"
        print(f"{status} {message}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✓ All configuration checks passed")
        sys.exit(0)
    else:
        print("\n✗ Configuration validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Usage

```bash
# Validate configuration
python validate_config.py

# Load configuration from file
source .env.production

# Check specific settings
echo "Database: $DATABASE_URL"
echo "Redis: $REDIS_URL"
echo "Environment: $ENVIRONMENT"
```

This configuration guide provides comprehensive settings for all aspects of the AI Data Readiness Platform. Adjust values based on your specific deployment requirements and environment constraints.