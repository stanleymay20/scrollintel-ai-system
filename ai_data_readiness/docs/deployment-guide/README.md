# Deployment Guide

This guide provides comprehensive instructions for deploying the AI Data Readiness Platform in various environments.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Prerequisites](#prerequisites)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Deployments](#cloud-deployments)
7. [Production Configuration](#production-configuration)
8. [Security Hardening](#security-hardening)
9. [Monitoring and Logging](#monitoring-and-logging)
10. [Backup and Recovery](#backup-and-recovery)
11. [Scaling and Performance](#scaling-and-performance)
12. [Maintenance](#maintenance)

## Deployment Overview

### Architecture Components

The AI Data Readiness Platform consists of several components:

- **API Server**: FastAPI application serving REST and GraphQL APIs
- **Worker Processes**: Background task processing (Celery)
- **Database**: PostgreSQL for persistent data storage
- **Cache**: Redis for caching and session storage
- **Message Queue**: Redis/RabbitMQ for task queuing
- **File Storage**: Local filesystem or cloud storage (S3, GCS, Azure)
- **Monitoring**: Prometheus, Grafana for observability

### Deployment Options

1. **Development**: Local setup with minimal dependencies
2. **Docker**: Containerized deployment for consistency
3. **Kubernetes**: Orchestrated deployment for scalability
4. **Cloud**: Managed services for production workloads

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 100 GB SSD
- Network: 1 Gbps

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 32+ GB
- Storage: 500+ GB SSD with backup
- Network: 10 Gbps

### Software Dependencies

**Required:**
- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Git

**Optional:**
- Docker 24+
- Kubernetes 1.28+
- Nginx (reverse proxy)
- SSL certificates

### Network Requirements

**Inbound Ports:**
- 80/443: HTTP/HTTPS traffic
- 8000: API server (development)
- 5432: PostgreSQL (if external access needed)
- 6379: Redis (if external access needed)

**Outbound Access:**
- Package repositories (PyPI, apt, yum)
- External APIs (if integrations enabled)
- Monitoring services
- Backup destinations

## Local Development

### Quick Start

1. **Clone Repository**
   ```bash
   git clone https://github.com/your-org/ai-data-readiness-platform.git
   cd ai-data-readiness-platform
   ```

2. **Set Up Python Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -r ai_data_readiness_requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

4. **Set Up Database**
   ```bash
   # Install PostgreSQL locally or use Docker
   docker run -d --name postgres \
     -e POSTGRES_DB=ai_data_readiness \
     -e POSTGRES_USER=ai_user \
     -e POSTGRES_PASSWORD=your_password \
     -p 5432:5432 postgres:15
   
   # Run migrations
   python init_database.py
   python -m alembic upgrade head
   ```

5. **Start Services**
   ```bash
   # Start Redis
   redis-server
   
   # Start API server
   python -m uvicorn ai_data_readiness.api.app:app --reload --host 0.0.0.0 --port 8000
   
   # Start worker (in another terminal)
   celery -A ai_data_readiness.core.celery_app worker --loglevel=info
   ```

### Development Configuration

**Environment Variables (.env):**
```bash
# Database
DATABASE_URL=postgresql://ai_user:your_password@localhost:5432/ai_data_readiness

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key

# Application
DEBUG=true
LOG_LEVEL=DEBUG
ENVIRONMENT=development

# File Storage
UPLOAD_PATH=./uploads
MAX_FILE_SIZE=1073741824  # 1GB

# Features
ENABLE_BIAS_DETECTION=true
ENABLE_COMPLIANCE_VALIDATION=true
ENABLE_DRIFT_MONITORING=true
```

## Docker Deployment

### Single Container

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ai_data_readiness_requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r ai_data_readiness_requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "ai_data_readiness.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run:**
```bash
# Build image
docker build -t ai-data-readiness:latest .

# Run container
docker run -d \
  --name ai-data-readiness \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e REDIS_URL=redis://host:6379/0 \
  -v $(pwd)/uploads:/app/uploads \
  ai-data-readiness:latest
```

### Docker Compose

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://ai_user:ai_password@postgres:5432/ai_data_readiness
      - REDIS_URL=redis://redis:6379/0
      - SECRET_KEY=your-secret-key
    depends_on:
      - postgres
      - redis
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    build: .
    command: celery -A ai_data_readiness.core.celery_app worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://ai_user:ai_password@postgres:5432/ai_data_readiness
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    restart: unless-stopped

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=ai_data_readiness
      - POSTGRES_USER=ai_user
      - POSTGRES_PASSWORD=ai_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./ai_data_readiness/migrations/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ai_user -d ai_data_readiness"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

**Start Services:**
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker=3

# Stop services
docker-compose down
```

## Kubernetes Deployment

### Namespace and ConfigMap

**namespace.yaml:**
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-data-readiness
```

**configmap.yaml:**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-data-readiness-config
  namespace: ai-data-readiness
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  ENABLE_BIAS_DETECTION: "true"
  ENABLE_COMPLIANCE_VALIDATION: "true"
  MAX_FILE_SIZE: "2147483648"  # 2GB
```

### Secrets

**secrets.yaml:**
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-data-readiness-secrets
  namespace: ai-data-readiness
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  SECRET_KEY: <base64-encoded-secret-key>
  JWT_SECRET_KEY: <base64-encoded-jwt-secret>
```

### Database Deployment

**postgres.yaml:**
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: ai-data-readiness
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15
        env:
        - name: POSTGRES_DB
          value: ai_data_readiness
        - name: POSTGRES_USER
          value: ai_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: ai-data-readiness
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

### Application Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-data-readiness-app
  namespace: ai-data-readiness
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-data-readiness-app
  template:
    metadata:
      labels:
        app: ai-data-readiness-app
    spec:
      containers:
      - name: app
        image: ai-data-readiness:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ai-data-readiness-secrets
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: ai-data-readiness-secrets
              key: REDIS_URL
        envFrom:
        - configMapRef:
            name: ai-data-readiness-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: uploads
          mountPath: /app/uploads
      volumes:
      - name: uploads
        persistentVolumeClaim:
          claimName: uploads-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ai-data-readiness-app
  namespace: ai-data-readiness
spec:
  selector:
    app: ai-data-readiness-app
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### Ingress

**ingress.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-data-readiness-ingress
  namespace: ai-data-readiness
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "2g"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
spec:
  tls:
  - hosts:
    - ai-data-readiness.example.com
    secretName: ai-data-readiness-tls
  rules:
  - host: ai-data-readiness.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ai-data-readiness-app
            port:
              number: 80
```

### Deploy to Kubernetes

```bash
# Apply configurations
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f postgres.yaml
kubectl apply -f deployment.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get pods -n ai-data-readiness
kubectl get services -n ai-data-readiness
kubectl get ingress -n ai-data-readiness

# View logs
kubectl logs -f deployment/ai-data-readiness-app -n ai-data-readiness

# Scale deployment
kubectl scale deployment ai-data-readiness-app --replicas=5 -n ai-data-readiness
```

## Cloud Deployments

### AWS Deployment

**Using ECS with Fargate:**

1. **Create Task Definition**
   ```json
   {
     "family": "ai-data-readiness",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "2048",
     "memory": "4096",
     "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "ai-data-readiness",
         "image": "your-account.dkr.ecr.region.amazonaws.com/ai-data-readiness:latest",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "ENVIRONMENT",
             "value": "production"
           }
         ],
         "secrets": [
           {
             "name": "DATABASE_URL",
             "valueFrom": "arn:aws:secretsmanager:region:account:secret:ai-data-readiness/database-url"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/ai-data-readiness",
             "awslogs-region": "us-west-2",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

2. **Create ECS Service**
   ```bash
   aws ecs create-service \
     --cluster ai-data-readiness-cluster \
     --service-name ai-data-readiness-service \
     --task-definition ai-data-readiness:1 \
     --desired-count 3 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-abcdef],assignPublicIp=ENABLED}"
   ```

**Using EKS:**

1. **Create EKS Cluster**
   ```bash
   eksctl create cluster \
     --name ai-data-readiness \
     --region us-west-2 \
     --nodegroup-name workers \
     --node-type m5.large \
     --nodes 3 \
     --nodes-min 1 \
     --nodes-max 10
   ```

2. **Deploy Application**
   ```bash
   kubectl apply -f k8s/
   ```

### Google Cloud Platform

**Using Cloud Run:**

1. **Build and Push Image**
   ```bash
   # Build image
   docker build -t gcr.io/your-project/ai-data-readiness:latest .
   
   # Push to Container Registry
   docker push gcr.io/your-project/ai-data-readiness:latest
   ```

2. **Deploy to Cloud Run**
   ```bash
   gcloud run deploy ai-data-readiness \
     --image gcr.io/your-project/ai-data-readiness:latest \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --memory 4Gi \
     --cpu 2 \
     --max-instances 10 \
     --set-env-vars ENVIRONMENT=production \
     --set-secrets DATABASE_URL=ai-data-readiness-db-url:latest
   ```

**Using GKE:**

1. **Create GKE Cluster**
   ```bash
   gcloud container clusters create ai-data-readiness \
     --zone us-central1-a \
     --num-nodes 3 \
     --machine-type n1-standard-4 \
     --enable-autoscaling \
     --min-nodes 1 \
     --max-nodes 10
   ```

2. **Deploy Application**
   ```bash
   kubectl apply -f k8s/
   ```

### Azure Deployment

**Using Container Instances:**

```bash
az container create \
  --resource-group ai-data-readiness-rg \
  --name ai-data-readiness \
  --image your-registry.azurecr.io/ai-data-readiness:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables ENVIRONMENT=production \
  --secure-environment-variables DATABASE_URL=$DATABASE_URL \
  --dns-name-label ai-data-readiness
```

**Using AKS:**

1. **Create AKS Cluster**
   ```bash
   az aks create \
     --resource-group ai-data-readiness-rg \
     --name ai-data-readiness-aks \
     --node-count 3 \
     --node-vm-size Standard_D4s_v3 \
     --enable-cluster-autoscaler \
     --min-count 1 \
     --max-count 10
   ```

2. **Deploy Application**
   ```bash
   kubectl apply -f k8s/
   ```

## Production Configuration

### Environment Variables

**Production .env:**
```bash
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-production-secret-key
JWT_SECRET_KEY=your-production-jwt-secret
JWT_EXPIRATION_HOURS=24
CORS_ORIGINS=https://your-domain.com

# Database
DATABASE_URL=postgresql://user:pass@prod-db:5432/ai_data_readiness
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://prod-redis:6379/0
REDIS_POOL_SIZE=10

# File Storage
STORAGE_BACKEND=s3  # or gcs, azure
AWS_S3_BUCKET=ai-data-readiness-prod
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Performance
MAX_WORKERS=4
WORKER_TIMEOUT=300
MAX_FILE_SIZE=5368709120  # 5GB
ENABLE_COMPRESSION=true

# Features
ENABLE_BIAS_DETECTION=true
ENABLE_COMPLIANCE_VALIDATION=true
ENABLE_DRIFT_MONITORING=true
ENABLE_ADVANCED_ANALYTICS=true

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
SENTRY_DSN=your-sentry-dsn
```

### Database Configuration

**PostgreSQL Production Settings:**
```sql
-- postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 4MB
min_wal_size = 1GB
max_wal_size = 4GB
max_worker_processes = 8
max_parallel_workers_per_gather = 2
max_parallel_workers = 8
max_parallel_maintenance_workers = 2
```

### Reverse Proxy Configuration

**Nginx Configuration:**
```nginx
upstream ai_data_readiness {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    server_name ai-data-readiness.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ai-data-readiness.example.com;

    ssl_certificate /etc/ssl/certs/ai-data-readiness.crt;
    ssl_certificate_key /etc/ssl/private/ai-data-readiness.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    client_max_body_size 5G;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;

    location / {
        proxy_pass http://ai_data_readiness;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        proxy_pass http://ai_data_readiness;
    }

    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://ai_data_readiness;
    }
}
```

## Security Hardening

### Application Security

1. **Environment Variables**
   ```bash
   # Use strong secrets
   SECRET_KEY=$(openssl rand -hex 32)
   JWT_SECRET_KEY=$(openssl rand -hex 32)
   
   # Secure database credentials
   DATABASE_PASSWORD=$(openssl rand -base64 32)
   ```

2. **HTTPS Configuration**
   ```python
   # Force HTTPS in production
   FORCE_HTTPS = True
   SECURE_SSL_REDIRECT = True
   SECURE_HSTS_SECONDS = 31536000
   SECURE_HSTS_INCLUDE_SUBDOMAINS = True
   ```

3. **CORS Configuration**
   ```python
   # Restrict CORS origins
   CORS_ORIGINS = [
       "https://your-frontend.com",
       "https://admin.your-domain.com"
   ]
   ```

### Infrastructure Security

1. **Network Security**
   - Use VPC/VNet with private subnets
   - Configure security groups/NSGs
   - Enable WAF for public endpoints
   - Use VPN for administrative access

2. **Database Security**
   ```sql
   -- Create read-only user for monitoring
   CREATE USER monitoring WITH PASSWORD 'secure_password';
   GRANT CONNECT ON DATABASE ai_data_readiness TO monitoring;
   GRANT USAGE ON SCHEMA public TO monitoring;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO monitoring;
   
   -- Revoke unnecessary permissions
   REVOKE ALL ON SCHEMA public FROM public;
   ```

3. **Container Security**
   ```dockerfile
   # Use non-root user
   RUN useradd -m -u 1000 appuser
   USER appuser
   
   # Use minimal base image
   FROM python:3.11-slim
   
   # Remove unnecessary packages
   RUN apt-get autoremove -y && apt-get clean
   ```

### Secrets Management

**Using Kubernetes Secrets:**
```bash
# Create secret from file
kubectl create secret generic ai-data-readiness-secrets \
  --from-env-file=.env.production \
  --namespace=ai-data-readiness

# Create TLS secret
kubectl create secret tls ai-data-readiness-tls \
  --cert=tls.crt \
  --key=tls.key \
  --namespace=ai-data-readiness
```

**Using AWS Secrets Manager:**
```python
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name, region_name="us-west-2"):
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response['SecretString']
    except ClientError as e:
        raise e
```

## Monitoring and Logging

### Application Monitoring

**Prometheus Configuration:**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-data-readiness'
    static_configs:
      - targets: ['app:9090']
    metrics_path: /metrics
    scrape_interval: 30s
```

**Grafana Dashboard:**
```json
{
  "dashboard": {
    "title": "AI Data Readiness Platform",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Centralized Logging

**ELK Stack Configuration:**
```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.5.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.5.0
    volumes:
      - ./logstash/pipeline:/usr/share/logstash/pipeline
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.5.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

### Health Checks

**Application Health Check:**
```python
from fastapi import APIRouter, HTTPException
from sqlalchemy import text
import redis

router = APIRouter()

@router.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "dependencies": {}
    }
    
    # Check database
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_status["dependencies"]["database"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Redis
    try:
        r = redis.Redis.from_url(REDIS_URL)
        r.ping()
        health_status["dependencies"]["redis"] = "healthy"
    except Exception as e:
        health_status["dependencies"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status
```

## Backup and Recovery

### Database Backup

**Automated Backup Script:**
```bash
#!/bin/bash
# backup-database.sh

BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="ai_data_readiness_backup_${TIMESTAMP}.sql"

# Create backup
pg_dump $DATABASE_URL > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to S3
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" s3://your-backup-bucket/database/

# Clean up old backups (keep last 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

**Cron Job:**
```bash
# Add to crontab
0 2 * * * /path/to/backup-database.sh
```

### File Storage Backup

**S3 Sync Script:**
```bash
#!/bin/bash
# backup-files.sh

SOURCE_DIR="/app/uploads"
S3_BUCKET="s3://your-backup-bucket/files"

# Sync files to S3
aws s3 sync $SOURCE_DIR $S3_BUCKET --delete

echo "File backup completed"
```

### Disaster Recovery

**Recovery Procedure:**

1. **Database Recovery**
   ```bash
   # Download backup
   aws s3 cp s3://your-backup-bucket/database/latest.sql.gz ./
   
   # Restore database
   gunzip latest.sql.gz
   psql $DATABASE_URL < latest.sql
   ```

2. **File Recovery**
   ```bash
   # Restore files from S3
   aws s3 sync s3://your-backup-bucket/files/ /app/uploads/
   ```

3. **Application Recovery**
   ```bash
   # Redeploy application
   kubectl rollout restart deployment/ai-data-readiness-app
   
   # Verify health
   kubectl get pods -l app=ai-data-readiness-app
   ```

## Scaling and Performance

### Horizontal Scaling

**Kubernetes HPA:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-data-readiness-hpa
  namespace: ai-data-readiness
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-data-readiness-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Database Scaling

**Read Replicas:**
```python
# Database configuration with read replicas
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'ai_data_readiness',
        'USER': 'ai_user',
        'PASSWORD': 'password',
        'HOST': 'primary-db.example.com',
        'PORT': '5432',
    },
    'read_replica': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'ai_data_readiness',
        'USER': 'ai_user',
        'PASSWORD': 'password',
        'HOST': 'replica-db.example.com',
        'PORT': '5432',
    }
}

DATABASE_ROUTERS = ['ai_data_readiness.routers.DatabaseRouter']
```

### Caching Strategy

**Redis Caching:**
```python
import redis
from functools import wraps

redis_client = redis.Redis.from_url(REDIS_URL)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(
                cache_key, 
                expiration, 
                json.dumps(result, default=str)
            )
            
            return result
        return wrapper
    return decorator
```

## Maintenance

### Regular Maintenance Tasks

**Weekly Tasks:**
```bash
#!/bin/bash
# weekly-maintenance.sh

# Update system packages
apt update && apt upgrade -y

# Clean up Docker images
docker system prune -f

# Vacuum database
psql $DATABASE_URL -c "VACUUM ANALYZE;"

# Clear old logs
find /var/log -name "*.log" -mtime +30 -delete

# Check disk space
df -h
```

**Monthly Tasks:**
```bash
#!/bin/bash
# monthly-maintenance.sh

# Update SSL certificates
certbot renew --quiet

# Backup configuration
tar -czf /backups/config_$(date +%Y%m%d).tar.gz /etc/ai-data-readiness/

# Performance analysis
pg_stat_statements_reset();

# Security updates
apt list --upgradable | grep -i security
```

### Monitoring Maintenance

**Automated Alerts:**
```yaml
# alertmanager.yml
groups:
- name: ai-data-readiness
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: High error rate detected
      
  - alert: DatabaseConnectionHigh
    expr: pg_stat_activity_count > 80
    for: 2m
    annotations:
      summary: High database connection count
      
  - alert: DiskSpaceLow
    expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
    for: 1m
    annotations:
      summary: Disk space is running low
```

### Update Procedures

**Rolling Updates:**
```bash
# Build new image
docker build -t ai-data-readiness:v1.1.0 .

# Push to registry
docker push your-registry/ai-data-readiness:v1.1.0

# Update Kubernetes deployment
kubectl set image deployment/ai-data-readiness-app \
  app=your-registry/ai-data-readiness:v1.1.0 \
  --namespace=ai-data-readiness

# Monitor rollout
kubectl rollout status deployment/ai-data-readiness-app \
  --namespace=ai-data-readiness

# Rollback if needed
kubectl rollout undo deployment/ai-data-readiness-app \
  --namespace=ai-data-readiness
```

This deployment guide provides comprehensive instructions for deploying the AI Data Readiness Platform across various environments. Choose the deployment method that best fits your infrastructure requirements and operational capabilities.