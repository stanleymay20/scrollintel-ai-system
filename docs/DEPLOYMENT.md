# ScrollIntel™ Deployment Guide

## Overview

This guide covers the deployment of ScrollIntel™ v4.0+ (ScrollSanctified HyperSovereign Edition™) across different environments including development, staging, and production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Configuration](#environment-configuration)
3. [Local Development](#local-development)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Monitoring and Health Checks](#monitoring-and-health-checks)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB minimum, SSD recommended
- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2

### Required Software

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Node.js**: 18+
- **Python**: 3.11+
- **PostgreSQL**: 15+
- **Redis**: 7+

### Optional Tools

- **Kubernetes**: 1.25+ (for production scaling)
- **NGINX**: 1.20+ (for reverse proxy)
- **Prometheus**: 2.40+ (for monitoring)
- **Grafana**: 9.0+ (for dashboards)

## Environment Configuration

### Environment Files

ScrollIntel uses different environment files for different deployment stages:

- `.env` - Development environment
- `.env.test` - Testing environment
- `.env.production` - Production environment

### Required Environment Variables

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Security
JWT_SECRET_KEY=your_jwt_secret_key

# AI Services
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
PINECONE_API_KEY=your_pinecone_api_key

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

### Setting Up Environment

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Fill in your configuration values in `.env`

3. Validate configuration:
   ```bash
   python -c "from scrollintel.core.config import get_settings; print('✅ Configuration valid')"
   ```

## Local Development

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/scrollintel.git
   cd scrollintel
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Run database migrations**:
   ```bash
   python scripts/migrate-database.py migrate
   ```

5. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Development Workflow

1. **Start development servers**:
   ```bash
   # Backend
   uvicorn scrollintel.api.gateway:app --reload --host 0.0.0.0 --port 8000
   
   # Frontend
   cd frontend && npm run dev
   ```

2. **Run tests**:
   ```bash
   # Backend tests
   pytest tests/ -v
   
   # Frontend tests
   cd frontend && npm test
   ```

3. **Database operations**:
   ```bash
   # Create migration
   python scripts/migrate-database.py create "description"
   
   # Apply migrations
   python scripts/migrate-database.py migrate
   
   # Seed database
   python scripts/migrate-database.py seed
   ```

## Docker Deployment

### Development with Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production Docker Build

```bash
# Build production images
docker build -t scrollintel-backend:latest --target production .
docker build -t scrollintel-frontend:latest ./frontend

# Run production containers
docker-compose -f docker-compose.prod.yml up -d
```

### Docker Health Checks

All containers include health checks:

```bash
# Check container health
docker ps
docker inspect <container_id> | grep Health

# Manual health check
curl http://localhost:8000/health
```

## Production Deployment

### Automated Production Deployment

Use the provided deployment script:

```bash
# Set environment variables
export DATABASE_URL="postgresql://user:pass@host:5432/db"
export REDIS_URL="redis://host:6379"

# Run deployment
./scripts/production-deploy.sh
```

### Manual Production Steps

1. **Prepare environment**:
   ```bash
   # Set production environment
   export ENVIRONMENT=production
   
   # Load production config
   cp .env.production .env
   ```

2. **Database setup**:
   ```bash
   # Run migrations
   python scripts/migrate-database.py --env production migrate
   
   # Seed initial data
   python scripts/migrate-database.py --env production seed
   ```

3. **Build and deploy**:
   ```bash
   # Build production images
   docker build -t scrollintel-backend:prod --target production .
   
   # Deploy with docker-compose
   docker-compose -f docker-compose.prod.yml up -d
   ```

4. **Verify deployment**:
   ```bash
   # Health checks
   curl http://your-domain.com/health
   curl http://your-domain.com/health/detailed
   
   # Load test
   ab -n 100 -c 10 http://your-domain.com/health
   ```

### Production Checklist

- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] SSL certificates installed
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Log aggregation setup
- [ ] Security headers configured
- [ ] Rate limiting enabled
- [ ] Health checks passing
- [ ] Performance tests completed

## Cloud Deployment

### Vercel (Frontend)

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Deploy frontend**:
   ```bash
   cd frontend
   vercel --prod
   ```

3. **Configure environment variables** in Vercel dashboard

### Render (Backend)

1. **Create render.yaml** (already provided)

2. **Deploy to Render**:
   ```bash
   # Using Render CLI
   render deploy
   
   # Or connect GitHub repository in Render dashboard
   ```

3. **Configure environment variables** in Render dashboard

### AWS/GCP/Azure

For cloud deployment, use the provided Kubernetes manifests:

```bash
# Apply Kubernetes configurations
kubectl apply -f k8s/

# Check deployment status
kubectl get pods
kubectl get services
```

## Monitoring and Health Checks

### Health Check Endpoints

- `/health` - Basic health check
- `/health/detailed` - Comprehensive system health
- `/health/agents` - Agent-specific health
- `/health/readiness` - Kubernetes readiness probe
- `/health/liveness` - Kubernetes liveness probe
- `/health/metrics` - System metrics
- `/health/dependencies` - External service health

### Monitoring Setup

1. **Prometheus configuration**:
   ```bash
   # Start monitoring stack
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

2. **Access monitoring dashboards**:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3001

### Key Metrics to Monitor

- **Application Metrics**:
  - Request rate and latency
  - Error rates
  - Agent response times
  - Database query performance

- **System Metrics**:
  - CPU and memory usage
  - Disk I/O and space
  - Network throughput
  - Container health

- **Business Metrics**:
  - Active users
  - API usage
  - Model training jobs
  - Data processing volume

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting guide.

### Quick Fixes

1. **Service not starting**:
   ```bash
   # Check logs
   docker-compose logs service_name
   
   # Restart service
   docker-compose restart service_name
   ```

2. **Database connection issues**:
   ```bash
   # Test database connection
   python -c "from scrollintel.core.config import get_settings; import psycopg2; psycopg2.connect(get_settings().database_url)"
   ```

3. **Health check failures**:
   ```bash
   # Check detailed health
   curl http://localhost:8000/health/detailed
   
   # Check dependencies
   curl http://localhost:8000/health/dependencies
   ```

## Security Considerations

### Production Security

- Use strong passwords and API keys
- Enable SSL/TLS encryption
- Configure firewall rules
- Regular security updates
- Monitor for suspicious activity
- Implement rate limiting
- Use secure headers

### Environment Security

- Never commit secrets to version control
- Use environment variables for sensitive data
- Rotate API keys regularly
- Implement proper access controls
- Enable audit logging

## Backup and Recovery

### Database Backups

```bash
# Create backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore backup
psql $DATABASE_URL < backup_file.sql
```

### File Backups

```bash
# Backup uploaded files
tar -czf uploads_backup_$(date +%Y%m%d_%H%M%S).tar.gz uploads/

# Backup models
tar -czf models_backup_$(date +%Y%m%d_%H%M%S).tar.gz models/
```

## Performance Optimization

### Production Optimizations

- Enable Redis caching
- Use connection pooling
- Optimize database queries
- Enable gzip compression
- Use CDN for static assets
- Implement horizontal scaling
- Monitor and tune performance

### Scaling Strategies

- **Horizontal scaling**: Add more backend instances
- **Database scaling**: Use read replicas
- **Cache scaling**: Redis cluster
- **Load balancing**: NGINX or cloud load balancer
- **Auto-scaling**: Kubernetes HPA

## Support

For deployment support:

- Check the troubleshooting guide
- Review application logs
- Monitor health check endpoints
- Contact support team

---

**ScrollIntel™ v4.0+ - The world's most advanced AI-CTO platform**