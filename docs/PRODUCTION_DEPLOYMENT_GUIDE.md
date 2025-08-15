# ScrollIntel Production Deployment Guide

## Overview

This guide covers the complete production deployment of ScrollIntel with enterprise-grade infrastructure including auto-scaling, load balancing, blue-green deployment, database replication, and comprehensive monitoring.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Auto Scaler   │    │   Monitoring    │
│   (Nginx)       │    │   (Python)      │    │   (Grafana)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
    ┌─────────────────────────────────────────────────────────┐
    │                Backend Instances                        │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
    │  │ Backend-1   │  │ Backend-2   │  │ Backend-3   │     │
    │  │ Port: 8000  │  │ Port: 8001  │  │ Port: 8002  │     │
    │  └─────────────┘  └─────────────┘  └─────────────┘     │
    └─────────────────────────────────────────────────────────┘
                                 │
    ┌─────────────────────────────────────────────────────────┐
    │                Database Layer                           │
    │  ┌─────────────┐                    ┌─────────────┐     │
    │  │ Master DB   │ ──── Replication ──│ Replica DB  │     │
    │  │ Port: 5432  │                    │ Port: 5433  │     │
    │  └─────────────┘                    └─────────────┘     │
    └─────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements
- **OS**: Linux/macOS (Windows with WSL2)
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: Minimum 50GB free space
- **CPU**: 4+ cores recommended

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Node.js 18+ (for frontend)
- Git

### Environment Variables
Create a `.env.production` file with the following variables:

```bash
# Database Configuration
POSTGRES_DB=scrollintel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
REPLICATION_USER=replicator
REPLICATION_PASSWORD=your_replication_password

# Application Configuration
JWT_SECRET_KEY=your_jwt_secret_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Infrastructure Configuration
BACKEND_INSTANCES=3
ENABLE_AUTO_SCALING=true
ENABLE_BLUE_GREEN=true
ENABLE_DB_REPLICATION=true
ENABLE_MONITORING=true
SSL_ENABLED=true

# Scaling Configuration
CPU_SCALE_UP_THRESHOLD=70
CPU_SCALE_DOWN_THRESHOLD=30
MEMORY_SCALE_UP_THRESHOLD=80
MEMORY_SCALE_DOWN_THRESHOLD=40
MIN_REPLICAS=2
MAX_REPLICAS=10

# Monitoring Configuration
GRAFANA_PASSWORD=your_grafana_password
SLACK_WEBHOOK_URL=your_slack_webhook_url
```

## Deployment Methods

### Method 1: Complete Automated Deployment (Recommended)

```bash
# 1. Clone and prepare the repository
git clone <repository-url>
cd scrollintel
cp .env.example .env.production
# Edit .env.production with your values

# 2. Run complete deployment
bash scripts/deploy-production-complete.sh

# 3. Validate deployment
python scripts/validate-deployment.py
```

### Method 2: Step-by-Step Deployment

#### Step 1: Infrastructure Setup
```bash
# Set up database replication
python scripts/database-replication-setup.py

# Set up load balancer
python scripts/load-balancer-setup.py

# Deploy infrastructure
python scripts/production-infrastructure-deploy.py --type full
```

#### Step 2: Application Deployment
```bash
# Option A: Blue-Green Deployment
python scripts/blue-green-deploy.py

# Option B: Standard Deployment
bash scripts/production-deploy.sh
```

#### Step 3: Auto-Scaling Setup
```bash
# Start auto-scaling manager
python scripts/auto-scaling-manager.py &
echo $! > logs/auto-scaling.pid
```

#### Step 4: Validation
```bash
python scripts/validate-deployment.py
```

### Method 3: Docker Compose Only (Minimal)

```bash
# Start with load-balanced configuration
docker-compose -f docker-compose.load-balanced.yml up -d

# Or start with database replication
docker-compose -f docker-compose.db-replication.yml up -d
```

## Configuration Options

### Deployment Types
- **full**: Complete deployment with all features
- **minimal**: Basic deployment without advanced features
- **update**: Update existing deployment

### Feature Toggles
- `ENABLE_AUTO_SCALING`: Enable/disable automatic scaling
- `ENABLE_BLUE_GREEN`: Enable/disable blue-green deployments
- `ENABLE_DB_REPLICATION`: Enable/disable database replication
- `ENABLE_MONITORING`: Enable/disable monitoring stack
- `SSL_ENABLED`: Enable/disable SSL/HTTPS

## Monitoring and Management

### Health Checks
```bash
# Application health
curl http://localhost/health

# Load balancer status
curl http://localhost:8080/nginx_status

# Upstream backend status
curl http://localhost:8080/upstream_check

# Comprehensive validation
python scripts/validate-deployment.py
```

### Monitoring Dashboards
- **Application**: http://localhost
- **Load Balancer Health**: http://localhost:8080/health
- **Grafana Monitoring**: http://localhost:3001
- **Prometheus Metrics**: http://localhost:9090

### Log Management
```bash
# View all service logs
docker-compose -f docker-compose.load-balanced.yml logs -f

# View specific service logs
docker-compose -f docker-compose.load-balanced.yml logs -f backend-1

# View auto-scaling logs
tail -f logs/auto-scaling.log

# View nginx access logs
tail -f logs/nginx/access.log
```

### Scaling Operations
```bash
# Manual scaling
docker-compose -f docker-compose.load-balanced.yml up -d --scale backend-1=5

# Check current scaling status
docker ps --filter name=scrollintel-backend

# View auto-scaling metrics
cat logs/scaling_metrics.jsonl | tail -10
```

## Backup and Recovery

### Database Backup
```bash
# Manual backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup (runs daily)
# Configured in database replication setup
```

### Application Backup
```bash
# Backup configuration and data
tar -czf scrollintel_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    .env.production \
    nginx/ \
    uploads/ \
    logs/ \
    deployments/
```

### Recovery Process
```bash
# 1. Stop services
docker-compose -f docker-compose.load-balanced.yml down

# 2. Restore database
psql $DATABASE_URL < backup_file.sql

# 3. Restore configuration
tar -xzf scrollintel_backup_file.tar.gz

# 4. Restart services
bash scripts/deploy-production-complete.sh
```

## Security Considerations

### SSL/TLS Configuration
- SSL certificates are automatically generated for development
- For production, replace with valid certificates from Let's Encrypt or CA
- Certificates location: `./nginx/ssl/`

### Network Security
- All services run in isolated Docker networks
- Rate limiting configured in nginx
- Database access restricted to application containers

### Access Control
- API key management for external access
- Role-based access control for users
- Audit logging for all actions

## Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check Docker daemon
docker info

# Check container logs
docker-compose -f docker-compose.load-balanced.yml logs

# Check system resources
df -h
free -h
```

#### Performance Issues
```bash
# Check auto-scaling status
cat logs/auto-scaling.log

# Check system metrics
python scripts/health-check-monitor.py

# Check database performance
docker exec scrollintel-postgres-master pg_stat_activity
```

#### Database Connection Issues
```bash
# Test database connectivity
python -c "
import psycopg2
conn = psycopg2.connect('$DATABASE_URL')
print('Database connection successful')
conn.close()
"

# Check replication status
docker exec scrollintel-postgres-master psql -U postgres -c "SELECT * FROM pg_stat_replication;"
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run deployment with verbose output
bash scripts/deploy-production-complete.sh --dry-run

# Check specific component
python scripts/validate-deployment.py --backend-url http://localhost
```

## Maintenance

### Regular Tasks
- Monitor disk space and clean old logs
- Update SSL certificates before expiration
- Review and rotate API keys
- Update dependencies and security patches
- Monitor performance metrics and optimize

### Scheduled Maintenance
```bash
# Weekly log rotation
find logs/ -name "*.log" -mtime +7 -delete

# Monthly backup cleanup
find backups/ -name "*.sql" -mtime +30 -delete

# Update system packages
apt update && apt upgrade -y
```

## Performance Optimization

### Auto-Scaling Tuning
Adjust thresholds in `.env.production`:
```bash
CPU_SCALE_UP_THRESHOLD=70      # Scale up when CPU > 70%
CPU_SCALE_DOWN_THRESHOLD=30    # Scale down when CPU < 30%
MEMORY_SCALE_UP_THRESHOLD=80   # Scale up when memory > 80%
MEMORY_SCALE_DOWN_THRESHOLD=40 # Scale down when memory < 40%
```

### Database Optimization
```bash
# Analyze database performance
docker exec scrollintel-postgres-master psql -U postgres -c "
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
"

# Optimize slow queries
# Add indexes for frequently queried columns
# Configure connection pooling
```

### Caching Configuration
- Redis caching is automatically configured
- Nginx caching for static assets
- Application-level caching for API responses

## Support and Documentation

### Additional Resources
- [API Documentation](./API_DOCUMENTATION.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [Architecture Overview](./ARCHITECTURE.md)

### Getting Help
1. Check logs for error messages
2. Run validation script for health status
3. Review this documentation
4. Contact support team with deployment ID and logs

---

**Note**: This deployment guide assumes a production environment. For development or testing, use simplified configurations with fewer resources and security requirements.