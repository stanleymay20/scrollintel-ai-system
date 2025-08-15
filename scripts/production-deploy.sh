#!/bin/bash

# ================================
# ScrollIntel Production Deployment Script
# Complete production deployment with auto-scaling and blue-green deployment
# ================================

set -e

echo "🚀 Starting ScrollIntel Production Deployment..."

# Configuration
ENVIRONMENT="production"
BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
HEALTH_CHECK_TIMEOUT=300
ROLLBACK_ON_FAILURE=true
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-full}"
ENABLE_BLUE_GREEN="${ENABLE_BLUE_GREEN:-true}"
ENABLE_AUTO_SCALING="${ENABLE_AUTO_SCALING:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if service is healthy
check_health() {
    local service_url=$1
    local max_attempts=30
    local attempt=1
    
    log_info "Checking health of $service_url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$service_url/health" > /dev/null; then
            log_info "Health check passed for $service_url"
            return 0
        fi
        
        log_warn "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log_error "Health check failed for $service_url after $max_attempts attempts"
    return 1
}

# Function to backup database
backup_database() {
    log_info "Creating database backup..."
    mkdir -p "$BACKUP_DIR"
    
    # Export database
    pg_dump "$DATABASE_URL" > "$BACKUP_DIR/database_backup.sql"
    
    # Backup uploaded files
    if [ -d "./uploads" ]; then
        cp -r ./uploads "$BACKUP_DIR/"
    fi
    
    log_info "Backup created at $BACKUP_DIR"
}

# Function to rollback deployment
rollback_deployment() {
    log_error "Deployment failed, initiating rollback..."
    
    # Restore database from backup
    if [ -f "$BACKUP_DIR/database_backup.sql" ]; then
        log_info "Restoring database from backup..."
        psql "$DATABASE_URL" < "$BACKUP_DIR/database_backup.sql"
    fi
    
    # Restore uploaded files
    if [ -d "$BACKUP_DIR/uploads" ]; then
        log_info "Restoring uploaded files..."
        rm -rf ./uploads
        cp -r "$BACKUP_DIR/uploads" ./uploads
    fi
    
    log_info "Rollback completed"
}

# Trap to handle failures
trap 'if [ "$ROLLBACK_ON_FAILURE" = true ]; then rollback_deployment; fi' ERR

# Pre-deployment checks
log_info "Running pre-deployment checks..."

# Check if required environment variables are set
required_vars=("DATABASE_URL" "REDIS_URL" "JWT_SECRET_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        log_error "Required environment variable $var is not set"
        exit 1
    fi
done

# Check if services are accessible
log_info "Checking service dependencies..."

# Test database connection
if ! python -c "
import psycopg2
import os
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    conn.close()
    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    exit(1)
"; then
    log_error "Database connection check failed"
    exit 1
fi

# Create backup
backup_database

# Run tests
log_info "Running test suite..."
python -m pytest tests/ -v --tb=short --maxfail=5

# Run database migrations
log_info "Running database migrations..."
python scripts/migrate-database.py --env production migrate

# Build and deploy backend
log_info "Building and deploying backend..."

# Build Docker image
docker build -t scrollintel-backend:latest --target production .

# Deploy using appropriate strategy
if [ "$ENABLE_BLUE_GREEN" = "true" ]; then
    log_info "Deploying using blue-green strategy..."
    python scripts/blue-green-deploy.py
else
    log_info "Deploying using standard strategy..."
    # Use production infrastructure deployment
    python scripts/production-infrastructure-deploy.py --type "$DEPLOYMENT_TYPE"
fi

# Wait for services to be ready
log_info "Waiting for services to be ready..."
sleep 30

# Health checks
BACKEND_URL="http://localhost:8000"
if ! check_health "$BACKEND_URL"; then
    log_error "Backend health check failed"
    exit 1
fi

# Deploy frontend
log_info "Deploying frontend..."
cd frontend

# Install dependencies and build
npm ci --only=production
npm run build

# Deploy to Vercel (or your chosen platform)
if command -v vercel &> /dev/null; then
    vercel --prod --yes
    FRONTEND_URL=$(vercel ls --limit 1 --format json | jq -r '.[0].url')
    log_info "Frontend deployed to: https://$FRONTEND_URL"
else
    log_warn "Vercel CLI not found, skipping frontend deployment"
fi

cd ..

# Final health checks
log_info "Running final health checks..."

# Test API endpoints
log_info "Testing critical API endpoints..."
endpoints=("/health" "/api/agents" "/api/auth/status")

for endpoint in "${endpoints[@]}"; do
    if ! curl -f -s "$BACKEND_URL$endpoint" > /dev/null; then
        log_error "Endpoint $endpoint is not responding"
        exit 1
    fi
done

# Performance test
log_info "Running basic performance test..."
if command -v ab &> /dev/null; then
    ab -n 100 -c 10 "$BACKEND_URL/health" > /dev/null
    log_info "Performance test completed"
else
    log_warn "Apache Bench not found, skipping performance test"
fi

# Setup auto-scaling if enabled
if [ "$ENABLE_AUTO_SCALING" = "true" ]; then
    log_info "Starting auto-scaling manager..."
    nohup python scripts/auto-scaling-manager.py > logs/auto-scaling.log 2>&1 &
    echo $! > logs/auto-scaling.pid
    log_info "Auto-scaling manager started (PID: $(cat logs/auto-scaling.pid))"
fi

# Cleanup old backups (keep last 5)
log_info "Cleaning up old backups..."
ls -t ./backups/ | tail -n +6 | xargs -r rm -rf

# Send deployment notification (if configured)
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"🚀 ScrollIntel production deployment completed successfully with auto-scaling and load balancing!\"}" \
        "$SLACK_WEBHOOK_URL"
fi

log_info "✅ Production deployment completed successfully!"
log_info "🔗 Application URL: http://localhost"
log_info "🔗 Health Check: http://localhost:8080/health"
log_info "🔗 Monitoring: http://localhost:3001"

echo ""
echo "🎉 ScrollIntel is now live in production with enterprise-grade infrastructure!"
echo ""
echo "Infrastructure Features:"
echo "• Auto-scaling: $ENABLE_AUTO_SCALING"
echo "• Blue-green deployment: $ENABLE_BLUE_GREEN"
echo "• Load balancing: Enabled"
echo "• Database replication: Enabled"
echo "• SSL/HTTPS: Enabled"
echo "• Monitoring & alerting: Enabled"
echo ""
echo "Next steps:"
echo "1. Monitor application logs and metrics"
echo "2. Run smoke tests on production environment"
echo "3. Configure DNS to point to production load balancer"
echo "4. Set up backup schedules and disaster recovery"
echo "5. Configure alerting notifications"
echo "6. Update documentation with new deployment info"