#!/bin/bash
# ScrollIntel.com Production Deployment Script

set -e

echo "🚀 Deploying ScrollIntel to scrollintel.com..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check prerequisites
print_status "Checking prerequisites..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if environment file exists
if [ ! -f .env.production ]; then
    print_warning ".env.production not found. Creating from template..."
    cp .env.scrollintel.com .env.production
    print_warning "Please edit .env.production with your actual values before continuing."
    print_warning "Required: OPENAI_API_KEY, JWT_SECRET_KEY, POSTGRES_PASSWORD"
    read -p "Press Enter after you've configured .env.production..."
fi

# Load environment variables
if [ -f .env.production ]; then
    export $(cat .env.production | grep -v '^#' | xargs)
fi

# Validate required environment variables
print_status "Validating environment variables..."

required_vars=("OPENAI_API_KEY" "JWT_SECRET_KEY" "POSTGRES_PASSWORD")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -ne 0 ]; then
    print_error "Missing required environment variables: ${missing_vars[*]}"
    print_error "Please set these in .env.production"
    exit 1
fi

print_success "Environment variables validated"

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p backups
mkdir -p letsencrypt
mkdir -p monitoring
mkdir -p logs

# Stop existing containers if running
print_status "Stopping existing containers..."
docker-compose -f docker-compose.scrollintel.com.yml down || true

# Pull latest images
print_status "Pulling latest Docker images..."
docker-compose -f docker-compose.scrollintel.com.yml pull

# Build containers
print_status "Building containers..."
docker-compose -f docker-compose.scrollintel.com.yml build

# Start services
print_status "Starting services..."
docker-compose -f docker-compose.scrollintel.com.yml up -d

# Wait for services to be ready
print_status "Waiting for services to start..."
sleep 30

# Check if database is ready
print_status "Checking database connectivity..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if docker-compose -f docker-compose.scrollintel.com.yml exec -T postgres pg_isready -U ${POSTGRES_USER:-scrollintel} > /dev/null 2>&1; then
        print_success "Database is ready"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        print_error "Database failed to start after $max_attempts attempts"
        exit 1
    fi
    
    print_status "Waiting for database... (attempt $attempt/$max_attempts)"
    sleep 2
    ((attempt++))
done

# Run database migrations
print_status "Running database migrations..."
docker-compose -f docker-compose.scrollintel.com.yml exec -T backend alembic upgrade head || {
    print_warning "Alembic migrations failed, trying to initialize database..."
    docker-compose -f docker-compose.scrollintel.com.yml exec -T backend python init_database.py
}

# Initialize database with seed data
print_status "Seeding database..."
docker-compose -f docker-compose.scrollintel.com.yml exec -T backend python init_database.py || print_warning "Database seeding failed (may already be initialized)"

# Health check
print_status "Running health checks..."
sleep 10

# Check backend health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_success "Backend health check passed"
else
    print_warning "Backend health check failed - checking logs..."
    docker-compose -f docker-compose.scrollintel.com.yml logs backend | tail -20
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    print_success "Frontend health check passed"
else
    print_warning "Frontend health check failed - checking logs..."
    docker-compose -f docker-compose.scrollintel.com.yml logs frontend | tail -20
fi

# Display status
print_status "Checking container status..."
docker-compose -f docker-compose.scrollintel.com.yml ps

print_success "Deployment complete!"
echo
echo "🌐 Your ScrollIntel platform is now running:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "   Grafana: http://localhost:3001 (admin/admin)"
echo "   MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo
echo "📋 Next steps:"
echo "1. Configure your DNS records to point to this server"
echo "2. Set up SSL certificates with Let's Encrypt"
echo "3. Configure your domain settings"
echo "4. Test the platform with sample data"
echo
echo "🔒 SSL Setup (run after DNS is configured):"
echo "   sudo certbot --nginx -d scrollintel.com -d app.scrollintel.com -d api.scrollintel.com"
echo
echo "📊 Monitor your deployment:"
echo "   docker-compose -f docker-compose.scrollintel.com.yml logs -f"
echo
echo "🎉 ScrollIntel.com is ready to replace your CTO with AI!"