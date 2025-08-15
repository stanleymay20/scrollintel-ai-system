#!/bin/bash

# ScrollIntel Core Startup Script

echo "🚀 Starting ScrollIntel Core..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.core.example .env
    echo "✅ Please edit .env file with your configuration"
fi

# Create necessary directories
mkdir -p uploads logs

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.core.yml up --build -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose -f docker-compose.core.yml exec -T postgres pg_isready -U scrollintel -d scrollintel_core > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

# Check Redis
if docker-compose -f docker-compose.core.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
fi

# Check API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is ready"
else
    echo "❌ API is not ready"
fi

echo ""
echo "🎉 ScrollIntel Core is starting up!"
echo ""
echo "📊 Services:"
echo "   • API: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo "   • Frontend: http://localhost:3000 (if enabled)"
echo "   • PostgreSQL: localhost:5432"
echo "   • Redis: localhost:6379"
echo ""
echo "📝 Logs:"
echo "   docker-compose -f docker-compose.core.yml logs -f"
echo ""
echo "🛑 Stop:"
echo "   docker-compose -f docker-compose.core.yml down"
echo ""

# Show logs
docker-compose -f docker-compose.core.yml logs -f