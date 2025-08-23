#!/bin/bash
# Docker Production Deployment Script

set -e

echo "🚀 Starting ScrollIntel Docker Deployment..."

# Check if required environment variables are set
required_vars=("POSTGRES_PASSWORD" "JWT_SECRET_KEY" "OPENAI_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Error: $var environment variable is not set"
        exit 1
    fi
done

# Build and deploy with Docker Compose
echo "📦 Building Docker images..."
docker-compose -f docker-compose.prod.yml build

echo "🚀 Starting services..."
docker-compose -f docker-compose.prod.yml up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

echo "🔍 Checking service health..."
docker-compose -f docker-compose.prod.yml ps

echo "✅ ScrollIntel deployed successfully!"
echo "🌐 Access your application at: http://localhost:8000"
