#!/bin/bash

# ================================
# ScrollIntel Render Deployment Script
# Deploys backend to Render with database migrations
# ================================

set -e

echo "🚀 Starting ScrollIntel Backend Deployment to Render..."

# Check if Render CLI is installed
if ! command -v render &> /dev/null; then
    echo "❌ Render CLI not found. Please install it first:"
    echo "   npm install -g @render/cli"
    exit 1
fi

# Run database migrations
echo "🗄️ Running database migrations..."
python -m alembic upgrade head

# Run tests
echo "🧪 Running backend tests..."
python -m pytest tests/ -v --tb=short

# Deploy to Render using render.yaml
echo "🌐 Deploying to Render..."
render deploy

echo "✅ Backend deployment completed successfully!"

# Health check
echo "🏥 Performing health check..."
sleep 30  # Wait for deployment to be ready

RENDER_URL="https://scrollintel-backend.onrender.com"
if curl -f "$RENDER_URL/health" > /dev/null 2>&1; then
    echo "✅ Health check passed!"
    echo "🔗 Backend URL: $RENDER_URL"
else
    echo "⚠️ Health check failed. Please check the deployment logs."
fi

echo "🎉 ScrollIntel Backend is now live!"