#!/bin/bash
# Railway Deployment Script

set -e

echo "🚀 Deploying ScrollIntel to Railway..."

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI is not installed. Install it from: https://railway.app/cli"
    exit 1
fi

# Login to Railway (if not already logged in)
railway login

# Set environment variables
echo "🔧 Setting environment variables..."
railway variables set ENVIRONMENT=production
railway variables set DEBUG=false
railway variables set API_HOST=0.0.0.0
railway variables set API_PORT=8000

# Deploy
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment complete!"
echo "🌐 Your app will be available at the Railway-provided URL"
