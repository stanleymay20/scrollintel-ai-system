#!/bin/bash

# ================================
# ScrollIntel Vercel Deployment Script
# Deploys frontend to Vercel with environment configuration
# ================================

set -e

echo "🚀 Starting ScrollIntel Frontend Deployment to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Navigate to frontend directory
cd frontend

# Install dependencies
echo "📦 Installing dependencies..."
npm ci

# Run tests
echo "🧪 Running tests..."
npm run test -- --run

# Build the application
echo "🔨 Building application..."
npm run build

# Deploy to Vercel
echo "🌐 Deploying to Vercel..."

if [ "$1" = "production" ]; then
    echo "🎯 Deploying to production..."
    vercel --prod --yes
else
    echo "🔧 Deploying to preview..."
    vercel --yes
fi

echo "✅ Frontend deployment completed successfully!"

# Get deployment URL
DEPLOYMENT_URL=$(vercel ls --limit 1 --format json | jq -r '.[0].url')
echo "🔗 Deployment URL: https://$DEPLOYMENT_URL"

cd ..

echo "🎉 ScrollIntel Frontend is now live!"