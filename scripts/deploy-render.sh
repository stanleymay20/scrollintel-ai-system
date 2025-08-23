#!/bin/bash
# Render Deployment Script

set -e

echo "🚀 Deploying ScrollIntel to Render..."

# Check if render.yaml exists
if [ ! -f "render.yaml" ]; then
    echo "❌ render.yaml not found. Run configure_for_deployment.py first."
    exit 1
fi

echo "📝 render.yaml configuration found"
echo "🌐 Go to https://render.com and create a new service using this repository"
echo "📋 Use the render.yaml file for automatic configuration"
echo "🔧 Don't forget to set your environment variables in the Render dashboard"

echo "Required environment variables:"
echo "  - POSTGRES_PASSWORD"
echo "  - JWT_SECRET_KEY"
echo "  - OPENAI_API_KEY"
echo "  - ANTHROPIC_API_KEY"
echo "  - EMAIL_PASSWORD"

echo "✅ Render configuration ready!"
