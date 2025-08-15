#!/bin/bash

# ================================
# ScrollIntel Core - Startup Script
# Starts the focused AI-CTO platform
# ================================

set -e

echo "🚀 Starting ScrollIntel Core..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration before running again."
    exit 1
fi

# Load environment variables
source .env

# Check required environment variables
if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "❌ Error: At least one AI service API key is required (OPENAI_API_KEY or ANTHROPIC_API_KEY)"
    echo "   Please set these in your .env file"
    exit 1
fi

# Create necessary directories
mkdir -p uploads logs

echo "🐳 Starting Docker services..."

# Start with Docker Compose
if [ "$1" = "dev" ]; then
    echo "🔧 Starting in development mode..."
    docker-compose up --build
elif [ "$1" = "prod" ]; then
    echo "🏭 Starting in production mode..."
    docker-compose -f docker-compose.yml up -d --build
else
    echo "📖 Usage: $0 [dev|prod]"
    echo "   dev  - Start in development mode (with logs)"
    echo "   prod - Start in production mode (detached)"
    exit 1
fi