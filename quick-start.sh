#!/bin/bash

# ================================
# ScrollIntel™ Quick Start Script
# Minimal setup for immediate launch
# ================================

set -e

echo "🚀 ScrollIntel™ Quick Start"
echo "=========================="

# Check Docker
if ! command -v docker &> /dev/null || ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker or Docker Compose not found. Please install Docker first."
    exit 1
fi

# Setup environment
if [ ! -f .env ]; then
    echo "📝 Setting up environment..."
    cp .env.example .env
    
    # Generate JWT secret
    JWT_SECRET=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || date +%s | sha256sum | base64 | head -c 64)
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env
    else
        sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env
    fi
    
    echo "✅ Environment configured"
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d --build

echo "⏳ Waiting for services..."
sleep 20

# Check health
if curl -s http://localhost:8000/health &> /dev/null; then
    echo "✅ Backend ready"
else
    echo "⚠️  Backend starting (may take a moment)"
fi

if curl -s http://localhost:3000 &> /dev/null; then
    echo "✅ Frontend ready"
else
    echo "⚠️  Frontend starting (may take a moment)"
fi

echo ""
echo "🎉 ScrollIntel™ is launching!"
echo ""
echo "📱 Access:"
echo "   Frontend: http://localhost:3000"
echo "   API:      http://localhost:8000"
echo "   Docs:     http://localhost:8000/docs"
echo ""
echo "🛠️  Commands:"
echo "   Logs:     docker-compose logs -f"
echo "   Stop:     docker-compose down"
echo ""

# Try to open browser (optional)
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000 &> /dev/null &
elif command -v open &> /dev/null; then
    open http://localhost:3000 &> /dev/null &
fi

echo "Ready! 🌟"