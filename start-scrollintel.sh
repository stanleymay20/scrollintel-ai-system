#!/bin/bash

# ScrollIntel Quick Start Script
# This script sets up and runs the complete ScrollIntel system

echo "🚀 Starting ScrollIntel AI System..."

# Check if .env exists, if not copy from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys before continuing!"
    echo "   Required: OPENAI_API_KEY, JWT_SECRET_KEY"
    read -p "Press Enter after updating .env file..."
fi

# Start all services
echo "🐳 Starting Docker containers..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check backend health
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is healthy"
else
    echo "❌ Backend API is not responding"
fi

# Check frontend health
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend is not responding"
fi

# Display access information
echo ""
echo "🎉 ScrollIntel is ready!"
echo ""
echo "📱 Frontend UI:      http://localhost:3000"
echo "🔧 Backend API:      http://localhost:8000"
echo "📚 API Docs:         http://localhost:8000/docs"
echo "💾 Database:         localhost:5432"
echo "🗄️  Redis Cache:      localhost:6379"
echo ""
echo "🔍 To view logs:     docker-compose logs -f"
echo "🛑 To stop:          docker-compose down"
echo ""
echo "Happy coding! 🚀"