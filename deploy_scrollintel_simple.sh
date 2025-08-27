#!/bin/bash
# ScrollIntel.com Simple Deployment Script

echo "Deploying ScrollIntel to scrollintel.com..."

# Load environment
export $(cat .env.scrollintel.com | grep -v '^#' | xargs)

# Build and start services
echo "Building containers..."
docker-compose -f docker-compose.scrollintel.yml build

echo "Starting services..."
docker-compose -f docker-compose.scrollintel.yml up -d

# Wait for services
echo "Waiting for services to start..."
sleep 30

# Initialize database
echo "Initializing database..."
docker-compose -f docker-compose.scrollintel.yml exec scrollintel-backend python init_database.py

echo "Deployment complete!"
echo "ScrollIntel is now live at:"
echo "   Main: https://scrollintel.com"
echo "   API: https://api.scrollintel.com"
echo "   Docs: https://api.scrollintel.com/docs"