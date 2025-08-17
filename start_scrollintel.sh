#!/bin/bash
echo "Starting ScrollIntel..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Docker not available, starting simple mode..."
    python3 run_simple.py
    exit 0
fi

# Try Docker Compose
echo "Starting with Docker Compose..."
docker-compose -f docker-compose.minimal.yml up -d postgres redis
sleep 10

# Start Python backend
echo "Starting ScrollIntel backend..."
python3 run_simple.py
