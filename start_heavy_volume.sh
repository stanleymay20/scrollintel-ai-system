#!/bin/bash
echo "🚀 Starting ScrollIntel Heavy Volume Mode..."

# Check system resources
echo "📊 System Resources:"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "  CPU Cores: $(nproc)"
echo "  Disk Space: $(df -h . | tail -1 | awk '{print $4}')"

# Start infrastructure
echo "🐳 Starting infrastructure services..."
docker-compose -f docker-compose.heavy-volume.yml up -d postgres redis minio

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Start monitoring
echo "📊 Starting monitoring services..."
docker-compose -f docker-compose.heavy-volume.yml up -d prometheus grafana

# Start background workers
echo "👷 Starting background workers..."
docker-compose -f docker-compose.heavy-volume.yml up -d celery-worker celery-beat

# Start main application
echo "🌟 Starting ScrollIntel application..."
python run_simple.py

echo "✅ ScrollIntel Heavy Volume Mode started successfully!"
echo "📊 Grafana Dashboard: http://localhost:3001 (admin/admin)"
echo "📈 Prometheus: http://localhost:9090"
echo "💾 MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
