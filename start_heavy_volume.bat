@echo off
echo Starting ScrollIntel Heavy Volume Mode...

echo System Resources:
wmic computersystem get TotalPhysicalMemory
wmic cpu get NumberOfCores

echo Starting infrastructure services...
docker-compose -f docker-compose.heavy-volume.yml up -d postgres redis minio

echo Waiting for services...
timeout /t 30 /nobreak

echo Starting monitoring services...
docker-compose -f docker-compose.heavy-volume.yml up -d prometheus grafana

echo Starting background workers...
docker-compose -f docker-compose.heavy-volume.yml up -d celery-worker celery-beat

echo Starting ScrollIntel application...
python run_simple.py

echo ScrollIntel Heavy Volume Mode started successfully!
echo Grafana Dashboard: http://localhost:3001 (admin/admin)
echo Prometheus: http://localhost:9090
echo MinIO Console: http://localhost:9001 (minioadmin/minioadmin)
pause
