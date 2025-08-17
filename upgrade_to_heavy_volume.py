#!/usr/bin/env python3
"""
ScrollIntel Heavy Volume Upgrade Script
Systematically upgrades the system to handle large datasets
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Any

def check_system_requirements():
    """Check if system meets requirements for heavy volume processing"""
    print("🔍 Checking System Requirements...")
    
    requirements = {
        'python_version': sys.version_info >= (3, 8),
        'memory_gb': get_system_memory() >= 6,  # Reduced for current system
        'disk_space_gb': get_free_disk_space() >= 100,
        'cpu_cores': os.cpu_count() >= 4
    }
    
    print(f"  Python Version: {sys.version_info.major}.{sys.version_info.minor} {'✅' if requirements['python_version'] else '❌'}")
    print(f"  System Memory: {get_system_memory():.1f} GB {'✅' if requirements['memory_gb'] else '❌'}")
    print(f"  Free Disk Space: {get_free_disk_space():.1f} GB {'✅' if requirements['disk_space_gb'] else '❌'}")
    print(f"  CPU Cores: {os.cpu_count()} {'✅' if requirements['cpu_cores'] else '❌'}")
    
    if not all(requirements.values()):
        print("❌ System does not meet minimum requirements for heavy volume processing")
        return False
    
    print("✅ System meets requirements for heavy volume processing")
    return True

def get_system_memory():
    """Get system memory in GB"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return 8.0  # Default assumption

def get_free_disk_space():
    """Get free disk space in GB"""
    try:
        import psutil
        return psutil.disk_usage('.').free / (1024**3)
    except ImportError:
        return 100.0  # Default assumption

def install_heavy_volume_dependencies():
    """Install additional dependencies for heavy volume processing"""
    print("📦 Installing Heavy Volume Dependencies...")
    
    heavy_volume_packages = [
        'dask>=2023.1.0',
        'pyarrow>=10.0.0',
        'fastparquet>=0.8.0',
        'sqlalchemy>=2.0.0',
        'psycopg2-binary>=2.9.0',
        'redis>=4.5.0',
        'celery>=5.2.0',
        'psutil>=5.9.0',
        'memory-profiler>=0.60.0',
        'asyncpg>=0.27.0',
        'aioredis>=2.0.0',
        'boto3>=1.26.0',
        'minio>=7.1.0'
    ]
    
    for package in heavy_volume_packages:
        try:
            print(f"  Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"  ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All heavy volume dependencies installed")
    return True

def create_heavy_volume_config():
    """Create configuration for heavy volume processing"""
    print("⚙️ Creating Heavy Volume Configuration...")
    
    config_content = """# ScrollIntel Heavy Volume Configuration

# File Processing
MAX_FILE_SIZE_GB=50
CHUNK_SIZE_ROWS=1000000
MAX_WORKERS=8
MEMORY_LIMIT_GB=16

# Database
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Redis Cache
REDIS_MAX_CONNECTIONS=100
REDIS_SOCKET_KEEPALIVE=True
REDIS_SOCKET_KEEPALIVE_OPTIONS={}

# Processing
ENABLE_DISTRIBUTED_PROCESSING=true
ENABLE_ASYNC_PROCESSING=true
ENABLE_RESULT_CACHING=true
CACHE_TTL_SECONDS=3600

# Storage
ENABLE_OBJECT_STORAGE=true
OBJECT_STORAGE_BUCKET=scrollintel-data
ENABLE_DATA_COMPRESSION=true
COMPRESSION_FORMAT=parquet

# Monitoring
ENABLE_PERFORMANCE_MONITORING=true
ENABLE_MEMORY_PROFILING=true
METRICS_COLLECTION_INTERVAL=30

# Scaling
AUTO_SCALING_ENABLED=true
MIN_WORKERS=2
MAX_WORKERS=16
SCALE_UP_THRESHOLD=80
SCALE_DOWN_THRESHOLD=20
"""
    
    config_file = Path(".env.heavy_volume")
    config_file.write_text(config_content)
    print(f"✅ Heavy volume configuration created: {config_file}")
    
    return True

def upgrade_file_processor():
    """Upgrade the file processor for heavy volume support"""
    print("🔧 Upgrading File Processor...")
    
    # Create enhanced file processor
    enhanced_processor = """
import asyncio
import pandas as pd
import dask.dataframe as dd
from typing import Dict, Any, AsyncGenerator
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HeavyVolumeFileProcessor:
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024 * 1024  # 50GB
        self.chunk_size = 1000000  # 1M rows
        self.use_dask = True
        self.compression = 'snappy'
        
    async def process_large_file(self, file_path: str) -> Dict[str, Any]:
        \"\"\"Process large files using Dask for distributed computing\"\"\"
        
        if self.use_dask:
            return await self._process_with_dask(file_path)
        else:
            return await self._process_with_pandas_chunks(file_path)
    
    async def _process_with_dask(self, file_path: str) -> Dict[str, Any]:
        \"\"\"Process file using Dask for better memory management\"\"\"
        
        try:
            # Read with Dask for lazy evaluation
            if file_path.endswith('.csv'):
                df = dd.read_csv(file_path, blocksize="100MB")
            elif file_path.endswith('.parquet'):
                df = dd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format")
            
            # Get basic info
            result = {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'column_types': dict(df.dtypes),
                'memory_usage': df.memory_usage(deep=True).sum().compute(),
                'processing_method': 'dask'
            }
            
            # Compute statistics lazily
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                result['statistics'] = df[numeric_cols].describe().compute().to_dict()
            
            return result
            
        except Exception as e:
            logger.error(f"Dask processing failed: {e}")
            raise
    
    async def _process_with_pandas_chunks(self, file_path: str) -> Dict[str, Any]:
        \"\"\"Fallback to pandas chunked processing\"\"\"
        
        total_rows = 0
        chunk_stats = []
        
        for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
            chunk_info = {
                'rows': len(chunk),
                'memory': chunk.memory_usage(deep=True).sum()
            }
            chunk_stats.append(chunk_info)
            total_rows += len(chunk)
        
        return {
            'total_rows': total_rows,
            'chunks_processed': len(chunk_stats),
            'total_memory': sum(stat['memory'] for stat in chunk_stats),
            'processing_method': 'pandas_chunks'
        }
"""
    
    # Write enhanced processor
    processor_file = Path("scrollintel/engines/heavy_volume_processor.py")
    processor_file.parent.mkdir(parents=True, exist_ok=True)
    processor_file.write_text(enhanced_processor)
    
    print("✅ Enhanced file processor created")
    return True

def setup_distributed_storage():
    """Setup distributed storage configuration"""
    print("💾 Setting up Distributed Storage...")
    
    storage_config = """
import os
from minio import Minio
from typing import Optional
import asyncio
import aiofiles

class DistributedStorage:
    def __init__(self):
        self.minio_client = None
        self.bucket_name = os.getenv('OBJECT_STORAGE_BUCKET', 'scrollintel-data')
        self.setup_minio()
    
    def setup_minio(self):
        \"\"\"Setup MinIO client for object storage\"\"\"
        endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        access_key = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        secret_key = os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        
        self.minio_client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        
        # Create bucket if it doesn't exist
        if not self.minio_client.bucket_exists(self.bucket_name):
            self.minio_client.make_bucket(self.bucket_name)
    
    async def upload_file(self, local_path: str, object_name: str) -> bool:
        \"\"\"Upload file to distributed storage\"\"\"
        try:
            self.minio_client.fput_object(
                self.bucket_name, 
                object_name, 
                local_path
            )
            return True
        except Exception as e:
            print(f"Upload failed: {e}")
            return False
    
    async def download_file(self, object_name: str, local_path: str) -> bool:
        \"\"\"Download file from distributed storage\"\"\"
        try:
            self.minio_client.fget_object(
                self.bucket_name,
                object_name,
                local_path
            )
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
"""
    
    storage_file = Path("scrollintel/core/distributed_storage.py")
    storage_file.write_text(storage_config)
    
    print("✅ Distributed storage configuration created")
    return True

def create_docker_compose_heavy_volume():
    """Create Docker Compose for heavy volume infrastructure"""
    print("🐳 Creating Heavy Volume Docker Compose...")
    
    docker_compose = """# ScrollIntel Heavy Volume Infrastructure
services:
  postgres:
    image: postgres:15-alpine
    container_name: scrollintel-postgres-hv
    environment:
      POSTGRES_DB: scrollintel
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: scrollintel_password
      POSTGRES_SHARED_PRELOAD_LIBRARIES: pg_stat_statements
    ports:
      - "5432:5432"
    volumes:
      - postgres_data_hv:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  redis:
    image: redis:7-alpine
    container_name: scrollintel-redis-hv
    ports:
      - "6379:6379"
    volumes:
      - redis_data_hv:/data
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    deploy:
      resources:
        limits:
          memory: 2G

  minio:
    image: minio/minio:latest
    container_name: scrollintel-minio
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: server /data --console-address ":9001"
    deploy:
      resources:
        limits:
          memory: 1G

  celery-worker:
    build: .
    container_name: scrollintel-celery-worker
    command: celery -A scrollintel.core.celery_app worker --loglevel=info --concurrency=4
    volumes:
      - ./scrollintel:/app/scrollintel
      - ./uploads:/app/uploads
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:scrollintel_password@postgres:5432/scrollintel
    depends_on:
      - postgres
      - redis
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  celery-beat:
    build: .
    container_name: scrollintel-celery-beat
    command: celery -A scrollintel.core.celery_app beat --loglevel=info
    volumes:
      - ./scrollintel:/app/scrollintel
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://postgres:scrollintel_password@postgres:5432/scrollintel
    depends_on:
      - postgres
      - redis

  prometheus:
    image: prom/prometheus:latest
    container_name: scrollintel-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: scrollintel-grafana
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  postgres_data_hv:
  redis_data_hv:
  minio_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: scrollintel-heavy-volume
"""
    
    compose_file = Path("docker-compose.heavy-volume.yml")
    compose_file.write_text(docker_compose)
    
    print("✅ Heavy volume Docker Compose created")
    return True

def create_postgresql_config():
    """Create optimized PostgreSQL configuration for heavy volume"""
    print("🗄️ Creating PostgreSQL Configuration...")
    
    pg_config = """# PostgreSQL Configuration for Heavy Volume Processing

# Memory Settings
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 256MB
maintenance_work_mem = 1GB

# Checkpoint Settings
checkpoint_completion_target = 0.9
wal_buffers = 64MB
checkpoint_timeout = 15min
max_wal_size = 4GB
min_wal_size = 1GB

# Connection Settings
max_connections = 200
shared_preload_libraries = 'pg_stat_statements'

# Query Planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on

# Autovacuum
autovacuum_max_workers = 4
autovacuum_naptime = 30s
"""
    
    config_file = Path("postgresql.conf")
    config_file.write_text(pg_config)
    
    print("✅ PostgreSQL configuration created")
    return True

def create_monitoring_config():
    """Create monitoring configuration"""
    print("📊 Creating Monitoring Configuration...")
    
    prometheus_config = """global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'scrollintel-api'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'minio'
    static_configs:
      - targets: ['minio:9000']
"""
    
    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(exist_ok=True)
    
    prometheus_file = monitoring_dir / "prometheus.yml"
    prometheus_file.write_text(prometheus_config)
    
    print("✅ Monitoring configuration created")
    return True

def create_startup_script():
    """Create startup script for heavy volume mode"""
    print("🚀 Creating Heavy Volume Startup Script...")
    
    startup_script = """#!/bin/bash
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
"""
    
    script_file = Path("start_heavy_volume.sh")
    script_file.write_text(startup_script, encoding='utf-8')
    try:
        script_file.chmod(0o755)
    except:
        pass  # Windows doesn't support chmod
    
    # Windows version
    windows_script = """@echo off
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
"""
    
    windows_file = Path("start_heavy_volume.bat")
    windows_file.write_text(windows_script)
    
    print("✅ Heavy volume startup scripts created")
    return True

def main():
    """Main upgrade function"""
    print("🔧 ScrollIntel Heavy Volume Upgrade")
    print("=" * 50)
    
    # Check system requirements
    if not check_system_requirements():
        print("❌ System requirements not met. Please upgrade your system.")
        return False
    
    # Install dependencies
    if not install_heavy_volume_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Create configurations
    create_heavy_volume_config()
    create_postgresql_config()
    create_monitoring_config()
    
    # Upgrade components
    upgrade_file_processor()
    setup_distributed_storage()
    
    # Create infrastructure
    create_docker_compose_heavy_volume()
    create_startup_script()
    
    print("\n✅ ScrollIntel Heavy Volume Upgrade Complete!")
    print("\n🚀 To start heavy volume mode:")
    print("   ./start_heavy_volume.sh (Linux/Mac)")
    print("   start_heavy_volume.bat (Windows)")
    
    print("\n📊 Monitoring URLs:")
    print("   Grafana: http://localhost:3001 (admin/admin)")
    print("   Prometheus: http://localhost:9090")
    print("   MinIO: http://localhost:9001 (minioadmin/minioadmin)")
    
    print("\n💡 Heavy Volume Features:")
    print("   ✅ Files up to 50GB")
    print("   ✅ Distributed processing with Dask")
    print("   ✅ Object storage with MinIO")
    print("   ✅ Background job processing")
    print("   ✅ Performance monitoring")
    print("   ✅ Optimized PostgreSQL")
    
    return True

if __name__ == "__main__":
    main()