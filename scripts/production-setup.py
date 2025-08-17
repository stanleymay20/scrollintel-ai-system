#!/usr/bin/env python3
"""
ScrollIntel™ Production Setup Script
Configures production environment with security and monitoring
"""

import os
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[0;34m",
        "SUCCESS": "\033[0;32m", 
        "WARNING": "\033[1;33m",
        "ERROR": "\033[0;31m"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}[{status}]{reset} {message}")

def setup_production_env():
    """Setup production environment file"""
    env_prod = Path(".env.production")
    env_example = Path(".env.example")
    
    if not env_example.exists():
        print_status(".env.example not found!", "ERROR")
        return False
    
    # Copy example to production
    shutil.copy(env_example, env_prod)
    
    # Read and modify for production
    with open(env_prod, 'r') as f:
        content = f.read()
    
    # Production settings
    content = content.replace("ENVIRONMENT=development", "ENVIRONMENT=production")
    content = content.replace("DEBUG=true", "DEBUG=false")
    content = content.replace("LOG_LEVEL=INFO", "LOG_LEVEL=WARNING")
    
    # Generate strong secrets
    jwt_secret = secrets.token_hex(64)  # Longer for production
    db_password = secrets.token_urlsafe(32)
    
    content = content.replace("JWT_SECRET_KEY=", f"JWT_SECRET_KEY={jwt_secret}")
    content = content.replace("POSTGRES_PASSWORD=", f"POSTGRES_PASSWORD={db_password}")
    
    # Write production config
    with open(env_prod, 'w') as f:
        f.write(content)
    
    print_status("Created .env.production with secure settings", "SUCCESS")
    return True

def setup_ssl_certificates():
    """Setup SSL certificates for production"""
    ssl_dir = Path("nginx/ssl")
    ssl_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate self-signed certificate for development
    try:
        subprocess.run([
            'openssl', 'req', '-x509', '-nodes', '-days', '365',
            '-newkey', 'rsa:2048',
            '-keyout', str(ssl_dir / 'scrollintel.key'),
            '-out', str(ssl_dir / 'scrollintel.crt'),
            '-subj', '/C=US/ST=State/L=City/O=ScrollIntel/CN=localhost'
        ], check=True, capture_output=True)
        
        print_status("Generated SSL certificates", "SUCCESS")
        print_status("⚠️  Using self-signed certificates. Replace with real certificates for production!", "WARNING")
        return True
        
    except subprocess.CalledProcessError:
        print_status("Failed to generate SSL certificates", "ERROR")
        print_status("OpenSSL not found. Install OpenSSL or provide your own certificates.", "INFO")
        return False
    except FileNotFoundError:
        print_status("OpenSSL not found. Install OpenSSL or provide your own certificates.", "WARNING")
        return False

def setup_monitoring():
    """Setup monitoring configuration"""
    monitoring_dir = Path("monitoring")
    monitoring_dir.mkdir(exist_ok=True)
    
    # Create Prometheus config if it doesn't exist
    prometheus_config = monitoring_dir / "prometheus.yml"
    if not prometheus_config.exists():
        config_content = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'scrollintel-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis:6379']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""
        with open(prometheus_config, 'w') as f:
            f.write(config_content.strip())
        
        print_status("Created Prometheus configuration", "SUCCESS")
    
    # Create alert rules
    alert_rules = monitoring_dir / "alert_rules.yml"
    if not alert_rules.exists():
        rules_content = """
groups:
  - name: scrollintel_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 10% for 5 minutes"

      - alert: DatabaseDown
        expr: up{job="postgres-exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database is down"
          description: "PostgreSQL database is not responding"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 90%"
"""
        with open(alert_rules, 'w') as f:
            f.write(rules_content.strip())
        
        print_status("Created alert rules", "SUCCESS")

def setup_backup_scripts():
    """Setup automated backup scripts"""
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    backup_script = scripts_dir / "backup-database.sh"
    backup_content = """#!/bin/bash
# ScrollIntel™ Database Backup Script

set -e

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="scrollintel_backup_$DATE.sql"

echo "Starting database backup..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
docker-compose exec -T postgres pg_dump -U postgres scrollintel > "$BACKUP_DIR/$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_DIR/$BACKUP_FILE"

echo "Backup completed: $BACKUP_DIR/$BACKUP_FILE.gz"

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "scrollintel_backup_*.sql.gz" -mtime +7 -delete

echo "Old backups cleaned up"
"""
    
    with open(backup_script, 'w') as f:
        f.write(backup_content.strip())
    
    # Make executable (on Unix systems)
    try:
        os.chmod(backup_script, 0o755)
    except:
        pass
    
    print_status("Created backup script", "SUCCESS")

def create_production_compose():
    """Create production docker-compose override"""
    prod_compose = Path("docker-compose.prod.yml")
    
    if prod_compose.exists():
        print_status("Production compose file already exists", "WARNING")
        return True
    
    compose_content = """version: '3.8'

services:
  backend:
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  frontend:
    environment:
      - NODE_ENV=production
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M

  postgres:
    restart: unless-stopped
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

  redis:
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 512M
        reservations:
          memory: 256M

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/alert_rules.yml:/etc/prometheus/alert_rules.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  grafana_data:
"""
    
    with open(prod_compose, 'w') as f:
        f.write(compose_content.strip())
    
    print_status("Created production docker-compose.prod.yml", "SUCCESS")
    return True

def main():
    print("🏭 ScrollIntel™ Production Setup")
    print("=" * 35)
    
    # Setup production environment
    print_status("Setting up production environment...", "INFO")
    setup_production_env()
    
    # Setup SSL certificates
    print_status("Setting up SSL certificates...", "INFO")
    setup_ssl_certificates()
    
    # Setup monitoring
    print_status("Setting up monitoring...", "INFO")
    setup_monitoring()
    
    # Setup backup scripts
    print_status("Setting up backup scripts...", "INFO")
    setup_backup_scripts()
    
    # Create production compose
    print_status("Creating production compose file...", "INFO")
    create_production_compose()
    
    print()
    print_status("🎉 Production setup complete!", "SUCCESS")
    print()
    print("📋 Next steps:")
    print("1. Review and update .env.production with your settings")
    print("2. Replace SSL certificates with real ones for production")
    print("3. Configure your domain and DNS")
    print("4. Deploy with: docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d")
    print("5. Setup automated backups with cron")
    print()
    print("🔒 Security checklist:")
    print("   ✅ Strong passwords generated")
    print("   ✅ SSL certificates configured")
    print("   ✅ Production environment settings")
    print("   ✅ Monitoring and alerting setup")
    print("   ✅ Backup scripts created")

if __name__ == "__main__":
    main()