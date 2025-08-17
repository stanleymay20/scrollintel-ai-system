#!/usr/bin/env python3
"""
ScrollIntel™ Health Check Script
Validates all services are running correctly
"""

import requests
import time
import sys
import subprocess
import json
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

def check_docker_services():
    """Check if Docker services are running"""
    try:
        result = subprocess.run(['docker-compose', 'ps', '--format', 'json'], 
                              capture_output=True, text=True, check=True)
        services = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    service = json.loads(line)
                    services.append(service)
                except json.JSONDecodeError:
                    continue
        
        required_services = ['postgres', 'redis', 'backend', 'frontend']
        running_services = []
        
        for service in services:
            service_name = service.get('Service', '')
            state = service.get('State', '')
            
            if service_name in required_services:
                if 'running' in state.lower():
                    print_status(f"✅ {service_name}: {state}", "SUCCESS")
                    running_services.append(service_name)
                else:
                    print_status(f"❌ {service_name}: {state}", "ERROR")
        
        missing_services = set(required_services) - set(running_services)
        if missing_services:
            print_status(f"Missing services: {', '.join(missing_services)}", "ERROR")
            return False
        
        return True
        
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to check Docker services: {e}", "ERROR")
        return False
    except Exception as e:
        print_status(f"Error checking services: {e}", "ERROR")
        return False

def check_endpoint(url, name, timeout=5):
    """Check if an endpoint is responding"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print_status(f"✅ {name}: {url} (Status: {response.status_code})", "SUCCESS")
            return True
        else:
            print_status(f"⚠️  {name}: {url} (Status: {response.status_code})", "WARNING")
            return False
    except requests.exceptions.RequestException as e:
        print_status(f"❌ {name}: {url} - {str(e)}", "ERROR")
        return False

def check_database_connection():
    """Check database connection via Docker"""
    try:
        result = subprocess.run([
            'docker-compose', 'exec', '-T', 'postgres', 
            'pg_isready', '-U', 'postgres'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print_status("✅ PostgreSQL: Connection successful", "SUCCESS")
            return True
        else:
            print_status("❌ PostgreSQL: Connection failed", "ERROR")
            return False
    except Exception as e:
        print_status(f"❌ PostgreSQL: Error checking connection - {e}", "ERROR")
        return False

def check_redis_connection():
    """Check Redis connection via Docker"""
    try:
        result = subprocess.run([
            'docker-compose', 'exec', '-T', 'redis', 
            'redis-cli', 'ping'
        ], capture_output=True, text=True)
        
        if 'PONG' in result.stdout:
            print_status("✅ Redis: Connection successful", "SUCCESS")
            return True
        else:
            print_status("❌ Redis: Connection failed", "ERROR")
            return False
    except Exception as e:
        print_status(f"❌ Redis: Error checking connection - {e}", "ERROR")
        return False

def main():
    print("🏥 ScrollIntel™ Health Check")
    print("=" * 30)
    
    all_healthy = True
    
    # Check Docker services
    print_status("Checking Docker services...", "INFO")
    if not check_docker_services():
        all_healthy = False
    
    print()
    
    # Check database
    print_status("Checking database connection...", "INFO")
    if not check_database_connection():
        all_healthy = False
    
    # Check Redis
    print_status("Checking Redis connection...", "INFO")
    if not check_redis_connection():
        all_healthy = False
    
    print()
    
    # Check web endpoints
    print_status("Checking web endpoints...", "INFO")
    endpoints = [
        ("http://localhost:8000/health", "Backend Health"),
        ("http://localhost:8000/docs", "API Documentation"),
        ("http://localhost:3000", "Frontend"),
    ]
    
    for url, name in endpoints:
        if not check_endpoint(url, name):
            all_healthy = False
    
    print()
    
    # Final status
    if all_healthy:
        print_status("🎉 All systems healthy! ScrollIntel™ is ready to use.", "SUCCESS")
        print()
        print("📱 Access Points:")
        print("   🌐 Frontend:    http://localhost:3000")
        print("   🔧 API:         http://localhost:8000")
        print("   📚 API Docs:    http://localhost:8000/docs")
        print("   ❤️  Health:     http://localhost:8000/health")
        sys.exit(0)
    else:
        print_status("❌ Some systems are not healthy. Check the logs above.", "ERROR")
        print()
        print("🛠️  Troubleshooting:")
        print("   📊 View logs:   docker-compose logs -f")
        print("   🔄 Restart:     docker-compose restart")
        print("   🛑 Reset:       docker-compose down -v && docker-compose up -d")
        sys.exit(1)

if __name__ == "__main__":
    main()