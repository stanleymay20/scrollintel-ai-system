#!/usr/bin/env python3
"""
ScrollIntel.com Status Checker
Verify that scrollintel.com is accessible and all services are running
"""

import requests
import subprocess
import sys
import time
from datetime import datetime

def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                ScrollIntel.com Status Checker               ║
║              Verify Platform Accessibility                  ║
╚══════════════════════════════════════════════════════════════╝
    """)

def log(message, status="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    status_icon = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARNING": "⚠️"}
    print(f"[{timestamp}] {status_icon.get(status, 'ℹ️')} {message}")

def check_url(url, name, timeout=10):
    """Check if a URL is accessible."""
    try:
        response = requests.get(url, timeout=timeout, verify=False)
        if response.status_code == 200:
            log(f"{name}: {url} - Status {response.status_code}", "SUCCESS")
            return True
        else:
            log(f"{name}: {url} - Status {response.status_code}", "WARNING")
            return False
    except requests.exceptions.RequestException as e:
        log(f"{name}: {url} - Failed: {str(e)}", "ERROR")
        return False

def check_docker_services():
    """Check Docker services status."""
    log("Checking Docker services...")
    
    try:
        # Check if Docker is running
        result = subprocess.run("docker ps", shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            log("Docker is not running or not accessible", "ERROR")
            return False
        
        # Check ScrollIntel containers
        containers = [
            "scrollintel-frontend",
            "scrollintel-backend", 
            "scrollintel-db",
            "scrollintel-redis"
        ]
        
        running_containers = result.stdout
        for container in containers:
            if container in running_containers:
                log(f"Container {container} is running", "SUCCESS")
            else:
                log(f"Container {container} is not running", "WARNING")
        
        return True
    except Exception as e:
        log(f"Error checking Docker services: {e}", "ERROR")
        return False

def check_local_services():
    """Check local services."""
    log("Checking local services...")
    
    local_endpoints = [
        ("http://localhost:3000", "Frontend (Local)"),
        ("http://localhost:8000", "Backend API (Local)"),
        ("http://localhost:8000/health", "Backend Health (Local)"),
        ("http://localhost:8000/docs", "API Documentation (Local)"),
        ("http://localhost:3001", "Grafana (Local)")
    ]
    
    results = []
    for url, name in local_endpoints:
        result = check_url(url, name, timeout=5)
        results.append(result)
    
    return any(results)

def check_domain_services():
    """Check domain services."""
    log("Checking domain services...")
    
    domain_endpoints = [
        ("https://scrollintel.com", "Main Website"),
        ("https://app.scrollintel.com", "Application"),
        ("https://api.scrollintel.com", "API"),
        ("https://api.scrollintel.com/health", "API Health"),
        ("https://api.scrollintel.com/docs", "API Documentation"),
        ("https://grafana.scrollintel.com", "Grafana Dashboard")
    ]
    
    results = []
    for url, name in domain_endpoints:
        result = check_url(url, name)
        results.append(result)
    
    return any(results)

def check_dns_resolution():
    """Check DNS resolution for scrollintel.com domains."""
    log("Checking DNS resolution...")
    
    domains = [
        "scrollintel.com",
        "app.scrollintel.com", 
        "api.scrollintel.com",
        "grafana.scrollintel.com"
    ]
    
    for domain in domains:
        try:
            result = subprocess.run(f"nslookup {domain}", shell=True, capture_output=True, text=True)
            if result.returncode == 0 and "NXDOMAIN" not in result.stdout:
                log(f"DNS resolution for {domain} - OK", "SUCCESS")
            else:
                log(f"DNS resolution for {domain} - Failed", "ERROR")
        except Exception as e:
            log(f"DNS check for {domain} failed: {e}", "ERROR")

def get_system_info():
    """Get system information."""
    log("Getting system information...")
    
    try:
        # Get server IP
        result = subprocess.run("curl -s ifconfig.me", shell=True, capture_output=True, text=True)
        server_ip = result.stdout.strip() if result.returncode == 0 else "Unknown"
        log(f"Server IP: {server_ip}")
        
        # Get disk usage
        result = subprocess.run("df -h /", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                disk_info = lines[1].split()
                log(f"Disk Usage: {disk_info[4]} used of {disk_info[1]}")
        
        # Get memory usage
        result = subprocess.run("free -h", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                mem_info = lines[1].split()
                log(f"Memory Usage: {mem_info[2]} used of {mem_info[1]}")
        
    except Exception as e:
        log(f"Error getting system info: {e}", "ERROR")

def main():
    """Main status check function."""
    print_banner()
    
    log("Starting ScrollIntel.com status check...")
    
    # Check system info
    get_system_info()
    
    # Check DNS resolution
    check_dns_resolution()
    
    # Check Docker services
    docker_ok = check_docker_services()
    
    # Check local services
    local_ok = check_local_services()
    
    # Check domain services
    domain_ok = check_domain_services()
    
    # Summary
    print("\n" + "="*60)
    print("📊 STATUS SUMMARY")
    print("="*60)
    
    if docker_ok:
        log("Docker Services: Running", "SUCCESS")
    else:
        log("Docker Services: Issues detected", "WARNING")
    
    if local_ok:
        log("Local Access: Available", "SUCCESS")
    else:
        log("Local Access: Not available", "ERROR")
    
    if domain_ok:
        log("Domain Access: Available", "SUCCESS")
    else:
        log("Domain Access: Not available (DNS/SSL setup needed)", "WARNING")
    
    print("\n🌐 ACCESS POINTS:")
    if local_ok:
        print("   • Local Frontend: http://localhost:3000")
        print("   • Local API: http://localhost:8000")
        print("   • Local Docs: http://localhost:8000/docs")
        print("   • Local Grafana: http://localhost:3001")
    
    if domain_ok:
        print("   • Website: https://scrollintel.com")
        print("   • Application: https://app.scrollintel.com")
        print("   • API: https://api.scrollintel.com")
        print("   • Monitoring: https://grafana.scrollintel.com")
    
    print("\n🔧 NEXT STEPS:")
    if not docker_ok:
        print("   1. Start Docker services: docker-compose up -d")
    if not local_ok and docker_ok:
        print("   1. Wait for services to initialize (may take 1-2 minutes)")
        print("   2. Check logs: docker-compose logs -f")
    if not domain_ok:
        print("   1. Configure DNS records to point to your server")
        print("   2. Set up SSL certificates with Let's Encrypt")
        print("   3. Update environment variables with your API keys")
    
    # Overall status
    if local_ok or domain_ok:
        log("ScrollIntel.com is accessible! 🎉", "SUCCESS")
        return 0
    else:
        log("ScrollIntel.com is not accessible. Check the issues above.", "ERROR")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Status check interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)