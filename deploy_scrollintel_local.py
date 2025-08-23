#!/usr/bin/env python3
"""
ScrollIntel Local Deployment Script
Deploy ScrollIntel locally using Docker for immediate testing
"""

import os
import subprocess
import time
import requests
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_docker():
    """Check if Docker is running"""
    print("🐳 Checking Docker...")
    try:
        result = subprocess.run("docker info", shell=True, check=True, capture_output=True, text=True)
        print("✅ Docker is running")
        return True
    except subprocess.CalledProcessError:
        print("❌ Docker is not running. Please start Docker Desktop.")
        return False

def wait_for_service(url, service_name, max_attempts=30):
    """Wait for a service to be ready"""
    print(f"⏳ Waiting for {service_name} to be ready...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print(f"❌ {service_name} failed to start within {max_attempts * 2} seconds")
    return False

def main():
    """Main deployment function"""
    
    print("🚀 ScrollIntel Local Deployment")
    print("=" * 50)
    
    # Check Docker
    if not check_docker():
        return False
    
    # Stop any existing containers
    print("🛑 Stopping existing containers...")
    subprocess.run("docker-compose down", shell=True, capture_output=True)
    
    # Build and start services
    if not run_command("docker-compose build", "Building Docker images"):
        return False
    
    if not run_command("docker-compose up -d", "Starting services"):
        return False
    
    # Wait for services to be ready
    print("\n⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Check service health
    services = [
        ("http://localhost:8000/health", "Backend API"),
        ("http://localhost:3000", "Frontend")
    ]
    
    all_ready = True
    for url, name in services:
        if not wait_for_service(url, name):
            all_ready = False
    
    if all_ready:
        print("\n" + "=" * 50)
        print("🎉 ScrollIntel Deployed Successfully!")
        print("\n🌐 Access URLs:")
        print("   • Frontend: http://localhost:3000")
        print("   • Backend API: http://localhost:8000")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Health Check: http://localhost:8000/health")
        
        print("\n📊 Service Status:")
        subprocess.run("docker-compose ps", shell=True)
        
        print("\n💡 Next Steps:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Test the API at http://localhost:8000/docs")
        print("   3. Check logs: docker-compose logs -f")
        print("   4. Stop services: docker-compose down")
        
        return True
    else:
        print("\n❌ Some services failed to start")
        print("🔍 Check logs: docker-compose logs")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)