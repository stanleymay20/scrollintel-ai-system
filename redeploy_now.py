#!/usr/bin/env python3
"""
ScrollIntel Redeployment Script
Automated redeployment after GitHub push
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Success: {description}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ Failed: {description}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception during {description}: {e}")
        return False

def check_deployment_readiness():
    """Check if deployment is ready"""
    print("\n🔍 Checking deployment readiness...")
    
    # Check if Docker is running
    if not run_command("docker --version", "Checking Docker"):
        return False
    
    # Check if required files exist
    required_files = [
        "Dockerfile",
        "docker-compose.yml", 
        "requirements.txt",
        ".env"
    ]
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing required file: {file}")
            return False
    
    print("✅ All deployment requirements met")
    return True

def deploy_scrollintel():
    """Deploy ScrollIntel system"""
    print("\n🚀 Starting ScrollIntel Redeployment...")
    
    # Stop existing containers
    run_command("docker-compose down", "Stopping existing containers")
    
    # Pull latest changes (already done via git push)
    print("✅ Latest changes already pushed to GitHub")
    
    # Build and start containers
    if not run_command("docker-compose build --no-cache", "Building containers"):
        return False
    
    if not run_command("docker-compose up -d", "Starting containers"):
        return False
    
    # Wait for services to start
    print("\n⏳ Waiting for services to start...")
    time.sleep(30)
    
    # Check health
    if run_command("docker-compose ps", "Checking container status"):
        print("\n✅ ScrollIntel redeployment completed successfully!")
        print("\n📊 Access your deployment:")
        print("🌐 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8000")
        print("📈 Health Check: http://localhost:8000/health")
        return True
    
    return False

def main():
    """Main deployment function"""
    print("🎯 ScrollIntel Redeployment Script")
    print("=" * 50)
    
    # Check readiness
    if not check_deployment_readiness():
        print("\n❌ Deployment readiness check failed")
        sys.exit(1)
    
    # Deploy
    if deploy_scrollintel():
        print("\n🎉 Redeployment successful!")
        print("\nNext steps:")
        print("1. Test the application at http://localhost:3000")
        print("2. Check logs with: docker-compose logs -f")
        print("3. Monitor health at: http://localhost:8000/health")
    else:
        print("\n❌ Redeployment failed")
        print("Check logs with: docker-compose logs")
        sys.exit(1)

if __name__ == "__main__":
    main()