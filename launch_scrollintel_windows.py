#!/usr/bin/env python3
"""
Windows-compatible ScrollIntel launcher.
"""

import os
import sys
import subprocess
import time

def main():
    """Main launcher function."""
    print("=" * 60)
    print("           SCROLLINTEL WINDOWS LAUNCHER")
    print("=" * 60)
    
    print("\n[ROCKET] Starting ScrollIntel deployment...")
    
    # Check if Docker is running
    try:
        result = subprocess.run(
            "docker --version", 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"[CHECK] Docker is available: {result.stdout.strip()}")
    except subprocess.CalledProcessError:
        print("[X] Docker is not available. Please install Docker Desktop.")
        return False
    
    # Check for existing container
    print("\n[SEARCH] Checking for existing containers...")
    try:
        result = subprocess.run(
            "docker ps -a --filter name=scrollintel", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        if "scrollintel" in result.stdout:
            print("[CHECK] Found existing ScrollIntel containers")
            
            # Stop existing containers
            print("[WRENCH] Stopping existing containers...")
            subprocess.run("docker stop scrollintel-test 2>nul || echo No test container", shell=True)
            subprocess.run("docker stop scrollintel-graphql-test 2>nul || echo No GraphQL container", shell=True)
            subprocess.run("docker rm scrollintel-test 2>nul || echo No test container", shell=True)
            subprocess.run("docker rm scrollintel-graphql-test 2>nul || echo No GraphQL container", shell=True)
    except Exception as e:
        print(f"[WARNING] Could not check containers: {e}")
    
    # Build new image
    print("\n[DOCKER] Building ScrollIntel Docker image...")
    build_result = subprocess.run(
        "docker build -t scrollintel:windows-fixed .", 
        shell=True
    )
    
    if build_result.returncode == 0:
        print("[CHECK] Docker image built successfully!")
        
        # Run the container
        print("\n[ROCKET] Starting ScrollIntel container...")
        run_result = subprocess.run(
            "docker run -d --name scrollintel-windows -p 8000:8000 scrollintel:windows-fixed",
            shell=True
        )
        
        if run_result.returncode == 0:
            print("[CHECK] ScrollIntel container started successfully!")
            print("\n[PARTY] ScrollIntel is now running!")
            print("\nAccess your application at:")
            print("- Main API: http://localhost:8000")
            print("- Health Check: http://localhost:8000/health")
            print("- GraphQL: http://localhost:8000/graphql")
            
            # Wait and test
            print("\n[HOURGLASS] Waiting for application to start...")
            time.sleep(10)
            
            # Test health endpoint
            try:
                test_result = subprocess.run(
                    "curl -f http://localhost:8000/health",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if test_result.returncode == 0:
                    print("[CHECK] Health check passed!")
                    print(f"Response: {test_result.stdout}")
                else:
                    print("[WARNING] Health check failed, but container is running")
            except Exception as e:
                print(f"[WARNING] Could not test health endpoint: {e}")
            
            print("\n[TROPHY] Deployment completed successfully!")
            print("\nTo stop ScrollIntel:")
            print("docker stop scrollintel-windows")
            print("\nTo restart ScrollIntel:")
            print("docker start scrollintel-windows")
            
            return True
        else:
            print("[X] Failed to start container")
            return False
    else:
        print("[X] Failed to build Docker image")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n[X] Deployment failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\n[STAR] All done! ScrollIntel is ready to use.")
