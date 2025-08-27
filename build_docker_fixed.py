#!/usr/bin/env python3
"""
Simple Docker build fix for TensorFlow/Keras compatibility.
"""

import os
import subprocess
import sys

def run_command_simple(command):
    """Run a command with simple output handling."""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True)
        print("✅ Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        return False

def main():
    """Main execution function."""
    print("🚀 Building Fixed ScrollIntel Docker Image")
    print("=" * 50)
    
    # Step 1: Build the Docker image
    print("\n🐳 Building Docker image...")
    if run_command_simple("docker build -t scrollintel:fixed2025 ."):
        print("✅ Docker image built successfully!")
        
        # Step 2: Test the container
        print("\n🧪 Testing the container...")
        
        # Stop any existing container
        run_command_simple("docker stop scrollintel-test 2>nul || echo Container not running")
        run_command_simple("docker rm scrollintel-test 2>nul || echo Container not found")
        
        # Run the container
        if run_command_simple("docker run -d --name scrollintel-test -p 8001:8000 scrollintel:fixed2025"):
            print("✅ Container started successfully!")
            print("\n🎉 Your fixed Docker container is ready!")
            print("\nTo run it:")
            print("docker run -p 8000:8000 scrollintel:fixed2025")
            print("\nTo stop the test container:")
            print("docker stop scrollintel-test && docker rm scrollintel-test")
        else:
            print("❌ Container failed to start")
    else:
        print("❌ Docker build failed")

if __name__ == "__main__":
    main()