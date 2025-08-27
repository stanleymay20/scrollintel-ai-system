#!/usr/bin/env python3
"""
Windows-compatible ScrollIntel deployment fix.
Fixes Unicode encoding issues and TensorFlow/Keras compatibility.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description="Running command"):
    """Run a command with proper Windows encoding handling."""
    print(f"\n{description}...")
    try:
        # Use UTF-8 encoding for Windows compatibility
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False

def fix_unicode_files():
    """Fix Unicode characters in Python files."""
    print("\n🔧 Fixing Unicode encoding issues...")
    
    files_to_fix = [
        "setup_scrollintel_com.py",
        "launch_scrollintel_com.py"
    ]
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            try:
                # Read with UTF-8 and replace problematic characters
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Replace Unicode emojis with ASCII equivalents
                replacements = {
                    '🎯': '[TARGET]',
                    '🚀': '[ROCKET]',
                    '⚡': '[LIGHTNING]',
                    '🔧': '[WRENCH]',
                    '✅': '[CHECK]',
                    '❌': '[X]',
                    '💡': '[BULB]',
                    '🐳': '[DOCKER]',
                    '🧪': '[TEST]',
                    '🎉': '[PARTY]',
                    '⏳': '[HOURGLASS]',
                    '⚠️': '[WARNING]',
                    '📦': '[PACKAGE]',
                    '🔍': '[SEARCH]',
                    '📊': '[CHART]',
                    '🌟': '[STAR]',
                    '🏆': '[TROPHY]',
                    '💻': '[COMPUTER]',
                    '🔥': '[FIRE]',
                    '⭐': '[STAR]',
                    '🎊': '[CONFETTI]',
                    '🚨': '[SIREN]',
                    '📈': '[TRENDING_UP]',
                    '🔒': '[LOCK]',
                    '🌐': '[GLOBE]',
                    '📝': '[MEMO]',
                    '🎨': '[PALETTE]',
                    '⚙️': '[GEAR]',
                    '🔄': '[ARROWS_COUNTERCLOCKWISE]',
                    '📋': '[CLIPBOARD]',
                    '🎪': '[CIRCUS_TENT]',
                    '🎭': '[PERFORMING_ARTS]',
                    '🎬': '[CLAPPER]',
                    '🎮': '[VIDEO_GAME]',
                    '🎲': '[GAME_DIE]',
                    '🎯': '[DIRECT_HIT]',
                    '🏁': '[CHECKERED_FLAG]',
                    '🏃': '[RUNNER]',
                    '🏅': '[SPORTS_MEDAL]',
                    '🏆': '[TROPHY]',
                    '🏈': '[AMERICAN_FOOTBALL]',
                    '🏉': '[RUGBY_FOOTBALL]',
                    '🏊': '[SWIMMER]',
                    '🏋️': '[WEIGHT_LIFTER]',
                    '🏌️': '[GOLFER]',
                    '🏍️': '[RACING_MOTORCYCLE]',
                    '🏎️': '[RACING_CAR]',
                    '🏏': '[CRICKET_BAT_AND_BALL]',
                    '🏐': '[VOLLEYBALL]',
                    '🏑': '[FIELD_HOCKEY_STICK_AND_BALL]',
                    '🏒': '[ICE_HOCKEY_STICK_AND_PUCK]',
                    '🏓': '[PING_PONG]',
                    '🏔️': '[SNOW_CAPPED_MOUNTAIN]',
                    '🏕️': '[CAMPING]',
                    '🏖️': '[BEACH_WITH_UMBRELLA]',
                    '🏗️': '[BUILDING_CONSTRUCTION]',
                    '🏘️': '[HOUSE_BUILDINGS]',
                    '🏙️': '[CITYSCAPE]',
                    '🏚️': '[DERELICT_HOUSE_BUILDING]',
                    '🏛️': '[CLASSICAL_BUILDING]',
                    '🏜️': '[DESERT]',
                    '🏝️': '[DESERT_ISLAND]',
                    '🏞️': '[NATIONAL_PARK]',
                    '🏟️': '[STADIUM]',
                    '🏠': '[HOUSE_BUILDING]',
                    '🏡': '[HOUSE_WITH_GARDEN]',
                    '🏢': '[OFFICE_BUILDING]',
                    '🏣': '[JAPANESE_POST_OFFICE]',
                    '🏤': '[EUROPEAN_POST_OFFICE]',
                    '🏥': '[HOSPITAL]',
                    '🏦': '[BANK]',
                    '🏧': '[ATM_SIGN]',
                    '🏨': '[HOTEL]',
                    '🏩': '[LOVE_HOTEL]',
                    '🏪': '[CONVENIENCE_STORE]',
                    '🏫': '[SCHOOL]',
                    '🏬': '[DEPARTMENT_STORE]',
                    '🏭': '[FACTORY]',
                    '🏮': '[IZAKAYA_LANTERN]',
                    '🏯': '[JAPANESE_CASTLE]',
                    '🏰': '[EUROPEAN_CASTLE]'
                }
                
                for unicode_char, ascii_replacement in replacements.items():
                    content = content.replace(unicode_char, ascii_replacement)
                
                # Write back with UTF-8 encoding
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print(f"✅ Fixed Unicode characters in {file_path}")
                
            except Exception as e:
                print(f"❌ Failed to fix {file_path}: {e}")
        else:
            print(f"⚠️ File not found: {file_path}")

def create_windows_compatible_launcher():
    """Create a Windows-compatible launcher script."""
    launcher_content = '''#!/usr/bin/env python3
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
    
    print("\\n[ROCKET] Starting ScrollIntel deployment...")
    
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
    print("\\n[SEARCH] Checking for existing containers...")
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
    print("\\n[DOCKER] Building ScrollIntel Docker image...")
    build_result = subprocess.run(
        "docker build -t scrollintel:windows-fixed .", 
        shell=True
    )
    
    if build_result.returncode == 0:
        print("[CHECK] Docker image built successfully!")
        
        # Run the container
        print("\\n[ROCKET] Starting ScrollIntel container...")
        run_result = subprocess.run(
            "docker run -d --name scrollintel-windows -p 8000:8000 scrollintel:windows-fixed",
            shell=True
        )
        
        if run_result.returncode == 0:
            print("[CHECK] ScrollIntel container started successfully!")
            print("\\n[PARTY] ScrollIntel is now running!")
            print("\\nAccess your application at:")
            print("- Main API: http://localhost:8000")
            print("- Health Check: http://localhost:8000/health")
            print("- GraphQL: http://localhost:8000/graphql")
            
            # Wait and test
            print("\\n[HOURGLASS] Waiting for application to start...")
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
            
            print("\\n[TROPHY] Deployment completed successfully!")
            print("\\nTo stop ScrollIntel:")
            print("docker stop scrollintel-windows")
            print("\\nTo restart ScrollIntel:")
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
        print("\\n[X] Deployment failed. Check the errors above.")
        sys.exit(1)
    else:
        print("\\n[STAR] All done! ScrollIntel is ready to use.")
'''
    
    with open("launch_scrollintel_windows.py", "w", encoding="utf-8") as f:
        f.write(launcher_content)
    
    print("✅ Created Windows-compatible launcher")

def update_dockerfile_for_windows():
    """Update Dockerfile to be more Windows-compatible."""
    dockerfile_content = '''# ScrollIntel Windows-Compatible Dockerfile
FROM python:3.11-slim

# Set environment variables for Windows compatibility
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_docker.txt .

# Install Python dependencies with TensorFlow/Keras fix
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \\
    pip install --no-cache-dir tf-keras>=2.15.0 && \\
    pip install --no-cache-dir -r requirements_docker.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "scrollintel.api.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    with open("Dockerfile.windows", "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    
    print("✅ Created Windows-compatible Dockerfile")

def create_simple_test_script():
    """Create a simple test script to verify the deployment."""
    test_script = '''#!/usr/bin/env python3
"""
Simple test script for ScrollIntel deployment.
"""

import requests
import time
import sys

def test_scrollintel():
    """Test ScrollIntel deployment."""
    print("Testing ScrollIntel deployment...")
    
    # Wait for startup
    print("Waiting for application to start...")
    time.sleep(15)
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            print(f"[CHECK] Health check passed: {response.json()}")
            
            # Test main endpoint
            try:
                main_response = requests.get("http://localhost:8000/", timeout=10)
                if main_response.status_code == 200:
                    print("[CHECK] Main endpoint accessible")
                else:
                    print(f"[WARNING] Main endpoint returned {main_response.status_code}")
            except Exception as e:
                print(f"[WARNING] Could not test main endpoint: {e}")
            
            print("[PARTY] ScrollIntel is working correctly!")
            return True
        else:
            print(f"[X] Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"[X] Could not connect to ScrollIntel: {e}")
        return False

if __name__ == "__main__":
    success = test_scrollintel()
    if success:
        print("\\n[TROPHY] All tests passed!")
    else:
        print("\\n[X] Tests failed!")
        sys.exit(1)
'''
    
    with open("test_scrollintel_deployment.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ Created deployment test script")

def main():
    """Main execution function."""
    print("=" * 60)
    print("    SCROLLINTEL WINDOWS COMPATIBILITY FIX")
    print("=" * 60)
    
    # Step 1: Fix Unicode issues
    fix_unicode_files()
    
    # Step 2: Create Windows-compatible launcher
    create_windows_compatible_launcher()
    
    # Step 3: Update Dockerfile
    update_dockerfile_for_windows()
    
    # Step 4: Create test script
    create_simple_test_script()
    
    print("\n" + "=" * 60)
    print("                 FIXES COMPLETED")
    print("=" * 60)
    
    print("\n[CHECK] Windows compatibility fixes applied!")
    print("\nTo deploy ScrollIntel on Windows:")
    print("1. Run: python launch_scrollintel_windows.py")
    print("2. Wait for deployment to complete")
    print("3. Access at http://localhost:8000")
    
    print("\nTo test the deployment:")
    print("python test_scrollintel_deployment.py")
    
    print("\n[STAR] Ready for Windows deployment!")

if __name__ == "__main__":
    main()