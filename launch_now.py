#!/usr/bin/env python3
"""
ScrollIntel Quick Launch - Get running in 30 seconds!
Minimal dependencies, maximum functionality
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    print("🚀 ScrollIntel Quick Launch")
    print("=" * 40)
    print("Getting you running in 30 seconds!")
    print()

def install_minimal_deps():
    """Install only essential dependencies"""
    print("📦 Installing minimal dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0", 
            "pydantic>=2.5.0",
            "python-dotenv>=1.0.0",
            "sqlalchemy>=2.0.0",
            "aiosqlite>=0.19.0",
            "openai>=1.3.0",
            "pandas>=2.1.0",
            "numpy>=1.24.0",
            "python-jose[cryptography]>=3.3.0",
            "passlib[bcrypt]>=1.7.4",
            "python-multipart>=0.0.6",
            "httpx>=0.25.0",
            "aiofiles>=23.2.0",
            "requests>=2.31.0"
        ], check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("Trying alternative approach...")
        return False

def setup_env():
    """Setup environment file"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating environment file...")
        with open(env_file, 'w') as f:
            f.write("""# ScrollIntel Environment
ENVIRONMENT=development
DEBUG=true
DATABASE_URL=sqlite:///./scrollintel.db
JWT_SECRET_KEY=your_jwt_secret_key_here_change_in_production
OPENAI_API_KEY=your_openai_api_key_here
""")
        print("✅ Environment file created!")
    else:
        print("✅ Environment file exists!")

def start_app():
    """Start the ScrollIntel application"""
    print("🚀 Starting ScrollIntel...")
    print("📍 Backend will be available at: http://localhost:8000")
    print("📚 API docs will be at: http://localhost:8000/docs")
    print("❤️  Health check at: http://localhost:8000/health")
    print()
    print("🎉 ScrollIntel is starting!")
    print("Press Ctrl+C to stop")
    print()
    
    # Set environment
    os.environ.setdefault("PYTHONPATH", ".")
    
    try:
        # Import and run directly
        import uvicorn
        uvicorn.run(
            "scrollintel.api.simple_main:app",
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False
        )
    except ImportError:
        print("❌ FastAPI not available. Installing now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"], check=True)
        import uvicorn
        uvicorn.run(
            "scrollintel.api.simple_main:app",
            host="127.0.0.1",
            port=8000,
            log_level="info",
            reload=False
        )
    except KeyboardInterrupt:
        print("\n👋 ScrollIntel stopped")
    except Exception as e:
        print(f"❌ Error starting app: {e}")
        print("💡 Try running: python scrollintel/api/simple_main.py")

def main():
    print_banner()
    
    # Install minimal dependencies
    if not install_minimal_deps():
        print("⚠️ Continuing with existing packages...")
    
    # Setup environment
    setup_env()
    
    # Start the app
    start_app()

if __name__ == "__main__":
    main()