#!/usr/bin/env python3
"""
Simple ScrollIntel Launcher
Starts the application without Docker for development
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file from template...")
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
        else:
            # Create minimal .env
            with open(env_file, 'w') as f:
                f.write("""# ScrollIntel Environment
ENVIRONMENT=development
DEBUG=true
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scrollintel
POSTGRES_USER=postgres
POSTGRES_PASSWORD=scrollintel_password
REDIS_HOST=localhost
REDIS_PORT=6379
JWT_SECRET_KEY=your_jwt_secret_key_here
OPENAI_API_KEY=your_openai_api_key_here
""")
    print("✅ Environment configured")

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    return True

def start_database():
    """Start PostgreSQL database (if available)"""
    print("🗄️ Checking database...")
    
    # Check if DATABASE_URL is already set
    database_url = os.environ.get("DATABASE_URL")
    if database_url and database_url.startswith("postgresql"):
        try:
            import psycopg2
            # Parse the DATABASE_URL
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                database=parsed.path[1:],  # Remove leading slash
                user=parsed.username,
                password=parsed.password
            )
            conn.close()
            print("✅ Database connection successful (PostgreSQL)")
            return True
        except Exception as e:
            print(f"⚠️ PostgreSQL connection failed: {e}")
            print("⚠️ Falling back to SQLite")
            os.environ["DATABASE_URL"] = "sqlite:///./scrollintel.db"
            return True
    else:
        # Try default PostgreSQL connection
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="scrollintel",
                user="postgres",
                password="boatemaa1612"  # Updated to match .env
            )
            conn.close()
            print("✅ Database connection successful (PostgreSQL)")
            os.environ["DATABASE_URL"] = "postgresql://postgres:boatemaa1612@localhost:5432/scrollintel"
            return True
        except Exception as e:
            print(f"⚠️ Database not available - using SQLite fallback: {e}")
            os.environ["DATABASE_URL"] = "sqlite:///./scrollintel.db"
            return True

def initialize_database():
    """Initialize database schema"""
    print("🔧 Initializing database...")
    try:
        # For simple mode, we'll skip complex database initialization
        # The simple_main.py uses in-memory storage
        print("✅ Database initialized (simple mode)")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")
        # Continue anyway - might work

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting ScrollIntel backend...")
    
    # Set environment variables
    os.environ.setdefault("PYTHONPATH", ".")
    
    try:
        print("✅ Backend starting on http://localhost:8000")
        print("📚 API docs available at http://localhost:8000/docs")
        print("🏥 Health check at http://localhost:8000/health")
        print("\n🎉 ScrollIntel is ready!")
        print("Press Ctrl+C to stop")
        
        # Import and run the app
        import uvicorn
        
        # Run the simple FastAPI app directly
        uvicorn.run(
            "scrollintel.api.simple_main:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 ScrollIntel stopped")
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🌟 ScrollIntel Simple Launcher")
    print("=" * 40)
    
    # Check requirements
    check_python_version()
    setup_environment()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Setup database
    start_database()
    initialize_database()
    
    # Start backend
    start_backend()

if __name__ == "__main__":
    main()