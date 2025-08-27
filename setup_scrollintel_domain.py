#!/usr/bin/env python3
"""
Quick Setup for ScrollIntel.com Domain Deployment
"""

import os
import secrets
import string

def generate_secure_key(length=64):
    """Generate a secure random key"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def setup_environment():
    """Setup environment variables for scrollintel.com"""
    
    print("🔧 Setting up ScrollIntel.com environment...")
    
    # Generate secure JWT secret
    jwt_secret = generate_secure_key(64)
    
    # Create production environment file
    env_content = f"""# ScrollIntel.com Production Environment
# Generated on: {os.popen('date').read().strip()}

# ===== REQUIRED: AI Services =====
# Get your OpenAI API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-api-key-here

# ===== REQUIRED: Security =====
JWT_SECRET_KEY={jwt_secret}

# ===== REQUIRED: Database =====
# For local development (will auto-create SQLite)
DATABASE_URL=sqlite:///./scrollintel.db

# For production PostgreSQL (uncomment and configure):
# DATABASE_URL=postgresql://username:password@localhost:5432/scrollintel

# ===== Optional: Advanced Features =====
# Redis for caching (optional)
REDIS_URL=redis://localhost:6379

# Email notifications (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password

# ===== Domain Configuration =====
DOMAIN=scrollintel.com
API_DOMAIN=api.scrollintel.com
APP_DOMAIN=app.scrollintel.com

# ===== Production Settings =====
NODE_ENV=production
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# CORS origins
CORS_ORIGINS=https://scrollintel.com,https://app.scrollintel.com

# ===== Monitoring =====
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true

# ===== Storage =====
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY={generate_secure_key(32)}

# ===== Database Credentials =====
POSTGRES_USER=scrollintel
POSTGRES_PASSWORD={generate_secure_key(32)}
POSTGRES_DB=scrollintel

# ===== Grafana =====
GRAFANA_PASSWORD={generate_secure_key(16)}
"""
    
    # Write environment file
    with open('.env.scrollintel.com', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env.scrollintel.com")
    
    # Create quick start script
    quick_start = """#!/bin/bash
# ScrollIntel.com Quick Start

echo "🚀 Starting ScrollIntel for scrollintel.com..."

# Load environment
export $(cat .env.scrollintel.com | grep -v '^#' | xargs)

# Start with simple backend first
echo "📡 Starting API server..."
python -c "
import uvicorn
from scrollintel.api.simple_main import app
print('🌐 ScrollIntel API starting at http://localhost:8000')
print('📖 API docs available at http://localhost:8000/docs')
print('🔍 Health check at http://localhost:8000/health')
uvicorn.run(app, host='0.0.0.0', port=8000)
"
"""
    
    with open('start_scrollintel_domain.sh', 'w') as f:
        f.write(quick_start)
    
    os.chmod('start_scrollintel_domain.sh', 0o755)
    
    print("✅ Created start_scrollintel_domain.sh")
    
    # Create Windows batch file
    windows_start = """@echo off
echo 🚀 Starting ScrollIntel for scrollintel.com...

REM Load environment from file
for /f "tokens=*" %%i in (.env.scrollintel.com) do set %%i

echo 📡 Starting API server...
python -c "import uvicorn; from scrollintel.api.simple_main import app; print('🌐 ScrollIntel API starting at http://localhost:8000'); print('📖 API docs available at http://localhost:8000/docs'); print('🔍 Health check at http://localhost:8000/health'); uvicorn.run(app, host='0.0.0.0', port=8000)"

pause
"""
    
    with open('start_scrollintel_domain.bat', 'w') as f:
        f.write(windows_start)
    
    print("✅ Created start_scrollintel_domain.bat")
    
    print("\n🎯 Next Steps:")
    print("1. Edit .env.scrollintel.com and add your OpenAI API key")
    print("2. Run: python start_scrollintel_domain.py (or .bat on Windows)")
    print("3. Test locally at http://localhost:8000")
    print("4. Deploy to your server with the domain pointing to it")
    
    return True

def create_simple_launcher():
    """Create a simple Python launcher"""
    launcher_content = """#!/usr/bin/env python3
'''
ScrollIntel.com Simple Launcher
Start ScrollIntel locally for testing before domain deployment
'''

import os
import sys
import uvicorn
from pathlib import Path

def load_env_file(env_file='.env.scrollintel.com'):
    '''Load environment variables from file'''
    if Path(env_file).exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✅ Loaded environment from {env_file}")
    else:
        print(f"⚠️  Environment file {env_file} not found")

def main():
    print("🚀 ScrollIntel.com Launcher")
    print("=" * 50)
    
    # Load environment
    load_env_file()
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'sk-your-openai-api-key-here':
        print("⚠️  Please set your OPENAI_API_KEY in .env.scrollintel.com")
        print("   Get your API key from: https://platform.openai.com/api-keys")
        print()
    
    try:
        # Import the app
        from scrollintel.api.simple_main import app
        
        print("🌐 Starting ScrollIntel API server...")
        print("📡 API: http://localhost:8000")
        print("📖 Docs: http://localhost:8000/docs") 
        print("🔍 Health: http://localhost:8000/health")
        print("🤖 Chat with AI agents at /chat")
        print("📊 Upload files at /upload")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 50)
        
        # Start the server
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=8000,
            reload=False,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the ScrollIntel directory")
    except KeyboardInterrupt:
        print("\\n👋 ScrollIntel stopped")
    except Exception as e:
        print(f"❌ Error starting ScrollIntel: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open('start_scrollintel_domain.py', 'w') as f:
        f.write(launcher_content)
    
    print("✅ Created start_scrollintel_domain.py")

if __name__ == "__main__":
    setup_environment()
    create_simple_launcher()
    
    print("\n🎉 ScrollIntel.com setup complete!")
    print("\n📋 Quick Start:")
    print("1. Edit .env.scrollintel.com - add your OpenAI API key")
    print("2. Run: python start_scrollintel_domain.py")
    print("3. Open: http://localhost:8000")
    print("4. Test the AI agents and file upload")
    print("5. Deploy to your server when ready!")