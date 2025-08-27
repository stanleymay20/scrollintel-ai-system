#!/usr/bin/env python3
"""
Start ScrollIntel backend from current directory
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

print("🚀 Starting ScrollIntel Backend...")
print(f"📁 Working directory: {current_dir}")
print("🌐 API will be available at: http://localhost:8000")
print("📚 API docs will be available at: http://localhost:8000/docs")
print()

try:
    import uvicorn
    
    # Start the server
    uvicorn.run(
        "scrollintel.api.simple_main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
    
except ImportError:
    print("Installing uvicorn...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "uvicorn[standard]"])
    
    import uvicorn
    uvicorn.run(
        "scrollintel.api.simple_main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
    
except Exception as e:
    print(f"❌ Error starting backend: {e}")
    print("💡 Try installing requirements: pip install -r requirements.txt")