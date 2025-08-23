#!/usr/bin/env python3
"""
ScrollIntel Local Development Launcher
Starts both backend and frontend with SQLite database for local development
"""

import os
import sys
import time
import subprocess
import threading
import signal
from pathlib import Path

def setup_local_environment():
    """Setup environment variables for local development"""
    os.environ.update({
        'ENVIRONMENT': 'development',
        'DEBUG': 'true',
        'DATABASE_URL': 'sqlite:///./scrollintel_dev.db',
        'API_HOST': '127.0.0.1',
        'API_PORT': '8000',
        'NEXT_PUBLIC_API_URL': 'http://localhost:8000',
        'NEXT_PUBLIC_WS_URL': 'ws://localhost:8000',
        'ENABLE_CORS': 'true',
        'CORS_ORIGINS': 'http://localhost:3000,http://localhost:3001',
        'LOG_LEVEL': 'INFO'
    })

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting ScrollIntel Backend...")
    
    # Change to project root
    os.