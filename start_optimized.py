#!/usr/bin/env python3
"""
Optimized ScrollIntel Startup Script
"""
import os
import sys
import asyncio
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def optimize_startup():
    """Optimize startup performance"""
    print("🚀 Starting ScrollIntel with optimizations...")
    
    # Memory optimization
    try:
        from scrollintel.core.memory_optimizer import start_memory_monitoring, optimize_memory
        optimize_memory()
        start_memory_monitoring()
        print("  ✅ Memory optimization enabled")
    except ImportError:
        print("  ⚠️ Memory optimizer not available")
    
    # Lazy configuration
    try:
        from scrollintel.core.optimized_config import get_settings
        settings = get_settings()
        print(f"  ✅ Configuration loaded (Environment: {settings.ENVIRONMENT})")
    except ImportError:
        print("  ⚠️ Using standard configuration")
    
    # Test agent system
    try:
        from scrollintel.agents.concrete_agent import QuickTestAgent
        agent = QuickTestAgent()
        print(f"  ✅ Agent system working: {agent.name}")
    except ImportError:
        print("  ⚠️ Agent system not available")

async def start_application():
    """Start the application with optimizations"""
    optimize_startup()
    
    try:
        # Import and start FastAPI app
        from scrollintel.api.main import app
        import uvicorn
        
        print("🌐 Starting FastAPI server...")
        
        # Get configuration
        try:
            from scrollintel.core.optimized_config import get_settings
            settings = get_settings()
            host = settings.API_HOST
            port = settings.API_PORT
        except:
            host = "0.0.0.0"
            port = 8000
        
        # Start server
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=False  # Reduce logging overhead
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        return 1
    
    return 0

def main():
    """Main function"""
    try:
        return asyncio.run(start_application())
    except KeyboardInterrupt:
        print("\n👋 ScrollIntel shutdown complete")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
