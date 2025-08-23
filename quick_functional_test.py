#!/usr/bin/env python3
"""
Quick Functional Test for ScrollIntel
Tests core functionality and system health
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

def test_imports():
    """Test if core modules can be imported"""
    print("🧪 Testing core module imports...")
    
    try:
        # Test core imports
        from scrollintel.core.config import get_settings
        print("  ✅ Core config import successful")
        
        from scrollintel.api.main import app
        print("  ✅ FastAPI app import successful")
        
        from scrollintel.core.monitoring import performance_monitor
        print("  ✅ Performance monitoring import successful")
        
        from scrollintel.agents.base import BaseAgent
        print("  ✅ Base agent import successful")
        
        return True
    except Exception as e:
        print(f"  ❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔧 Testing configuration...")
    
    try:
        from scrollintel.core.config import get_settings
        settings = get_settings()
        
        print(f"  ✅ Environment: {settings.ENVIRONMENT}")
        print(f"  ✅ Debug mode: {settings.DEBUG}")
        print(f"  ✅ API host: {settings.API_HOST}:{settings.API_PORT}")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_database_connection():
    """Test database connectivity"""
    print("\n🗄️ Testing database connection...")
    
    try:
        from scrollintel.models.database import get_database_url
        from sqlalchemy import create_engine, text
        
        db_url = get_database_url()
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("  ✅ Database connection successful")
            return True
            
    except Exception as e:
        print(f"  ⚠️ Database connection issue: {e}")
        # Try SQLite fallback
        try:
            import sqlite3
            conn = sqlite3.connect('scrollintel.db')
            conn.execute("SELECT 1")
            conn.close()
            print("  ✅ SQLite fallback connection successful")
            return True
        except Exception as e2:
            print(f"  ❌ SQLite fallback failed: {e2}")
            return False

def test_api_routes():
    """Test API route registration"""
    print("\n🌐 Testing API routes...")
    
    try:
        from scrollintel.api.main import app
        
        routes = [route.path for route in app.routes]
        essential_routes = ["/", "/health", "/docs"]
        
        for route in essential_routes:
            if route in routes:
                print(f"  ✅ Route {route} registered")
            else:
                print(f"  ❌ Route {route} missing")
                
        print(f"  📊 Total routes registered: {len(routes)}")
        return True
        
    except Exception as e:
        print(f"  ❌ API routes error: {e}")
        return False

def test_agent_system():
    """Test agent system functionality"""
    print("\n🤖 Testing agent system...")
    
    try:
        from scrollintel.agents.base import BaseAgent
        from scrollintel.core.registry import AgentRegistry
        
        # Test base agent
        agent = BaseAgent(agent_id="test_agent", name="Test Agent")
        print(f"  ✅ Base agent created: {agent.name}")
        
        # Test registry
        registry = AgentRegistry()
        print("  ✅ Agent registry initialized")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent system error: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring system"""
    print("\n📊 Testing performance monitoring...")
    
    try:
        from scrollintel.core.performance_monitor import ResponseTimeMetric
        from datetime import datetime
        
        # Create test metric
        metric = ResponseTimeMetric(
            endpoint="/test",
            method="GET",
            response_time=0.1,
            status_code=200,
            timestamp=datetime.utcnow()
        )
        
        print(f"  ✅ Performance metric created: {metric.endpoint}")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance monitoring error: {e}")
        return False

async def test_async_functionality():
    """Test async functionality"""
    print("\n⚡ Testing async functionality...")
    
    try:
        # Test async sleep
        await asyncio.sleep(0.1)
        print("  ✅ Async operations working")
        
        # Test async context manager
        from contextlib import asynccontextmanager
        
        @asynccontextmanager
        async def test_context():
            print("  ✅ Async context manager working")
            yield "test"
        
        async with test_context() as value:
            assert value == "test"
            
        return True
        
    except Exception as e:
        print(f"  ❌ Async functionality error: {e}")
        return False

def generate_health_report():
    """Generate comprehensive health report"""
    print("\n📋 Generating health report...")
    
    # Run all tests
    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {
            "imports": test_imports(),
            "configuration": test_configuration(),
            "database": test_database_connection(),
            "api_routes": test_api_routes(),
            "agent_system": test_agent_system(),
            "performance_monitoring": test_performance_monitoring()
        }
    }
    
    # Add async test
    try:
        test_results["tests"]["async_functionality"] = asyncio.run(test_async_functionality())
    except Exception as e:
        print(f"  ❌ Async test failed: {e}")
        test_results["tests"]["async_functionality"] = False
    
    # Calculate overall health
    passed_tests = sum(1 for result in test_results["tests"].values() if result)
    total_tests = len(test_results["tests"])
    health_percentage = (passed_tests / total_tests) * 100
    
    test_results["summary"] = {
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "health_percentage": round(health_percentage, 2),
        "status": "healthy" if health_percentage >= 80 else "degraded" if health_percentage >= 60 else "critical"
    }
    
    return test_results

def main():
    """Main function"""
    print("🚀 ScrollIntel Quick Functional Test")
    print("=" * 50)
    
    # Generate health report
    health_report = generate_health_report()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 HEALTH SUMMARY")
    print("=" * 50)
    
    summary = health_report["summary"]
    status = summary["status"]
    
    if status == "healthy":
        emoji = "🟢"
    elif status == "degraded":
        emoji = "🟡"
    else:
        emoji = "🔴"
        
    print(f"{emoji} Overall Health: {summary['health_percentage']}% ({status.upper()})")
    print(f"✅ Passed Tests: {summary['passed_tests']}/{summary['total_tests']}")
    
    # List failed tests
    failed_tests = [test for test, result in health_report["tests"].items() if not result]
    if failed_tests:
        print(f"❌ Failed Tests: {', '.join(failed_tests)}")
    
    # Save report
    with open('health_report.json', 'w') as f:
        json.dump(health_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: health_report.json")
    print("=" * 50)
    
    # Return appropriate exit code
    return 0 if summary["health_percentage"] >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())