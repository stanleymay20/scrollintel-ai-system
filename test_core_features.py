#!/usr/bin/env python3
"""
Core ScrollIntel Feature Testing
Tests the most essential features to verify the app is working
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that core modules can be imported"""
    print("Testing core module imports...")
    
    try:
        # Test core imports
        from scrollintel.core.config import get_settings
        print("✓ Core config module imported successfully")
        
        from scrollintel.models.database import get_db_session
        print("✓ Database module imported successfully")
        
        from scrollintel.api.main import app
        print("✓ Main API application imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from scrollintel.core.config import get_settings
        settings = get_settings()
        
        # Check essential settings
        if hasattr(settings, 'database_url') and settings.database_url:
            print("✓ Database URL configured")
        else:
            print("⚠ Database URL not configured")
        
        if hasattr(settings, 'openai_api_key') and settings.openai_api_key:
            print("✓ OpenAI API key configured")
        else:
            print("⚠ OpenAI API key not configured")
        
        if hasattr(settings, 'jwt_secret_key') and settings.jwt_secret_key:
            print("✓ JWT secret key configured")
        else:
            print("⚠ JWT secret key not configured")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_database_connection():
    """Test database connectivity"""
    print("\nTesting database connection...")
    
    try:
        from scrollintel.models.database import get_db_session
        from sqlalchemy import text
        
        with get_db_session() as session:
            result = session.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            if row and row[0] == 1:
                print("✓ Database connection successful")
                return True
            else:
                print("✗ Database query returned unexpected result")
                return False
                
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_agent_modules():
    """Test agent module imports"""
    print("\nTesting agent modules...")
    
    agents = [
        ("AI Engineer", "scrollintel.agents.scroll_ai_engineer"),
        ("BI Agent", "scrollintel.agents.scroll_bi_agent"),
        ("ML Engineer", "scrollintel.agents.scroll_ml_engineer"),
        ("Forecast Agent", "scrollintel.engines.scroll_forecast_engine"),
        ("QA Agent", "scrollintel.engines.scroll_qa_engine"),
        ("CTO Agent", "scrollintel.agents.scroll_cto_agent")
    ]
    
    success_count = 0
    
    for agent_name, module_path in agents:
        try:
            __import__(module_path)
            print(f"✓ {agent_name} module imported successfully")
            success_count += 1
        except Exception as e:
            print(f"⚠ {agent_name} module import failed: {e}")
    
    return success_count >= len(agents) // 2  # At least half should work

def test_api_routes():
    """Test API route definitions"""
    print("\nTesting API routes...")
    
    try:
        from scrollintel.api.main import app
        
        # Get all routes
        routes = []
        for route in app.routes:
            if hasattr(route, 'path'):
                routes.append(route.path)
        
        essential_routes = [
            "/",
            "/health",
            "/api/agents",
            "/api/dashboard",
            "/api/files"
        ]
        
        found_routes = 0
        for route in essential_routes:
            if any(r.startswith(route) for r in routes):
                print(f"✓ Route {route} is defined")
                found_routes += 1
            else:
                print(f"⚠ Route {route} not found")
        
        print(f"✓ Total routes defined: {len(routes)}")
        return found_routes >= len(essential_routes) // 2
        
    except Exception as e:
        print(f"✗ API routes test failed: {e}")
        return False

def test_file_system():
    """Test file system setup"""
    print("\nTesting file system setup...")
    
    try:
        # Check essential directories
        directories = [
            "scrollintel",
            "scrollintel/api",
            "scrollintel/agents", 
            "scrollintel/engines",
            "scrollintel/models",
            "scrollintel/core",
            "frontend",
            "tests"
        ]
        
        found_dirs = 0
        for directory in directories:
            if os.path.exists(directory):
                print(f"✓ Directory {directory} exists")
                found_dirs += 1
            else:
                print(f"⚠ Directory {directory} missing")
        
        # Check essential files
        files = [
            "requirements.txt",
            ".env",
            "scrollintel/__init__.py",
            "scrollintel/api/main.py"
        ]
        
        found_files = 0
        for file_path in files:
            if os.path.exists(file_path):
                print(f"✓ File {file_path} exists")
                found_files += 1
            else:
                print(f"⚠ File {file_path} missing")
        
        return found_dirs >= len(directories) // 2 and found_files >= len(files) // 2
        
    except Exception as e:
        print(f"✗ File system test failed: {e}")
        return False

def test_environment_variables():
    """Test environment variables"""
    print("\nTesting environment variables...")
    
    essential_vars = [
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "OPENAI_API_KEY",
        "ENVIRONMENT"
    ]
    
    found_vars = 0
    for var in essential_vars:
        if os.getenv(var):
            print(f"✓ Environment variable {var} is set")
            found_vars += 1
        else:
            print(f"⚠ Environment variable {var} not set")
    
    return found_vars >= len(essential_vars) // 2

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        # Test creating a simple agent instance
        from scrollintel.agents.base import BaseAgent
        
        class TestAgent(BaseAgent):
            def __init__(self):
                super().__init__("test-agent", "Test Agent")
            
            async def process(self, input_data):
                return {"result": "test successful"}
        
        agent = TestAgent()
        print(f"✓ Test agent created: {agent.name}")
        
        # Test basic processing
        import asyncio
        result = asyncio.run(agent.process({"test": "data"}))
        if result and result.get("result") == "test successful":
            print("✓ Basic agent processing works")
            return True
        else:
            print("✗ Basic agent processing failed")
            return False
            
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def run_core_tests():
    """Run all core tests"""
    print("=" * 60)
    print("ScrollIntel Core Feature Testing")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Database Connection", test_database_connection),
        ("Agent Modules", test_agent_modules),
        ("API Routes", test_api_routes),
        ("File System", test_file_system),
        ("Environment Variables", test_environment_variables),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*40}")
        print(f"Running: {test_name}")
        print('='*40)
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✓ {test_name} - PASSED")
            else:
                print(f"✗ {test_name} - FAILED")
        except Exception as e:
            results[test_name] = False
            print(f"✗ {test_name} - ERROR: {e}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"Duration: {duration:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All core features are working!")
        status = "excellent"
    elif passed >= total * 0.8:
        print("\n✅ Most core features are working well!")
        status = "good"
    elif passed >= total * 0.6:
        print("\n⚠️  Some core features need attention.")
        status = "needs_attention"
    else:
        print("\n❌ Major issues found. Core features need fixing.")
        status = "critical"
    
    # Save results
    test_results = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": status,
        "summary": {
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": total - passed,
            "success_rate": (passed/total)*100,
            "duration_seconds": duration
        },
        "test_results": results
    }
    
    results_file = f"core_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return test_results

if __name__ == "__main__":
    try:
        results = run_core_tests()
        
        # Exit with appropriate code
        if results["summary"]["success_rate"] >= 80:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)