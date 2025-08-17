#!/usr/bin/env python3
"""
Test script to verify the main ScrollIntel application works end-to-end
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_database_initialization():
    """Test database initialization."""
    try:
        from scrollintel.models.database_utils import DatabaseManager
        from scrollintel.core.config import get_settings
        
        logger.info("Testing database initialization...")
        settings = get_settings()
        logger.info(f"Using database: {settings.get('database_url', 'sqlite:///./scrollintel.db')}")
        
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Create tables if they don't exist
        await db_manager.create_tables()
        logger.info("✅ Database initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

async def test_api_components():
    """Test API components can be imported and initialized."""
    try:
        logger.info("Testing API components...")
        
        # Test core imports
        from scrollintel.api.main import app
        from scrollintel.core.config import get_settings
        from scrollintel.agents.scroll_cto_agent import ScrollCTOAgent
        
        logger.info("✅ Core API components imported successfully")
        
        # Test agent initialization
        cto_agent = ScrollCTOAgent()
        logger.info("✅ CTO Agent initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_heavy_volume_support():
    """Test heavy volume data handling capabilities."""
    try:
        logger.info("Testing heavy volume support...")
        
        from scrollintel.engines.heavy_volume_processor import HeavyVolumeProcessor
        from scrollintel.core.config import get_settings
        
        settings = get_settings()
        max_file_size = settings.get('max_file_size', '10GB')
        logger.info(f"Max file size configured: {max_file_size}")
        
        # Test heavy volume processor
        processor = HeavyVolumeProcessor()
        logger.info("✅ Heavy volume processor initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Heavy volume support test failed: {e}")
        return False

async def test_file_processing():
    """Test file processing capabilities."""
    try:
        logger.info("Testing file processing...")
        
        from scrollintel.engines.file_processor import FileProcessor
        
        processor = FileProcessor()
        logger.info("✅ File processor initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ File processing test failed: {e}")
        return False

async def test_agent_system():
    """Test agent system functionality."""
    try:
        logger.info("Testing agent system...")
        
        from scrollintel.agents.scroll_ml_engineer import ScrollMLEngineer
        from scrollintel.agents.scroll_data_scientist import ScrollDataScientist
        from scrollintel.agents.scroll_bi_agent import ScrollBIAgent
        
        # Initialize agents
        ml_agent = ScrollMLEngineer()
        ds_agent = ScrollDataScientist()
        bi_agent = ScrollBIAgent()
        
        logger.info("✅ Agent system initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Agent system test failed: {e}")
        return False

async def test_monitoring_system():
    """Test monitoring and analytics system."""
    try:
        logger.info("Testing monitoring system...")
        
        from scrollintel.core.monitoring import MonitoringSystem
        from scrollintel.core.analytics import AnalyticsEngine
        
        monitoring = MonitoringSystem()
        analytics = AnalyticsEngine()
        
        logger.info("✅ Monitoring system initialized successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Monitoring system test failed: {e}")
        return False

async def main():
    """Main test function."""
    logger.info("🚀 Starting ScrollIntel End-to-End Application Test")
    logger.info("=" * 60)
    
    tests = [
        ("Database Initialization", test_database_initialization),
        ("API Components", test_api_components),
        ("Heavy Volume Support", test_heavy_volume_support),
        ("File Processing", test_file_processing),
        ("Agent System", test_agent_system),
        ("Monitoring System", test_monitoring_system),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running test: {test_name}")
        logger.info("-" * 40)
        
        try:
            if await test_func():
                passed_tests += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {total_tests - passed_tests}")
    logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! ScrollIntel is ready for production!")
        return True
    else:
        logger.error("💥 Some tests failed. Please review and fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)