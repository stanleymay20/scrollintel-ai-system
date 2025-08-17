#!/usr/bin/env python3
"""
Test script for the configuration system
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_configuration_manager():
    """Test the new configuration manager."""
    try:
        logger.info("Testing new configuration manager...")
        
        from scrollintel.core.configuration_manager import ConfigurationManager
        
        # Create configuration manager
        config_manager = ConfigurationManager()
        
        # Load configuration
        config = config_manager.load_config()
        
        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  Environment: {config.environment}")
        logger.info(f"  Debug: {config.debug}")
        logger.info(f"  Database Primary: {config.database.primary_url}")
        logger.info(f"  Database Fallback: {config.database.fallback_url}")
        logger.info(f"  Session Timeout: {config.session.timeout_minutes} minutes")
        logger.info(f"  API Port: {config.services.api_port}")
        logger.info(f"  Frontend Port: {config.services.frontend_port}")
        
        # Check validation
        if config_manager.is_valid():
            logger.info("✓ Configuration is valid")
        else:
            logger.warning("⚠ Configuration has issues:")
            for error in config_manager.get_validation_errors():
                logger.warning(f"  Error: {error}")
            for warning in config_manager.get_validation_warnings():
                logger.warning(f"  Warning: {warning}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration manager test failed: {e}")
        return False


def test_legacy_config():
    """Test the legacy configuration system."""
    try:
        logger.info("Testing legacy configuration system...")
        
        from scrollintel.core.config import get_config
        
        # Load configuration
        config = get_config()
        
        logger.info(f"Legacy configuration loaded successfully:")
        logger.info(f"  Environment: {config.get('environment')}")
        logger.info(f"  Debug: {config.get('debug')}")
        logger.info(f"  Database URL: {config.get('database_url')}")
        logger.info(f"  Session Timeout: {config.get('session_timeout_minutes')} minutes")
        
        return True
        
    except Exception as e:
        logger.error(f"Legacy configuration test failed: {e}")
        return False


def test_database_connection_manager():
    """Test the database connection manager."""
    try:
        logger.info("Testing database connection manager...")
        
        from scrollintel.core.database_connection_manager import DatabaseConnectionManager
        
        # Create database manager
        db_manager = DatabaseConnectionManager()
        
        # Get connection info
        info = db_manager.get_connection_info()
        logger.info(f"Database connection info: {info}")
        
        logger.info("✓ Database connection manager created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database connection manager test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting configuration system tests...")
    
    tests = [
        ("Configuration Manager", test_configuration_manager),
        ("Legacy Config", test_legacy_config),
        ("Database Connection Manager", test_database_connection_manager),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                logger.info(f"✓ {test_name} passed")
            else:
                logger.error(f"✗ {test_name} failed")
        except Exception as e:
            logger.error(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n--- Test Summary ---")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())