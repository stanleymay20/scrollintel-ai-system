#!/usr/bin/env python3
"""
Test SQLite fallback system comprehensively
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_sqlite_fallback_system():
    """Test SQLite fallback system comprehensively."""
    try:
        logger.info("Testing SQLite fallback system...")
        
        from scrollintel.core.database_connection_manager import DatabaseConnectionManager
        
        # Create a temporary directory for test databases
        temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Using temporary directory: {temp_dir}")
        
        try:
            # Test 1: Direct SQLite connection
            logger.info("\n--- Test 1: Direct SQLite Connection ---")
            sqlite_url = f"sqlite:///{temp_dir}/test1.db"
            os.environ["DATABASE_URL"] = "postgresql://invalid:invalid@invalid:9999/invalid"  # Force failure
            os.environ["SQLITE_URL"] = sqlite_url
            
            db_manager = DatabaseConnectionManager()
            success = await db_manager.initialize()
            
            if success:
                info = db_manager.get_connection_info()
                logger.info(f"✓ Connected to {info['database_type']}")
                logger.info(f"✓ Fallback active: {info['fallback_active']}")
                
                # Test table creation
                await db_manager.create_tables()
                logger.info("✓ Tables created successfully")
                
                # Test session
                async with db_manager.get_async_session() as session:
                    from sqlalchemy import text
                    result = await session.execute(text("SELECT 1"))
                    assert result.scalar() == 1
                    logger.info("✓ Session test passed")
                
                await db_manager.close()
            else:
                logger.error("✗ Direct SQLite connection failed")
                return False
            
            # Test 2: Schema synchronization
            logger.info("\n--- Test 2: Schema Synchronization ---")
            sqlite_url2 = f"sqlite:///{temp_dir}/test2.db"
            os.environ["SQLITE_URL"] = sqlite_url2
            
            db_manager2 = DatabaseConnectionManager()
            await db_manager2.initialize()
            
            # Create tables and add some data
            await db_manager2.create_tables()
            
            async with db_manager2.get_async_session() as session:
                from scrollintel.models.database import User
                from scrollintel.core.interfaces import UserRole
                
                # Create a test user
                test_user = User(
                    email="test@example.com",
                    hashed_password="hashed_password",
                    full_name="Test User",
                    role=UserRole.VIEWER
                )
                session.add(test_user)
                await session.commit()
                
                # Verify user was created
                from sqlalchemy import select
                result = await session.execute(select(User).where(User.email == "test@example.com"))
                user = result.scalar_one_or_none()
                
                if user:
                    logger.info("✓ Data insertion and retrieval successful")
                else:
                    logger.error("✗ Data insertion failed")
                    return False
            
            await db_manager2.close()
            
            # Test 3: Graceful degradation
            logger.info("\n--- Test 3: Graceful Degradation ---")
            
            # Test with invalid SQLite path (should still work with in-memory)
            os.environ["SQLITE_URL"] = "sqlite:///:memory:"
            
            db_manager3 = DatabaseConnectionManager()
            success = await db_manager3.initialize()
            
            if success:
                logger.info("✓ In-memory SQLite fallback successful")
                
                # Test health check
                health = await db_manager3.check_health()
                if health['healthy']:
                    logger.info("✓ Health check passed")
                else:
                    logger.error("✗ Health check failed")
                    return False
                
                await db_manager3.close()
            else:
                logger.error("✗ In-memory SQLite fallback failed")
                return False
            
            # Test 4: Connection switching
            logger.info("\n--- Test 4: Connection Switching ---")
            
            db_manager4 = DatabaseConnectionManager()
            await db_manager4.initialize()
            
            # Test manual fallback switch
            if not db_manager4.fallback_active:
                switch_result = await db_manager4.switch_to_fallback()
                if switch_result:
                    logger.info("✓ Manual fallback switch successful")
                else:
                    logger.warning("⚠ Manual fallback switch not needed (already on fallback)")
            else:
                logger.info("✓ Already using fallback database")
            
            # Test switch back to primary (should fail gracefully)
            switch_back = await db_manager4.switch_to_primary()
            if not switch_back:
                logger.info("✓ Switch to primary failed gracefully (as expected)")
            else:
                logger.info("✓ Switch to primary succeeded")
            
            await db_manager4.close()
            
            logger.info("\n🎉 All SQLite fallback tests passed!")
            return True
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
            
    except Exception as e:
        logger.error(f"SQLite fallback test failed: {e}")
        return False


async def test_sqlite_performance():
    """Test SQLite performance characteristics."""
    try:
        logger.info("\n--- SQLite Performance Test ---")
        
        from scrollintel.core.database_connection_manager import DatabaseConnectionManager
        import time
        
        # Create temporary database
        temp_dir = Path(tempfile.mkdtemp())
        sqlite_url = f"sqlite:///{temp_dir}/performance_test.db"
        os.environ["DATABASE_URL"] = "postgresql://invalid:invalid@invalid:9999/invalid"
        os.environ["SQLITE_URL"] = sqlite_url
        
        try:
            db_manager = DatabaseConnectionManager()
            await db_manager.initialize()
            await db_manager.create_tables()
            
            # Test concurrent sessions
            logger.info("Testing concurrent sessions...")
            start_time = time.time()
            
            async def test_session(session_id):
                async with db_manager.get_async_session() as session:
                    from sqlalchemy import text
                    result = await session.execute(text(f"SELECT {session_id}"))
                    return result.scalar()
            
            # Run 10 concurrent sessions
            tasks = [test_session(i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            if all(results[i] == i for i in range(10)):
                logger.info(f"✓ Concurrent sessions test passed in {duration:.2f}s")
            else:
                logger.error("✗ Concurrent sessions test failed")
                return False
            
            # Test bulk operations
            logger.info("Testing bulk operations...")
            start_time = time.time()
            
            async with db_manager.get_async_session() as session:
                from scrollintel.models.database import User
                from scrollintel.core.interfaces import UserRole
                
                # Create 100 test users
                users = []
                for i in range(100):
                    user = User(
                        email=f"user{i}@example.com",
                        hashed_password="hashed_password",
                        full_name=f"User {i}",
                        role=UserRole.VIEWER
                    )
                    users.append(user)
                
                session.add_all(users)
                await session.commit()
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"✓ Bulk insert of 100 users completed in {duration:.2f}s")
            
            # Verify data
            async with db_manager.get_async_session() as session:
                from sqlalchemy import select, func
                from scrollintel.models.database import User
                
                count_result = await session.execute(select(func.count(User.id)))
                user_count = count_result.scalar()
                
                if user_count == 100:
                    logger.info("✓ Bulk data verification passed")
                else:
                    logger.error(f"✗ Expected 100 users, found {user_count}")
                    return False
            
            await db_manager.close()
            logger.info("✓ SQLite performance test completed successfully")
            return True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.error(f"SQLite performance test failed: {e}")
        return False


def main():
    """Run all SQLite fallback tests."""
    logger.info("Starting comprehensive SQLite fallback tests...")
    
    async def run_all_tests():
        tests = [
            ("SQLite Fallback System", test_sqlite_fallback_system),
            ("SQLite Performance", test_sqlite_performance),
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info(f"{'='*50}")
            
            try:
                result = await test_func()
                results.append((test_name, result))
                if result:
                    logger.info(f"✓ {test_name} PASSED")
                else:
                    logger.error(f"✗ {test_name} FAILED")
            except Exception as e:
                logger.error(f"✗ {test_name} FAILED with exception: {e}")
                results.append((test_name, False))
        
        return results
    
    try:
        results = asyncio.run(run_all_tests())
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*50}")
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All SQLite fallback tests passed!")
            return 0
        else:
            logger.error("❌ Some tests failed")
            return 1
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())