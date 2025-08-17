#!/usr/bin/env python3
"""
Comprehensive PostgreSQL Default Database Test Suite
Tests to ensure PostgreSQL is working as the default database
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
import asyncpg
import psycopg2

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class PostgreSQLDefaultTests:
    """Comprehensive test suite for PostgreSQL as default database"""
    
    def __init__(self):
        """Initialize test suite"""
        load_dotenv()
        self.database_url = os.getenv('DATABASE_URL')
        self.postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.postgres_port = int(os.getenv('POSTGRES_PORT', 5432))
        self.postgres_db = os.getenv('POSTGRES_DB', 'scrollintel')
        self.postgres_user = os.getenv('POSTGRES_USER', 'postgres')
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', '')
        
        self.test_results = []
    
    def log_test_result(self, test_name: str, success: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        self.test_results.append((test_name, success, message))
        logger.info(f"{status}: {test_name} - {message}")
    
    def test_environment_variables(self):
        """Test 1: Environment variables are properly set"""
        try:
            assert self.database_url, "DATABASE_URL not set"
            assert self.postgres_password, "POSTGRES_PASSWORD not set"
            assert "postgresql://" in self.database_url, "DATABASE_URL is not PostgreSQL"
            assert "boatemaa1612" in self.database_url, "Password not in DATABASE_URL"
            
            self.log_test_result(
                "Environment Variables", 
                True, 
                f"DATABASE_URL: {self.database_url[:30]}..."
            )
            return True
        except Exception as e:
            self.log_test_result("Environment Variables", False, str(e))
            return False
    
    def test_direct_psycopg2_connection(self):
        """Test 2: Direct psycopg2 connection"""
        try:
            conn = psycopg2.connect(
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            self.log_test_result(
                "Direct psycopg2 Connection", 
                True, 
                f"PostgreSQL {version.split()[1]}"
            )
            return True
        except Exception as e:
            self.log_test_result("Direct psycopg2 Connection", False, str(e))
            return False
    
    async def test_direct_asyncpg_connection(self):
        """Test 3: Direct asyncpg connection"""
        try:
            conn = await asyncpg.connect(
                host=self.postgres_host,
                port=self.postgres_port,
                database=self.postgres_db,
                user=self.postgres_user,
                password=self.postgres_password
            )
            
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            
            self.log_test_result(
                "Direct asyncpg Connection", 
                True, 
                f"PostgreSQL {version.split()[1]}"
            )
            return True
        except Exception as e:
            self.log_test_result("Direct asyncpg Connection", False, str(e))
            return False
    
    async def test_database_url_connection(self):
        """Test 4: Connection using DATABASE_URL"""
        try:
            conn = await asyncpg.connect(self.database_url)
            result = await conn.fetchval("SELECT current_database()")
            await conn.close()
            
            self.log_test_result(
                "DATABASE_URL Connection", 
                True, 
                f"Connected to database: {result}"
            )
            return True
        except Exception as e:
            self.log_test_result("DATABASE_URL Connection", False, str(e))
            return False
    
    async def test_database_operations(self):
        """Test 5: Basic database operations"""
        try:
            conn = await asyncpg.connect(self.database_url)
            
            # Create test table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS test_postgresql_default (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert test data
            await conn.execute(
                "INSERT INTO test_postgresql_default (name) VALUES ($1)",
                "PostgreSQL Test"
            )
            
            # Query test data
            result = await conn.fetchrow(
                "SELECT name, created_at FROM test_postgresql_default WHERE name = $1",
                "PostgreSQL Test"
            )
            
            # Clean up
            await conn.execute("DROP TABLE test_postgresql_default")
            await conn.close()
            
            assert result['name'] == "PostgreSQL Test"
            
            self.log_test_result(
                "Database Operations", 
                True, 
                f"CRUD operations successful"
            )
            return True
        except Exception as e:
            self.log_test_result("Database Operations", False, str(e))
            return False
    
    async def test_scrollintel_database_manager(self):
        """Test 6: ScrollIntel database connection manager"""
        try:
            from scrollintel.core.database_connection_manager import DatabaseConnectionManager
            
            # Create database manager
            db_manager = DatabaseConnectionManager()
            
            # Initialize connection
            success = await db_manager.initialize()
            
            if success:
                # Get connection info
                info = db_manager.get_connection_info()
                
                # Check if using PostgreSQL
                is_postgresql = info.get('database_type') == 'postgresql'
                
                # Test health check
                health = await db_manager.check_health()
                
                await db_manager.close()
                
                if is_postgresql:
                    self.log_test_result(
                        "ScrollIntel Database Manager", 
                        True, 
                        f"Using PostgreSQL, Health: {health.get('healthy', False)}"
                    )
                    return True
                else:
                    self.log_test_result(
                        "ScrollIntel Database Manager", 
                        False, 
                        f"Using {info.get('database_type', 'unknown')} instead of PostgreSQL"
                    )
                    return False
            else:
                self.log_test_result("ScrollIntel Database Manager", False, "Failed to initialize")
                return False
                
        except Exception as e:
            self.log_test_result("ScrollIntel Database Manager", False, str(e))
            return False
    
    def test_postgresql_service_status(self):
        """Test 7: PostgreSQL service status"""
        try:
            import subprocess
            
            # Check if PostgreSQL service is running (Windows)
            result = subprocess.run(
                ['powershell', '-Command', 'Get-Service -Name "*postgres*" | Select-Object Status, Name'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and "Running" in result.stdout:
                self.log_test_result(
                    "PostgreSQL Service Status", 
                    True, 
                    "PostgreSQL service is running"
                )
                return True
            else:
                self.log_test_result(
                    "PostgreSQL Service Status", 
                    False, 
                    "PostgreSQL service not running or not found"
                )
                return False
                
        except Exception as e:
            self.log_test_result("PostgreSQL Service Status", False, str(e))
            return False
    
    def test_port_availability(self):
        """Test 8: PostgreSQL port availability"""
        try:
            import socket
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.postgres_host, self.postgres_port))
            sock.close()
            
            if result == 0:
                self.log_test_result(
                    "Port Availability", 
                    True, 
                    f"Port {self.postgres_port} is open and accessible"
                )
                return True
            else:
                self.log_test_result(
                    "Port Availability", 
                    False, 
                    f"Port {self.postgres_port} is not accessible"
                )
                return False
                
        except Exception as e:
            self.log_test_result("Port Availability", False, str(e))
            return False
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 Starting PostgreSQL Default Database Test Suite")
        logger.info("=" * 60)
        
        # Run tests
        test_1 = self.test_environment_variables()
        test_2 = self.test_direct_psycopg2_connection()
        test_3 = await self.test_direct_asyncpg_connection()
        test_4 = await self.test_database_url_connection()
        test_5 = await self.test_database_operations()
        test_6 = await self.test_scrollintel_database_manager()
        test_7 = self.test_postgresql_service_status()
        test_8 = self.test_port_availability()
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status}: {test_name}")
            if not success and message:
                logger.info(f"    Error: {message}")
            
            if success:
                passed += 1
            else:
                failed += 1
        
        logger.info("=" * 60)
        logger.info(f"📈 TOTAL: {len(self.test_results)} tests")
        logger.info(f"✅ PASSED: {passed}")
        logger.info(f"❌ FAILED: {failed}")
        
        if failed == 0:
            logger.info("🎉 ALL TESTS PASSED! PostgreSQL is working as the default database!")
            return True
        else:
            logger.info(f"⚠️  {failed} test(s) failed. PostgreSQL may not be properly configured as default.")
            return False


async def main():
    """Main test function"""
    test_suite = PostgreSQLDefaultTests()
    success = await test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)