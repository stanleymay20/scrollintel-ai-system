#!/usr/bin/env python3
"""
Quick fix for database and import issues in ScrollIntel
"""

import os
import sys
from pathlib import Path

def fix_database_session_function():
    """Add the missing get_db_session function to database.py"""
    
    database_file = Path("scrollintel/models/database.py")
    
    if not database_file.exists():
        print("❌ Database file not found")
        return False
    
    # Read the current content
    with open(database_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if get_db_session already exists
    if 'def get_db_session' in content:
        print("✅ get_db_session function already exists")
        return True
    
    # Add the missing function at the end of the file
    additional_code = '''

# Database session management functions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from typing import Generator

# Global session factory
_SessionLocal = None
_engine = None

def init_database_session(database_url: str = None):
    """Initialize database session factory"""
    global _SessionLocal, _engine
    
    if database_url is None:
        # Try to get from environment or use default
        database_url = os.getenv('DATABASE_URL', 'sqlite:///./scrollintel.db')
    
    # Create engine
    if database_url.startswith('sqlite'):
        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            echo=False
        )
    else:
        _engine = create_engine(database_url, echo=False)
    
    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    
    # Create tables
    Base.metadata.create_all(bind=_engine)

@contextmanager
def get_db_session() -> Generator:
    """Get database session with automatic cleanup"""
    global _SessionLocal
    
    if _SessionLocal is None:
        init_database_session()
    
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def get_engine():
    """Get database engine"""
    global _engine
    if _engine is None:
        init_database_session()
    return _engine
'''
    
    # Append the additional code
    with open(database_file, 'w', encoding='utf-8') as f:
        f.write(content + additional_code)
    
    print("✅ Added get_db_session function to database.py")
    return True

def create_base_agent_class():
    """Create a simple base agent class"""
    
    agents_dir = Path("scrollintel/agents")
    agents_dir.mkdir(exist_ok=True)
    
    base_agent_file = agents_dir / "base.py"
    
    if base_agent_file.exists():
        print("✅ Base agent class already exists")
        return True
    
    base_agent_code = '''"""
Base agent class for ScrollIntel agents
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all ScrollIntel agents"""
    
    def __init__(self, agent_id: str, name: str, description: str = ""):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.status = "inactive"
        self.created_at = datetime.utcnow()
        self.last_activity = None
        
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return results"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health status"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "last_activity": self.last_activity,
            "healthy": True
        }
    
    def activate(self):
        """Activate the agent"""
        self.status = "active"
        self.last_activity = datetime.utcnow()
        logger.info(f"Agent {self.name} activated")
    
    def deactivate(self):
        """Deactivate the agent"""
        self.status = "inactive"
        logger.info(f"Agent {self.name} deactivated")
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.agent_id}, name={self.name}, status={self.status})>"
'''
    
    with open(base_agent_file, 'w', encoding='utf-8') as f:
        f.write(base_agent_code)
    
    print("✅ Created base agent class")
    return True

def fix_pinecone_imports():
    """Fix Pinecone import issues by updating requirements"""
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("⚠️ requirements.txt not found")
        return False
    
    # Read current requirements
    with open(requirements_file, 'r') as f:
        lines = f.readlines()
    
    # Update pinecone-client to pinecone
    updated_lines = []
    pinecone_updated = False
    
    for line in lines:
        if 'pinecone-client' in line:
            updated_lines.append(line.replace('pinecone-client', 'pinecone'))
            pinecone_updated = True
        else:
            updated_lines.append(line)
    
    # Add pinecone if not present
    if not pinecone_updated and not any('pinecone' in line for line in lines):
        updated_lines.append('pinecone>=3.0.0\n')
        pinecone_updated = True
    
    if pinecone_updated:
        with open(requirements_file, 'w') as f:
            f.writelines(updated_lines)
        print("✅ Updated Pinecone package in requirements.txt")
    else:
        print("✅ Pinecone package already correct")
    
    return True

def fix_config_issues():
    """Fix configuration issues"""
    
    config_file = Path("scrollintel/core/config.py")
    
    if not config_file.exists():
        print("⚠️ Config file not found")
        return False
    
    # Read current config
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if get_settings function exists and works properly
    if 'def get_settings' not in content:
        print("⚠️ get_settings function not found in config")
        return False
    
    # Add a simple fallback configuration if needed
    fallback_config = '''

# Fallback configuration for testing
class FallbackSettings:
    """Fallback settings when main config fails"""
    
    def __init__(self):
        import os
        self.database_url = os.getenv('DATABASE_URL', 'sqlite:///./scrollintel.db')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.jwt_secret_key = os.getenv('JWT_SECRET_KEY', 'fallback-secret-key')
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.db_pool_size = int(os.getenv('DB_POOL_SIZE', '5'))
        self.db_max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '10'))
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))
        self.redis_password = os.getenv('REDIS_PASSWORD', '')
        self.redis_db = int(os.getenv('REDIS_DB', '0'))
        self.skip_redis = os.getenv('SKIP_REDIS', 'false').lower() == 'true'
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return getattr(self, key, default)

def get_fallback_settings():
    """Get fallback settings for testing"""
    return FallbackSettings()
'''
    
    if 'class FallbackSettings' not in content:
        with open(config_file, 'a', encoding='utf-8') as f:
            f.write(fallback_config)
        print("✅ Added fallback configuration")
    else:
        print("✅ Fallback configuration already exists")
    
    return True

def run_fixes():
    """Run all fixes"""
    print("🔧 Running ScrollIntel fixes...")
    print("=" * 50)
    
    fixes = [
        ("Database session function", fix_database_session_function),
        ("Base agent class", create_base_agent_class),
        ("Pinecone imports", fix_pinecone_imports),
        ("Configuration issues", fix_config_issues),
    ]
    
    success_count = 0
    
    for fix_name, fix_func in fixes:
        print(f"\n🔧 {fix_name}...")
        try:
            if fix_func():
                success_count += 1
            else:
                print(f"❌ {fix_name} failed")
        except Exception as e:
            print(f"❌ {fix_name} failed with error: {e}")
    
    print(f"\n{'='*50}")
    print(f"✅ Completed {success_count}/{len(fixes)} fixes successfully")
    
    if success_count == len(fixes):
        print("🎉 All fixes applied successfully!")
        print("\nYou can now run the test again:")
        print("python test_core_features.py")
    else:
        print("⚠️ Some fixes failed. Please check the errors above.")
    
    return success_count == len(fixes)

if __name__ == "__main__":
    run_fixes()