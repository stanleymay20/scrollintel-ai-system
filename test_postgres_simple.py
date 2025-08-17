#!/usr/bin/env python3
"""
Simple PostgreSQL connection test
"""

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

async def test_connection():
    """Test PostgreSQL connection with current environment"""
    
    # Load environment variables
    load_dotenv()
    
    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    print(f"Database URL: {database_url}")
    
    if not database_url:
        print("No DATABASE_URL found")
        return False
    
    try:
        # Test connection
        conn = await asyncpg.connect(database_url)
        result = await conn.fetchval("SELECT version()")
        print(f"✓ Connection successful: {result[:50]}...")
        await conn.close()
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_connection())
    if success:
        print("🎉 PostgreSQL is working!")
    else:
        print("❌ PostgreSQL connection failed")