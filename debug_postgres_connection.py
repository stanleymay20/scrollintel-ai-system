#!/usr/bin/env python3
"""
Debug PostgreSQL connection issues
"""

import os
import asyncio
import asyncpg
import psycopg2
from urllib.parse import urlparse

def test_direct_connection():
    """Test direct connection with psycopg2"""
    try:
        print("Testing direct psycopg2 connection...")
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="scrollintel",
            user="postgres",
            password="boatemaa1612"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        result = cursor.fetchone()
        print(f"✓ Direct connection successful: {result[0][:50]}...")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"✗ Direct connection failed: {e}")
        return False

async def test_asyncpg_connection():
    """Test asyncpg connection"""
    try:
        print("Testing asyncpg connection...")
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            database="scrollintel",
            user="postgres",
            password="boatemaa1612"
        )
        result = await conn.fetchval("SELECT version()")
        print(f"✓ Asyncpg connection successful: {result[:50]}...")
        await conn.close()
        return True
    except Exception as e:
        print(f"✗ Asyncpg connection failed: {e}")
        return False

def test_url_parsing():
    """Test URL parsing"""
    urls = [
        "postgresql://postgres:Apply@2025@localhost:5432/scrollintel",
        "postgresql://postgres:Apply%402025@localhost:5432/scrollintel",
        os.getenv("DATABASE_URL", "")
    ]
    
    for url in urls:
        if not url:
            continue
        print(f"\nTesting URL: {url}")
        try:
            parsed = urlparse(url)
            print(f"  Host: {parsed.hostname}")
            print(f"  Port: {parsed.port}")
            print(f"  Database: {parsed.path.lstrip('/')}")
            print(f"  Username: {parsed.username}")
            print(f"  Password: {parsed.password}")
        except Exception as e:
            print(f"  URL parsing failed: {e}")

async def main():
    """Main test function"""
    print("=== PostgreSQL Connection Debug ===")
    
    # Test environment variables
    print(f"DATABASE_URL: {os.getenv('DATABASE_URL', 'Not set')}")
    print(f"POSTGRES_PASSWORD: {os.getenv('POSTGRES_PASSWORD', 'Not set')}")
    
    # Test URL parsing
    test_url_parsing()
    
    # Test direct connections
    print("\n=== Direct Connection Tests ===")
    test_direct_connection()
    await test_asyncpg_connection()

if __name__ == "__main__":
    asyncio.run(main())