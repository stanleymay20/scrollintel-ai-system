#!/usr/bin/env python3
"""
Database initialization script for ScrollIntel system.
This script can be used to initialize, reset, or check the database status.
"""

import asyncio
import sys
import argparse
import logging
from scrollintel.models.init_db import (
    initialize_database,
    reset_database,
    clear_all_data,
    check_database_status,
    migrate_database
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main function to handle database operations."""
    parser = argparse.ArgumentParser(description="ScrollIntel Database Management")
    parser.add_argument(
        "command", 
        choices=["init", "reset", "clear", "status", "migrate"],
        help="Command to execute"
    )
    parser.add_argument(
        "--no-seed", 
        action="store_true",
        help="Skip seeding data (for init and reset commands)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force recreate tables (for init command)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "init":
            logger.info("Initializing database...")
            result = await initialize_database(
                seed=not args.no_seed,
                force_recreate=args.force
            )
        elif args.command == "reset":
            logger.info("Resetting database...")
            result = await reset_database(reseed=not args.no_seed)
        elif args.command == "clear":
            logger.info("Clearing database data...")
            result = await clear_all_data()
        elif args.command == "status":
            logger.info("Checking database status...")
            result = await check_database_status()
        elif args.command == "migrate":
            logger.info("Running database migrations...")
            result = migrate_database()
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
        
        # Print result
        print(f"\nResult: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Message: {result.get('message', 'No message')}")
        
        if args.command == "status" and result["success"]:
            print(f"Database Connected: {result['database_connected']}")
            print(f"Redis Connected: {result['redis_connected']}")
            print(f"Tables Exist: {result['tables_exist']}")
            print(f"Table Count: {result['table_count']}")
            print(f"Status: {result['status']}")
            if result.get('data_counts'):
                print("Data Counts:")
                for table, count in result['data_counts'].items():
                    print(f"  {table}: {count}")
        
        if result.get('seed_data'):
            seed_data = result['seed_data']
            if seed_data.get('data'):
                print("Seed Data:")
                for key, value in seed_data['data'].items():
                    print(f"  {key}: {value}")
        
        sys.exit(0 if result["success"] else 1)
        
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())