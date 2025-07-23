"""
Database initialization script for ScrollIntel system.
"""

import logging
from typing import Dict, Any
from sqlalchemy.exc import SQLAlchemyError

from .database_utils import db_manager, init_database
from .seed_data import seed_database, clear_seed_data

logger = logging.getLogger(__name__)


def initialize_database(seed: bool = True, force_recreate: bool = False) -> Dict[str, Any]:
    """
    Initialize the database with tables and optional seed data.
    
    Args:
        seed: Whether to populate with seed data
        force_recreate: Whether to drop and recreate all tables
        
    Returns:
        Dictionary with initialization results
    """
    try:
        logger.info("Starting database initialization...")
        
        if force_recreate:
            logger.warning("Force recreate enabled - dropping all tables")
            db_manager.drop_tables()
        
        # Initialize database and create tables
        init_database()
        
        result = {
            "success": True,
            "message": "Database initialized successfully",
            "tables_created": True,
            "seed_data": None
        }
        
        # Seed database if requested
        if seed:
            logger.info("Seeding database with initial data...")
            with db_manager.session_scope() as session:
                seed_result = seed_database(session)
                result["seed_data"] = seed_result
                
                if not seed_result["success"]:
                    logger.error(f"Failed to seed database: {seed_result['message']}")
                    result["success"] = False
                    result["message"] = f"Database initialized but seeding failed: {seed_result['message']}"
                else:
                    logger.info("Database seeded successfully")
        
        logger.info("Database initialization completed")
        return result
        
    except SQLAlchemyError as e:
        logger.error(f"Database error during initialization: {e}")
        return {
            "success": False,
            "message": f"Database initialization failed: {str(e)}",
            "tables_created": False,
            "seed_data": None
        }
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        return {
            "success": False,
            "message": f"Initialization failed: {str(e)}",
            "tables_created": False,
            "seed_data": None
        }


def reset_database(reseed: bool = True) -> Dict[str, Any]:
    """
    Reset the database by dropping and recreating all tables.
    
    Args:
        reseed: Whether to populate with seed data after reset
        
    Returns:
        Dictionary with reset results
    """
    try:
        logger.warning("Resetting database - all data will be lost!")
        
        # Drop all tables
        db_manager.drop_tables()
        logger.info("All tables dropped")
        
        # Reinitialize
        return initialize_database(seed=reseed, force_recreate=False)
        
    except Exception as e:
        logger.error(f"Failed to reset database: {e}")
        return {
            "success": False,
            "message": f"Database reset failed: {str(e)}"
        }


def clear_all_data() -> Dict[str, Any]:
    """
    Clear all data from the database without dropping tables.
    
    Returns:
        Dictionary with clear results
    """
    try:
        logger.warning("Clearing all data from database...")
        
        with db_manager.session_scope() as session:
            result = clear_seed_data(session)
            
        if result["success"]:
            logger.info("All data cleared successfully")
        else:
            logger.error(f"Failed to clear data: {result['message']}")
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to clear database data: {e}")
        return {
            "success": False,
            "message": f"Data clearing failed: {str(e)}"
        }


def check_database_status() -> Dict[str, Any]:
    """
    Check the current status of the database.
    
    Returns:
        Dictionary with database status information
    """
    try:
        # Test database connection
        db_connected = db_manager.test_connection()
        redis_connected = db_manager.test_redis_connection()
        
        # Check if tables exist
        tables_exist = False
        table_count = 0
        
        if db_connected:
            try:
                with db_manager.session_scope() as session:
                    # Try to query a table to see if schema exists
                    from .database import User
                    result = session.query(User).limit(1).all()
                    tables_exist = True
                    
                    # Count tables (approximate)
                    from .database import Base
                    table_count = len(Base.metadata.tables)
                    
            except Exception:
                tables_exist = False
        
        # Check data counts
        data_counts = {}
        if tables_exist:
            try:
                with db_manager.session_scope() as session:
                    from .database import User, Agent, Dataset, Dashboard, AuditLog
                    
                    data_counts = {
                        "users": session.query(User).count(),
                        "agents": session.query(Agent).count(),
                        "datasets": session.query(Dataset).count(),
                        "dashboards": session.query(Dashboard).count(),
                        "audit_logs": session.query(AuditLog).count()
                    }
            except Exception as e:
                logger.warning(f"Could not get data counts: {e}")
        
        return {
            "success": True,
            "database_connected": db_connected,
            "redis_connected": redis_connected,
            "tables_exist": tables_exist,
            "table_count": table_count,
            "data_counts": data_counts,
            "status": "healthy" if db_connected and tables_exist else "needs_initialization"
        }
        
    except Exception as e:
        logger.error(f"Failed to check database status: {e}")
        return {
            "success": False,
            "message": f"Status check failed: {str(e)}",
            "status": "error"
        }


def migrate_database() -> Dict[str, Any]:
    """
    Run database migrations using Alembic.
    
    Returns:
        Dictionary with migration results
    """
    try:
        logger.info("Running database migrations...")
        
        from .database_utils import run_migrations
        run_migrations()
        
        return {
            "success": True,
            "message": "Database migrations completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to run migrations: {e}")
        return {
            "success": False,
            "message": f"Migration failed: {str(e)}"
        }


# CLI-style functions for direct execution
if __name__ == "__main__":
    import sys
    import argparse
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="ScrollIntel Database Management")
    parser.add_argument("command", choices=["init", "reset", "clear", "status", "migrate"],
                       help="Command to execute")
    parser.add_argument("--no-seed", action="store_true",
                       help="Skip seeding data (for init and reset commands)")
    parser.add_argument("--force", action="store_true",
                       help="Force recreate tables (for init command)")
    
    args = parser.parse_args()
    
    if args.command == "init":
        result = initialize_database(
            seed=not args.no_seed,
            force_recreate=args.force
        )
    elif args.command == "reset":
        result = reset_database(reseed=not args.no_seed)
    elif args.command == "clear":
        result = clear_all_data()
    elif args.command == "status":
        result = check_database_status()
    elif args.command == "migrate":
        result = migrate_database()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)
    
    # Print result
    print(f"Success: {result['success']}")
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