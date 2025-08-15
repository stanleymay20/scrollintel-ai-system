"""Migration runner for AI Data Readiness Platform."""

import logging
from typing import Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from ..core.config import Config
from ..models.database import Database, Base

logger = logging.getLogger(__name__)


class MigrationRunner:
    """Handles database migrations and schema management."""
    
    def __init__(self, config: Config):
        self.config = config
        self.database = Database(config.database.connection_string)
    
    def create_database_if_not_exists(self) -> bool:
        """Create the database if it doesn't exist."""
        try:
            # Connect to postgres database to create our target database
            postgres_url = self.config.database.connection_string.replace(
                f"/{self.config.database.database}", "/postgres"
            )
            engine = create_engine(postgres_url)
            
            with engine.connect() as conn:
                # Check if database exists
                result = conn.execute(text(
                    "SELECT 1 FROM pg_database WHERE datname = :db_name"
                ), {"db_name": self.config.database.database})
                
                if not result.fetchone():
                    # Create database
                    conn.execute(text("COMMIT"))  # End any existing transaction
                    conn.execute(text(f"CREATE DATABASE {self.config.database.database}"))
                    logger.info(f"Created database: {self.config.database.database}")
                else:
                    logger.info(f"Database {self.config.database.database} already exists")
            
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database: {e}")
            return False
    
    def run_initial_migration(self) -> bool:
        """Run the initial database migration."""
        try:
            # Create all tables
            self.database.create_tables()
            logger.info("Successfully created all database tables")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to run initial migration: {e}")
            return False
    
    def check_migration_status(self) -> dict:
        """Check the current migration status."""
        status = {
            "database_exists": False,
            "tables_exist": False,
            "migration_needed": True
        }
        
        try:
            # Check if we can connect to the database
            if self.database.health_check():
                status["database_exists"] = True
                
                # Check if tables exist
                with self.database.get_session() as session:
                    result = session.execute(text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' AND table_name = 'datasets'"
                    ))
                    
                    if result.fetchone():
                        status["tables_exist"] = True
                        status["migration_needed"] = False
            
        except Exception as e:
            logger.warning(f"Could not check migration status: {e}")
        
        return status
    
    def migrate(self) -> bool:
        """Run complete migration process."""
        logger.info("Starting database migration...")
        
        # Check current status
        status = self.check_migration_status()
        
        if not status["database_exists"]:
            logger.info("Creating database...")
            if not self.create_database_if_not_exists():
                return False
        
        if status["migration_needed"]:
            logger.info("Running initial migration...")
            if not self.run_initial_migration():
                return False
        else:
            logger.info("Database is already up to date")
        
        logger.info("Migration completed successfully")
        return True
    
    def reset_database(self) -> bool:
        """Reset the database by dropping and recreating all tables."""
        try:
            logger.warning("Resetting database - all data will be lost!")
            self.database.drop_tables()
            self.database.create_tables()
            logger.info("Database reset completed")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"Failed to reset database: {e}")
            return False


def run_migrations(config: Optional[Config] = None) -> bool:
    """Convenience function to run migrations."""
    if config is None:
        config = Config()
    
    runner = MigrationRunner(config)
    return runner.migrate()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run migrations
    success = run_migrations()
    exit(0 if success else 1)