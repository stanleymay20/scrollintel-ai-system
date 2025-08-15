#!/usr/bin/env python3
"""
ScrollIntel Database Migration Script
Handles database migrations for different environments
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from alembic.config import Config
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scrollintel.core.config import get_settings, reload_settings
from scrollintel.models.database import engine, Base
from scrollintel.models.init_db import init_database


class DatabaseMigrator:
    """Handles database migrations and setup"""
    
    def __init__(self, environment: str = None):
        self.settings = reload_settings(environment) if environment else get_settings()
        self.alembic_cfg = Config("alembic.ini")
        self.alembic_cfg.set_main_option("sqlalchemy.url", self.settings.database_url)
    
    def check_database_connection(self) -> bool:
        """Check if database is accessible"""
        try:
            test_engine = create_engine(self.settings.database_url)
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            return True
        except OperationalError as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist"""
        try:
            # Connect to postgres database to create our database
            postgres_url = self.settings.database_url.replace(
                f"/{self.settings.postgres_db}", "/postgres"
            )
            postgres_engine = create_engine(postgres_url)
            
            with postgres_engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": self.settings.postgres_db}
                )
                
                if not result.fetchone():
                    # Create database
                    conn.execute(text("COMMIT"))  # End transaction
                    conn.execute(text(f"CREATE DATABASE {self.settings.postgres_db}"))
                    print(f"✅ Created database: {self.settings.postgres_db}")
                else:
                    print(f"✅ Database already exists: {self.settings.postgres_db}")
                    
        except Exception as e:
            print(f"⚠️ Could not create database: {e}")
            print("Please ensure the database exists before running migrations")
    
    def run_migrations(self):
        """Run Alembic migrations"""
        try:
            print("🔄 Running database migrations...")
            command.upgrade(self.alembic_cfg, "head")
            print("✅ Migrations completed successfully")
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            raise
    
    def create_migration(self, message: str):
        """Create a new migration"""
        try:
            print(f"📝 Creating migration: {message}")
            command.revision(self.alembic_cfg, message=message, autogenerate=True)
            print("✅ Migration created successfully")
        except Exception as e:
            print(f"❌ Migration creation failed: {e}")
            raise
    
    def rollback_migration(self, revision: str = "-1"):
        """Rollback to specific revision"""
        try:
            print(f"⏪ Rolling back to revision: {revision}")
            command.downgrade(self.alembic_cfg, revision)
            print("✅ Rollback completed successfully")
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            raise
    
    def show_migration_history(self):
        """Show migration history"""
        try:
            print("📋 Migration history:")
            command.history(self.alembic_cfg)
        except Exception as e:
            print(f"❌ Could not show history: {e}")
    
    def show_current_revision(self):
        """Show current database revision"""
        try:
            print("📍 Current revision:")
            command.current(self.alembic_cfg)
        except Exception as e:
            print(f"❌ Could not show current revision: {e}")
    
    def seed_database(self):
        """Seed database with initial data"""
        try:
            print("🌱 Seeding database with initial data...")
            asyncio.run(init_database())
            print("✅ Database seeded successfully")
        except Exception as e:
            print(f"❌ Database seeding failed: {e}")
            raise
    
    def reset_database(self):
        """Reset database (drop all tables and recreate)"""
        try:
            print("⚠️ Resetting database (this will delete all data)...")
            
            # Drop all tables
            Base.metadata.drop_all(bind=engine)
            print("🗑️ Dropped all tables")
            
            # Run migrations
            self.run_migrations()
            
            # Seed database
            self.seed_database()
            
            print("✅ Database reset completed")
        except Exception as e:
            print(f"❌ Database reset failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="ScrollIntel Database Migration Tool")
    parser.add_argument(
        "--env", 
        choices=["development", "staging", "production", "test"],
        help="Environment to run migrations for"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations")
    
    # Create migration command
    create_parser = subparsers.add_parser("create", help="Create new migration")
    create_parser.add_argument("message", help="Migration message")
    
    # Rollback command
    rollback_parser = subparsers.add_parser("rollback", help="Rollback migration")
    rollback_parser.add_argument(
        "--revision", 
        default="-1", 
        help="Revision to rollback to (default: -1)"
    )
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show migration history")
    
    # Current command
    current_parser = subparsers.add_parser("current", help="Show current revision")
    
    # Seed command
    seed_parser = subparsers.add_parser("seed", help="Seed database with initial data")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset database (DANGER: deletes all data)")
    reset_parser.add_argument(
        "--confirm", 
        action="store_true", 
        help="Confirm database reset"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize migrator
    migrator = DatabaseMigrator(args.env)
    
    # Check database connection
    if not migrator.check_database_connection():
        if args.command == "migrate":
            migrator.create_database_if_not_exists()
            if not migrator.check_database_connection():
                print("❌ Could not establish database connection")
                sys.exit(1)
        else:
            print("❌ Database connection required for this command")
            sys.exit(1)
    
    try:
        if args.command == "migrate":
            migrator.run_migrations()
        elif args.command == "create":
            migrator.create_migration(args.message)
        elif args.command == "rollback":
            migrator.rollback_migration(args.revision)
        elif args.command == "history":
            migrator.show_migration_history()
        elif args.command == "current":
            migrator.show_current_revision()
        elif args.command == "seed":
            migrator.seed_database()
        elif args.command == "reset":
            if not args.confirm:
                print("⚠️ This will delete all data. Use --confirm to proceed.")
                sys.exit(1)
            migrator.reset_database()
            
    except Exception as e:
        print(f"❌ Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()