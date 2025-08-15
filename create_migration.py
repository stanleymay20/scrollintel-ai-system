#!/usr/bin/env python3
"""
Script to create the initial database migration for ScrollIntel.
"""

import os
import sys
from alembic.config import Config
from alembic import command

def create_initial_migration():
    """Create the initial database migration."""
    try:
        # Set up the alembic configuration
        alembic_cfg = Config("alembic.ini")
        
        # Create the initial migration
        print("Creating initial migration...")
        command.revision(
            alembic_cfg, 
            message="Initial migration - create all tables",
            autogenerate=True
        )
        print("Initial migration created successfully!")
        
    except Exception as e:
        print(f"Error creating migration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_initial_migration()