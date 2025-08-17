"""
Database connection and session management for ScrollIntel.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from .config import get_config

config = get_config()

# Create database engine
engine = create_engine(
    config.get('database_url', 'sqlite:///./scrollintel.db'),
    pool_pre_ping=True,
    pool_recycle=300,
    echo=config.get('debug', False)
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()