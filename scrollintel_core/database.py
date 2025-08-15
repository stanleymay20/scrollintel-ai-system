"""
Database configuration and session management
"""
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import redis
from typing import Generator

from .config import settings

# PostgreSQL Database Setup
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=QueuePool,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy Base
Base = declarative_base()

# Metadata for migrations
metadata = MetaData()

# Redis Setup
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
redis_session = redis.from_url(
    settings.REDIS_URL.replace("/0", f"/{settings.REDIS_SESSION_DB}"),
    decode_responses=True
)
redis_cache = redis.from_url(
    settings.REDIS_URL.replace("/0", f"/{settings.REDIS_CACHE_DB}"),
    decode_responses=True
)


def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Get Redis client"""
    return redis_client


def get_redis_session() -> redis.Redis:
    """Get Redis session client"""
    return redis_session


def get_redis_cache() -> redis.Redis:
    """Get Redis cache client"""
    return redis_cache


def create_tables():
    """Create all database tables"""
    # Import models to ensure they're registered
    from . import models
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)