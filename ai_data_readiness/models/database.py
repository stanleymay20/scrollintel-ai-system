"""Database models and connection management."""

from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Text, JSON, Enum as SQLEnum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
from typing import Optional

from .base_models import DatasetStatus, QualityDimension, BiasType

Base = declarative_base()


class DatasetModel(Base):
    """SQLAlchemy model for datasets."""
    __tablename__ = "datasets"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    schema_definition = Column(JSON)
    dataset_metadata = Column(JSON)  # Renamed from 'metadata' to avoid conflict
    quality_score = Column(Float, default=0.0)
    ai_readiness_score = Column(Float, default=0.0)
    status = Column(SQLEnum(DatasetStatus), default=DatasetStatus.PENDING)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    version = Column(String(50), default="1.0")
    lineage = Column(JSON)
    owner = Column(String(255))
    tags = Column(JSON)


class QualityReportModel(Base):
    """SQLAlchemy model for quality reports."""
    __tablename__ = "quality_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    overall_score = Column(Float, nullable=False)
    completeness_score = Column(Float, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    consistency_score = Column(Float, nullable=False)
    validity_score = Column(Float, nullable=False)
    uniqueness_score = Column(Float, default=0.0)
    timeliness_score = Column(Float, default=0.0)
    issues = Column(JSON)
    recommendations = Column(JSON)
    generated_at = Column(DateTime, default=datetime.utcnow)


class BiasReportModel(Base):
    """SQLAlchemy model for bias reports."""
    __tablename__ = "bias_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    protected_attributes = Column(JSON)
    bias_metrics = Column(JSON)
    fairness_violations = Column(JSON)
    mitigation_strategies = Column(JSON)
    generated_at = Column(DateTime, default=datetime.utcnow)


class AIReadinessScoreModel(Base):
    """SQLAlchemy model for AI readiness scores."""
    __tablename__ = "ai_readiness_scores"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    overall_score = Column(Float, nullable=False)
    data_quality_score = Column(Float, nullable=False)
    feature_quality_score = Column(Float, nullable=False)
    bias_score = Column(Float, nullable=False)
    compliance_score = Column(Float, nullable=False)
    scalability_score = Column(Float, nullable=False)
    dimensions = Column(JSON)
    improvement_areas = Column(JSON)
    generated_at = Column(DateTime, default=datetime.utcnow)


class DriftReportModel(Base):
    """SQLAlchemy model for drift reports."""
    __tablename__ = "drift_reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    reference_dataset_id = Column(UUID(as_uuid=True), nullable=False)
    drift_score = Column(Float, nullable=False)
    feature_drift_scores = Column(JSON)
    statistical_tests = Column(JSON)
    alerts = Column(JSON)
    recommendations = Column(JSON)
    generated_at = Column(DateTime, default=datetime.utcnow)


class ProcessingJobModel(Base):
    """SQLAlchemy model for processing jobs."""
    __tablename__ = "processing_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), nullable=False)
    job_type = Column(String(100), nullable=False)  # ingestion, quality_assessment, etc.
    status = Column(String(50), default="pending")
    progress = Column(Float, default=0.0)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    config = Column(JSON)
    results = Column(JSON)


class Database:
    """Database connection and session management."""
    
    def __init__(self, connection_string: str):
        self.engine = create_engine(connection_string, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception:
            return False


# Global database instance
_database_instance: Optional[Database] = None


def init_database(connection_string: str = None) -> Database:
    """Initialize the global database instance."""
    global _database_instance
    if connection_string is None:
        from ..core.config import get_settings
        settings = get_settings()
        connection_string = settings.database.connection_string
    
    _database_instance = Database(connection_string)
    _database_instance.create_tables()
    return _database_instance


def get_database() -> Database:
    """Get the global database instance."""
    global _database_instance
    if _database_instance is None:
        _database_instance = init_database()
    return _database_instance


def get_db_session() -> Session:
    """Get a database session."""
    return get_database().get_session()