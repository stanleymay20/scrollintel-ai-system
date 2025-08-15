"""
Data models for Automated Research Engine

Defines database models for:
- Research topics
- Literature sources and analysis
- Research hypotheses
- Research plans and methodologies
"""

from sqlalchemy import Column, String, Text, Float, Integer, DateTime, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid
import enum

from scrollintel.models.database import Base


class ResearchDomainEnum(enum.Enum):
    """Research domains enumeration"""
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    MACHINE_LEARNING = "machine_learning"
    QUANTUM_COMPUTING = "quantum_computing"
    BIOTECHNOLOGY = "biotechnology"
    NANOTECHNOLOGY = "nanotechnology"
    RENEWABLE_ENERGY = "renewable_energy"
    SPACE_TECHNOLOGY = "space_technology"
    ROBOTICS = "robotics"
    BLOCKCHAIN = "blockchain"
    CYBERSECURITY = "cybersecurity"


class ResearchStatusEnum(enum.Enum):
    """Research project status enumeration"""
    PLANNING = "planning"
    ACTIVE = "active"
    ANALYSIS = "analysis"
    COMPLETED = "completed"
    PAUSED = "paused"


class ResearchTopic(Base):
    """Research topic model"""
    __tablename__ = "research_topics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False, index=True)
    domain = Column(SQLEnum(ResearchDomainEnum), nullable=False, index=True)
    description = Column(Text)
    keywords = Column(JSON, default=list)
    novelty_score = Column(Float, default=0.0)
    feasibility_score = Column(Float, default=0.0)
    impact_potential = Column(Float, default=0.0)
    research_gaps = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    literature_analyses = relationship("LiteratureAnalysis", back_populates="topic", cascade="all, delete-orphan")
    hypotheses = relationship("ResearchHypothesis", back_populates="topic", cascade="all, delete-orphan")
    research_plans = relationship("ResearchPlan", back_populates="topic", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ResearchTopic(id={self.id}, title='{self.title[:50]}...', domain={self.domain.value})>"


class LiteratureSource(Base):
    """Literature source model"""
    __tablename__ = "literature_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(UUID(as_uuid=True), ForeignKey("literature_analyses.id"), nullable=False)
    title = Column(String(1000), nullable=False)
    authors = Column(JSON, default=list)
    publication_year = Column(Integer)
    journal = Column(String(500))
    doi = Column(String(200))
    abstract = Column(Text)
    keywords = Column(JSON, default=list)
    citation_count = Column(Integer, default=0)
    relevance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    analysis = relationship("LiteratureAnalysis", back_populates="sources")
    
    def __repr__(self):
        return f"<LiteratureSource(id={self.id}, title='{self.title[:50]}...', year={self.publication_year})>"


class LiteratureAnalysis(Base):
    """Literature analysis model"""
    __tablename__ = "literature_analyses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id = Column(UUID(as_uuid=True), ForeignKey("research_topics.id"), nullable=False)
    knowledge_gaps = Column(JSON, default=list)
    research_trends = Column(JSON, default=list)
    key_findings = Column(JSON, default=list)
    methodological_gaps = Column(JSON, default=list)
    theoretical_gaps = Column(JSON, default=list)
    empirical_gaps = Column(JSON, default=list)
    analysis_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    topic = relationship("ResearchTopic", back_populates="literature_analyses")
    sources = relationship("LiteratureSource", back_populates="analysis", cascade="all, delete-orphan")
    hypotheses = relationship("ResearchHypothesis", back_populates="literature_analysis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<LiteratureAnalysis(id={self.id}, topic_id={self.topic_id}, confidence={self.analysis_confidence})>"


class ResearchHypothesis(Base):
    """Research hypothesis model"""
    __tablename__ = "research_hypotheses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id = Column(UUID(as_uuid=True), ForeignKey("research_topics.id"), nullable=False)
    literature_analysis_id = Column(UUID(as_uuid=True), ForeignKey("literature_analyses.id"), nullable=False)
    statement = Column(Text, nullable=False)
    null_hypothesis = Column(Text)
    alternative_hypothesis = Column(Text)
    variables = Column(JSON, default=dict)
    testability_score = Column(Float, default=0.0)
    novelty_score = Column(Float, default=0.0)
    significance_potential = Column(Float, default=0.0)
    required_resources = Column(JSON, default=list)
    expected_timeline_days = Column(Integer, default=30)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    topic = relationship("ResearchTopic", back_populates="hypotheses")
    literature_analysis = relationship("LiteratureAnalysis", back_populates="hypotheses")
    research_plans = relationship("ResearchPlan", back_populates="hypothesis", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ResearchHypothesis(id={self.id}, statement='{self.statement[:50]}...', testability={self.testability_score})>"


class ResearchMethodology(Base):
    """Research methodology model"""
    __tablename__ = "research_methodologies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan_id = Column(UUID(as_uuid=True), ForeignKey("research_plans.id"), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    methodology_type = Column(String(100))  # experimental, observational, computational, theoretical
    data_collection_methods = Column(JSON, default=list)
    analysis_methods = Column(JSON, default=list)
    validation_approaches = Column(JSON, default=list)
    limitations = Column(JSON, default=list)
    ethical_considerations = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    plan = relationship("ResearchPlan", back_populates="methodology")
    
    def __repr__(self):
        return f"<ResearchMethodology(id={self.id}, name='{self.name}', type={self.methodology_type})>"


class ResearchPlan(Base):
    """Research plan model"""
    __tablename__ = "research_plans"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id = Column(UUID(as_uuid=True), ForeignKey("research_topics.id"), nullable=False)
    hypothesis_id = Column(UUID(as_uuid=True), ForeignKey("research_hypotheses.id"), nullable=False)
    title = Column(String(500), nullable=False)
    objectives = Column(JSON, default=list)
    timeline = Column(JSON, default=dict)
    milestones = Column(JSON, default=list)
    resource_requirements = Column(JSON, default=dict)
    success_criteria = Column(JSON, default=list)
    risk_assessment = Column(JSON, default=dict)
    status = Column(SQLEnum(ResearchStatusEnum), default=ResearchStatusEnum.PLANNING, index=True)
    progress_percentage = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    topic = relationship("ResearchTopic", back_populates="research_plans")
    hypothesis = relationship("ResearchHypothesis", back_populates="research_plans")
    methodology = relationship("ResearchMethodology", back_populates="plan", uselist=False, cascade="all, delete-orphan")
    experiments = relationship("ResearchExperiment", back_populates="plan", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ResearchPlan(id={self.id}, title='{self.title[:50]}...', status={self.status.value})>"


class ResearchExperiment(Base):
    """Research experiment model"""
    __tablename__ = "research_experiments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plan_id = Column(UUID(as_uuid=True), ForeignKey("research_plans.id"), nullable=False)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    experimental_design = Column(JSON, default=dict)
    protocol = Column(JSON, default=dict)
    results = Column(JSON, default=dict)
    analysis = Column(JSON, default=dict)
    conclusions = Column(JSON, default=list)
    confidence_level = Column(Float, default=0.0)
    status = Column(String(50), default="planned")
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    plan = relationship("ResearchPlan", back_populates="experiments")
    
    def __repr__(self):
        return f"<ResearchExperiment(id={self.id}, name='{self.name}', status={self.status})>"


class ResearchProject(Base):
    """Research project model for tracking autonomous research sessions"""
    __tablename__ = "research_projects"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    domain = Column(SQLEnum(ResearchDomainEnum), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    status = Column(SQLEnum(ResearchStatusEnum), default=ResearchStatusEnum.PLANNING, index=True)
    topic_count = Column(Integer, default=0)
    hypothesis_count = Column(Integer, default=0)
    plan_count = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    results_summary = Column(JSON, default=dict)
    started_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<ResearchProject(id={self.id}, title='{self.title[:50]}...', status={self.status.value})>"


class ResearchMetrics(Base):
    """Research metrics and analytics model"""
    __tablename__ = "research_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("research_projects.id"))
    metric_type = Column(String(100), nullable=False)  # topic_generation, literature_analysis, etc.
    metric_name = Column(String(200), nullable=False)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50))
    context = Column(JSON, default=dict)
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f"<ResearchMetrics(id={self.id}, type={self.metric_type}, name={self.metric_name}, value={self.metric_value})>"


# Database utility functions
def create_research_tables(engine):
    """Create all research-related tables"""
    Base.metadata.create_all(engine)


def get_research_topic_by_id(db_session, topic_id: str):
    """Get research topic by ID"""
    return db_session.query(ResearchTopic).filter(ResearchTopic.id == topic_id).first()


def get_literature_analysis_by_topic(db_session, topic_id: str):
    """Get literature analysis for a topic"""
    return db_session.query(LiteratureAnalysis).filter(LiteratureAnalysis.topic_id == topic_id).first()


def get_hypotheses_by_topic(db_session, topic_id: str):
    """Get all hypotheses for a topic"""
    return db_session.query(ResearchHypothesis).filter(ResearchHypothesis.topic_id == topic_id).all()


def get_research_plans_by_topic(db_session, topic_id: str):
    """Get all research plans for a topic"""
    return db_session.query(ResearchPlan).filter(ResearchPlan.topic_id == topic_id).all()


def get_active_research_projects(db_session):
    """Get all active research projects"""
    return db_session.query(ResearchProject).filter(
        ResearchProject.status.in_([ResearchStatusEnum.PLANNING, ResearchStatusEnum.ACTIVE])
    ).all()


def create_research_topic(db_session, topic_data: dict):
    """Create a new research topic"""
    topic = ResearchTopic(**topic_data)
    db_session.add(topic)
    db_session.commit()
    db_session.refresh(topic)
    return topic


def create_literature_analysis(db_session, analysis_data: dict):
    """Create a new literature analysis"""
    analysis = LiteratureAnalysis(**analysis_data)
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)
    return analysis


def create_research_hypothesis(db_session, hypothesis_data: dict):
    """Create a new research hypothesis"""
    hypothesis = ResearchHypothesis(**hypothesis_data)
    db_session.add(hypothesis)
    db_session.commit()
    db_session.refresh(hypothesis)
    return hypothesis


def create_research_plan(db_session, plan_data: dict):
    """Create a new research plan"""
    plan = ResearchPlan(**plan_data)
    db_session.add(plan)
    db_session.commit()
    db_session.refresh(plan)
    return plan


def update_research_project_status(db_session, project_id: str, status: ResearchStatusEnum, progress: float = None):
    """Update research project status and progress"""
    project = db_session.query(ResearchProject).filter(ResearchProject.id == project_id).first()
    if project:
        project.status = status
        if progress is not None:
            project.progress_percentage = progress
        project.updated_at = datetime.utcnow()
        db_session.commit()
        db_session.refresh(project)
    return project


def record_research_metric(db_session, metric_data: dict):
    """Record a research metric"""
    metric = ResearchMetrics(**metric_data)
    db_session.add(metric)
    db_session.commit()
    db_session.refresh(metric)
    return metric