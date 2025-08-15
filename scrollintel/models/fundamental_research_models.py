"""
Fundamental Research Models for Big Tech CTO Capabilities

This module defines data models for breakthrough research capabilities including
research breakthroughs, hypotheses, and experimental designs.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, DateTime, Float, Integer, Text, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class ResearchDomain(str, Enum):
    """Research domains for fundamental research"""
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    QUANTUM_COMPUTING = "quantum_computing"
    BIOTECHNOLOGY = "biotechnology"
    NANOTECHNOLOGY = "nanotechnology"
    MATERIALS_SCIENCE = "materials_science"
    ENERGY_SYSTEMS = "energy_systems"
    SPACE_TECHNOLOGY = "space_technology"
    NEUROSCIENCE = "neuroscience"
    ROBOTICS = "robotics"
    CRYPTOGRAPHY = "cryptography"

class ResearchMethodology(str, Enum):
    """Research methodologies"""
    EXPERIMENTAL = "experimental"
    THEORETICAL = "theoretical"
    COMPUTATIONAL = "computational"
    OBSERVATIONAL = "observational"
    MIXED_METHODS = "mixed_methods"

class HypothesisStatus(str, Enum):
    """Status of research hypothesis"""
    PROPOSED = "proposed"
    UNDER_INVESTIGATION = "under_investigation"
    VALIDATED = "validated"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"

class PublicationStatus(str, Enum):
    """Publication status of research"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    PUBLISHED = "published"
    REJECTED = "rejected"

# Pydantic Models for API
class HypothesisCreate(BaseModel):
    """Model for creating a research hypothesis"""
    title: str = Field(..., description="Title of the hypothesis")
    description: str = Field(..., description="Detailed description of the hypothesis")
    domain: ResearchDomain = Field(..., description="Research domain")
    theoretical_foundation: str = Field(..., description="Theoretical basis for the hypothesis")
    testable_predictions: List[str] = Field(..., description="Testable predictions derived from hypothesis")
    novelty_score: float = Field(ge=0.0, le=1.0, description="Novelty score (0-1)")
    feasibility_score: float = Field(ge=0.0, le=1.0, description="Feasibility score (0-1)")
    impact_potential: float = Field(ge=0.0, le=1.0, description="Potential impact score (0-1)")

class HypothesisResponse(BaseModel):
    """Response model for hypothesis"""
    id: str
    title: str
    description: str
    domain: ResearchDomain
    status: HypothesisStatus
    theoretical_foundation: str
    testable_predictions: List[str]
    novelty_score: float
    feasibility_score: float
    impact_potential: float
    created_at: datetime
    updated_at: datetime

class ExperimentDesign(BaseModel):
    """Model for experimental design"""
    hypothesis_id: str = Field(..., description="Associated hypothesis ID")
    methodology: ResearchMethodology = Field(..., description="Research methodology")
    experimental_setup: str = Field(..., description="Detailed experimental setup")
    variables: Dict[str, Any] = Field(..., description="Independent and dependent variables")
    controls: List[str] = Field(..., description="Control conditions")
    measurements: List[str] = Field(..., description="What will be measured")
    timeline: Dict[str, str] = Field(..., description="Experimental timeline")
    resources_required: Dict[str, Any] = Field(..., description="Required resources")
    success_criteria: List[str] = Field(..., description="Criteria for success")

class ExperimentResults(BaseModel):
    """Model for experiment results"""
    experiment_id: str = Field(..., description="Associated experiment ID")
    raw_data: Dict[str, Any] = Field(..., description="Raw experimental data")
    processed_data: Dict[str, Any] = Field(..., description="Processed data")
    statistical_analysis: Dict[str, Any] = Field(..., description="Statistical analysis results")
    observations: List[str] = Field(..., description="Key observations")
    anomalies: List[str] = Field(default=[], description="Observed anomalies")
    confidence_level: float = Field(ge=0.0, le=1.0, description="Confidence in results")

class ResearchInsight(BaseModel):
    """Model for research insights"""
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed insight description")
    significance: float = Field(ge=0.0, le=1.0, description="Significance score")
    implications: List[str] = Field(..., description="Implications of the insight")
    related_work: List[str] = Field(default=[], description="Related research work")

class ResearchBreakthroughCreate(BaseModel):
    """Model for creating a research breakthrough"""
    title: str = Field(..., description="Title of the breakthrough")
    domain: ResearchDomain = Field(..., description="Research domain")
    hypothesis_id: str = Field(..., description="Associated hypothesis ID")
    methodology: ResearchMethodology = Field(..., description="Research methodology used")
    key_findings: List[str] = Field(..., description="Key research findings")
    insights: List[ResearchInsight] = Field(..., description="Research insights")
    implications: List[str] = Field(..., description="Broader implications")
    novelty_assessment: float = Field(ge=0.0, le=1.0, description="Novelty assessment score")
    impact_assessment: float = Field(ge=0.0, le=1.0, description="Impact assessment score")
    reproducibility_score: float = Field(ge=0.0, le=1.0, description="Reproducibility score")

class ResearchBreakthroughResponse(BaseModel):
    """Response model for research breakthrough"""
    id: str
    title: str
    domain: ResearchDomain
    hypothesis_id: str
    methodology: ResearchMethodology
    key_findings: List[str]
    insights: List[ResearchInsight]
    implications: List[str]
    novelty_assessment: float
    impact_assessment: float
    reproducibility_score: float
    publication_status: PublicationStatus
    created_at: datetime
    updated_at: datetime

class ResearchPaper(BaseModel):
    """Model for generated research paper"""
    breakthrough_id: str = Field(..., description="Associated breakthrough ID")
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    introduction: str = Field(..., description="Introduction section")
    methodology: str = Field(..., description="Methodology section")
    results: str = Field(..., description="Results section")
    discussion: str = Field(..., description="Discussion section")
    conclusion: str = Field(..., description="Conclusion section")
    references: List[str] = Field(..., description="References")
    keywords: List[str] = Field(..., description="Keywords")
    publication_readiness: float = Field(ge=0.0, le=1.0, description="Publication readiness score")

# SQLAlchemy Models
class Hypothesis(Base):
    """SQLAlchemy model for research hypothesis"""
    __tablename__ = "hypotheses"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    domain = Column(String, nullable=False)
    status = Column(String, default=HypothesisStatus.PROPOSED)
    theoretical_foundation = Column(Text, nullable=False)
    testable_predictions = Column(JSON, nullable=False)
    novelty_score = Column(Float, nullable=False)
    feasibility_score = Column(Float, nullable=False)
    impact_potential = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    breakthroughs = relationship("ResearchBreakthrough", back_populates="hypothesis")

class ResearchBreakthrough(Base):
    """SQLAlchemy model for research breakthrough"""
    __tablename__ = "research_breakthroughs"
    
    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    hypothesis_id = Column(String, ForeignKey("hypotheses.id"), nullable=False)
    methodology = Column(String, nullable=False)
    key_findings = Column(JSON, nullable=False)
    insights = Column(JSON, nullable=False)
    implications = Column(JSON, nullable=False)
    novelty_assessment = Column(Float, nullable=False)
    impact_assessment = Column(Float, nullable=False)
    reproducibility_score = Column(Float, nullable=False)
    publication_status = Column(String, default=PublicationStatus.DRAFT)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    hypothesis = relationship("Hypothesis", back_populates="breakthroughs")