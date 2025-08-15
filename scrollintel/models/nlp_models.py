"""
Data models for Natural Language Processing in the Automated Code Generation System.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class RequirementType(str, Enum):
    """Types of requirements that can be extracted."""
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS = "business"
    TECHNICAL = "technical"
    UI_UX = "ui_ux"
    DATA = "data"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"


class IntentType(str, Enum):
    """Types of intents that can be classified."""
    CREATE_APPLICATION = "create_application"
    MODIFY_FEATURE = "modify_feature"
    ADD_FUNCTIONALITY = "add_functionality"
    INTEGRATE_SYSTEM = "integrate_system"
    OPTIMIZE_PERFORMANCE = "optimize_performance"
    ENHANCE_SECURITY = "enhance_security"
    IMPROVE_UI = "improve_ui"
    MANAGE_DATA = "manage_data"
    DEPLOY_APPLICATION = "deploy_application"
    CLARIFY_REQUIREMENT = "clarify_requirement"


class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    USER_ROLE = "user_role"
    BUSINESS_OBJECT = "business_object"
    SYSTEM_COMPONENT = "system_component"
    DATA_ENTITY = "data_entity"
    TECHNOLOGY = "technology"
    FEATURE = "feature"
    CONSTRAINT = "constraint"
    METRIC = "metric"
    INTEGRATION_POINT = "integration_point"


class ConfidenceLevel(str, Enum):
    """Confidence levels for NLP processing results."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# SQLAlchemy Models
class Requirements(Base):
    """Database model for storing requirements."""
    __tablename__ = "requirements"
    
    id = Column(Integer, primary_key=True, index=True)
    raw_text = Column(Text, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow)
    confidence_score = Column(Float, default=0.0)
    is_complete = Column(Boolean, default=False)
    needs_clarification = Column(Boolean, default=False)
    
    # Relationships
    parsed_requirements = relationship("ParsedRequirement", back_populates="requirements")
    entities = relationship("Entity", back_populates="requirements")
    clarifications = relationship("Clarification", back_populates="requirements")


class ParsedRequirement(Base):
    """Database model for individual parsed requirements."""
    __tablename__ = "parsed_requirements"
    
    id = Column(Integer, primary_key=True, index=True)
    requirements_id = Column(Integer, ForeignKey("requirements.id"))
    requirement_type = Column(String(50), nullable=False)
    intent = Column(String(50), nullable=False)
    description = Column(Text, nullable=False)
    priority = Column(String(20), default="medium")
    confidence_score = Column(Float, default=0.0)
    acceptance_criteria = Column(JSON)
    dependencies = Column(JSON)
    
    # Relationships
    requirements = relationship("Requirements", back_populates="parsed_requirements")


class Entity(Base):
    """Database model for extracted entities."""
    __tablename__ = "entities"
    
    id = Column(Integer, primary_key=True, index=True)
    requirements_id = Column(Integer, ForeignKey("requirements.id"))
    entity_type = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    attributes = Column(JSON)
    confidence_score = Column(Float, default=0.0)
    
    # Relationships
    requirements = relationship("Requirements", back_populates="entities")
    relationships = relationship("EntityRelationship", 
                               foreign_keys="EntityRelationship.source_entity_id",
                               back_populates="source_entity")


class EntityRelationship(Base):
    """Database model for relationships between entities."""
    __tablename__ = "entity_relationships"
    
    id = Column(Integer, primary_key=True, index=True)
    source_entity_id = Column(Integer, ForeignKey("entities.id"))
    target_entity_id = Column(Integer, ForeignKey("entities.id"))
    relationship_type = Column(String(50), nullable=False)
    description = Column(Text)
    confidence_score = Column(Float, default=0.0)
    
    # Relationships
    source_entity = relationship("Entity", foreign_keys=[source_entity_id])
    target_entity = relationship("Entity", foreign_keys=[target_entity_id])


class Clarification(Base):
    """Database model for clarification questions and responses."""
    __tablename__ = "clarifications"
    
    id = Column(Integer, primary_key=True, index=True)
    requirements_id = Column(Integer, ForeignKey("requirements.id"))
    question = Column(Text, nullable=False)
    answer = Column(Text)
    is_answered = Column(Boolean, default=False)
    priority = Column(String(20), default="medium")
    created_at = Column(DateTime, default=datetime.utcnow)
    answered_at = Column(DateTime)
    
    # Relationships
    requirements = relationship("Requirements", back_populates="clarifications")


# Pydantic Models for API
class EntityModel(BaseModel):
    """Pydantic model for entities."""
    id: Optional[int] = None
    entity_type: EntityType
    name: str
    description: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    confidence_score: float = 0.0
    
    class Config:
        from_attributes = True


class EntityRelationshipModel(BaseModel):
    """Pydantic model for entity relationships."""
    id: Optional[int] = None
    source_entity_id: int
    target_entity_id: int
    relationship_type: str
    description: Optional[str] = None
    confidence_score: float = 0.0
    
    class Config:
        from_attributes = True


class ParsedRequirementModel(BaseModel):
    """Pydantic model for parsed requirements."""
    id: Optional[int] = None
    requirement_type: RequirementType
    intent: IntentType
    description: str
    priority: str = "medium"
    confidence_score: float = 0.0
    acceptance_criteria: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class ClarificationModel(BaseModel):
    """Pydantic model for clarifications."""
    id: Optional[int] = None
    question: str
    answer: Optional[str] = None
    is_answered: bool = False
    priority: str = "medium"
    created_at: Optional[datetime] = None
    answered_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class RequirementsModel(BaseModel):
    """Pydantic model for requirements."""
    id: Optional[int] = None
    raw_text: str
    processed_at: Optional[datetime] = None
    confidence_score: float = 0.0
    is_complete: bool = False
    needs_clarification: bool = False
    parsed_requirements: List[ParsedRequirementModel] = Field(default_factory=list)
    entities: List[EntityModel] = Field(default_factory=list)
    clarifications: List[ClarificationModel] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class ProcessingResult(BaseModel):
    """Result of NLP processing."""
    requirements: RequirementsModel
    processing_time: float
    confidence_level: ConfidenceLevel
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Result of requirements validation."""
    is_valid: bool
    completeness_score: float
    missing_elements: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)
    confidence_score: float = 0.0