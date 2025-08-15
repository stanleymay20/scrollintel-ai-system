"""
Data models for Board Dynamics Analysis System

This module defines the data models used by the board dynamics analysis engine
for storing and managing board member information, meeting data, and analysis results.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

Base = declarative_base()


class BoardMemberModel(Base):
    """Database model for board members"""
    __tablename__ = "board_members"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    background_data = Column(JSON)  # Stores Background dataclass as JSON
    expertise_areas = Column(JSON)  # List of expertise areas
    influence_level = Column(String)  # InfluenceLevel enum value
    communication_style = Column(String)  # CommunicationStyle enum value
    decision_making_pattern = Column(String)  # DecisionPattern enum value
    tenure = Column(Integer, default=0)
    committee_memberships = Column(JSON)  # List of committee names
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    relationships = relationship("BoardRelationshipModel", foreign_keys="BoardRelationshipModel.member_id")
    priorities = relationship("BoardMemberPriorityModel", back_populates="member")


class BoardRelationshipModel(Base):
    """Database model for board member relationships"""
    __tablename__ = "board_relationships"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(String, ForeignKey("board_members.id"), nullable=False)
    related_member_id = Column(String, ForeignKey("board_members.id"), nullable=False)
    relationship_type = Column(String, nullable=False)
    strength = Column(Float, nullable=False)
    influence_direction = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BoardMemberPriorityModel(Base):
    """Database model for board member priorities"""
    __tablename__ = "board_member_priorities"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    member_id = Column(String, ForeignKey("board_members.id"), nullable=False)
    area = Column(String, nullable=False)
    importance = Column(Float, nullable=False)
    description = Column(Text)
    timeline = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    member = relationship("BoardMemberModel", back_populates="priorities")


class BoardMeetingModel(Base):
    """Database model for board meetings"""
    __tablename__ = "board_meetings"
    
    id = Column(String, primary_key=True)
    meeting_date = Column(DateTime, nullable=False)
    meeting_type = Column(String)  # regular, special, emergency
    agenda_items = Column(JSON)  # List of agenda items
    attendance = Column(JSON)  # List of attending member IDs
    meeting_duration = Column(Integer)  # Duration in minutes
    meeting_data = Column(JSON)  # Comprehensive meeting data for analysis
    effectiveness_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    participation_records = relationship("MeetingParticipationModel", back_populates="meeting")


class MeetingParticipationModel(Base):
    """Database model for meeting participation records"""
    __tablename__ = "meeting_participation"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(String, ForeignKey("board_meetings.id"), nullable=False)
    member_id = Column(String, ForeignKey("board_members.id"), nullable=False)
    speaking_time = Column(Integer)  # Speaking time in seconds
    questions_asked = Column(Integer, default=0)
    contributions = Column(Integer, default=0)
    engagement_score = Column(Float)
    participation_data = Column(JSON)  # Additional participation metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    meeting = relationship("BoardMeetingModel", back_populates="participation_records")


class BoardAnalysisModel(Base):
    """Database model for storing board analysis results"""
    __tablename__ = "board_analyses"
    
    id = Column(String, primary_key=True)
    analysis_type = Column(String, nullable=False)  # composition, power_structure, dynamics, governance
    analysis_date = Column(DateTime, default=datetime.utcnow)
    board_members_analyzed = Column(JSON)  # List of member IDs included in analysis
    analysis_results = Column(JSON)  # Complete analysis results
    insights = Column(JSON)  # Key insights from analysis
    recommendations = Column(JSON)  # Actionable recommendations
    overall_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class GovernanceFrameworkModel(Base):
    """Database model for governance framework information"""
    __tablename__ = "governance_frameworks"
    
    id = Column(String, primary_key=True)
    framework_name = Column(String, nullable=False)
    board_structure = Column(JSON)
    committee_structure = Column(JSON)
    decision_processes = Column(JSON)
    reporting_requirements = Column(JSON)
    compliance_frameworks = Column(JSON)
    effectiveness_scores = Column(JSON)
    governance_gaps = Column(JSON)
    last_review_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class BoardCommitteeModel(Base):
    """Database model for board committees"""
    __tablename__ = "board_committees"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    committee_type = Column(String)  # audit, compensation, nominating, etc.
    chair_member_id = Column(String, ForeignKey("board_members.id"))
    members = Column(JSON)  # List of member IDs
    charter = Column(Text)
    meeting_frequency = Column(String)
    responsibilities = Column(JSON)
    performance_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PowerStructureModel(Base):
    """Database model for power structure analysis results"""
    __tablename__ = "power_structures"
    
    id = Column(String, primary_key=True)
    analysis_date = Column(DateTime, default=datetime.utcnow)
    influence_networks = Column(JSON)
    decision_makers = Column(JSON)
    coalition_groups = Column(JSON)
    influence_flows = Column(JSON)
    key_relationships = Column(JSON)
    power_distribution_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class MeetingDynamicsModel(Base):
    """Database model for meeting dynamics analysis results"""
    __tablename__ = "meeting_dynamics"
    
    id = Column(String, primary_key=True)
    meeting_id = Column(String, ForeignKey("board_meetings.id"))
    analysis_date = Column(DateTime, default=datetime.utcnow)
    participation_patterns = Column(JSON)
    communication_patterns = Column(JSON)
    engagement_levels = Column(JSON)
    conflict_indicators = Column(JSON)
    collaboration_quality = Column(Float)
    meeting_effectiveness = Column(Float)
    decision_efficiency = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# Pydantic models for API serialization
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class BackgroundSchema(BaseModel):
    """Schema for board member background information"""
    industry_experience: List[str]
    functional_expertise: List[str]
    education: List[str]
    previous_roles: List[str]
    years_experience: int


class PrioritySchema(BaseModel):
    """Schema for board member priorities"""
    area: str
    importance: float
    description: str
    timeline: str


class RelationshipSchema(BaseModel):
    """Schema for board member relationships"""
    member_id: str
    relationship_type: str
    strength: float
    influence_direction: str


class BoardMemberSchema(BaseModel):
    """Schema for board member information"""
    id: str
    name: str
    background: BackgroundSchema
    expertise_areas: List[str]
    influence_level: str
    communication_style: str
    decision_making_pattern: str
    relationships: List[RelationshipSchema] = []
    priorities: List[PrioritySchema] = []
    tenure: int = 0
    committee_memberships: List[str] = []
    
    class Config:
        from_attributes = True


class MeetingParticipationSchema(BaseModel):
    """Schema for meeting participation data"""
    member_id: str
    speaking_time: int
    questions_asked: int
    contributions: int
    engagement_score: float
    participation_data: Dict[str, Any] = {}


class BoardMeetingSchema(BaseModel):
    """Schema for board meeting information"""
    id: str
    meeting_date: datetime
    meeting_type: str
    agenda_items: List[str]
    attendance: List[str]
    meeting_duration: int
    meeting_data: Dict[str, Any]
    effectiveness_score: Optional[float] = None
    participation_records: List[MeetingParticipationSchema] = []
    
    class Config:
        from_attributes = True


class CompositionAnalysisSchema(BaseModel):
    """Schema for board composition analysis results"""
    member_profiles: List[BoardMemberSchema]
    expertise_coverage: Dict[str, List[str]]
    experience_distribution: Dict[str, int]
    diversity_metrics: Dict[str, Any]
    skill_gaps: List[str]
    strengths: List[str]


class PowerStructureMapSchema(BaseModel):
    """Schema for power structure mapping results"""
    influence_networks: Dict[str, List[str]]
    decision_makers: List[str]
    coalition_groups: List[List[str]]
    influence_flows: Dict[str, Dict[str, float]]
    key_relationships: List[RelationshipSchema]


class DynamicsAssessmentSchema(BaseModel):
    """Schema for meeting dynamics assessment results"""
    meeting_effectiveness: float
    engagement_levels: Dict[str, float]
    communication_patterns: Dict[str, Any]
    decision_efficiency: float
    conflict_indicators: List[str]
    collaboration_quality: float


class GovernanceFrameworkSchema(BaseModel):
    """Schema for governance framework information"""
    board_structure: Dict[str, Any]
    committee_structure: Dict[str, Any]
    decision_processes: Dict[str, Any]
    reporting_requirements: List[str]
    compliance_frameworks: List[str]


class BoardAnalysisResultSchema(BaseModel):
    """Schema for comprehensive board analysis results"""
    composition_analysis: CompositionAnalysisSchema
    power_structure_map: PowerStructureMapSchema
    dynamics_assessment: DynamicsAssessmentSchema
    governance_analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[Dict[str, Any]]
    analysis_timestamp: str


class BoardCommitteeSchema(BaseModel):
    """Schema for board committee information"""
    id: str
    name: str
    committee_type: str
    chair_member_id: Optional[str] = None
    members: List[str]
    charter: Optional[str] = None
    meeting_frequency: str
    responsibilities: List[str]
    performance_metrics: Dict[str, Any] = {}
    
    class Config:
        from_attributes = True


class AnalysisRequestSchema(BaseModel):
    """Schema for analysis request parameters"""
    analysis_type: str  # composition, power_structure, dynamics, governance, comprehensive
    member_ids: List[str]
    meeting_ids: Optional[List[str]] = None
    include_historical: bool = False
    analysis_parameters: Dict[str, Any] = {}


class AnalysisResponseSchema(BaseModel):
    """Schema for analysis response"""
    analysis_id: str
    analysis_type: str
    status: str
    results: Dict[str, Any]
    insights: List[str]
    recommendations: List[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


@dataclass
class Board:
    """Represents a board of directors"""
    id: str
    name: str
    members: List['BoardMember']
    committees: List[str]
    governance_structure: Dict[str, Any]
    meeting_schedule: List[datetime]
    created_at: datetime
    updated_at: datetime


# Import BoardMember from the engine since it's defined there
from ..engines.board_dynamics_engine import BoardMember


# Utility functions for model conversion
def board_member_to_schema(member_model: BoardMemberModel) -> BoardMemberSchema:
    """Convert database model to schema"""
    background_data = member_model.background_data or {}
    background = BackgroundSchema(
        industry_experience=background_data.get("industry_experience", []),
        functional_expertise=background_data.get("functional_expertise", []),
        education=background_data.get("education", []),
        previous_roles=background_data.get("previous_roles", []),
        years_experience=background_data.get("years_experience", 0)
    )
    
    relationships = []
    for rel_model in member_model.relationships:
        relationships.append(RelationshipSchema(
            member_id=rel_model.related_member_id,
            relationship_type=rel_model.relationship_type,
            strength=rel_model.strength,
            influence_direction=rel_model.influence_direction
        ))
    
    priorities = []
    for priority_model in member_model.priorities:
        priorities.append(PrioritySchema(
            area=priority_model.area,
            importance=priority_model.importance,
            description=priority_model.description,
            timeline=priority_model.timeline
        ))
    
    return BoardMemberSchema(
        id=member_model.id,
        name=member_model.name,
        background=background,
        expertise_areas=member_model.expertise_areas or [],
        influence_level=member_model.influence_level,
        communication_style=member_model.communication_style,
        decision_making_pattern=member_model.decision_making_pattern,
        relationships=relationships,
        priorities=priorities,
        tenure=member_model.tenure,
        committee_memberships=member_model.committee_memberships or []
    )


def schema_to_board_member(member_schema: BoardMemberSchema) -> BoardMemberModel:
    """Convert schema to database model"""
    background_data = {
        "industry_experience": member_schema.background.industry_experience,
        "functional_expertise": member_schema.background.functional_expertise,
        "education": member_schema.background.education,
        "previous_roles": member_schema.background.previous_roles,
        "years_experience": member_schema.background.years_experience
    }
    
    return BoardMemberModel(
        id=member_schema.id,
        name=member_schema.name,
        background_data=background_data,
        expertise_areas=member_schema.expertise_areas,
        influence_level=member_schema.influence_level,
        communication_style=member_schema.communication_style,
        decision_making_pattern=member_schema.decision_making_pattern,
        tenure=member_schema.tenure,
        committee_memberships=member_schema.committee_memberships
    )


def meeting_to_schema(meeting_model: BoardMeetingModel) -> BoardMeetingSchema:
    """Convert meeting database model to schema"""
    participation_records = []
    for participation in meeting_model.participation_records:
        participation_records.append(MeetingParticipationSchema(
            member_id=participation.member_id,
            speaking_time=participation.speaking_time,
            questions_asked=participation.questions_asked,
            contributions=participation.contributions,
            engagement_score=participation.engagement_score,
            participation_data=participation.participation_data or {}
        ))
    
    return BoardMeetingSchema(
        id=meeting_model.id,
        meeting_date=meeting_model.meeting_date,
        meeting_type=meeting_model.meeting_type,
        agenda_items=meeting_model.agenda_items or [],
        attendance=meeting_model.attendance or [],
        meeting_duration=meeting_model.meeting_duration,
        meeting_data=meeting_model.meeting_data or {},
        effectiveness_score=meeting_model.effectiveness_score,
        participation_records=participation_records
    )