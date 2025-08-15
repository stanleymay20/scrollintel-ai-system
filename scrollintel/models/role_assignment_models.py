"""
Data models for Role Assignment Engine
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Optional

Base = declarative_base()


class PersonModel(Base):
    """Database model for person information"""
    __tablename__ = "people"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    current_availability = Column(Float, nullable=False, default=1.0)
    stress_tolerance = Column(Float, nullable=False, default=0.5)
    leadership_experience = Column(Integer, nullable=False, default=0)
    current_workload = Column(Float, nullable=False, default=0.0)
    crisis_history = Column(JSON, nullable=True)  # List of crisis IDs
    preferred_roles = Column(JSON, nullable=True)  # List of role types
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    skills = relationship("PersonSkillModel", back_populates="person", cascade="all, delete-orphan")
    assignments = relationship("RoleAssignmentModel", back_populates="person")


class PersonSkillModel(Base):
    """Database model for person skills"""
    __tablename__ = "person_skills"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String, ForeignKey("people.id"), nullable=False)
    skill_name = Column(String, nullable=False)
    skill_level = Column(Integer, nullable=False)  # 1-5 scale
    years_experience = Column(Integer, nullable=False, default=0)
    recent_performance = Column(Float, nullable=False, default=0.5)
    crisis_experience = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    person = relationship("PersonModel", back_populates="skills")


class CrisisModel(Base):
    """Database model for crisis information"""
    __tablename__ = "crises"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    crisis_type = Column(String, nullable=False)
    severity_level = Column(Float, nullable=False)
    status = Column(String, nullable=False, default="active")
    description = Column(Text, nullable=True)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    affected_areas = Column(JSON, nullable=True)  # List of affected areas
    stakeholders_impacted = Column(JSON, nullable=True)  # List of stakeholder IDs
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assignments = relationship("RoleAssignmentModel", back_populates="crisis")


class RoleAssignmentModel(Base):
    """Database model for role assignments"""
    __tablename__ = "role_assignments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crisis_id = Column(String, ForeignKey("crises.id"), nullable=False)
    person_id = Column(String, ForeignKey("people.id"), nullable=False)
    role_type = Column(String, nullable=False)
    assignment_confidence = Column(Float, nullable=False)
    responsibilities = Column(JSON, nullable=True)  # List of responsibilities
    reporting_structure = Column(JSON, nullable=True)  # Dict of reporting relationships
    assignment_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    confirmation_status = Column(String, nullable=False, default="pending")  # pending, confirmed, declined
    confirmation_time = Column(DateTime, nullable=True)
    expected_duration = Column(Integer, nullable=True)  # hours
    actual_duration = Column(Integer, nullable=True)  # hours
    performance_rating = Column(Float, nullable=True)  # 0.0 to 1.0
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    crisis = relationship("CrisisModel", back_populates="assignments")
    person = relationship("PersonModel", back_populates="assignments")


class BackupAssignmentModel(Base):
    """Database model for backup role assignments"""
    __tablename__ = "backup_assignments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crisis_id = Column(String, ForeignKey("crises.id"), nullable=False)
    role_type = Column(String, nullable=False)
    person_id = Column(String, ForeignKey("people.id"), nullable=False)
    backup_priority = Column(Integer, nullable=False)  # 1 = first backup, 2 = second backup
    compatibility_score = Column(Float, nullable=False)
    activated = Column(Boolean, nullable=False, default=False)
    activation_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RoleDefinitionModel(Base):
    """Database model for role definitions and requirements"""
    __tablename__ = "role_definitions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    role_type = Column(String, nullable=False, unique=True)
    role_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    required_skills = Column(JSON, nullable=False)  # List of required skills
    minimum_skill_level = Column(Integer, nullable=False)
    leadership_required = Column(Boolean, nullable=False, default=False)
    stress_tolerance_required = Column(Float, nullable=False, default=0.5)
    priority = Column(Integer, nullable=False, default=5)
    estimated_workload = Column(Float, nullable=False, default=0.5)
    standard_responsibilities = Column(JSON, nullable=True)  # List of responsibilities
    reporting_structure = Column(JSON, nullable=True)  # Dict of reporting relationships
    success_criteria = Column(JSON, nullable=True)  # List of success criteria
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AssignmentHistoryModel(Base):
    """Database model for assignment history and analytics"""
    __tablename__ = "assignment_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    crisis_id = Column(String, nullable=False)
    assignment_session_id = Column(String, nullable=False)  # Groups assignments from same session
    total_roles_required = Column(Integer, nullable=False)
    total_roles_assigned = Column(Integer, nullable=False)
    assignment_quality_score = Column(Float, nullable=False)
    assignment_time = Column(DateTime, nullable=False)
    crisis_severity = Column(Float, nullable=False)
    people_available = Column(Integer, nullable=False)
    recommendations = Column(JSON, nullable=True)  # List of recommendations
    assignment_algorithm_version = Column(String, nullable=False, default="1.0")
    created_at = Column(DateTime, default=datetime.utcnow)


class RolePerformanceModel(Base):
    """Database model for tracking role performance metrics"""
    __tablename__ = "role_performance"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    assignment_id = Column(Integer, ForeignKey("role_assignments.id"), nullable=False)
    person_id = Column(String, ForeignKey("people.id"), nullable=False)
    role_type = Column(String, nullable=False)
    crisis_id = Column(String, ForeignKey("crises.id"), nullable=False)
    
    # Performance metrics
    response_time = Column(Float, nullable=True)  # Time to start role activities (minutes)
    task_completion_rate = Column(Float, nullable=True)  # 0.0 to 1.0
    communication_effectiveness = Column(Float, nullable=True)  # 0.0 to 1.0
    stakeholder_satisfaction = Column(Float, nullable=True)  # 0.0 to 1.0
    stress_handling = Column(Float, nullable=True)  # 0.0 to 1.0
    decision_quality = Column(Float, nullable=True)  # 0.0 to 1.0
    team_coordination = Column(Float, nullable=True)  # 0.0 to 1.0
    overall_performance = Column(Float, nullable=True)  # 0.0 to 1.0
    
    # Feedback and notes
    supervisor_feedback = Column(Text, nullable=True)
    self_assessment = Column(Text, nullable=True)
    lessons_learned = Column(JSON, nullable=True)  # List of lessons
    improvement_areas = Column(JSON, nullable=True)  # List of improvement areas
    
    # Timestamps
    performance_period_start = Column(DateTime, nullable=False)
    performance_period_end = Column(DateTime, nullable=True)
    evaluation_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SkillDevelopmentModel(Base):
    """Database model for tracking skill development recommendations"""
    __tablename__ = "skill_development"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String, ForeignKey("people.id"), nullable=False)
    skill_name = Column(String, nullable=False)
    current_level = Column(Integer, nullable=False)
    target_level = Column(Integer, nullable=False)
    development_priority = Column(String, nullable=False)  # high, medium, low
    recommended_actions = Column(JSON, nullable=True)  # List of development actions
    training_resources = Column(JSON, nullable=True)  # List of training resources
    target_completion_date = Column(DateTime, nullable=True)
    progress_status = Column(String, nullable=False, default="not_started")
    progress_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Utility functions for model operations
def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)


def get_person_with_skills(session, person_id: str) -> Optional[PersonModel]:
    """Get person with all their skills"""
    return session.query(PersonModel).filter(PersonModel.id == person_id).first()


def get_crisis_assignments(session, crisis_id: str) -> List[RoleAssignmentModel]:
    """Get all assignments for a crisis"""
    return session.query(RoleAssignmentModel).filter(RoleAssignmentModel.crisis_id == crisis_id).all()


def get_person_assignment_history(session, person_id: str) -> List[RoleAssignmentModel]:
    """Get assignment history for a person"""
    return session.query(RoleAssignmentModel).filter(RoleAssignmentModel.person_id == person_id).all()


def get_role_performance_metrics(session, role_type: str) -> List[RolePerformanceModel]:
    """Get performance metrics for a specific role type"""
    return session.query(RolePerformanceModel).filter(RolePerformanceModel.role_type == role_type).all()


def update_person_availability(session, person_id: str, availability: float, workload: float):
    """Update person's availability and workload"""
    person = session.query(PersonModel).filter(PersonModel.id == person_id).first()
    if person:
        person.current_availability = availability
        person.current_workload = workload
        person.updated_at = datetime.utcnow()
        session.commit()


def confirm_assignment(session, assignment_id: int, confirmed: bool, confirmation_time: datetime = None):
    """Confirm or decline a role assignment"""
    assignment = session.query(RoleAssignmentModel).filter(RoleAssignmentModel.id == assignment_id).first()
    if assignment:
        assignment.confirmation_status = "confirmed" if confirmed else "declined"
        assignment.confirmation_time = confirmation_time or datetime.utcnow()
        assignment.updated_at = datetime.utcnow()
        session.commit()


def activate_backup_assignment(session, backup_assignment_id: int):
    """Activate a backup assignment"""
    backup = session.query(BackupAssignmentModel).filter(BackupAssignmentModel.id == backup_assignment_id).first()
    if backup:
        backup.activated = True
        backup.activation_time = datetime.utcnow()
        backup.updated_at = datetime.utcnow()
        session.commit()
        
        # Create primary assignment from backup
        primary_assignment = RoleAssignmentModel(
            crisis_id=backup.crisis_id,
            person_id=backup.person_id,
            role_type=backup.role_type,
            assignment_confidence=backup.compatibility_score,
            assignment_time=datetime.utcnow(),
            confirmation_status="pending"
        )
        session.add(primary_assignment)
        session.commit()
        
        return primary_assignment
    return None