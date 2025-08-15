"""
Crisis Response Effectiveness Testing Models

Data models for crisis response effectiveness testing and validation.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, List, Any, Optional

Base = declarative_base()

class CrisisResponseTest(Base):
    """Model for crisis response effectiveness tests"""
    __tablename__ = "crisis_response_tests"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String(100), unique=True, index=True, nullable=False)
    crisis_scenario = Column(Text, nullable=False)
    test_type = Column(String(50), nullable=False)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    phases_tested = Column(JSON, nullable=True)  # List of testing phases
    overall_score = Column(Float, nullable=True)
    recommendations = Column(JSON, nullable=True)  # List of recommendations
    test_status = Column(String(20), nullable=False, default="active")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    effectiveness_scores = relationship("EffectivenessScore", back_populates="test", cascade="all, delete-orphan")
    speed_measurements = relationship("ResponseSpeedMeasurement", back_populates="test", cascade="all, delete-orphan")
    quality_assessments = relationship("DecisionQualityAssessment", back_populates="test", cascade="all, delete-orphan")
    communication_evaluations = relationship("CommunicationEvaluation", back_populates="test", cascade="all, delete-orphan")
    outcome_assessments = relationship("OutcomeAssessment", back_populates="test", cascade="all, delete-orphan")
    leadership_evaluations = relationship("LeadershipEvaluation", back_populates="test", cascade="all, delete-orphan")

class EffectivenessScore(Base):
    """Model for individual effectiveness scores"""
    __tablename__ = "effectiveness_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String(100), ForeignKey("crisis_response_tests.test_id"), nullable=False)
    metric = Column(String(50), nullable=False)  # EffectivenessMetric enum value
    score = Column(Float, nullable=False)  # 0.0 to 1.0
    details = Column(JSON, nullable=True)
    measurement_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    confidence_level = Column(Float, nullable=False, default=0.9)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test = relationship("CrisisResponseTest", back_populates="effectiveness_scores")

class ResponseSpeedMeasurement(Base):
    """Model for response speed measurements"""
    __tablename__ = "response_speed_measurements"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String(100), ForeignKey("crisis_response_tests.test_id"), nullable=False)
    detection_time = Column(DateTime, nullable=False)
    first_response_time = Column(DateTime, nullable=False)
    full_response_time = Column(DateTime, nullable=False)
    detection_to_first_response_seconds = Column(Float, nullable=False)
    detection_to_full_response_seconds = Column(Float, nullable=False)
    first_response_score = Column(Float, nullable=False)
    full_response_score = Column(Float, nullable=False)
    overall_speed_score = Column(Float, nullable=False)
    target_first_response = Column(Float, nullable=False, default=300)  # 5 minutes
    target_full_response = Column(Float, nullable=False, default=1800)  # 30 minutes
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test = relationship("CrisisResponseTest", back_populates="speed_measurements")

class DecisionQualityAssessment(Base):
    """Model for decision quality assessments"""
    __tablename__ = "decision_quality_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String(100), ForeignKey("crisis_response_tests.test_id"), nullable=False)
    decision_id = Column(String(100), nullable=False)
    decision_type = Column(String(50), nullable=False)
    information_completeness = Column(Float, nullable=False, default=0.5)
    stakeholder_consideration = Column(Float, nullable=False, default=0.5)
    risk_assessment_accuracy = Column(Float, nullable=False, default=0.5)
    implementation_feasibility = Column(Float, nullable=False, default=0.5)
    outcome_effectiveness = Column(Float, nullable=False, default=0.5)
    overall_quality_score = Column(Float, nullable=False)
    decision_context = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test = relationship("CrisisResponseTest", back_populates="quality_assessments")

class CommunicationEvaluation(Base):
    """Model for communication effectiveness evaluations"""
    __tablename__ = "communication_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String(100), ForeignKey("crisis_response_tests.test_id"), nullable=False)
    communication_id = Column(String(100), nullable=False)
    channel = Column(String(50), nullable=False)
    audience = Column(String(100), nullable=False)
    clarity_score = Column(Float, nullable=False, default=0.5)
    timeliness_score = Column(Float, nullable=False, default=0.5)
    completeness_score = Column(Float, nullable=False, default=0.5)
    appropriateness_score = Column(Float, nullable=False, default=0.5)
    overall_communication_score = Column(Float, nullable=False)
    stakeholder_feedback = Column(JSON, nullable=True)
    message_content = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test = relationship("CrisisResponseTest", back_populates="communication_evaluations")

class OutcomeAssessment(Base):
    """Model for crisis outcome assessments"""
    __tablename__ = "outcome_assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String(100), ForeignKey("crisis_response_tests.test_id"), nullable=False)
    objective = Column(Text, nullable=False)
    completion_rate = Column(Float, nullable=False, default=0.0)
    quality_rating = Column(Float, nullable=False, default=0.5)
    stakeholder_satisfaction = Column(Float, nullable=False, default=0.5)
    long_term_impact_score = Column(Float, nullable=False, default=0.5)
    objective_score = Column(Float, nullable=False)
    outcome_details = Column(JSON, nullable=True)
    success_metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test = relationship("CrisisResponseTest", back_populates="outcome_assessments")

class LeadershipEvaluation(Base):
    """Model for leadership effectiveness evaluations"""
    __tablename__ = "leadership_evaluations"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String(100), ForeignKey("crisis_response_tests.test_id"), nullable=False)
    decision_making_score = Column(Float, nullable=False, default=0.5)
    communication_score = Column(Float, nullable=False, default=0.5)
    team_coordination_score = Column(Float, nullable=False, default=0.5)
    stakeholder_management_score = Column(Float, nullable=False, default=0.5)
    crisis_composure_score = Column(Float, nullable=False, default=0.5)
    leadership_clarity = Column(Float, nullable=False, default=0.5)
    decision_confidence = Column(Float, nullable=False, default=0.5)
    communication_effectiveness = Column(Float, nullable=False, default=0.5)
    overall_leadership_score = Column(Float, nullable=False)
    leadership_actions = Column(JSON, nullable=True)
    team_feedback = Column(JSON, nullable=True)
    stakeholder_confidence = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    test = relationship("CrisisResponseTest", back_populates="leadership_evaluations")

class EffectivenessBaseline(Base):
    """Model for effectiveness baseline metrics"""
    __tablename__ = "effectiveness_baselines"
    
    id = Column(Integer, primary_key=True, index=True)
    metric = Column(String(50), nullable=False, unique=True)  # EffectivenessMetric enum value
    baseline_score = Column(Float, nullable=False)
    measurement_period = Column(String(50), nullable=False)  # e.g., "monthly", "quarterly"
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)
    sample_size = Column(Integer, nullable=False, default=1)
    confidence_interval = Column(Float, nullable=False, default=0.95)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PerformanceThreshold(Base):
    """Model for performance thresholds"""
    __tablename__ = "performance_thresholds"
    
    id = Column(Integer, primary_key=True, index=True)
    metric = Column(String(50), nullable=False)  # EffectivenessMetric enum value
    excellent_threshold = Column(Float, nullable=False)
    good_threshold = Column(Float, nullable=False)
    acceptable_threshold = Column(Float, nullable=False)
    poor_threshold = Column(Float, nullable=False)
    threshold_rationale = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class TestingRecommendation(Base):
    """Model for testing recommendations"""
    __tablename__ = "testing_recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(String(100), ForeignKey("crisis_response_tests.test_id"), nullable=False)
    recommendation = Column(Text, nullable=False)
    priority = Column(String(20), nullable=False, default="medium")  # high, medium, low
    category = Column(String(50), nullable=False)  # e.g., "response_speed", "communication"
    implementation_effort = Column(String(20), nullable=False, default="medium")  # high, medium, low
    expected_impact = Column(Float, nullable=False, default=0.5)  # 0.0 to 1.0
    status = Column(String(20), nullable=False, default="pending")  # pending, in_progress, completed, dismissed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class EffectivenessTrend(Base):
    """Model for effectiveness trends over time"""
    __tablename__ = "effectiveness_trends"
    
    id = Column(Integer, primary_key=True, index=True)
    metric = Column(String(50), nullable=False)  # EffectivenessMetric enum value
    time_period = Column(String(20), nullable=False)  # daily, weekly, monthly
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    average_score = Column(Float, nullable=False)
    score_variance = Column(Float, nullable=False, default=0.0)
    test_count = Column(Integer, nullable=False, default=0)
    trend_direction = Column(String(20), nullable=False)  # improving, declining, stable
    trend_strength = Column(Float, nullable=False, default=0.0)  # -1.0 to 1.0
    created_at = Column(DateTime, default=datetime.utcnow)

# Utility functions for model operations
def create_effectiveness_test_record(
    test_id: str,
    crisis_scenario: str,
    test_type: str = "comprehensive"
) -> CrisisResponseTest:
    """Create a new crisis response test record"""
    return CrisisResponseTest(
        test_id=test_id,
        crisis_scenario=crisis_scenario,
        test_type=test_type,
        start_time=datetime.utcnow(),
        test_status="active"
    )

def create_effectiveness_score_record(
    test_id: str,
    metric: str,
    score: float,
    details: Dict[str, Any],
    confidence_level: float = 0.9
) -> EffectivenessScore:
    """Create a new effectiveness score record"""
    return EffectivenessScore(
        test_id=test_id,
        metric=metric,
        score=score,
        details=details,
        measurement_time=datetime.utcnow(),
        confidence_level=confidence_level
    )

def create_speed_measurement_record(
    test_id: str,
    detection_time: datetime,
    first_response_time: datetime,
    full_response_time: datetime,
    scores: Dict[str, float]
) -> ResponseSpeedMeasurement:
    """Create a new response speed measurement record"""
    return ResponseSpeedMeasurement(
        test_id=test_id,
        detection_time=detection_time,
        first_response_time=first_response_time,
        full_response_time=full_response_time,
        detection_to_first_response_seconds=(first_response_time - detection_time).total_seconds(),
        detection_to_full_response_seconds=(full_response_time - detection_time).total_seconds(),
        first_response_score=scores.get("first_response_score", 0.0),
        full_response_score=scores.get("full_response_score", 0.0),
        overall_speed_score=scores.get("overall_speed_score", 0.0)
    )