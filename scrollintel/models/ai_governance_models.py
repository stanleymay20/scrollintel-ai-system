"""
AI Governance and Ethics Framework Models

This module defines the data models for AI governance, ethics frameworks,
regulatory compliance, and public policy analysis systems.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
from sqlalchemy import Column, Integer, String, DateTime, Float, Text, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    REQUIRES_ACTION = "requires_action"


class EthicalPrinciple(Enum):
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    SAFETY = "safety"
    HUMAN_AUTONOMY = "human_autonomy"
    NON_MALEFICENCE = "non_maleficence"


class AIGovernance(Base):
    """AI Governance framework for managing AI systems at scale"""
    __tablename__ = 'ai_governance'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Governance configuration
    governance_policies = Column(JSON)  # Dict of policy configurations
    risk_thresholds = Column(JSON)  # Risk level thresholds
    approval_workflows = Column(JSON)  # Approval process definitions
    monitoring_requirements = Column(JSON)  # Monitoring specifications
    
    # Relationships
    ethics_frameworks = relationship("EthicsFramework", back_populates="governance")
    compliance_records = relationship("ComplianceRecord", back_populates="governance")
    policy_recommendations = relationship("PolicyRecommendation", back_populates="governance")


class EthicsFramework(Base):
    """Ethics framework for AI development and deployment"""
    __tablename__ = 'ethics_frameworks'
    
    id = Column(Integer, primary_key=True)
    governance_id = Column(Integer, ForeignKey('ai_governance.id'), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    version = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Ethics configuration
    ethical_principles = Column(JSON)  # List of ethical principles
    decision_criteria = Column(JSON)  # Decision-making criteria
    evaluation_metrics = Column(JSON)  # Ethics evaluation metrics
    stakeholder_considerations = Column(JSON)  # Stakeholder impact analysis
    
    # Relationships
    governance = relationship("AIGovernance", back_populates="ethics_frameworks")
    ethical_assessments = relationship("EthicalAssessment", back_populates="framework")


class ComplianceRecord(Base):
    """Regulatory compliance tracking and automation"""
    __tablename__ = 'compliance_records'
    
    id = Column(Integer, primary_key=True)
    governance_id = Column(Integer, ForeignKey('ai_governance.id'), nullable=False)
    regulation_name = Column(String(255), nullable=False)
    jurisdiction = Column(String(100), nullable=False)
    compliance_status = Column(String(50), nullable=False)
    assessment_date = Column(DateTime, default=datetime.utcnow)
    next_review_date = Column(DateTime)
    
    # Compliance details
    requirements = Column(JSON)  # Regulatory requirements
    evidence = Column(JSON)  # Compliance evidence
    gaps = Column(JSON)  # Compliance gaps
    remediation_plan = Column(JSON)  # Gap remediation plan
    risk_assessment = Column(JSON)  # Compliance risk assessment
    
    # Relationships
    governance = relationship("AIGovernance", back_populates="compliance_records")


class EthicalAssessment(Base):
    """Ethical decision-making assessments"""
    __tablename__ = 'ethical_assessments'
    
    id = Column(Integer, primary_key=True)
    framework_id = Column(Integer, ForeignKey('ethics_frameworks.id'), nullable=False)
    ai_system_id = Column(String(255), nullable=False)
    assessment_type = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Assessment details
    ethical_scores = Column(JSON)  # Scores per ethical principle
    risk_factors = Column(JSON)  # Identified risk factors
    mitigation_strategies = Column(JSON)  # Risk mitigation strategies
    stakeholder_impact = Column(JSON)  # Impact on different stakeholders
    recommendations = Column(JSON)  # Ethical recommendations
    
    # Assessment results
    overall_score = Column(Float)
    risk_level = Column(String(50))
    approval_status = Column(String(50))
    
    # Relationships
    framework = relationship("EthicsFramework", back_populates="ethical_assessments")


class PolicyRecommendation(Base):
    """Public policy analysis and recommendations"""
    __tablename__ = 'policy_recommendations'
    
    id = Column(Integer, primary_key=True)
    governance_id = Column(Integer, ForeignKey('ai_governance.id'), nullable=False)
    title = Column(String(255), nullable=False)
    policy_area = Column(String(100), nullable=False)
    jurisdiction = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Policy analysis
    current_landscape = Column(JSON)  # Current policy landscape
    gap_analysis = Column(JSON)  # Policy gaps identified
    recommendations = Column(JSON)  # Policy recommendations
    impact_assessment = Column(JSON)  # Expected policy impact
    implementation_plan = Column(JSON)  # Implementation strategy
    
    # Recommendation details
    priority_level = Column(String(50))
    expected_timeline = Column(String(100))
    stakeholders = Column(JSON)  # Affected stakeholders
    
    # Relationships
    governance = relationship("AIGovernance", back_populates="policy_recommendations")


@dataclass
class SafetyAlignment:
    """AI Safety and Alignment configuration"""
    safety_objectives: List[str]
    alignment_metrics: Dict[str, float]
    risk_mitigation_strategies: List[str]
    monitoring_protocols: List[str]
    escalation_procedures: List[str]


@dataclass
class RegulatoryCompliance:
    """Regulatory compliance configuration"""
    applicable_regulations: List[str]
    compliance_requirements: Dict[str, Any]
    audit_schedules: Dict[str, datetime]
    reporting_obligations: List[str]
    penalty_assessments: Dict[str, float]


@dataclass
class EthicalDecisionFramework:
    """Ethical decision-making framework"""
    decision_criteria: List[str]
    stakeholder_weights: Dict[str, float]
    ethical_constraints: List[str]
    evaluation_process: List[str]
    appeal_mechanisms: List[str]


@dataclass
class PublicPolicyAnalysis:
    """Public policy analysis results"""
    policy_landscape: Dict[str, Any]
    regulatory_trends: List[str]
    policy_gaps: List[str]
    recommendations: List[str]
    implementation_roadmap: Dict[str, Any]