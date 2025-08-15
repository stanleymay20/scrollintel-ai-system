"""
Post-Crisis Analysis Models

Data models for comprehensive crisis response analysis and evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class AnalysisType(Enum):
    RESPONSE_EFFECTIVENESS = "response_effectiveness"
    DECISION_QUALITY = "decision_quality"
    COMMUNICATION_ANALYSIS = "communication_analysis"
    RESOURCE_UTILIZATION = "resource_utilization"
    TEAM_PERFORMANCE = "team_performance"
    STAKEHOLDER_IMPACT = "stakeholder_impact"


class LessonCategory(Enum):
    PROCESS_IMPROVEMENT = "process_improvement"
    COMMUNICATION = "communication"
    RESOURCE_MANAGEMENT = "resource_management"
    DECISION_MAKING = "decision_making"
    TEAM_COORDINATION = "team_coordination"
    STAKEHOLDER_MANAGEMENT = "stakeholder_management"


class RecommendationPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class CrisisMetric:
    """Individual crisis performance metric"""
    metric_name: str
    target_value: float
    actual_value: float
    variance: float
    performance_score: float
    measurement_unit: str


@dataclass
class LessonLearned:
    """Individual lesson learned from crisis response"""
    id: str
    crisis_id: str
    category: LessonCategory
    title: str
    description: str
    root_cause: str
    impact_assessment: str
    evidence: List[str]
    identified_by: str
    identification_date: datetime
    validation_status: str


@dataclass
class ImprovementRecommendation:
    """Specific improvement recommendation"""
    id: str
    lesson_id: str
    title: str
    description: str
    priority: RecommendationPriority
    implementation_effort: str
    expected_impact: str
    success_metrics: List[str]
    responsible_team: str
    target_completion: datetime
    implementation_status: str


@dataclass
class PostCrisisAnalysis:
    """Comprehensive post-crisis analysis"""
    id: str
    crisis_id: str
    analysis_type: AnalysisType
    analyst_id: str
    analysis_date: datetime
    
    # Crisis overview
    crisis_summary: str
    crisis_duration: float
    crisis_severity: str
    
    # Performance metrics
    response_metrics: List[CrisisMetric]
    overall_performance_score: float
    
    # Analysis findings
    strengths_identified: List[str]
    weaknesses_identified: List[str]
    lessons_learned: List[LessonLearned]
    improvement_recommendations: List[ImprovementRecommendation]
    
    # Impact assessment
    stakeholder_impact: Dict[str, Any]
    business_impact: Dict[str, Any]
    reputation_impact: Dict[str, Any]
    
    # Analysis metadata
    analysis_methodology: str
    data_sources: List[str]
    confidence_level: float
    review_status: str


@dataclass
class AnalysisReport:
    """Formatted analysis report"""
    analysis_id: str
    report_title: str
    executive_summary: str
    detailed_findings: str
    recommendations_summary: str
    appendices: List[str]
    generated_date: datetime
    report_format: str