"""
Board Presentation Models for ScrollIntel

This module defines data models for board presentation framework components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class PresentationFormat(Enum):
    """Board presentation format types"""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis"
    STRATEGIC_OVERVIEW = "strategic_overview"
    FINANCIAL_REPORT = "financial_report"
    RISK_ASSESSMENT = "risk_assessment"


class BoardMemberType(Enum):
    """Types of board members for presentation customization"""
    INDEPENDENT_DIRECTOR = "independent_director"
    EXECUTIVE_DIRECTOR = "executive_director"
    INVESTOR_REPRESENTATIVE = "investor_representative"
    INDUSTRY_EXPERT = "industry_expert"
    FINANCIAL_EXPERT = "financial_expert"


@dataclass
class BoardMemberProfile:
    """Profile of a board member for presentation customization"""
    id: str
    name: str
    member_type: BoardMemberType
    expertise_areas: List[str]
    communication_preferences: Dict[str, Any]
    attention_span: int  # minutes
    detail_preference: str  # "high", "medium", "low"
    visual_preference: bool
    data_comfort_level: str  # "expert", "intermediate", "basic"


@dataclass
class PresentationTemplate:
    """Template for board presentations"""
    id: str
    name: str
    format_type: PresentationFormat
    target_audience: List[BoardMemberType]
    slide_structure: List[str]
    design_guidelines: Dict[str, Any]
    content_guidelines: Dict[str, str]
    timing_recommendations: Dict[str, int]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ContentSection:
    """Section of presentation content"""
    id: str
    title: str
    content_type: str  # "text", "chart", "table", "image"
    content: Any
    importance_level: str  # "critical", "important", "supporting"
    estimated_time: int  # seconds
    board_member_relevance: Dict[str, float]  # member_id -> relevance_score


@dataclass
class PresentationSlide:
    """Individual presentation slide"""
    id: str
    slide_number: int
    title: str
    sections: List[ContentSection]
    design_elements: Dict[str, Any]
    speaker_notes: str
    estimated_duration: int  # seconds
    interaction_points: List[str]


@dataclass
class BoardPresentation:
    """Complete board presentation"""
    id: str
    title: str
    board_id: str
    presenter_id: str
    presentation_date: datetime
    format_type: PresentationFormat
    slides: List[PresentationSlide]
    executive_summary: str
    key_messages: List[str]
    success_metrics: List[str]
    qa_preparation: Optional['QAPreparation'] = None
    quality_score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DesignPreferences:
    """Board design preferences"""
    color_scheme: str
    font_family: str
    chart_style: str
    layout_preference: str
    branding_requirements: Dict[str, Any]
    accessibility_requirements: List[str]


@dataclass
class QualityMetrics:
    """Presentation quality assessment metrics"""
    clarity_score: float
    relevance_score: float
    engagement_score: float
    professional_score: float
    time_efficiency_score: float
    overall_score: float
    improvement_suggestions: List[str]


@dataclass
class PresentationFeedback:
    """Feedback on presentation performance"""
    presentation_id: str
    board_member_id: str
    engagement_level: str  # "high", "medium", "low"
    comprehension_score: float
    feedback_comments: str
    suggested_improvements: List[str]
    timestamp: datetime = field(default_factory=datetime.now)