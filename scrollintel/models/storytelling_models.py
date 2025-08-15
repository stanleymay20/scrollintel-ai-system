"""
Storytelling Framework Models

Data models for storytelling framework including narrative structures,
story personalization, and impact measurement.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class StoryType(Enum):
    """Types of transformation stories"""
    TRANSFORMATION_JOURNEY = "transformation_journey"
    SUCCESS_STORY = "success_story"
    VISION_NARRATIVE = "vision_narrative"
    CHANGE_STORY = "change_story"
    HERO_JOURNEY = "hero_journey"
    CHALLENGE_OVERCOME = "challenge_overcome"
    FUTURE_STATE = "future_state"
    ORIGIN_STORY = "origin_story"


class NarrativeStructure(Enum):
    """Narrative structure types"""
    HERO_JOURNEY = "hero_journey"
    THREE_ACT = "three_act"
    PROBLEM_SOLUTION = "problem_solution"
    BEFORE_AFTER = "before_after"
    CHALLENGE_ACTION_RESULT = "challenge_action_result"
    SITUATION_COMPLICATION_RESOLUTION = "situation_complication_resolution"


class StoryElement(Enum):
    """Story elements"""
    PROTAGONIST = "protagonist"
    CONFLICT = "conflict"
    RESOLUTION = "resolution"
    TRANSFORMATION = "transformation"
    LESSON = "lesson"
    EMOTION = "emotion"
    VISION = "vision"
    ACTION = "action"


class DeliveryFormat(Enum):
    """Story delivery formats"""
    WRITTEN_NARRATIVE = "written_narrative"
    VIDEO_STORY = "video_story"
    PRESENTATION = "presentation"
    INTERACTIVE_STORY = "interactive_story"
    PODCAST = "podcast"
    INFOGRAPHIC = "infographic"
    CASE_STUDY = "case_study"


@dataclass
class StoryCharacter:
    """Story character definition"""
    id: str
    name: str
    role: str
    characteristics: Dict[str, Any]
    motivations: List[str]
    challenges: List[str]
    transformation_arc: Dict[str, str]
    relatability_factors: List[str]


@dataclass
class StoryPlot:
    """Story plot structure"""
    id: str
    structure: NarrativeStructure
    acts: List[Dict[str, Any]]
    key_moments: List[Dict[str, Any]]
    emotional_arc: List[Dict[str, float]]
    conflict_resolution: Dict[str, Any]
    transformation_points: List[Dict[str, Any]]


@dataclass
class TransformationStory:
    """Core transformation story"""
    id: str
    title: str
    story_type: StoryType
    narrative_structure: NarrativeStructure
    content: str
    characters: List[StoryCharacter]
    plot: StoryPlot
    cultural_themes: List[str]
    key_messages: List[str]
    emotional_tone: Dict[str, float]
    target_outcomes: List[str]
    created_at: datetime
    updated_at: datetime
    version: int = 1
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryPersonalization:
    """Story personalization for specific audience"""
    id: str
    base_story_id: str
    audience_id: str
    personalized_content: str
    character_adaptations: List[Dict[str, Any]]
    cultural_adaptations: Dict[str, Any]
    language_style: str
    delivery_format: DeliveryFormat
    personalization_score: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StoryDelivery:
    """Story delivery tracking"""
    id: str
    story_id: str
    audience_id: str
    delivery_format: DeliveryFormat
    delivery_channel: str
    delivered_at: datetime
    delivery_context: Dict[str, Any]
    recipient_count: int
    delivery_status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoryEngagement:
    """Story engagement metrics"""
    id: str
    story_id: str
    audience_id: str
    delivery_format: DeliveryFormat
    views: int = 0
    completion_rate: float = 0.0
    shares: int = 0
    comments: int = 0
    emotional_responses: Dict[str, int] = field(default_factory=dict)
    time_spent: int = 0
    interaction_points: List[Dict[str, Any]] = field(default_factory=list)
    measured_at: datetime = field(default_factory=datetime.now)


@dataclass
class StoryImpact:
    """Story impact measurement"""
    id: str
    story_id: str
    audience_id: str
    impact_score: float
    emotional_impact: Dict[str, float]
    behavioral_indicators: Dict[str, float]
    cultural_alignment: float
    message_retention: float
    transformation_influence: float
    feedback_summary: Dict[str, Any]
    measured_at: datetime = field(default_factory=datetime.now)


@dataclass
class StoryTemplate:
    """Reusable story template"""
    id: str
    name: str
    story_type: StoryType
    narrative_structure: NarrativeStructure
    template_content: str
    character_templates: List[Dict[str, Any]]
    plot_template: Dict[str, Any]
    customization_points: List[str]
    usage_guidelines: str
    effectiveness_data: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class StorytellingCampaign:
    """Storytelling campaign"""
    id: str
    name: str
    description: str
    transformation_objectives: List[str]
    target_audiences: List[str]
    stories: List[str]
    narrative_arc: Dict[str, Any]
    delivery_schedule: Dict[str, Any]
    success_metrics: Dict[str, float]
    start_date: datetime
    end_date: datetime
    status: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class NarrativeStrategy:
    """Overall narrative strategy"""
    id: str
    organization_id: str
    transformation_vision: str
    core_narratives: List[str]
    story_themes: List[str]
    character_archetypes: List[StoryCharacter]
    narrative_guidelines: Dict[str, Any]
    audience_story_preferences: Dict[str, Any]
    effectiveness_targets: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class StoryAnalytics:
    """Story analytics and insights"""
    id: str
    story_id: str
    time_period: Dict[str, datetime]
    engagement_metrics: Dict[str, float]
    impact_metrics: Dict[str, float]
    audience_insights: Dict[str, Any]
    optimization_recommendations: List[str]
    trend_analysis: Dict[str, Any]
    comparative_performance: Dict[str, float]
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class StoryFeedback:
    """Story feedback collection"""
    id: str
    story_id: str
    audience_id: str
    feedback_type: str
    rating: float
    emotional_response: Dict[str, float]
    comprehension_score: float
    relevance_score: float
    inspiration_score: float
    action_intent: float
    comments: str
    suggestions: List[str]
    collected_at: datetime = field(default_factory=datetime.now)