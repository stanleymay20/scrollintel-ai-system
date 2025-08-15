"""
ScrollIntel Competitive Intelligence Data Models
Pydantic models for competitive analysis and market positioning data structures
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class CompetitorTierEnum(str, Enum):
    TIER_1_DIRECT = "tier_1_direct"
    TIER_2_ADJACENT = "tier_2_adjacent"
    TIER_3_TRADITIONAL = "tier_3_traditional"
    EMERGING_THREAT = "emerging_threat"

class ThreatLevelEnum(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"

class MarketPositionEnum(str, Enum):
    LEADER = "leader"
    CHALLENGER = "challenger"
    FOLLOWER = "follower"
    NICHE = "niche"

class LeadershipTeamMember(BaseModel):
    name: str = Field(..., description="Executive name")
    role: str = Field(..., description="Executive role/title")
    background: str = Field(..., description="Professional background")

class RecentDevelopment(BaseModel):
    date: str = Field(..., description="Development date (YYYY-MM format)")
    event: str = Field(..., description="Description of the development")

class CompetitorProfileModel(BaseModel):
    competitor_id: str = Field(..., description="Unique competitor identifier")
    company_name: str = Field(..., description="Company name")
    tier: CompetitorTierEnum = Field(..., description="Competitor tier classification")
    market_position: MarketPositionEnum = Field(..., description="Market position")
    threat_level: ThreatLevelEnum = Field(..., description="Threat level assessment")
    annual_revenue: float = Field(..., description="Annual revenue in USD")
    employee_count: int = Field(..., description="Number of employees")
    funding_raised: float = Field(..., description="Total funding raised in USD")
    key_products: List[str] = Field(..., description="Key products and services")
    target_market: List[str] = Field(..., description="Target market segments")
    pricing_model: str = Field(..., description="Pricing model description")
    key_differentiators: List[str] = Field(..., description="Key competitive differentiators")
    weaknesses: List[str] = Field(..., description="Identified weaknesses")
    recent_developments: List[RecentDevelopment] = Field(..., description="Recent company developments")
    market_share: float = Field(..., description="Market share percentage (0-1)")
    customer_count: int = Field(..., description="Number of customers")
    geographic_presence: List[str] = Field(..., description="Geographic markets served")
    technology_stack: List[str] = Field(..., description="Technology stack components")
    leadership_team: List[LeadershipTeamMember] = Field(..., description="Key leadership team members")
    financial_health: str = Field(..., description="Financial health assessment")
    growth_trajectory: str = Field(..., description="Growth trajectory description")
    competitive_advantages: List[str] = Field(..., description="Competitive advantages")
    strategic_partnerships: List[str] = Field(..., description="Strategic partnerships")

class BuyerPersona(BaseModel):
    persona: str = Field(..., description="Buyer persona name")
    pain_points: List[str] = Field(..., description="Key pain points")
    decision_criteria: List[str] = Field(..., description="Decision criteria")
    budget_authority: str = Field(..., description="Budget authority level")
    decision_timeline: str = Field(..., description="Typical decision timeline")

class MarketIntelligenceModel(BaseModel):
    market_size: float = Field(..., description="Total addressable market size in USD")
    growth_rate: float = Field(..., description="Annual market growth rate (0-1)")
    key_trends: List[str] = Field(..., description="Key market trends")
    adoption_barriers: List[str] = Field(..., description="Market adoption barriers")
    buyer_personas: List[BuyerPersona] = Field(..., description="Buyer personas")
    decision_criteria: List[str] = Field(..., description="Common decision criteria")
    budget_allocation_patterns: Dict[str, float] = Field(..., description="Budget allocation patterns")
    technology_readiness: str = Field(..., description="Technology readiness assessment")
    regulatory_environment: List[str] = Field(..., description="Regulatory considerations")
    competitive_landscape_summary: str = Field(..., description="Competitive landscape summary")

class MessagingFramework(BaseModel):
    primary_message: str = Field(..., description="Primary marketing message")
    supporting_messages: List[str] = Field(..., description="Supporting messages")
    proof_points: List[str] = Field(..., description="Proof points for messaging")

class PositioningStrategyModel(BaseModel):
    unique_value_proposition: str = Field(..., description="Unique value proposition")
    key_differentiators: List[str] = Field(..., description="Key differentiators")
    target_segments: List[str] = Field(..., description="Target market segments")
    messaging_framework: MessagingFramework = Field(..., description="Messaging framework")
    competitive_advantages: List[str] = Field(..., description="Competitive advantages")
    proof_points: List[str] = Field(..., description="Proof points")
    objection_handling: Dict[str, str] = Field(..., description="Objection handling responses")
    pricing_strategy: str = Field(..., description="Pricing strategy")
    go_to_market_approach: str = Field(..., description="Go-to-market approach")

class ThreatAnalysisItem(BaseModel):
    competitor: str = Field(..., description="Competitor name")
    threat_level: str = Field(..., description="Threat level")
    threat_score: float = Field(..., description="Calculated threat score")
    key_concerns: List[str] = Field(..., description="Key concerns")
    mitigation_strategies: List[str] = Field(..., description="Mitigation strategies")
    market_impact: str = Field(..., description="Market impact description")

class EmergingThreat(BaseModel):
    competitor: str = Field(..., description="Competitor name")
    growth_trajectory: str = Field(..., description="Growth trajectory")
    funding_status: str = Field(..., description="Funding status")
    market_position: str = Field(..., description="Market position")
    watch_indicators: List[str] = Field(..., description="Indicators to watch")

class MarketOpportunity(BaseModel):
    opportunity: str = Field(..., description="Opportunity name")
    market_size: str = Field(..., description="Market size description")
    competitive_gap: str = Field(..., description="Competitive gap description")
    time_to_market: str = Field(..., description="Time to market")
    success_probability: str = Field(..., description="Success probability")

class StrategicRecommendation(BaseModel):
    priority: str = Field(..., description="Priority level")
    recommendation: str = Field(..., description="Recommendation description")
    rationale: str = Field(..., description="Rationale for recommendation")
    timeline: str = Field(..., description="Implementation timeline")
    investment: str = Field(..., description="Investment required")
    expected_impact: str = Field(..., description="Expected impact")

class ThreatAnalysisModel(BaseModel):
    immediate_threats: List[ThreatAnalysisItem] = Field(..., description="Immediate threats")
    emerging_threats: List[EmergingThreat] = Field(..., description="Emerging threats")
    market_opportunities: List[MarketOpportunity] = Field(..., description="Market opportunities")
    strategic_recommendations: List[StrategicRecommendation] = Field(..., description="Strategic recommendations")

class CompetitiveDashboardModel(BaseModel):
    competitive_overview: Dict[str, Any] = Field(..., description="Competitive overview metrics")
    market_intelligence: Dict[str, Any] = Field(..., description="Market intelligence summary")
    positioning_strength: Dict[str, Any] = Field(..., description="Positioning strength metrics")
    threat_monitoring: Dict[str, Any] = Field(..., description="Threat monitoring status")
    strategic_priorities: List[str] = Field(..., description="Strategic priorities")
    success_indicators: Dict[str, str] = Field(..., description="Success indicators")

class MarketPositioningReportModel(BaseModel):
    executive_summary: Dict[str, Any] = Field(..., description="Executive summary")
    market_analysis: Dict[str, Any] = Field(..., description="Market analysis")
    competitive_landscape: Dict[str, Any] = Field(..., description="Competitive landscape")
    positioning_strategy: Dict[str, Any] = Field(..., description="Positioning strategy")
    strategic_recommendations: List[StrategicRecommendation] = Field(..., description="Strategic recommendations")
    market_opportunities: List[MarketOpportunity] = Field(..., description="Market opportunities")
    success_metrics: Dict[str, str] = Field(..., description="Success metrics")
    risk_assessment: Dict[str, Any] = Field(..., description="Risk assessment")

class CompetitorFilterModel(BaseModel):
    tier: Optional[CompetitorTierEnum] = Field(None, description="Filter by competitor tier")
    threat_level: Optional[ThreatLevelEnum] = Field(None, description="Filter by threat level")
    market_position: Optional[MarketPositionEnum] = Field(None, description="Filter by market position")

class ThreatMonitoringModel(BaseModel):
    threat_summary: Dict[str, Any] = Field(..., description="Threat summary metrics")
    critical_threats: List[Dict[str, Any]] = Field(..., description="Critical threats")
    high_threats: List[Dict[str, Any]] = Field(..., description="High threats")
    monitoring_alerts: List[Dict[str, Any]] = Field(..., description="Monitoring alerts")

class MarketOpportunityAnalysisModel(BaseModel):
    opportunity_id: str = Field(..., description="Opportunity identifier")
    title: str = Field(..., description="Opportunity title")
    market_size: float = Field(..., description="Market size in USD")
    competitive_gap: str = Field(..., description="Competitive gap description")
    time_to_market: str = Field(..., description="Time to market")
    success_probability: float = Field(..., description="Success probability (0-1)")
    investment_required: float = Field(..., description="Investment required in USD")
    expected_roi: float = Field(..., description="Expected ROI multiple")
    strategic_importance: str = Field(..., description="Strategic importance level")
    key_success_factors: List[str] = Field(..., description="Key success factors")

class OpportunityAnalysisResponseModel(BaseModel):
    opportunities: List[MarketOpportunityAnalysisModel] = Field(..., description="Market opportunities")
    total_market_opportunity: float = Field(..., description="Total market opportunity in USD")
    weighted_success_probability: float = Field(..., description="Weighted success probability")
    total_investment_required: float = Field(..., description="Total investment required in USD")
    expected_combined_roi: float = Field(..., description="Expected combined ROI")

class HealthCheckModel(BaseModel):
    status: str = Field(..., description="Overall system status")
    components: Dict[str, str] = Field(..., description="Component status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    timestamp: str = Field(..., description="Health check timestamp")
    message: str = Field(..., description="Health check message")

# Response wrapper models
class CompetitiveIntelligenceResponse(BaseModel):
    status: str = Field(..., description="Response status")
    data: Any = Field(..., description="Response data")
    timestamp: str = Field(..., description="Response timestamp")
    message: str = Field(..., description="Response message")

class CompetitorListResponse(BaseModel):
    status: str = Field(..., description="Response status")
    data: Dict[str, Any] = Field(..., description="Competitor data with filters")
    timestamp: str = Field(..., description="Response timestamp")
    message: str = Field(..., description="Response message")

class CompetitorProfileResponse(BaseModel):
    status: str = Field(..., description="Response status")
    data: CompetitorProfileModel = Field(..., description="Competitor profile data")
    timestamp: str = Field(..., description="Response timestamp")
    message: str = Field(..., description="Response message")