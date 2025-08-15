"""
Crisis-Strategic Planning Integration Engine

This module integrates crisis leadership capabilities with strategic planning systems
to enable crisis-aware strategic adjustments and long-term planning integration.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from ..models.crisis_detection_models import Crisis, CrisisType, SeverityLevel
from ..models.strategic_planning_models import (
    StrategicRoadmap, TechnologyBet, StrategicMilestone, 
    StrategicPivot, MarketChange, InvestmentAnalysis
)
from ..engines.strategic_planner import StrategicPlanner
from ..engines.crisis_detection_engine import CrisisDetectionEngine

logger = logging.getLogger(__name__)


class CrisisImpactLevel(str, Enum):
    """Crisis impact levels on strategic plans"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"
    CATASTROPHIC = "catastrophic"


@dataclass
class CrisisStrategicImpact:
    """Assessment of crisis impact on strategic plans"""
    crisis_id: str
    strategic_plan_id: str
    impact_level: CrisisImpactLevel
    affected_milestones: List[str]
    affected_technology_bets: List[str]
    resource_reallocation_needed: float  # Percentage of resources to reallocate
    timeline_adjustments: Dict[str, int]  # Milestone delays in days
    risk_level_changes: Dict[str, float]  # Risk level adjustments
    strategic_recommendations: List[str]
    recovery_timeline: int  # Days to recover strategic momentum
    created_at: datetime


@dataclass
class CrisisAwareAdjustment:
    """Strategic adjustment recommendations during crisis"""
    adjustment_id: str
    crisis_id: str
    adjustment_type: str
    description: str
    priority: int  # 1-5, 1 being highest
    implementation_timeline: int  # Days to implement
    resource_requirements: Dict[str, Any]
    expected_benefits: List[str]
    risks: List[str]
    success_metrics: List[str]
    dependencies: List[str]
    created_at: datetime


@dataclass
class RecoveryIntegrationPlan:
    """Integration plan for crisis recovery with long-term planning"""
    plan_id: str
    crisis_id: str
    strategic_roadmap_id: str
    recovery_phases: List[Dict[str, Any]]
    milestone_realignment: Dict[str, datetime]
    resource_rebalancing: Dict[str, float]
    technology_bet_adjustments: List[Dict[str, Any]]
    stakeholder_communication_plan: Dict[str, Any]
    success_criteria: List[str]
    monitoring_framework: Dict[str, Any]
    created_at: datetime


class CrisisStrategicIntegration:
    """
    Integration engine for crisis leadership and strategic planning systems
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategic_planner = StrategicPlanner()
        self.crisis_detector = CrisisDetectionEngine()
        
    async def assess_crisis_impact_on_strategy(
        self, 
        crisis: Crisis, 
        strategic_roadmap: StrategicRoadmap
    ) -> CrisisStrategicImpact:
        """
        Assess how a crisis impacts strategic plans and roadmaps
        
        Args:
            crisis: Active crisis situation
            strategic_roadmap: Current strategic roadmap
            
        Returns:
            Comprehensive impact assessment
        """
        try:
            self.logger.info(f"Assessing crisis impact on strategic roadmap: {strategic_roadmap.id}")
            
            # Determine impact level based on crisis severity and type
            impact_level = self._calculate_impact_level(crisis, strategic_roadmap)
            
            # Identify affected milestones
            affected_milestones = self._identify_affected_milestones(
                crisis, strategic_roadmap.milestones
            )
            
            # Identify affected technology bets
            affected_technology_bets = self._identify_affected_technology_bets(
                crisis, strategic_roadmap.technology_bets
            )
            
            # Calculate resource reallocation needs
            resource_reallocation = self._calculate_resource_reallocation(
                crisis, impact_level
            )
            
            # Determine timeline adjustments
            timeline_adjustments = self._calculate_timeline_adjustments(
                crisis, affected_milestones, impact_level
            )
            
            # Assess risk level changes
            risk_level_changes = self._assess_risk_level_changes(
                crisis, strategic_roadmap.risk_assessments
            )
            
            # Generate strategic recommendations
            strategic_recommendations = self._generate_strategic_recommendations(
                crisis, impact_level, strategic_roadmap
            )
            
            # Estimate recovery timeline
            recovery_timeline = self._estimate_recovery_timeline(crisis, impact_level)
            
            impact_assessment = CrisisStrategicImpact(
                crisis_id=crisis.id,
                strategic_plan_id=strategic_roadmap.id,
                impact_level=impact_level,
                affected_milestones=affected_milestones,
                affected_technology_bets=affected_technology_bets,
                resource_reallocation_needed=resource_reallocation,
                timeline_adjustments=timeline_adjustments,
                risk_level_changes=risk_level_changes,
                strategic_recommendations=strategic_recommendations,
                recovery_timeline=recovery_timeline,
                created_at=datetime.now()
            )
            
            self.logger.info(f"Crisis impact assessment completed: {impact_level.value} impact")
            return impact_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing crisis impact on strategy: {str(e)}")
            raise
    
    async def generate_crisis_aware_adjustments(
        self, 
        crisis: Crisis, 
        strategic_roadmap: StrategicRoadmap,
        impact_assessment: CrisisStrategicImpact
    ) -> List[CrisisAwareAdjustment]:
        """
        Generate strategic adjustments that account for crisis conditions
        
        Args:
            crisis: Active crisis situation
            strategic_roadmap: Current strategic roadmap
            impact_assessment: Crisis impact assessment
            
        Returns:
            List of crisis-aware strategic adjustments
        """
        try:
            self.logger.info("Generating crisis-aware strategic adjustments")
            
            adjustments = []
            
            # Resource reallocation adjustments
            if impact_assessment.resource_reallocation_needed > 0.1:
                resource_adjustment = CrisisAwareAdjustment(
                    adjustment_id=f"adj_resource_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    crisis_id=crisis.id,
                    adjustment_type="resource_reallocation",
                    description=f"Reallocate {impact_assessment.resource_reallocation_needed*100:.1f}% of resources to crisis response",
                    priority=1,
                    implementation_timeline=7,  # 1 week
                    resource_requirements={
                        "budget_reallocation": impact_assessment.resource_reallocation_needed,
                        "personnel_reassignment": int(impact_assessment.resource_reallocation_needed * 100),
                        "infrastructure_adjustment": "moderate"
                    },
                    expected_benefits=[
                        "Faster crisis resolution",
                        "Maintained strategic momentum",
                        "Stakeholder confidence preservation"
                    ],
                    risks=[
                        "Delayed strategic milestones",
                        "Reduced innovation velocity",
                        "Competitive disadvantage"
                    ],
                    success_metrics=[
                        "Crisis resolution time",
                        "Strategic milestone recovery rate",
                        "Stakeholder satisfaction scores"
                    ],
                    dependencies=["crisis_team_formation", "stakeholder_approval"],
                    created_at=datetime.now()
                )
                adjustments.append(resource_adjustment)
            
            # Timeline adjustment for critical milestones
            if impact_assessment.timeline_adjustments:
                timeline_adjustment = CrisisAwareAdjustment(
                    adjustment_id=f"adj_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    crisis_id=crisis.id,
                    adjustment_type="milestone_timeline",
                    description="Adjust strategic milestone timelines to account for crisis impact",
                    priority=2,
                    implementation_timeline=14,  # 2 weeks
                    resource_requirements={
                        "planning_resources": "high",
                        "stakeholder_communication": "extensive",
                        "project_management": "enhanced"
                    },
                    expected_benefits=[
                        "Realistic timeline expectations",
                        "Improved resource allocation",
                        "Better stakeholder communication"
                    ],
                    risks=[
                        "Market opportunity loss",
                        "Competitive timing disadvantage",
                        "Team morale impact"
                    ],
                    success_metrics=[
                        "Milestone achievement rate",
                        "Resource utilization efficiency",
                        "Stakeholder acceptance rate"
                    ],
                    dependencies=["impact_assessment_approval", "resource_reallocation"],
                    created_at=datetime.now()
                )
                adjustments.append(timeline_adjustment)
            
            # Technology bet risk adjustments
            if impact_assessment.risk_level_changes:
                risk_adjustment = CrisisAwareAdjustment(
                    adjustment_id=f"adj_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    crisis_id=crisis.id,
                    adjustment_type="risk_rebalancing",
                    description="Rebalance technology investment risks based on crisis learnings",
                    priority=3,
                    implementation_timeline=30,  # 1 month
                    resource_requirements={
                        "risk_analysis": "comprehensive",
                        "portfolio_review": "detailed",
                        "investment_committee": "engaged"
                    },
                    expected_benefits=[
                        "Improved risk management",
                        "Better crisis resilience",
                        "Optimized investment returns"
                    ],
                    risks=[
                        "Reduced innovation potential",
                        "Conservative bias development",
                        "Competitive disadvantage"
                    ],
                    success_metrics=[
                        "Portfolio risk-adjusted returns",
                        "Crisis resilience score",
                        "Investment performance metrics"
                    ],
                    dependencies=["crisis_resolution", "strategic_review_completion"],
                    created_at=datetime.now()
                )
                adjustments.append(risk_adjustment)
            
            # Communication strategy adjustment
            communication_adjustment = CrisisAwareAdjustment(
                adjustment_id=f"adj_comm_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                crisis_id=crisis.id,
                adjustment_type="communication_strategy",
                description="Enhance strategic communication to address crisis-related concerns",
                priority=2,
                implementation_timeline=5,  # 5 days
                resource_requirements={
                    "communication_team": "dedicated",
                    "messaging_framework": "crisis_aware",
                    "stakeholder_engagement": "intensive"
                },
                expected_benefits=[
                    "Maintained stakeholder confidence",
                    "Clear strategic direction",
                    "Reduced uncertainty"
                ],
                risks=[
                    "Message inconsistency",
                    "Stakeholder confusion",
                    "Reputation impact"
                ],
                success_metrics=[
                    "Stakeholder confidence scores",
                    "Message clarity ratings",
                    "Strategic alignment perception"
                ],
                dependencies=["crisis_communication_plan", "leadership_alignment"],
                created_at=datetime.now()
            )
            adjustments.append(communication_adjustment)
            
            self.logger.info(f"Generated {len(adjustments)} crisis-aware adjustments")
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error generating crisis-aware adjustments: {str(e)}")
            raise
    
    async def create_recovery_integration_plan(
        self, 
        crisis: Crisis, 
        strategic_roadmap: StrategicRoadmap,
        impact_assessment: CrisisStrategicImpact
    ) -> RecoveryIntegrationPlan:
        """
        Create integrated plan for crisis recovery and long-term strategic planning
        
        Args:
            crisis: Crisis situation
            strategic_roadmap: Strategic roadmap
            impact_assessment: Crisis impact assessment
            
        Returns:
            Comprehensive recovery integration plan
        """
        try:
            self.logger.info("Creating crisis recovery integration plan")
            
            # Define recovery phases
            recovery_phases = [
                {
                    "phase": "immediate_stabilization",
                    "duration_days": 30,
                    "objectives": [
                        "Stabilize crisis situation",
                        "Restore basic operations",
                        "Communicate with stakeholders"
                    ],
                    "strategic_focus": "damage_control",
                    "resource_allocation": 0.7  # 70% to crisis response
                },
                {
                    "phase": "strategic_realignment",
                    "duration_days": 90,
                    "objectives": [
                        "Realign strategic priorities",
                        "Adjust milestone timelines",
                        "Rebalance resource allocation"
                    ],
                    "strategic_focus": "adaptation",
                    "resource_allocation": 0.5  # 50% to crisis recovery
                },
                {
                    "phase": "momentum_restoration",
                    "duration_days": 180,
                    "objectives": [
                        "Restore strategic momentum",
                        "Accelerate key initiatives",
                        "Rebuild stakeholder confidence"
                    ],
                    "strategic_focus": "acceleration",
                    "resource_allocation": 0.2  # 20% to ongoing crisis management
                },
                {
                    "phase": "enhanced_resilience",
                    "duration_days": 365,
                    "objectives": [
                        "Build crisis resilience",
                        "Integrate learnings",
                        "Strengthen strategic capabilities"
                    ],
                    "strategic_focus": "enhancement",
                    "resource_allocation": 0.1  # 10% to resilience building
                }
            ]
            
            # Create milestone realignment plan
            milestone_realignment = {}
            for milestone_id in impact_assessment.affected_milestones:
                delay_days = impact_assessment.timeline_adjustments.get(milestone_id, 0)
                milestone = next(
                    (m for m in strategic_roadmap.milestones if m.id == milestone_id), 
                    None
                )
                if milestone:
                    new_date = milestone.target_date + timedelta(days=delay_days)
                    milestone_realignment[milestone_id] = new_date
            
            # Create resource rebalancing plan
            resource_rebalancing = {
                "crisis_response": impact_assessment.resource_reallocation_needed,
                "strategic_initiatives": 1.0 - impact_assessment.resource_reallocation_needed,
                "resilience_building": 0.1,
                "stakeholder_management": 0.15,
                "risk_management": 0.2
            }
            
            # Technology bet adjustments
            technology_bet_adjustments = []
            for bet_id in impact_assessment.affected_technology_bets:
                bet = next(
                    (b for b in strategic_roadmap.technology_bets if b.id == bet_id), 
                    None
                )
                if bet:
                    adjustment = {
                        "bet_id": bet_id,
                        "risk_adjustment": impact_assessment.risk_level_changes.get(bet_id, 0),
                        "timeline_adjustment": impact_assessment.timeline_adjustments.get(bet_id, 0),
                        "investment_adjustment": -0.1 if impact_assessment.impact_level in [
                            CrisisImpactLevel.SEVERE, CrisisImpactLevel.CATASTROPHIC
                        ] else 0,
                        "focus_adjustment": "increased_resilience"
                    }
                    technology_bet_adjustments.append(adjustment)
            
            # Stakeholder communication plan
            stakeholder_communication_plan = {
                "frequency": "weekly" if impact_assessment.impact_level in [
                    CrisisImpactLevel.SEVERE, CrisisImpactLevel.CATASTROPHIC
                ] else "bi_weekly",
                "channels": ["executive_briefings", "board_updates", "team_communications"],
                "key_messages": [
                    "Crisis response progress",
                    "Strategic plan adjustments",
                    "Recovery timeline updates",
                    "Long-term vision reaffirmation"
                ],
                "stakeholder_groups": {
                    "board": "strategic_impact_focus",
                    "executives": "operational_recovery_focus",
                    "teams": "tactical_adjustment_focus",
                    "investors": "financial_impact_focus"
                }
            }
            
            # Success criteria
            success_criteria = [
                "Crisis fully resolved within recovery timeline",
                "Strategic milestones achieved within adjusted timelines",
                "Stakeholder confidence restored to pre-crisis levels",
                "Enhanced crisis resilience capabilities developed",
                "Long-term strategic vision maintained and strengthened"
            ]
            
            # Monitoring framework
            monitoring_framework = {
                "crisis_metrics": [
                    "crisis_resolution_progress",
                    "operational_stability_index",
                    "stakeholder_confidence_score"
                ],
                "strategic_metrics": [
                    "milestone_achievement_rate",
                    "resource_utilization_efficiency",
                    "strategic_momentum_index"
                ],
                "integration_metrics": [
                    "recovery_plan_adherence",
                    "strategic_alignment_score",
                    "resilience_capability_index"
                ],
                "reporting_frequency": "weekly",
                "escalation_triggers": [
                    "recovery_timeline_deviation > 20%",
                    "stakeholder_confidence < 70%",
                    "strategic_milestone_delay > 30 days"
                ]
            }
            
            recovery_plan = RecoveryIntegrationPlan(
                plan_id=f"recovery_{crisis.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                crisis_id=crisis.id,
                strategic_roadmap_id=strategic_roadmap.id,
                recovery_phases=recovery_phases,
                milestone_realignment=milestone_realignment,
                resource_rebalancing=resource_rebalancing,
                technology_bet_adjustments=technology_bet_adjustments,
                stakeholder_communication_plan=stakeholder_communication_plan,
                success_criteria=success_criteria,
                monitoring_framework=monitoring_framework,
                created_at=datetime.now()
            )
            
            self.logger.info("Crisis recovery integration plan created successfully")
            return recovery_plan
            
        except Exception as e:
            self.logger.error(f"Error creating recovery integration plan: {str(e)}")
            raise
    
    def _calculate_impact_level(
        self, 
        crisis: Crisis, 
        strategic_roadmap: StrategicRoadmap
    ) -> CrisisImpactLevel:
        """Calculate the impact level of crisis on strategic plans"""
        
        # Base impact from crisis severity
        severity_impact = {
            SeverityLevel.LOW: CrisisImpactLevel.MINIMAL,
            SeverityLevel.MEDIUM: CrisisImpactLevel.MODERATE,
            SeverityLevel.HIGH: CrisisImpactLevel.SIGNIFICANT,
            SeverityLevel.CRITICAL: CrisisImpactLevel.SEVERE
        }
        
        base_impact = severity_impact.get(crisis.severity_level, CrisisImpactLevel.MODERATE)
        
        # Adjust based on crisis type
        if crisis.crisis_type in [CrisisType.SECURITY_BREACH, CrisisType.REGULATORY_VIOLATION]:
            # These types have higher strategic impact
            impact_levels = list(CrisisImpactLevel)
            current_index = impact_levels.index(base_impact)
            if current_index < len(impact_levels) - 1:
                base_impact = impact_levels[current_index + 1]
        
        # Adjust based on strategic roadmap characteristics
        if strategic_roadmap.time_horizon > 10:
            # Longer roadmaps are more resilient to short-term crises
            impact_levels = list(CrisisImpactLevel)
            current_index = impact_levels.index(base_impact)
            if current_index > 0:
                base_impact = impact_levels[current_index - 1]
        
        return base_impact
    
    def _identify_affected_milestones(
        self, 
        crisis: Crisis, 
        milestones: List[StrategicMilestone]
    ) -> List[str]:
        """Identify which strategic milestones are affected by the crisis"""
        affected = []
        
        for milestone in milestones:
            # Check if milestone is within crisis impact timeframe
            days_until_milestone = (milestone.target_date - datetime.now().date()).days
            
            if days_until_milestone <= 365:  # Within 1 year
                # Check if milestone has dependencies that could be affected
                if any(risk in milestone.risk_factors for risk in [
                    "resource_constraints", "team_disruption", "market_volatility"
                ]):
                    affected.append(milestone.id)
                
                # Check if milestone requires resources that might be reallocated
                if milestone.resource_requirements.get("budget", 0) > 100e6:  # >$100M
                    affected.append(milestone.id)
        
        return affected
    
    def _identify_affected_technology_bets(
        self, 
        crisis: Crisis, 
        technology_bets: List[TechnologyBet]
    ) -> List[str]:
        """Identify which technology bets are affected by the crisis"""
        affected = []
        
        for bet in technology_bets:
            # High-risk bets are more vulnerable during crisis
            if bet.risk_level in [InvestmentRisk.HIGH, InvestmentRisk.EXTREME]:
                affected.append(bet.id)
            
            # Large investments might need reallocation
            if bet.investment_amount > 1e9:  # >$1B
                affected.append(bet.id)
            
            # Bets with external dependencies are vulnerable
            if len(bet.dependencies) > 2:
                affected.append(bet.id)
        
        return affected
    
    def _calculate_resource_reallocation(
        self, 
        crisis: Crisis, 
        impact_level: CrisisImpactLevel
    ) -> float:
        """Calculate percentage of resources that need reallocation"""
        
        reallocation_map = {
            CrisisImpactLevel.MINIMAL: 0.05,      # 5%
            CrisisImpactLevel.MODERATE: 0.15,     # 15%
            CrisisImpactLevel.SIGNIFICANT: 0.30,  # 30%
            CrisisImpactLevel.SEVERE: 0.50,       # 50%
            CrisisImpactLevel.CATASTROPHIC: 0.70  # 70%
        }
        
        base_reallocation = reallocation_map.get(impact_level, 0.15)
        
        # Adjust based on crisis type
        if crisis.crisis_type == CrisisType.SECURITY_BREACH:
            base_reallocation *= 1.2  # Security crises need more resources
        elif crisis.crisis_type == CrisisType.NATURAL_DISASTER:
            base_reallocation *= 1.5  # Natural disasters need significant reallocation
        
        return min(base_reallocation, 0.8)  # Cap at 80%
    
    def _calculate_timeline_adjustments(
        self, 
        crisis: Crisis, 
        affected_milestones: List[str], 
        impact_level: CrisisImpactLevel
    ) -> Dict[str, int]:
        """Calculate timeline adjustments for affected milestones"""
        
        delay_map = {
            CrisisImpactLevel.MINIMAL: 7,        # 1 week
            CrisisImpactLevel.MODERATE: 30,      # 1 month
            CrisisImpactLevel.SIGNIFICANT: 90,   # 3 months
            CrisisImpactLevel.SEVERE: 180,       # 6 months
            CrisisImpactLevel.CATASTROPHIC: 365  # 1 year
        }
        
        base_delay = delay_map.get(impact_level, 30)
        
        adjustments = {}
        for milestone_id in affected_milestones:
            # Add some randomization to avoid all milestones having same delay
            milestone_delay = int(base_delay * (0.8 + 0.4 * hash(milestone_id) % 100 / 100))
            adjustments[milestone_id] = milestone_delay
        
        return adjustments
    
    def _assess_risk_level_changes(
        self, 
        crisis: Crisis, 
        risk_assessments: List[RiskAssessment]
    ) -> Dict[str, float]:
        """Assess how crisis changes risk levels for strategic initiatives"""
        
        risk_changes = {}
        
        for risk in risk_assessments:
            # Increase risk levels based on crisis type and severity
            if crisis.crisis_type == CrisisType.SECURITY_BREACH and "security" in risk.risk_type.lower():
                risk_changes[risk.id] = 0.2  # Increase by 20%
            elif crisis.crisis_type == CrisisType.MARKET_VOLATILITY and "market" in risk.risk_type.lower():
                risk_changes[risk.id] = 0.3  # Increase by 30%
            elif crisis.crisis_type == CrisisType.REGULATORY_VIOLATION and "regulatory" in risk.risk_type.lower():
                risk_changes[risk.id] = 0.25  # Increase by 25%
            else:
                # General risk increase during crisis
                risk_changes[risk.id] = 0.1  # Increase by 10%
        
        return risk_changes
    
    def _generate_strategic_recommendations(
        self, 
        crisis: Crisis, 
        impact_level: CrisisImpactLevel, 
        strategic_roadmap: StrategicRoadmap
    ) -> List[str]:
        """Generate strategic recommendations based on crisis impact"""
        
        recommendations = []
        
        # Base recommendations by impact level
        if impact_level in [CrisisImpactLevel.SEVERE, CrisisImpactLevel.CATASTROPHIC]:
            recommendations.extend([
                "Immediately pause non-critical strategic initiatives",
                "Reallocate majority of resources to crisis response",
                "Establish crisis-strategic integration task force",
                "Implement emergency decision-making protocols"
            ])
        elif impact_level == CrisisImpactLevel.SIGNIFICANT:
            recommendations.extend([
                "Delay non-essential strategic milestones",
                "Increase risk monitoring for all technology bets",
                "Enhance stakeholder communication frequency",
                "Review and adjust resource allocation priorities"
            ])
        else:
            recommendations.extend([
                "Monitor crisis impact on strategic initiatives",
                "Maintain strategic momentum while addressing crisis",
                "Document lessons learned for future planning",
                "Strengthen crisis resilience capabilities"
            ])
        
        # Crisis-type specific recommendations
        if crisis.crisis_type == CrisisType.SECURITY_BREACH:
            recommendations.extend([
                "Accelerate cybersecurity technology investments",
                "Review all technology bets for security implications",
                "Integrate security-by-design into strategic planning"
            ])
        elif crisis.crisis_type == CrisisType.TALENT_SHORTAGE:
            recommendations.extend([
                "Prioritize talent acquisition and retention initiatives",
                "Accelerate automation and AI technology bets",
                "Review talent requirements for all strategic milestones"
            ])
        
        return recommendations
    
    def _estimate_recovery_timeline(
        self, 
        crisis: Crisis, 
        impact_level: CrisisImpactLevel
    ) -> int:
        """Estimate timeline for strategic momentum recovery"""
        
        base_timeline = {
            CrisisImpactLevel.MINIMAL: 30,        # 1 month
            CrisisImpactLevel.MODERATE: 90,       # 3 months
            CrisisImpactLevel.SIGNIFICANT: 180,   # 6 months
            CrisisImpactLevel.SEVERE: 365,        # 1 year
            CrisisImpactLevel.CATASTROPHIC: 730   # 2 years
        }
        
        recovery_days = base_timeline.get(impact_level, 90)
        
        # Adjust based on crisis type
        if crisis.crisis_type == CrisisType.NATURAL_DISASTER:
            recovery_days *= 1.5  # Physical disasters take longer to recover from
        elif crisis.crisis_type == CrisisType.SECURITY_BREACH:
            recovery_days *= 1.2  # Security breaches need trust rebuilding
        
        return int(recovery_days)