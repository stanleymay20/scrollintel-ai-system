"""
Relationship Building Engine for board and executive relationship development.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import asdict

from ..models.relationship_models import (
    RelationshipProfile, RelationshipAction, RelationshipInsight,
    RelationshipNetwork, RelationshipMaintenancePlan, RelationshipHistory,
    RelationshipGoal, TrustMetrics, PersonalityProfile, RelationshipType,
    RelationshipStatus, InteractionType, CommunicationStyle
)


class RelationshipDevelopmentFramework:
    """Framework for long-term board member relationship development."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_relationship_profile(self, stakeholder_data: Dict[str, Any]) -> RelationshipProfile:
        """Create comprehensive relationship profile for a stakeholder."""
        try:
            # Analyze personality and communication style
            personality_profile = self._analyze_personality_profile(stakeholder_data)
            
            # Initialize trust metrics
            trust_metrics = TrustMetrics(
                overall_trust_score=0.5,  # Start neutral
                competence_trust=0.5,
                benevolence_trust=0.5,
                integrity_trust=0.5,
                predictability_trust=0.5,
                transparency_score=0.5,
                reliability_score=0.5,
                last_updated=datetime.now()
            )
            
            # Create relationship profile
            profile = RelationshipProfile(
                stakeholder_id=stakeholder_data['id'],
                name=stakeholder_data['name'],
                title=stakeholder_data['title'],
                organization=stakeholder_data['organization'],
                relationship_type=RelationshipType(stakeholder_data['type']),
                relationship_status=RelationshipStatus.INITIAL,
                personality_profile=personality_profile,
                influence_level=stakeholder_data.get('influence_level', 0.5),
                decision_making_power=stakeholder_data.get('decision_power', 0.5),
                network_connections=stakeholder_data.get('connections', []),
                trust_metrics=trust_metrics,
                relationship_strength=0.3,  # Start low
                engagement_frequency=0.0,
                response_rate=0.0,
                relationship_start_date=datetime.now(),
                last_interaction_date=None,
                interaction_history=[],
                relationship_goals=[],
                development_strategy="",
                next_planned_interaction=None,
                key_interests=stakeholder_data.get('interests', []),
                business_priorities=stakeholder_data.get('priorities', []),
                personal_interests=stakeholder_data.get('personal_interests', []),
                communication_cadence="monthly"
            )
            
            # Generate initial development strategy
            profile.development_strategy = self._create_development_strategy(profile)
            
            # Set initial relationship goals
            profile.relationship_goals = self._create_initial_goals(profile)
            
            self.logger.info(f"Created relationship profile for {profile.name}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error creating relationship profile: {str(e)}")
            raise
    
    def _analyze_personality_profile(self, stakeholder_data: Dict[str, Any]) -> PersonalityProfile:
        """Analyze stakeholder data to determine personality profile."""
        # Determine communication style based on role and background
        communication_style = CommunicationStyle.DIRECT
        if stakeholder_data.get('type') == 'board_member':
            communication_style = CommunicationStyle.DIPLOMATIC
        elif stakeholder_data.get('background') == 'technical':
            communication_style = CommunicationStyle.DATA_DRIVEN
        elif stakeholder_data.get('role') == 'investor':
            communication_style = CommunicationStyle.RESULTS_ORIENTED
        
        return PersonalityProfile(
            communication_style=communication_style,
            decision_making_style=stakeholder_data.get('decision_style', 'analytical'),
            key_motivators=stakeholder_data.get('motivators', ['success', 'growth']),
            concerns=stakeholder_data.get('concerns', ['risk', 'compliance']),
            preferred_interaction_frequency=stakeholder_data.get('frequency', 'monthly'),
            optimal_meeting_times=stakeholder_data.get('meeting_times', ['morning']),
            communication_preferences=stakeholder_data.get('comm_prefs', {})
        )
    
    def _create_development_strategy(self, profile: RelationshipProfile) -> str:
        """Create tailored development strategy based on stakeholder profile."""
        strategies = []
        
        if profile.relationship_type == RelationshipType.BOARD_MEMBER:
            strategies.append("Focus on strategic value demonstration")
            strategies.append("Provide regular governance insights")
            strategies.append("Build trust through transparency")
        elif profile.relationship_type == RelationshipType.INVESTOR:
            strategies.append("Emphasize ROI and growth metrics")
            strategies.append("Provide market opportunity analysis")
            strategies.append("Demonstrate competitive advantages")
        elif profile.relationship_type == RelationshipType.EXECUTIVE:
            strategies.append("Collaborate on strategic initiatives")
            strategies.append("Support cross-functional alignment")
            strategies.append("Share technology insights")
        
        # Add personality-based strategies
        if profile.personality_profile.communication_style == CommunicationStyle.DATA_DRIVEN:
            strategies.append("Lead with data and analytics")
        elif profile.personality_profile.communication_style == CommunicationStyle.RELATIONSHIP_FOCUSED:
            strategies.append("Invest in personal connection building")
        
        return "; ".join(strategies)
    
    def _create_initial_goals(self, profile: RelationshipProfile) -> List[RelationshipGoal]:
        """Create initial relationship development goals."""
        goals = []
        
        # Trust building goal
        goals.append(RelationshipGoal(
            goal_id=f"{profile.stakeholder_id}_trust_1",
            description="Establish baseline trust and credibility",
            target_date=datetime.now() + timedelta(days=90),
            priority="high",
            success_metrics=["Trust score > 0.7", "Positive interaction feedback"],
            current_progress=0.0,
            action_items=["Schedule introductory meeting", "Prepare capability overview"]
        ))
        
        # Engagement goal
        goals.append(RelationshipGoal(
            goal_id=f"{profile.stakeholder_id}_engagement_1",
            description="Establish regular communication cadence",
            target_date=datetime.now() + timedelta(days=60),
            priority="medium",
            success_metrics=["Monthly touchpoints", "Response rate > 80%"],
            current_progress=0.0,
            action_items=["Set up recurring meetings", "Create communication calendar"]
        ))
        
        return goals
    
    def develop_relationship_roadmap(self, profile: RelationshipProfile, 
                                   timeline_months: int = 12) -> List[RelationshipAction]:
        """Develop comprehensive relationship building roadmap."""
        try:
            actions = []
            current_date = datetime.now()
            
            # Phase 1: Foundation (0-3 months)
            actions.extend(self._create_foundation_actions(profile, current_date))
            
            # Phase 2: Development (3-6 months)
            actions.extend(self._create_development_actions(profile, current_date + timedelta(days=90)))
            
            # Phase 3: Strengthening (6-12 months)
            actions.extend(self._create_strengthening_actions(profile, current_date + timedelta(days=180)))
            
            self.logger.info(f"Created {len(actions)} relationship actions for {profile.name}")
            return actions
            
        except Exception as e:
            self.logger.error(f"Error developing relationship roadmap: {str(e)}")
            raise
    
    def _create_foundation_actions(self, profile: RelationshipProfile, 
                                 start_date: datetime) -> List[RelationshipAction]:
        """Create foundation phase actions (0-3 months)."""
        actions = []
        
        # Initial meeting
        actions.append(RelationshipAction(
            action_id=f"{profile.stakeholder_id}_foundation_1",
            stakeholder_id=profile.stakeholder_id,
            action_type="initial_meeting",
            description="Conduct comprehensive introductory meeting",
            scheduled_date=start_date + timedelta(days=7),
            priority="high",
            expected_outcome="Establish rapport and understand priorities",
            preparation_required=[
                "Research stakeholder background",
                "Prepare capability overview",
                "Identify mutual interests"
            ],
            success_criteria=[
                "Positive meeting feedback",
                "Follow-up meeting scheduled",
                "Key priorities identified"
            ],
            status="planned"
        ))
        
        # Value demonstration
        actions.append(RelationshipAction(
            action_id=f"{profile.stakeholder_id}_foundation_2",
            stakeholder_id=profile.stakeholder_id,
            action_type="value_demonstration",
            description="Provide strategic insight relevant to their priorities",
            scheduled_date=start_date + timedelta(days=21),
            priority="high",
            expected_outcome="Demonstrate strategic value and expertise",
            preparation_required=[
                "Analyze their business challenges",
                "Prepare strategic recommendations",
                "Create executive summary"
            ],
            success_criteria=[
                "Positive feedback on insights",
                "Request for additional analysis",
                "Trust score improvement"
            ],
            status="planned"
        ))
        
        return actions
    
    def _create_development_actions(self, profile: RelationshipProfile, 
                                  start_date: datetime) -> List[RelationshipAction]:
        """Create development phase actions (3-6 months)."""
        actions = []
        
        # Regular touchpoints
        for i in range(3):
            actions.append(RelationshipAction(
                action_id=f"{profile.stakeholder_id}_development_{i+1}",
                stakeholder_id=profile.stakeholder_id,
                action_type="regular_touchpoint",
                description=f"Monthly strategic update and consultation #{i+1}",
                scheduled_date=start_date + timedelta(days=30*i),
                priority="medium",
                expected_outcome="Maintain engagement and provide ongoing value",
                preparation_required=[
                    "Prepare monthly insights",
                    "Review their recent activities",
                    "Identify collaboration opportunities"
                ],
                success_criteria=[
                    "Consistent engagement",
                    "Positive feedback",
                    "Relationship strength increase"
                ],
                status="planned"
            ))
        
        return actions
    
    def _create_strengthening_actions(self, profile: RelationshipProfile, 
                                    start_date: datetime) -> List[RelationshipAction]:
        """Create strengthening phase actions (6-12 months)."""
        actions = []
        
        # Strategic collaboration
        actions.append(RelationshipAction(
            action_id=f"{profile.stakeholder_id}_strengthen_1",
            stakeholder_id=profile.stakeholder_id,
            action_type="strategic_collaboration",
            description="Initiate joint strategic initiative",
            scheduled_date=start_date + timedelta(days=30),
            priority="high",
            expected_outcome="Deepen partnership through collaboration",
            preparation_required=[
                "Identify collaboration opportunities",
                "Develop joint proposal",
                "Align on success metrics"
            ],
            success_criteria=[
                "Joint initiative launched",
                "Strong partnership established",
                "Mutual value creation"
            ],
            status="planned"
        ))
        
        return actions


class RelationshipMaintenanceSystem:
    """System for maintaining and enhancing existing relationships."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_maintenance_plan(self, profile: RelationshipProfile) -> RelationshipMaintenancePlan:
        """Create systematic maintenance plan for relationship."""
        try:
            # Determine maintenance frequency based on relationship importance
            frequency = self._determine_maintenance_frequency(profile)
            
            # Define touch point types
            touch_points = self._define_touch_points(profile)
            
            # Create content themes
            content_themes = self._create_content_themes(profile)
            
            plan = RelationshipMaintenancePlan(
                plan_id=f"{profile.stakeholder_id}_maintenance",
                stakeholder_id=profile.stakeholder_id,
                maintenance_frequency=frequency,
                touch_point_types=touch_points,
                content_themes=content_themes,
                seasonal_considerations=self._identify_seasonal_considerations(profile),
                escalation_triggers=self._define_escalation_triggers(profile),
                success_indicators=self._define_success_indicators(profile),
                next_review_date=datetime.now() + timedelta(days=90)
            )
            
            self.logger.info(f"Created maintenance plan for {profile.name}")
            return plan
            
        except Exception as e:
            self.logger.error(f"Error creating maintenance plan: {str(e)}")
            raise
    
    def _determine_maintenance_frequency(self, profile: RelationshipProfile) -> str:
        """Determine optimal maintenance frequency."""
        if profile.influence_level > 0.8:
            return "weekly"
        elif profile.influence_level > 0.6:
            return "bi-weekly"
        elif profile.relationship_strength > 0.7:
            return "monthly"
        else:
            return "quarterly"
    
    def _define_touch_points(self, profile: RelationshipProfile) -> List[str]:
        """Define appropriate touch point types."""
        touch_points = ["email_update", "strategic_insight"]
        
        if profile.relationship_type == RelationshipType.BOARD_MEMBER:
            touch_points.extend(["board_preparation", "governance_update"])
        elif profile.relationship_type == RelationshipType.INVESTOR:
            touch_points.extend(["performance_update", "market_analysis"])
        elif profile.relationship_type == RelationshipType.EXECUTIVE:
            touch_points.extend(["collaboration_opportunity", "strategic_consultation"])
        
        return touch_points
    
    def _create_content_themes(self, profile: RelationshipProfile) -> List[str]:
        """Create relevant content themes."""
        themes = ["strategic_insights", "industry_trends", "technology_updates"]
        
        # Add personalized themes based on interests
        themes.extend(profile.key_interests[:3])  # Top 3 interests
        
        return themes
    
    def _identify_seasonal_considerations(self, profile: RelationshipProfile) -> List[str]:
        """Identify seasonal relationship considerations."""
        considerations = [
            "Q4 board planning season",
            "Annual review periods",
            "Budget planning cycles"
        ]
        
        if profile.relationship_type == RelationshipType.INVESTOR:
            considerations.append("Earnings seasons")
        
        return considerations
    
    def _define_escalation_triggers(self, profile: RelationshipProfile) -> List[str]:
        """Define triggers for relationship escalation."""
        return [
            "Trust score drops below 0.6",
            "No response to 3 consecutive communications",
            "Negative feedback received",
            "Relationship strength decreases by 20%"
        ]
    
    def _define_success_indicators(self, profile: RelationshipProfile) -> List[str]:
        """Define success indicators for relationship maintenance."""
        return [
            "Consistent engagement levels",
            "Positive interaction sentiment",
            "Trust score maintenance above 0.7",
            "Proactive communication from stakeholder"
        ]
    
    def execute_maintenance_action(self, action: RelationshipAction, 
                                 profile: RelationshipProfile) -> Dict[str, Any]:
        """Execute a specific maintenance action."""
        try:
            result = {
                'action_id': action.action_id,
                'executed_at': datetime.now(),
                'status': 'completed',
                'outcomes': [],
                'next_actions': []
            }
            
            if action.action_type == "email_update":
                result['outcomes'].append("Strategic update email sent")
                result['next_actions'].append("Monitor response and engagement")
            
            elif action.action_type == "strategic_insight":
                result['outcomes'].append("Strategic insight shared")
                result['next_actions'].append("Follow up on insight application")
            
            elif action.action_type == "board_preparation":
                result['outcomes'].append("Board meeting preparation materials provided")
                result['next_actions'].append("Gather feedback on materials")
            
            self.logger.info(f"Executed maintenance action {action.action_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing maintenance action: {str(e)}")
            raise


class RelationshipQualityAssessment:
    """System for assessing and optimizing relationship quality."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def assess_relationship_quality(self, profile: RelationshipProfile) -> Dict[str, Any]:
        """Comprehensive assessment of relationship quality."""
        try:
            assessment = {
                'stakeholder_id': profile.stakeholder_id,
                'assessment_date': datetime.now(),
                'overall_score': 0.0,
                'dimension_scores': {},
                'strengths': [],
                'weaknesses': [],
                'recommendations': [],
                'risk_factors': []
            }
            
            # Assess different dimensions
            trust_score = self._assess_trust_dimension(profile)
            engagement_score = self._assess_engagement_dimension(profile)
            value_score = self._assess_value_dimension(profile)
            communication_score = self._assess_communication_dimension(profile)
            
            assessment['dimension_scores'] = {
                'trust': trust_score,
                'engagement': engagement_score,
                'value_delivery': value_score,
                'communication': communication_score
            }
            
            # Calculate overall score
            assessment['overall_score'] = (
                trust_score * 0.3 + 
                engagement_score * 0.25 + 
                value_score * 0.25 + 
                communication_score * 0.2
            )
            
            # Identify strengths and weaknesses
            assessment['strengths'] = self._identify_strengths(assessment['dimension_scores'])
            assessment['weaknesses'] = self._identify_weaknesses(assessment['dimension_scores'])
            
            # Generate recommendations
            assessment['recommendations'] = self._generate_recommendations(
                profile, assessment['dimension_scores']
            )
            
            # Identify risk factors
            assessment['risk_factors'] = self._identify_risk_factors(profile)
            
            self.logger.info(f"Completed quality assessment for {profile.name}")
            return assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing relationship quality: {str(e)}")
            raise
    
    def _assess_trust_dimension(self, profile: RelationshipProfile) -> float:
        """Assess trust dimension of relationship."""
        return profile.trust_metrics.overall_trust_score
    
    def _assess_engagement_dimension(self, profile: RelationshipProfile) -> float:
        """Assess engagement dimension of relationship."""
        # Consider response rate, interaction frequency, and recency
        recency_score = 1.0
        if profile.last_interaction_date:
            days_since = (datetime.now() - profile.last_interaction_date).days
            recency_score = max(0.0, 1.0 - (days_since / 90))  # Decay over 90 days
        
        return (profile.response_rate * 0.4 + 
                profile.engagement_frequency * 0.4 + 
                recency_score * 0.2)
    
    def _assess_value_dimension(self, profile: RelationshipProfile) -> float:
        """Assess value delivery dimension."""
        # Based on goal achievement and positive outcomes
        goal_achievement = 0.0
        if profile.relationship_goals:
            total_progress = sum(goal.current_progress for goal in profile.relationship_goals)
            goal_achievement = total_progress / len(profile.relationship_goals)
        
        return min(1.0, goal_achievement + profile.relationship_strength * 0.3)
    
    def _assess_communication_dimension(self, profile: RelationshipProfile) -> float:
        """Assess communication effectiveness."""
        # Based on interaction sentiment and communication alignment
        if not profile.interaction_history:
            return 0.5  # Neutral if no history
        
        recent_interactions = profile.interaction_history[-5:]  # Last 5 interactions
        avg_sentiment = sum(interaction.sentiment_score for interaction in recent_interactions) / len(recent_interactions)
        
        # Normalize sentiment from [-1, 1] to [0, 1]
        return (avg_sentiment + 1) / 2
    
    def _identify_strengths(self, dimension_scores: Dict[str, float]) -> List[str]:
        """Identify relationship strengths."""
        strengths = []
        for dimension, score in dimension_scores.items():
            if score > 0.8:
                strengths.append(f"Excellent {dimension.replace('_', ' ')}")
            elif score > 0.7:
                strengths.append(f"Strong {dimension.replace('_', ' ')}")
        return strengths
    
    def _identify_weaknesses(self, dimension_scores: Dict[str, float]) -> List[str]:
        """Identify relationship weaknesses."""
        weaknesses = []
        for dimension, score in dimension_scores.items():
            if score < 0.4:
                weaknesses.append(f"Poor {dimension.replace('_', ' ')}")
            elif score < 0.6:
                weaknesses.append(f"Weak {dimension.replace('_', ' ')}")
        return weaknesses
    
    def _generate_recommendations(self, profile: RelationshipProfile, 
                                dimension_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if dimension_scores['trust'] < 0.6:
            recommendations.append("Focus on trust building through transparency and reliability")
        
        if dimension_scores['engagement'] < 0.6:
            recommendations.append("Increase engagement frequency and improve response times")
        
        if dimension_scores['value_delivery'] < 0.6:
            recommendations.append("Enhance value proposition and demonstrate concrete benefits")
        
        if dimension_scores['communication'] < 0.6:
            recommendations.append("Improve communication style alignment and message clarity")
        
        return recommendations
    
    def _identify_risk_factors(self, profile: RelationshipProfile) -> List[str]:
        """Identify relationship risk factors."""
        risks = []
        
        if profile.trust_metrics.overall_trust_score < 0.5:
            risks.append("Low trust levels")
        
        if profile.response_rate < 0.5:
            risks.append("Poor response rate")
        
        if profile.last_interaction_date and (datetime.now() - profile.last_interaction_date).days > 60:
            risks.append("Extended period without interaction")
        
        if profile.relationship_strength < 0.4:
            risks.append("Weak relationship foundation")
        
        return risks


class RelationshipBuildingEngine:
    """Main engine orchestrating relationship building activities."""
    
    def __init__(self):
        self.development_framework = RelationshipDevelopmentFramework()
        self.maintenance_system = RelationshipMaintenanceSystem()
        self.quality_assessment = RelationshipQualityAssessment()
        self.logger = logging.getLogger(__name__)
    
    def initialize_relationship(self, stakeholder_data: Dict[str, Any]) -> RelationshipProfile:
        """Initialize a new relationship with comprehensive planning."""
        try:
            # Create relationship profile
            profile = self.development_framework.create_relationship_profile(stakeholder_data)
            
            # Develop relationship roadmap
            roadmap = self.development_framework.develop_relationship_roadmap(profile)
            
            # Create maintenance plan
            maintenance_plan = self.maintenance_system.create_maintenance_plan(profile)
            
            self.logger.info(f"Initialized relationship with {profile.name}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Error initializing relationship: {str(e)}")
            raise
    
    def optimize_relationship(self, profile: RelationshipProfile) -> Dict[str, Any]:
        """Optimize existing relationship based on quality assessment."""
        try:
            # Assess current quality
            assessment = self.quality_assessment.assess_relationship_quality(profile)
            
            # Generate optimization plan
            optimization_plan = {
                'assessment': assessment,
                'optimization_actions': [],
                'updated_goals': [],
                'revised_strategy': ""
            }
            
            # Create optimization actions based on weaknesses
            for weakness in assessment['weaknesses']:
                if 'trust' in weakness.lower():
                    optimization_plan['optimization_actions'].append(
                        self._create_trust_building_action(profile)
                    )
                elif 'engagement' in weakness.lower():
                    optimization_plan['optimization_actions'].append(
                        self._create_engagement_action(profile)
                    )
                elif 'value' in weakness.lower():
                    optimization_plan['optimization_actions'].append(
                        self._create_value_demonstration_action(profile)
                    )
            
            self.logger.info(f"Created optimization plan for {profile.name}")
            return optimization_plan
            
        except Exception as e:
            self.logger.error(f"Error optimizing relationship: {str(e)}")
            raise
    
    def _create_trust_building_action(self, profile: RelationshipProfile) -> RelationshipAction:
        """Create trust building action."""
        return RelationshipAction(
            action_id=f"{profile.stakeholder_id}_trust_building",
            stakeholder_id=profile.stakeholder_id,
            action_type="trust_building",
            description="Implement trust building initiative",
            scheduled_date=datetime.now() + timedelta(days=7),
            priority="high",
            expected_outcome="Improved trust metrics",
            preparation_required=[
                "Identify trust gaps",
                "Prepare transparency report",
                "Schedule trust-building conversation"
            ],
            success_criteria=["Trust score improvement", "Positive feedback"],
            status="planned"
        )
    
    def _create_engagement_action(self, profile: RelationshipProfile) -> RelationshipAction:
        """Create engagement improvement action."""
        return RelationshipAction(
            action_id=f"{profile.stakeholder_id}_engagement_boost",
            stakeholder_id=profile.stakeholder_id,
            action_type="engagement_improvement",
            description="Enhance engagement through personalized outreach",
            scheduled_date=datetime.now() + timedelta(days=3),
            priority="medium",
            expected_outcome="Increased engagement levels",
            preparation_required=[
                "Analyze engagement patterns",
                "Personalize communication approach",
                "Create engaging content"
            ],
            success_criteria=["Higher response rate", "More frequent interactions"],
            status="planned"
        )
    
    def _create_value_demonstration_action(self, profile: RelationshipProfile) -> RelationshipAction:
        """Create value demonstration action."""
        return RelationshipAction(
            action_id=f"{profile.stakeholder_id}_value_demo",
            stakeholder_id=profile.stakeholder_id,
            action_type="value_demonstration",
            description="Demonstrate concrete value through strategic insights",
            scheduled_date=datetime.now() + timedelta(days=5),
            priority="high",
            expected_outcome="Clear value recognition",
            preparation_required=[
                "Identify value opportunities",
                "Prepare strategic analysis",
                "Create value proposition"
            ],
            success_criteria=["Positive value feedback", "Request for more insights"],
            status="planned"
        )