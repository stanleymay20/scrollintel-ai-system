"""
Cultural Leadership Assessment Engine

Comprehensive system for assessing cultural leadership capabilities,
creating development plans, and measuring leadership effectiveness.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import asdict

from ..models.cultural_leadership_models import (
    CulturalLeadershipAssessment, CompetencyScore, CulturalCompetency,
    LeadershipLevel, LeadershipDevelopmentPlan, LearningActivity,
    CoachingSession, DevelopmentMilestone, CulturalLeadershipProfile,
    AssessmentFramework, LeadershipEffectivenessMetrics
)

logger = logging.getLogger(__name__)


class CulturalLeadershipAssessmentEngine:
    """Engine for cultural leadership assessment and development"""
    
    def __init__(self):
        self.assessment_frameworks = {}
        self.competency_weights = self._initialize_competency_weights()
        self.scoring_rubrics = self._initialize_scoring_rubrics()
        self.development_resources = self._initialize_development_resources()
        
    def _initialize_competency_weights(self) -> Dict[CulturalCompetency, float]:
        """Initialize competency weights for overall scoring"""
        return {
            CulturalCompetency.VISION_CREATION: 0.15,
            CulturalCompetency.VALUES_ALIGNMENT: 0.12,
            CulturalCompetency.CHANGE_LEADERSHIP: 0.13,
            CulturalCompetency.COMMUNICATION: 0.14,
            CulturalCompetency.INFLUENCE: 0.11,
            CulturalCompetency.EMPATHY: 0.10,
            CulturalCompetency.AUTHENTICITY: 0.08,
            CulturalCompetency.RESILIENCE: 0.07,
            CulturalCompetency.ADAPTABILITY: 0.06,
            CulturalCompetency.SYSTEMS_THINKING: 0.04
        }
    
    def _initialize_scoring_rubrics(self) -> Dict[CulturalCompetency, Dict[LeadershipLevel, Dict[str, Any]]]:
        """Initialize detailed scoring rubrics for each competency"""
        return {
            CulturalCompetency.VISION_CREATION: {
                LeadershipLevel.EMERGING: {
                    "score_range": (0, 20),
                    "indicators": ["Shows interest in organizational vision", "Asks questions about direction"],
                    "behaviors": ["Participates in vision discussions", "Seeks clarity on goals"]
                },
                LeadershipLevel.DEVELOPING: {
                    "score_range": (21, 40),
                    "indicators": ["Understands organizational vision", "Can articulate basic vision elements"],
                    "behaviors": ["Communicates vision to immediate team", "Aligns work with vision"]
                },
                LeadershipLevel.PROFICIENT: {
                    "score_range": (41, 60),
                    "indicators": ["Creates compelling team vision", "Connects vision to daily work"],
                    "behaviors": ["Facilitates vision development", "Inspires others with vision"]
                },
                LeadershipLevel.ADVANCED: {
                    "score_range": (61, 80),
                    "indicators": ["Develops organizational vision", "Creates vision alignment across teams"],
                    "behaviors": ["Leads vision transformation", "Builds vision consensus"]
                },
                LeadershipLevel.EXPERT: {
                    "score_range": (81, 100),
                    "indicators": ["Creates transformational vision", "Influences industry vision"],
                    "behaviors": ["Shapes organizational future", "Inspires cultural transformation"]
                }
            },
            CulturalCompetency.COMMUNICATION: {
                LeadershipLevel.EMERGING: {
                    "score_range": (0, 20),
                    "indicators": ["Basic communication skills", "Listens to others"],
                    "behaviors": ["Participates in meetings", "Asks clarifying questions"]
                },
                LeadershipLevel.DEVELOPING: {
                    "score_range": (21, 40),
                    "indicators": ["Clear verbal communication", "Adapts message to audience"],
                    "behaviors": ["Facilitates team discussions", "Provides regular updates"]
                },
                LeadershipLevel.PROFICIENT: {
                    "score_range": (41, 60),
                    "indicators": ["Persuasive communication", "Builds rapport across levels"],
                    "behaviors": ["Influences through communication", "Resolves conflicts effectively"]
                },
                LeadershipLevel.ADVANCED: {
                    "score_range": (61, 80),
                    "indicators": ["Inspirational communication", "Creates emotional connection"],
                    "behaviors": ["Drives change through communication", "Builds organizational alignment"]
                },
                LeadershipLevel.EXPERT: {
                    "score_range": (81, 100),
                    "indicators": ["Transformational communication", "Shapes organizational narrative"],
                    "behaviors": ["Creates cultural movement", "Influences external stakeholders"]
                }
            }
        }
    
    def _initialize_development_resources(self) -> Dict[CulturalCompetency, List[Dict[str, Any]]]:
        """Initialize development resources for each competency"""
        return {
            CulturalCompetency.VISION_CREATION: [
                {
                    "type": "workshop",
                    "title": "Vision Development Masterclass",
                    "duration": 16,
                    "description": "Learn to create compelling organizational visions"
                },
                {
                    "type": "reading",
                    "title": "Visionary Leadership Principles",
                    "duration": 8,
                    "description": "Study proven vision creation methodologies"
                },
                {
                    "type": "project",
                    "title": "Team Vision Creation",
                    "duration": 40,
                    "description": "Lead your team through vision development process"
                }
            ],
            CulturalCompetency.COMMUNICATION: [
                {
                    "type": "coaching",
                    "title": "Executive Communication Coaching",
                    "duration": 20,
                    "description": "One-on-one coaching for leadership communication"
                },
                {
                    "type": "workshop",
                    "title": "Influential Communication Skills",
                    "duration": 12,
                    "description": "Master persuasive and inspiring communication"
                }
            ]
        }
    
    def assess_cultural_leadership(
        self,
        leader_id: str,
        organization_id: str,
        assessment_data: Dict[str, Any],
        framework_id: Optional[str] = None
    ) -> CulturalLeadershipAssessment:
        """Conduct comprehensive cultural leadership assessment"""
        try:
            logger.info(f"Starting cultural leadership assessment for leader {leader_id}")
            
            # Create competency scores
            competency_scores = []
            total_weighted_score = 0
            
            for competency in CulturalCompetency:
                score = self._assess_competency(
                    competency,
                    assessment_data.get(competency.value, {}),
                    assessment_data.get('assessment_method', 'comprehensive')
                )
                competency_scores.append(score)
                total_weighted_score += score.score * self.competency_weights[competency]
            
            # Determine overall leadership level
            leadership_level = self._determine_leadership_level(total_weighted_score)
            
            # Calculate specific scores
            cultural_impact_score = self._calculate_cultural_impact_score(assessment_data)
            vision_clarity_score = self._calculate_vision_clarity_score(assessment_data)
            communication_effectiveness = self._calculate_communication_effectiveness(assessment_data)
            change_readiness = self._calculate_change_readiness_score(assessment_data)
            team_engagement_score = self._calculate_team_engagement_score(assessment_data)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(competency_scores, assessment_data)
            
            # Create assessment
            assessment = CulturalLeadershipAssessment(
                id=str(uuid.uuid4()),
                leader_id=leader_id,
                organization_id=organization_id,
                assessment_date=datetime.now(),
                competency_scores=competency_scores,
                overall_score=total_weighted_score,
                leadership_level=leadership_level,
                cultural_impact_score=cultural_impact_score,
                vision_clarity_score=vision_clarity_score,
                communication_effectiveness=communication_effectiveness,
                change_readiness=change_readiness,
                team_engagement_score=team_engagement_score,
                assessment_method=assessment_data.get('assessment_method', 'comprehensive'),
                assessor_id=assessment_data.get('assessor_id'),
                self_assessment=assessment_data.get('self_assessment', False),
                peer_feedback=assessment_data.get('peer_feedback', []),
                direct_report_feedback=assessment_data.get('direct_report_feedback', []),
                recommendations=recommendations
            )
            
            logger.info(f"Cultural leadership assessment completed for leader {leader_id}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error in cultural leadership assessment: {str(e)}")
            raise
    
    def _assess_competency(
        self,
        competency: CulturalCompetency,
        competency_data: Dict[str, Any],
        assessment_method: str
    ) -> CompetencyScore:
        """Assess individual competency"""
        # Get behavioral indicators
        behaviors = competency_data.get('behaviors', [])
        self_rating = competency_data.get('self_rating', 0)
        peer_ratings = competency_data.get('peer_ratings', [])
        manager_rating = competency_data.get('manager_rating', 0)
        
        # Calculate composite score
        if assessment_method == 'comprehensive':
            score = self._calculate_comprehensive_score(
                self_rating, peer_ratings, manager_rating, behaviors
            )
        elif assessment_method == '360_feedback':
            score = self._calculate_360_score(self_rating, peer_ratings, manager_rating)
        else:
            score = self_rating
        
        # Determine levels
        current_level = self._score_to_level(score)
        target_level = self._determine_target_level(current_level, competency_data)
        
        # Extract evidence and development areas
        evidence = competency_data.get('evidence', [])
        development_areas = self._identify_development_areas(competency, score, behaviors)
        strengths = self._identify_strengths(competency, score, behaviors)
        
        return CompetencyScore(
            competency=competency,
            current_level=current_level,
            target_level=target_level,
            score=score,
            evidence=evidence,
            development_areas=development_areas,
            strengths=strengths
        )
    
    def _calculate_comprehensive_score(
        self,
        self_rating: float,
        peer_ratings: List[float],
        manager_rating: float,
        behaviors: List[str]
    ) -> float:
        """Calculate comprehensive competency score"""
        # Weight different sources
        weights = {
            'self': 0.2,
            'peers': 0.4,
            'manager': 0.3,
            'behaviors': 0.1
        }
        
        # Calculate weighted average
        peer_avg = sum(peer_ratings) / len(peer_ratings) if peer_ratings else 0
        behavior_score = len(behaviors) * 10  # Simple behavior count scoring
        
        total_score = (
            self_rating * weights['self'] +
            peer_avg * weights['peers'] +
            manager_rating * weights['manager'] +
            min(behavior_score, 100) * weights['behaviors']
        )
        
        return min(max(total_score, 0), 100)
    
    def _calculate_360_score(
        self,
        self_rating: float,
        peer_ratings: List[float],
        manager_rating: float
    ) -> float:
        """Calculate 360-degree feedback score"""
        weights = {'self': 0.25, 'peers': 0.5, 'manager': 0.25}
        
        peer_avg = sum(peer_ratings) / len(peer_ratings) if peer_ratings else 0
        
        return (
            self_rating * weights['self'] +
            peer_avg * weights['peers'] +
            manager_rating * weights['manager']
        )
    
    def _score_to_level(self, score: float) -> LeadershipLevel:
        """Convert numeric score to leadership level"""
        if score >= 81:
            return LeadershipLevel.EXPERT
        elif score >= 61:
            return LeadershipLevel.ADVANCED
        elif score >= 41:
            return LeadershipLevel.PROFICIENT
        elif score >= 21:
            return LeadershipLevel.DEVELOPING
        else:
            return LeadershipLevel.EMERGING
    
    def _determine_target_level(
        self,
        current_level: LeadershipLevel,
        competency_data: Dict[str, Any]
    ) -> LeadershipLevel:
        """Determine appropriate target level for development"""
        target_preference = competency_data.get('target_level')
        if target_preference:
            return LeadershipLevel(target_preference)
        
        # Default to next level up
        level_progression = {
            LeadershipLevel.EMERGING: LeadershipLevel.DEVELOPING,
            LeadershipLevel.DEVELOPING: LeadershipLevel.PROFICIENT,
            LeadershipLevel.PROFICIENT: LeadershipLevel.ADVANCED,
            LeadershipLevel.ADVANCED: LeadershipLevel.EXPERT,
            LeadershipLevel.EXPERT: LeadershipLevel.EXPERT
        }
        
        return level_progression.get(current_level, LeadershipLevel.PROFICIENT)
    
    def _determine_leadership_level(self, overall_score: float) -> LeadershipLevel:
        """Determine overall leadership level from composite score"""
        return self._score_to_level(overall_score)
    
    def _calculate_cultural_impact_score(self, assessment_data: Dict[str, Any]) -> float:
        """Calculate cultural impact score"""
        impact_indicators = assessment_data.get('cultural_impact', {})
        
        # Key impact metrics
        team_culture_improvement = impact_indicators.get('team_culture_improvement', 0)
        cultural_initiative_success = impact_indicators.get('cultural_initiative_success', 0)
        employee_engagement_change = impact_indicators.get('employee_engagement_change', 0)
        cultural_alignment_score = impact_indicators.get('cultural_alignment_score', 0)
        
        return (
            team_culture_improvement * 0.3 +
            cultural_initiative_success * 0.3 +
            employee_engagement_change * 0.2 +
            cultural_alignment_score * 0.2
        )
    
    def _calculate_vision_clarity_score(self, assessment_data: Dict[str, Any]) -> float:
        """Calculate vision clarity and communication score"""
        vision_data = assessment_data.get('vision_clarity', {})
        
        clarity_rating = vision_data.get('clarity_rating', 0)
        alignment_rating = vision_data.get('alignment_rating', 0)
        inspiration_rating = vision_data.get('inspiration_rating', 0)
        
        return (clarity_rating + alignment_rating + inspiration_rating) / 3
    
    def _calculate_communication_effectiveness(self, assessment_data: Dict[str, Any]) -> float:
        """Calculate communication effectiveness score"""
        comm_data = assessment_data.get('communication', {})
        
        clarity = comm_data.get('clarity', 0)
        influence = comm_data.get('influence', 0)
        engagement = comm_data.get('engagement', 0)
        feedback_quality = comm_data.get('feedback_quality', 0)
        
        return (clarity + influence + engagement + feedback_quality) / 4
    
    def _calculate_change_readiness_score(self, assessment_data: Dict[str, Any]) -> float:
        """Calculate change leadership readiness score"""
        change_data = assessment_data.get('change_readiness', {})
        
        adaptability = change_data.get('adaptability', 0)
        resilience = change_data.get('resilience', 0)
        change_advocacy = change_data.get('change_advocacy', 0)
        
        return (adaptability + resilience + change_advocacy) / 3
    
    def _calculate_team_engagement_score(self, assessment_data: Dict[str, Any]) -> float:
        """Calculate team engagement score"""
        engagement_data = assessment_data.get('team_engagement', {})
        
        return engagement_data.get('overall_engagement', 0)
    
    def _identify_development_areas(
        self,
        competency: CulturalCompetency,
        score: float,
        behaviors: List[str]
    ) -> List[str]:
        """Identify specific development areas for competency"""
        development_areas = []
        
        if score < 60:  # Below proficient level
            if competency == CulturalCompetency.VISION_CREATION:
                development_areas.extend([
                    "Develop vision articulation skills",
                    "Practice connecting vision to daily work",
                    "Learn stakeholder alignment techniques"
                ])
            elif competency == CulturalCompetency.COMMUNICATION:
                development_areas.extend([
                    "Improve message clarity and structure",
                    "Develop active listening skills",
                    "Practice influential communication techniques"
                ])
        
        return development_areas
    
    def _identify_strengths(
        self,
        competency: CulturalCompetency,
        score: float,
        behaviors: List[str]
    ) -> List[str]:
        """Identify strengths in competency"""
        strengths = []
        
        if score >= 70:  # Strong performance
            if competency == CulturalCompetency.VISION_CREATION:
                strengths.extend([
                    "Creates compelling and inspiring visions",
                    "Effectively communicates vision to others",
                    "Builds strong vision alignment"
                ])
            elif competency == CulturalCompetency.COMMUNICATION:
                strengths.extend([
                    "Communicates with clarity and impact",
                    "Builds strong relationships through communication",
                    "Influences and inspires others effectively"
                ])
        
        return strengths
    
    def _generate_recommendations(
        self,
        competency_scores: List[CompetencyScore],
        assessment_data: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized development recommendations"""
        recommendations = []
        
        # Identify lowest scoring competencies
        lowest_scores = sorted(competency_scores, key=lambda x: x.score)[:3]
        
        for comp_score in lowest_scores:
            if comp_score.score < 60:
                recommendations.append(
                    f"Focus on developing {comp_score.competency.value.replace('_', ' ').title()} "
                    f"through targeted learning and practice"
                )
        
        # Add specific recommendations based on role and context
        role = assessment_data.get('role', '')
        if 'senior' in role.lower() or 'executive' in role.lower():
            recommendations.append(
                "Develop advanced cultural transformation and organizational change capabilities"
            )
        
        return recommendations
    
    def create_development_plan(
        self,
        assessment: CulturalLeadershipAssessment,
        preferences: Dict[str, Any]
    ) -> LeadershipDevelopmentPlan:
        """Create personalized leadership development plan"""
        try:
            logger.info(f"Creating development plan for leader {assessment.leader_id}")
            
            # Identify priority competencies (lowest scores)
            priority_competencies = [
                score.competency for score in 
                sorted(assessment.competency_scores, key=lambda x: x.score)[:3]
            ]
            
            # Generate development goals
            development_goals = self._generate_development_goals(
                assessment, priority_competencies, preferences
            )
            
            # Create learning activities
            learning_activities = self._create_learning_activities(
                priority_competencies, preferences
            )
            
            # Plan coaching sessions
            coaching_sessions = self._plan_coaching_sessions(
                assessment, priority_competencies, preferences
            )
            
            # Define progress milestones
            milestones = self._define_development_milestones(
                priority_competencies, preferences
            )
            
            # Set success metrics
            success_metrics = self._define_success_metrics(assessment, priority_competencies)
            
            plan = LeadershipDevelopmentPlan(
                id=str(uuid.uuid4()),
                leader_id=assessment.leader_id,
                assessment_id=assessment.id,
                created_date=datetime.now(),
                target_completion=datetime.now() + timedelta(days=preferences.get('duration_days', 180)),
                priority_competencies=priority_competencies,
                development_goals=development_goals,
                learning_activities=learning_activities,
                coaching_sessions=coaching_sessions,
                progress_milestones=milestones,
                success_metrics=success_metrics
            )
            
            logger.info(f"Development plan created for leader {assessment.leader_id}")
            return plan
            
        except Exception as e:
            logger.error(f"Error creating development plan: {str(e)}")
            raise
    
    def _generate_development_goals(
        self,
        assessment: CulturalLeadershipAssessment,
        priority_competencies: List[CulturalCompetency],
        preferences: Dict[str, Any]
    ) -> List[str]:
        """Generate specific development goals"""
        goals = []
        
        for competency in priority_competencies:
            comp_score = next(
                (score for score in assessment.competency_scores 
                 if score.competency == competency), None
            )
            
            if comp_score:
                target_score = self._level_to_score(comp_score.target_level)
                improvement = target_score - comp_score.score
                
                goals.append(
                    f"Improve {competency.value.replace('_', ' ').title()} "
                    f"from {comp_score.current_level.value} to {comp_score.target_level.value} "
                    f"level (target improvement: {improvement:.0f} points)"
                )
        
        return goals
    
    def _level_to_score(self, level: LeadershipLevel) -> float:
        """Convert leadership level to target score"""
        level_scores = {
            LeadershipLevel.EMERGING: 15,
            LeadershipLevel.DEVELOPING: 35,
            LeadershipLevel.PROFICIENT: 55,
            LeadershipLevel.ADVANCED: 75,
            LeadershipLevel.EXPERT: 95
        }
        return level_scores.get(level, 55)
    
    def _create_learning_activities(
        self,
        priority_competencies: List[CulturalCompetency],
        preferences: Dict[str, Any]
    ) -> List[LearningActivity]:
        """Create learning activities for development plan"""
        activities = []
        
        for competency in priority_competencies:
            resources = self.development_resources.get(competency.value, [])
            
            # If no specific resources, create default activity
            if not resources:
                resources = [{
                    'title': f'{competency.value.replace("_", " ").title()} Development Workshop',
                    'description': f'Comprehensive workshop on {competency.value.replace("_", " ")}',
                    'type': 'workshop',
                    'duration': 8
                }]
            
            for resource in resources:
                activity = LearningActivity(
                    id=str(uuid.uuid4()),
                    title=resource['title'],
                    description=resource['description'],
                    activity_type=resource['type'],
                    target_competencies=[competency],
                    estimated_duration=resource['duration'],
                    resources=[resource.get('url', ''), resource.get('materials', '')],
                    completion_criteria=[
                        "Complete all learning materials",
                        "Demonstrate competency improvement",
                        "Apply learning in work context"
                    ]
                )
                activities.append(activity)
        
        return activities
    
    def _plan_coaching_sessions(
        self,
        assessment: CulturalLeadershipAssessment,
        priority_competencies: List[CulturalCompetency],
        preferences: Dict[str, Any]
    ) -> List[CoachingSession]:
        """Plan coaching sessions for development"""
        sessions = []
        
        # Plan monthly coaching sessions
        session_count = preferences.get('coaching_sessions', 6)
        
        for i in range(session_count):
            session_date = datetime.now() + timedelta(days=30 * (i + 1))
            
            session = CoachingSession(
                id=str(uuid.uuid4()),
                leader_id=assessment.leader_id,
                coach_id=preferences.get('coach_id', 'tbd'),
                session_date=session_date,
                duration=90,
                focus_areas=priority_competencies[:2],  # Focus on top 2 priorities
                objectives=[
                    f"Review progress on {comp.value.replace('_', ' ')}" 
                    for comp in priority_competencies[:2]
                ],
                activities=[
                    "Progress review and feedback",
                    "Skill practice and role-playing",
                    "Action planning for next period"
                ],
                insights=[],
                action_items=[],
                progress_notes=""
            )
            sessions.append(session)
        
        return sessions
    
    def _define_development_milestones(
        self,
        priority_competencies: List[CulturalCompetency],
        preferences: Dict[str, Any]
    ) -> List[DevelopmentMilestone]:
        """Define development progress milestones"""
        milestones = []
        
        # 30-day milestone
        milestone_30 = DevelopmentMilestone(
            id=str(uuid.uuid4()),
            title="30-Day Progress Check",
            description="Initial progress assessment and plan adjustment",
            target_date=datetime.now() + timedelta(days=30),
            completion_criteria=[
                "Complete initial learning activities",
                "Demonstrate basic skill improvement",
                "Receive feedback from manager and peers"
            ],
            success_metrics=[
                "10-point improvement in priority competencies",
                "Positive feedback from stakeholders",
                "Successful application of new skills"
            ]
        )
        milestones.append(milestone_30)
        
        # 90-day milestone
        milestone_90 = DevelopmentMilestone(
            id=str(uuid.uuid4()),
            title="90-Day Competency Assessment",
            description="Mid-point competency reassessment",
            target_date=datetime.now() + timedelta(days=90),
            completion_criteria=[
                "Complete 50% of learning activities",
                "Show measurable competency improvement",
                "Lead cultural initiative or project"
            ],
            success_metrics=[
                "20-point improvement in priority competencies",
                "Successful cultural leadership demonstration",
                "Positive impact on team culture"
            ]
        )
        milestones.append(milestone_90)
        
        return milestones
    
    def _define_success_metrics(
        self,
        assessment: CulturalLeadershipAssessment,
        priority_competencies: List[CulturalCompetency]
    ) -> List[str]:
        """Define success metrics for development plan"""
        return [
            f"Achieve {comp.value.replace('_', ' ').title()} competency improvement of 25+ points"
            for comp in priority_competencies
        ] + [
            "Increase overall cultural leadership score by 20+ points",
            "Receive positive feedback from team on cultural leadership",
            "Successfully lead at least one cultural transformation initiative",
            "Demonstrate sustained behavior change over 6 months"
        ]
    
    def measure_leadership_effectiveness(
        self,
        leader_id: str,
        measurement_period: str,
        metrics_data: Dict[str, Any]
    ) -> LeadershipEffectivenessMetrics:
        """Measure cultural leadership effectiveness"""
        try:
            logger.info(f"Measuring leadership effectiveness for leader {leader_id}")
            
            metrics = LeadershipEffectivenessMetrics(
                leader_id=leader_id,
                measurement_period=measurement_period,
                team_engagement_score=metrics_data.get('team_engagement_score', 0),
                cultural_alignment_score=metrics_data.get('cultural_alignment_score', 0),
                change_success_rate=metrics_data.get('change_success_rate', 0),
                vision_clarity_rating=metrics_data.get('vision_clarity_rating', 0),
                communication_effectiveness=metrics_data.get('communication_effectiveness', 0),
                influence_reach=metrics_data.get('influence_reach', 0),
                retention_rate=metrics_data.get('retention_rate', 0),
                promotion_rate=metrics_data.get('promotion_rate', 0),
                peer_leadership_rating=metrics_data.get('peer_leadership_rating', 0),
                direct_report_satisfaction=metrics_data.get('direct_report_satisfaction', 0),
                cultural_initiative_success=metrics_data.get('cultural_initiative_success', 0),
                innovation_fostered=metrics_data.get('innovation_fostered', 0),
                conflict_resolution_success=metrics_data.get('conflict_resolution_success', 0)
            )
            
            logger.info(f"Leadership effectiveness measured for leader {leader_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error measuring leadership effectiveness: {str(e)}")
            raise
    
    def get_assessment_insights(
        self,
        assessment: CulturalLeadershipAssessment
    ) -> Dict[str, Any]:
        """Generate insights from leadership assessment"""
        insights = {
            'strengths': [],
            'development_opportunities': [],
            'leadership_style': '',
            'cultural_impact_potential': '',
            'recommended_focus_areas': [],
            'career_development_suggestions': []
        }
        
        # Analyze strengths
        strong_competencies = [
            score for score in assessment.competency_scores 
            if score.score >= 70
        ]
        insights['strengths'] = [
            f"Strong {comp.competency.value.replace('_', ' ').title()} capabilities"
            for comp in strong_competencies
        ]
        
        # Identify development opportunities
        weak_competencies = [
            score for score in assessment.competency_scores 
            if score.score < 50
        ]
        insights['development_opportunities'] = [
            f"Opportunity to strengthen {comp.competency.value.replace('_', ' ').title()}"
            for comp in weak_competencies
        ]
        
        # Determine leadership style
        if assessment.vision_clarity_score > 80:
            insights['leadership_style'] = 'Visionary Leader'
        elif assessment.communication_effectiveness > 80:
            insights['leadership_style'] = 'Inspirational Communicator'
        elif assessment.change_readiness > 80:
            insights['leadership_style'] = 'Change Champion'
        else:
            insights['leadership_style'] = 'Developing Leader'
        
        # Assess cultural impact potential
        if assessment.cultural_impact_score > 80:
            insights['cultural_impact_potential'] = 'High impact - Ready for major cultural transformation roles'
        elif assessment.cultural_impact_score > 60:
            insights['cultural_impact_potential'] = 'Medium impact - Can lead team-level cultural initiatives'
        else:
            insights['cultural_impact_potential'] = 'Developing impact - Focus on building cultural influence'
        
        return insights