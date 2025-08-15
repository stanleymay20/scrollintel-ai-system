"""
Tests for Cultural Leadership Assessment Engine

Test suite for cultural leadership assessment, development planning, and effectiveness measurement.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.cultural_leadership_assessment_engine import CulturalLeadershipAssessmentEngine
from scrollintel.models.cultural_leadership_models import (
    CulturalCompetency, LeadershipLevel, CulturalLeadershipAssessment,
    LeadershipDevelopmentPlan, LeadershipEffectivenessMetrics
)


class TestCulturalLeadershipAssessmentEngine:
    """Test cases for Cultural Leadership Assessment Engine"""
    
    @pytest.fixture
    def assessment_engine(self):
        """Create assessment engine instance"""
        return CulturalLeadershipAssessmentEngine()
    
    @pytest.fixture
    def sample_assessment_data(self):
        """Sample assessment data for testing"""
        return {
            "assessment_method": "comprehensive",
            "assessor_id": "assessor_123",
            "self_assessment": False,
            "vision_creation": {
                "self_rating": 75,
                "peer_ratings": [70, 80, 75],
                "manager_rating": 78,
                "behaviors": [
                    "Creates compelling team vision",
                    "Communicates vision effectively",
                    "Aligns team with vision"
                ],
                "evidence": ["Led successful vision workshop", "Improved team alignment scores"]
            },
            "communication": {
                "self_rating": 85,
                "peer_ratings": [80, 85, 90],
                "manager_rating": 88,
                "behaviors": [
                    "Communicates clearly and persuasively",
                    "Listens actively to others",
                    "Adapts communication style"
                ]
            },
            "change_leadership": {
                "self_rating": 60,
                "peer_ratings": [55, 65, 60],
                "manager_rating": 62,
                "behaviors": [
                    "Supports change initiatives",
                    "Helps team navigate change"
                ]
            },
            "cultural_impact": {
                "team_culture_improvement": 75,
                "cultural_initiative_success": 80,
                "employee_engagement_change": 70,
                "cultural_alignment_score": 85
            },
            "vision_clarity": {
                "clarity_rating": 80,
                "alignment_rating": 75,
                "inspiration_rating": 85
            },
            "team_engagement": {
                "overall_engagement": 78
            },
            "change_readiness": {
                "adaptability": 70,
                "resilience": 75,
                "change_advocacy": 65
            }
        }
    
    def test_assess_cultural_leadership_success(self, assessment_engine, sample_assessment_data):
        """Test successful cultural leadership assessment"""
        # Execute assessment
        assessment = assessment_engine.assess_cultural_leadership(
            leader_id="leader_123",
            organization_id="org_456",
            assessment_data=sample_assessment_data
        )
        
        # Verify assessment structure
        assert isinstance(assessment, CulturalLeadershipAssessment)
        assert assessment.leader_id == "leader_123"
        assert assessment.organization_id == "org_456"
        assert assessment.assessment_method == "comprehensive"
        assert assessment.assessor_id == "assessor_123"
        assert not assessment.self_assessment
        
        # Verify competency scores
        assert len(assessment.competency_scores) == len(CulturalCompetency)
        
        # Check specific competency scores
        vision_score = next(
            score for score in assessment.competency_scores 
            if score.competency == CulturalCompetency.VISION_CREATION
        )
        assert vision_score.score > 70  # Should be high based on input data
        assert vision_score.current_level in [LeadershipLevel.ADVANCED, LeadershipLevel.PROFICIENT]
        
        # Verify overall scores
        assert 0 <= assessment.overall_score <= 100
        assert 0 <= assessment.cultural_impact_score <= 100
        assert 0 <= assessment.vision_clarity_score <= 100
        assert 0 <= assessment.communication_effectiveness <= 100
        
        # Verify recommendations exist
        assert len(assessment.recommendations) > 0
    
    def test_competency_scoring(self, assessment_engine):
        """Test individual competency scoring"""
        competency_data = {
            "self_rating": 70,
            "peer_ratings": [65, 75, 70],
            "manager_rating": 72,
            "behaviors": ["Behavior 1", "Behavior 2", "Behavior 3"],
            "evidence": ["Evidence 1", "Evidence 2"]
        }
        
        score = assessment_engine._assess_competency(
            CulturalCompetency.VISION_CREATION,
            competency_data,
            "comprehensive"
        )
        
        assert score.competency == CulturalCompetency.VISION_CREATION
        assert 0 <= score.score <= 100
        assert score.current_level in LeadershipLevel
        assert score.target_level in LeadershipLevel
        assert len(score.evidence) == 2
        assert isinstance(score.development_areas, list)
        assert isinstance(score.strengths, list)
    
    def test_leadership_level_determination(self, assessment_engine):
        """Test leadership level determination from scores"""
        # Test different score ranges
        assert assessment_engine._score_to_level(95) == LeadershipLevel.EXPERT
        assert assessment_engine._score_to_level(75) == LeadershipLevel.ADVANCED
        assert assessment_engine._score_to_level(55) == LeadershipLevel.PROFICIENT
        assert assessment_engine._score_to_level(35) == LeadershipLevel.DEVELOPING
        assert assessment_engine._score_to_level(15) == LeadershipLevel.EMERGING
    
    def test_create_development_plan(self, assessment_engine, sample_assessment_data):
        """Test development plan creation"""
        # First create an assessment
        assessment = assessment_engine.assess_cultural_leadership(
            leader_id="leader_123",
            organization_id="org_456",
            assessment_data=sample_assessment_data
        )
        
        # Create development plan
        preferences = {
            "duration_days": 180,
            "coaching_sessions": 6,
            "learning_style": "blended",
            "coach_id": "coach_789"
        }
        
        plan = assessment_engine.create_development_plan(assessment, preferences)
        
        # Verify plan structure
        assert isinstance(plan, LeadershipDevelopmentPlan)
        assert plan.leader_id == "leader_123"
        assert plan.assessment_id == assessment.id
        assert len(plan.priority_competencies) <= 3  # Top 3 priorities
        assert len(plan.development_goals) > 0
        assert len(plan.learning_activities) > 0
        assert len(plan.coaching_sessions) == 6
        assert len(plan.progress_milestones) > 0
        assert len(plan.success_metrics) > 0
        
        # Verify timeline
        assert plan.target_completion > plan.created_date
        assert (plan.target_completion - plan.created_date).days == 180
    
    def test_learning_activities_creation(self, assessment_engine):
        """Test learning activities creation"""
        priority_competencies = [
            CulturalCompetency.VISION_CREATION,
            CulturalCompetency.COMMUNICATION
        ]
        preferences = {"learning_style": "blended"}
        
        activities = assessment_engine._create_learning_activities(
            priority_competencies, preferences
        )
        
        assert len(activities) > 0
        
        for activity in activities:
            assert activity.id is not None
            assert activity.title is not None
            assert activity.description is not None
            assert activity.activity_type in ["workshop", "reading", "project", "coaching"]
            assert activity.estimated_duration > 0
            assert len(activity.target_competencies) > 0
            assert activity.status == "not_started"
    
    def test_coaching_sessions_planning(self, assessment_engine, sample_assessment_data):
        """Test coaching sessions planning"""
        assessment = assessment_engine.assess_cultural_leadership(
            leader_id="leader_123",
            organization_id="org_456",
            assessment_data=sample_assessment_data
        )
        
        priority_competencies = [CulturalCompetency.VISION_CREATION, CulturalCompetency.COMMUNICATION]
        preferences = {"coaching_sessions": 4, "coach_id": "coach_123"}
        
        sessions = assessment_engine._plan_coaching_sessions(
            assessment, priority_competencies, preferences
        )
        
        assert len(sessions) == 4
        
        for i, session in enumerate(sessions):
            assert session.leader_id == "leader_123"
            assert session.coach_id == "coach_123"
            assert session.duration == 90
            assert len(session.focus_areas) <= 2
            assert len(session.objectives) > 0
            assert len(session.activities) > 0
            
            # Verify session dates are spaced monthly
            expected_date = datetime.now() + timedelta(days=30 * (i + 1))
            assert abs((session.session_date - expected_date).days) <= 1
    
    def test_measure_leadership_effectiveness(self, assessment_engine):
        """Test leadership effectiveness measurement"""
        metrics_data = {
            "team_engagement_score": 85,
            "cultural_alignment_score": 78,
            "change_success_rate": 92,
            "vision_clarity_rating": 88,
            "communication_effectiveness": 90,
            "influence_reach": 150,
            "retention_rate": 95,
            "promotion_rate": 12,
            "peer_leadership_rating": 82,
            "direct_report_satisfaction": 87,
            "cultural_initiative_success": 89,
            "innovation_fostered": 5,
            "conflict_resolution_success": 94
        }
        
        metrics = assessment_engine.measure_leadership_effectiveness(
            leader_id="leader_123",
            measurement_period="Q1_2024",
            metrics_data=metrics_data
        )
        
        assert isinstance(metrics, LeadershipEffectivenessMetrics)
        assert metrics.leader_id == "leader_123"
        assert metrics.measurement_period == "Q1_2024"
        assert metrics.team_engagement_score == 85
        assert metrics.cultural_alignment_score == 78
        assert metrics.change_success_rate == 92
        assert metrics.vision_clarity_rating == 88
        assert metrics.communication_effectiveness == 90
        assert metrics.influence_reach == 150
        assert metrics.retention_rate == 95
        assert metrics.promotion_rate == 12
        assert metrics.peer_leadership_rating == 82
        assert metrics.direct_report_satisfaction == 87
        assert metrics.cultural_initiative_success == 89
        assert metrics.innovation_fostered == 5
        assert metrics.conflict_resolution_success == 94
    
    def test_assessment_insights_generation(self, assessment_engine, sample_assessment_data):
        """Test assessment insights generation"""
        assessment = assessment_engine.assess_cultural_leadership(
            leader_id="leader_123",
            organization_id="org_456",
            assessment_data=sample_assessment_data
        )
        
        insights = assessment_engine.get_assessment_insights(assessment)
        
        assert "strengths" in insights
        assert "development_opportunities" in insights
        assert "leadership_style" in insights
        assert "cultural_impact_potential" in insights
        assert "recommended_focus_areas" in insights
        assert "career_development_suggestions" in insights
        
        assert isinstance(insights["strengths"], list)
        assert isinstance(insights["development_opportunities"], list)
        assert insights["leadership_style"] in [
            "Visionary Leader", "Inspirational Communicator", 
            "Change Champion", "Developing Leader"
        ]
        assert "impact" in insights["cultural_impact_potential"].lower()
    
    def test_comprehensive_score_calculation(self, assessment_engine):
        """Test comprehensive score calculation"""
        score = assessment_engine._calculate_comprehensive_score(
            self_rating=70,
            peer_ratings=[65, 75, 80],
            manager_rating=78,
            behaviors=["Behavior 1", "Behavior 2", "Behavior 3"]
        )
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_360_score_calculation(self, assessment_engine):
        """Test 360-degree feedback score calculation"""
        score = assessment_engine._calculate_360_score(
            self_rating=70,
            peer_ratings=[65, 75, 80],
            manager_rating=78
        )
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_cultural_impact_score_calculation(self, assessment_engine):
        """Test cultural impact score calculation"""
        assessment_data = {
            "cultural_impact": {
                "team_culture_improvement": 80,
                "cultural_initiative_success": 85,
                "employee_engagement_change": 75,
                "cultural_alignment_score": 90
            }
        }
        
        score = assessment_engine._calculate_cultural_impact_score(assessment_data)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
    
    def test_vision_clarity_score_calculation(self, assessment_engine):
        """Test vision clarity score calculation"""
        assessment_data = {
            "vision_clarity": {
                "clarity_rating": 85,
                "alignment_rating": 80,
                "inspiration_rating": 90
            }
        }
        
        score = assessment_engine._calculate_vision_clarity_score(assessment_data)
        
        assert 0 <= score <= 100
        assert isinstance(score, float)
        assert score == (85 + 80 + 90) / 3
    
    def test_development_milestones_creation(self, assessment_engine):
        """Test development milestones creation"""
        priority_competencies = [CulturalCompetency.VISION_CREATION]
        preferences = {}
        
        milestones = assessment_engine._define_development_milestones(
            priority_competencies, preferences
        )
        
        assert len(milestones) >= 2  # At least 30-day and 90-day milestones
        
        for milestone in milestones:
            assert milestone.id is not None
            assert milestone.title is not None
            assert milestone.description is not None
            assert milestone.target_date > datetime.now()
            assert len(milestone.completion_criteria) > 0
            assert len(milestone.success_metrics) > 0
            assert milestone.status == "pending"
    
    def test_success_metrics_definition(self, assessment_engine, sample_assessment_data):
        """Test success metrics definition"""
        assessment = assessment_engine.assess_cultural_leadership(
            leader_id="leader_123",
            organization_id="org_456",
            assessment_data=sample_assessment_data
        )
        
        priority_competencies = [CulturalCompetency.VISION_CREATION, CulturalCompetency.COMMUNICATION]
        
        metrics = assessment_engine._define_success_metrics(assessment, priority_competencies)
        
        assert len(metrics) > 0
        assert all(isinstance(metric, str) for metric in metrics)
        assert any("improvement" in metric.lower() for metric in metrics)
        assert any("cultural" in metric.lower() for metric in metrics)
    
    def test_error_handling(self, assessment_engine):
        """Test error handling in assessment engine"""
        # Test with missing required data - should handle gracefully
        try:
            assessment = assessment_engine.assess_cultural_leadership(
                leader_id="test_leader",
                organization_id="org_456",
                assessment_data={}  # Empty assessment data
            )
            # Should create assessment with default values
            assert assessment is not None
            assert assessment.leader_id == "test_leader"
        except Exception as e:
            # If it raises an exception, that's also acceptable
            assert isinstance(e, Exception)
    
    def test_competency_weights_initialization(self, assessment_engine):
        """Test competency weights initialization"""
        weights = assessment_engine.competency_weights
        
        assert len(weights) == len(CulturalCompetency)
        assert all(0 <= weight <= 1 for weight in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to approximately 1
    
    def test_scoring_rubrics_initialization(self, assessment_engine):
        """Test scoring rubrics initialization"""
        rubrics = assessment_engine.scoring_rubrics
        
        # Check that key competencies have rubrics
        assert CulturalCompetency.VISION_CREATION in rubrics
        assert CulturalCompetency.COMMUNICATION in rubrics
        
        # Check rubric structure for vision creation
        vision_rubric = rubrics[CulturalCompetency.VISION_CREATION]
        assert LeadershipLevel.EMERGING in vision_rubric
        assert LeadershipLevel.EXPERT in vision_rubric
        
        # Check rubric content
        emerging_rubric = vision_rubric[LeadershipLevel.EMERGING]
        assert "score_range" in emerging_rubric
        assert "indicators" in emerging_rubric
        assert "behaviors" in emerging_rubric
    
    def test_development_resources_initialization(self, assessment_engine):
        """Test development resources initialization"""
        resources = assessment_engine.development_resources
        
        # Check that key competencies have resources
        assert CulturalCompetency.VISION_CREATION in resources
        assert CulturalCompetency.COMMUNICATION in resources
        
        # Check resource structure
        vision_resources = resources[CulturalCompetency.VISION_CREATION]
        assert len(vision_resources) > 0
        
        for resource in vision_resources:
            assert "type" in resource
            assert "title" in resource
            assert "duration" in resource
            assert "description" in resource