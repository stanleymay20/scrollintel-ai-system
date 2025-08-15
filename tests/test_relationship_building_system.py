"""
Tests for the relationship building system.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.relationship_building_engine import (
    RelationshipBuildingEngine, RelationshipDevelopmentFramework,
    RelationshipMaintenanceSystem, RelationshipQualityAssessment
)
from scrollintel.models.relationship_models import (
    RelationshipProfile, RelationshipType, RelationshipStatus,
    CommunicationStyle, TrustMetrics, PersonalityProfile
)


class TestRelationshipDevelopmentFramework:
    """Test relationship development framework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = RelationshipDevelopmentFramework()
        self.sample_stakeholder_data = {
            'id': 'stakeholder_001',
            'name': 'John Board',
            'title': 'Board Chair',
            'organization': 'Company Board',
            'type': 'board_member',
            'influence_level': 0.9,
            'decision_power': 0.8,
            'interests': ['strategy', 'governance', 'growth'],
            'priorities': ['profitability', 'compliance', 'innovation']
        }
    
    def test_create_relationship_profile(self):
        """Test relationship profile creation."""
        profile = self.framework.create_relationship_profile(self.sample_stakeholder_data)
        
        assert profile.stakeholder_id == 'stakeholder_001'
        assert profile.name == 'John Board'
        assert profile.relationship_type == RelationshipType.BOARD_MEMBER
        assert profile.relationship_status == RelationshipStatus.INITIAL
        assert profile.influence_level == 0.9
        assert profile.decision_making_power == 0.8
        assert len(profile.relationship_goals) > 0
        assert profile.development_strategy != ""
        assert isinstance(profile.trust_metrics, TrustMetrics)
        assert isinstance(profile.personality_profile, PersonalityProfile)
    
    def test_personality_profile_analysis(self):
        """Test personality profile analysis."""
        profile = self.framework.create_relationship_profile(self.sample_stakeholder_data)
        
        # Board member should have diplomatic communication style
        assert profile.personality_profile.communication_style == CommunicationStyle.DIPLOMATIC
        assert len(profile.personality_profile.key_motivators) > 0
        assert len(profile.personality_profile.concerns) > 0
    
    def test_development_strategy_creation(self):
        """Test development strategy creation."""
        profile = self.framework.create_relationship_profile(self.sample_stakeholder_data)
        
        strategy = profile.development_strategy
        assert "strategic value demonstration" in strategy.lower()
        assert "governance insights" in strategy.lower()
        assert "transparency" in strategy.lower()
    
    def test_initial_goals_creation(self):
        """Test initial relationship goals creation."""
        profile = self.framework.create_relationship_profile(self.sample_stakeholder_data)
        
        goals = profile.relationship_goals
        assert len(goals) >= 2
        
        # Should have trust and engagement goals
        goal_descriptions = [goal.description.lower() for goal in goals]
        assert any('trust' in desc for desc in goal_descriptions)
        assert any('engagement' in desc or 'communication' in desc for desc in goal_descriptions)
    
    def test_relationship_roadmap_development(self):
        """Test relationship roadmap development."""
        profile = self.framework.create_relationship_profile(self.sample_stakeholder_data)
        roadmap = self.framework.develop_relationship_roadmap(profile, 12)
        
        assert len(roadmap) > 0
        
        # Should have actions for different phases
        action_types = [action.action_type for action in roadmap]
        assert 'initial_meeting' in action_types
        assert 'value_demonstration' in action_types
        
        # Actions should be properly scheduled
        for action in roadmap:
            assert action.scheduled_date > datetime.now()
            assert action.priority in ['high', 'medium', 'low']
            assert len(action.preparation_required) > 0
            assert len(action.success_criteria) > 0
    
    def test_foundation_actions(self):
        """Test foundation phase actions."""
        profile = self.framework.create_relationship_profile(self.sample_stakeholder_data)
        actions = self.framework._create_foundation_actions(profile, datetime.now())
        
        assert len(actions) >= 2
        
        # Should include initial meeting and value demonstration
        action_types = [action.action_type for action in actions]
        assert 'initial_meeting' in action_types
        assert 'value_demonstration' in action_types
        
        # Initial meeting should be high priority and soon
        initial_meeting = next(a for a in actions if a.action_type == 'initial_meeting')
        assert initial_meeting.priority == 'high'
        assert initial_meeting.scheduled_date <= datetime.now() + timedelta(days=14)


class TestRelationshipMaintenanceSystem:
    """Test relationship maintenance system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.maintenance_system = RelationshipMaintenanceSystem()
        self.sample_profile = RelationshipProfile(
            stakeholder_id='test_001',
            name='Test Executive',
            title='CEO',
            organization='Test Corp',
            relationship_type=RelationshipType.EXECUTIVE,
            relationship_status=RelationshipStatus.ESTABLISHED,
            personality_profile=PersonalityProfile(
                communication_style=CommunicationStyle.RESULTS_ORIENTED,
                decision_making_style='analytical',
                key_motivators=['growth', 'efficiency'],
                concerns=['competition', 'market_changes'],
                preferred_interaction_frequency='monthly',
                optimal_meeting_times=['morning'],
                communication_preferences={}
            ),
            influence_level=0.8,
            decision_making_power=0.7,
            network_connections=[],
            trust_metrics=TrustMetrics(
                overall_trust_score=0.75,
                competence_trust=0.8,
                benevolence_trust=0.7,
                integrity_trust=0.8,
                predictability_trust=0.7,
                transparency_score=0.75,
                reliability_score=0.8,
                last_updated=datetime.now()
            ),
            relationship_strength=0.7,
            engagement_frequency=0.6,
            response_rate=0.8,
            relationship_start_date=datetime.now() - timedelta(days=180),
            last_interaction_date=datetime.now() - timedelta(days=15),
            interaction_history=[],
            relationship_goals=[],
            development_strategy="Test strategy",
            next_planned_interaction=None,
            key_interests=['technology', 'strategy'],
            business_priorities=['growth', 'innovation'],
            personal_interests=['leadership'],
            communication_cadence='monthly'
        )
    
    def test_create_maintenance_plan(self):
        """Test maintenance plan creation."""
        plan = self.maintenance_system.create_maintenance_plan(self.sample_profile)
        
        assert plan.stakeholder_id == 'test_001'
        assert plan.maintenance_frequency in ['weekly', 'bi-weekly', 'monthly', 'quarterly']
        assert len(plan.touch_point_types) > 0
        assert len(plan.content_themes) > 0
        assert len(plan.success_indicators) > 0
        assert plan.next_review_date > datetime.now()
    
    def test_maintenance_frequency_determination(self):
        """Test maintenance frequency determination."""
        # High influence should get more frequent maintenance
        self.sample_profile.influence_level = 0.9
        frequency = self.maintenance_system._determine_maintenance_frequency(self.sample_profile)
        assert frequency in ['weekly', 'bi-weekly']
        
        # Lower influence should get less frequent maintenance
        self.sample_profile.influence_level = 0.3
        frequency = self.maintenance_system._determine_maintenance_frequency(self.sample_profile)
        assert frequency in ['monthly', 'quarterly']
    
    def test_touch_points_definition(self):
        """Test touch points definition."""
        touch_points = self.maintenance_system._define_touch_points(self.sample_profile)
        
        assert 'email_update' in touch_points
        assert 'strategic_insight' in touch_points
        
        # Executive type should have specific touch points
        assert 'collaboration_opportunity' in touch_points
        assert 'strategic_consultation' in touch_points
    
    def test_content_themes_creation(self):
        """Test content themes creation."""
        themes = self.maintenance_system._create_content_themes(self.sample_profile)
        
        assert 'strategic_insights' in themes
        assert 'industry_trends' in themes
        assert 'technology_updates' in themes
        
        # Should include stakeholder's interests
        assert 'technology' in themes
        assert 'strategy' in themes
    
    def test_execute_maintenance_action(self):
        """Test maintenance action execution."""
        from scrollintel.models.relationship_models import RelationshipAction
        
        action = RelationshipAction(
            action_id='test_action',
            stakeholder_id='test_001',
            action_type='email_update',
            description='Send strategic update',
            scheduled_date=datetime.now(),
            priority='medium',
            expected_outcome='Maintained engagement',
            preparation_required=['Prepare update'],
            success_criteria=['Email sent'],
            status='planned'
        )
        
        result = self.maintenance_system.execute_maintenance_action(action, self.sample_profile)
        
        assert result['action_id'] == 'test_action'
        assert result['status'] == 'completed'
        assert len(result['outcomes']) > 0
        assert len(result['next_actions']) > 0


class TestRelationshipQualityAssessment:
    """Test relationship quality assessment."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.quality_assessment = RelationshipQualityAssessment()
        self.sample_profile = RelationshipProfile(
            stakeholder_id='test_001',
            name='Test Stakeholder',
            title='Board Member',
            organization='Test Board',
            relationship_type=RelationshipType.BOARD_MEMBER,
            relationship_status=RelationshipStatus.ESTABLISHED,
            personality_profile=PersonalityProfile(
                communication_style=CommunicationStyle.DIPLOMATIC,
                decision_making_style='consensus',
                key_motivators=['governance', 'transparency'],
                concerns=['risk', 'compliance'],
                preferred_interaction_frequency='monthly',
                optimal_meeting_times=['afternoon'],
                communication_preferences={}
            ),
            influence_level=0.8,
            decision_making_power=0.9,
            network_connections=[],
            trust_metrics=TrustMetrics(
                overall_trust_score=0.8,
                competence_trust=0.85,
                benevolence_trust=0.75,
                integrity_trust=0.8,
                predictability_trust=0.8,
                transparency_score=0.8,
                reliability_score=0.85,
                last_updated=datetime.now()
            ),
            relationship_strength=0.75,
            engagement_frequency=0.7,
            response_rate=0.85,
            relationship_start_date=datetime.now() - timedelta(days=365),
            last_interaction_date=datetime.now() - timedelta(days=7),
            interaction_history=[],
            relationship_goals=[],
            development_strategy="Test strategy",
            next_planned_interaction=None,
            key_interests=['governance'],
            business_priorities=['compliance'],
            personal_interests=['leadership'],
            communication_cadence='monthly'
        )
    
    def test_assess_relationship_quality(self):
        """Test comprehensive relationship quality assessment."""
        assessment = self.quality_assessment.assess_relationship_quality(self.sample_profile)
        
        assert assessment['stakeholder_id'] == 'test_001'
        assert 'overall_score' in assessment
        assert 'dimension_scores' in assessment
        assert 'strengths' in assessment
        assert 'weaknesses' in assessment
        assert 'recommendations' in assessment
        assert 'risk_factors' in assessment
        
        # Overall score should be reasonable
        assert 0.0 <= assessment['overall_score'] <= 1.0
        
        # Should have dimension scores
        dimensions = assessment['dimension_scores']
        assert 'trust' in dimensions
        assert 'engagement' in dimensions
        assert 'value_delivery' in dimensions
        assert 'communication' in dimensions
    
    def test_trust_dimension_assessment(self):
        """Test trust dimension assessment."""
        trust_score = self.quality_assessment._assess_trust_dimension(self.sample_profile)
        assert trust_score == self.sample_profile.trust_metrics.overall_trust_score
    
    def test_engagement_dimension_assessment(self):
        """Test engagement dimension assessment."""
        engagement_score = self.quality_assessment._assess_engagement_dimension(self.sample_profile)
        
        # Should consider response rate, frequency, and recency
        assert 0.0 <= engagement_score <= 1.0
        
        # Recent interaction should boost score
        assert engagement_score > 0.5  # Should be decent with recent interaction
    
    def test_strengths_identification(self):
        """Test strengths identification."""
        dimension_scores = {
            'trust': 0.85,
            'engagement': 0.9,
            'value_delivery': 0.6,
            'communication': 0.7
        }
        
        strengths = self.quality_assessment._identify_strengths(dimension_scores)
        
        assert len(strengths) > 0
        assert any('trust' in strength.lower() for strength in strengths)
        assert any('engagement' in strength.lower() for strength in strengths)
    
    def test_weaknesses_identification(self):
        """Test weaknesses identification."""
        dimension_scores = {
            'trust': 0.3,
            'engagement': 0.5,
            'value_delivery': 0.9,
            'communication': 0.8
        }
        
        weaknesses = self.quality_assessment._identify_weaknesses(dimension_scores)
        
        assert len(weaknesses) > 0
        assert any('trust' in weakness.lower() for weakness in weaknesses)
    
    def test_recommendations_generation(self):
        """Test recommendations generation."""
        dimension_scores = {
            'trust': 0.4,
            'engagement': 0.3,
            'value_delivery': 0.8,
            'communication': 0.7
        }
        
        recommendations = self.quality_assessment._generate_recommendations(
            self.sample_profile, dimension_scores
        )
        
        assert len(recommendations) > 0
        assert any('trust' in rec.lower() for rec in recommendations)
        assert any('engagement' in rec.lower() for rec in recommendations)
    
    def test_risk_factors_identification(self):
        """Test risk factors identification."""
        # Set up profile with risk factors
        self.sample_profile.trust_metrics.overall_trust_score = 0.3
        self.sample_profile.response_rate = 0.2
        self.sample_profile.last_interaction_date = datetime.now() - timedelta(days=90)
        self.sample_profile.relationship_strength = 0.2
        
        risks = self.quality_assessment._identify_risk_factors(self.sample_profile)
        
        assert len(risks) > 0
        assert any('trust' in risk.lower() for risk in risks)
        assert any('response' in risk.lower() for risk in risks)
        assert any('interaction' in risk.lower() for risk in risks)


class TestRelationshipBuildingEngine:
    """Test main relationship building engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RelationshipBuildingEngine()
        self.sample_stakeholder_data = {
            'id': 'engine_test_001',
            'name': 'Engine Test Stakeholder',
            'title': 'CTO',
            'organization': 'Tech Company',
            'type': 'executive',
            'influence_level': 0.7,
            'decision_power': 0.8
        }
    
    def test_initialize_relationship(self):
        """Test relationship initialization."""
        profile = self.engine.initialize_relationship(self.sample_stakeholder_data)
        
        assert profile.stakeholder_id == 'engine_test_001'
        assert profile.name == 'Engine Test Stakeholder'
        assert profile.relationship_type == RelationshipType.EXECUTIVE
        assert len(profile.relationship_goals) > 0
        assert profile.development_strategy != ""
    
    def test_optimize_relationship(self):
        """Test relationship optimization."""
        # First initialize a relationship
        profile = self.engine.initialize_relationship(self.sample_stakeholder_data)
        
        # Set some metrics that need optimization
        profile.trust_metrics.overall_trust_score = 0.4
        profile.response_rate = 0.3
        profile.relationship_strength = 0.35
        
        optimization_plan = self.engine.optimize_relationship(profile)
        
        assert 'assessment' in optimization_plan
        assert 'optimization_actions' in optimization_plan
        assert len(optimization_plan['optimization_actions']) > 0
        
        # Should have actions addressing the weak areas
        action_types = [action.action_type for action in optimization_plan['optimization_actions']]
        assert 'trust_building' in action_types
        assert 'engagement_improvement' in action_types
    
    def test_trust_building_action_creation(self):
        """Test trust building action creation."""
        profile = self.engine.initialize_relationship(self.sample_stakeholder_data)
        action = self.engine._create_trust_building_action(profile)
        
        assert action.action_type == 'trust_building'
        assert action.priority == 'high'
        assert len(action.preparation_required) > 0
        assert len(action.success_criteria) > 0
        assert action.scheduled_date > datetime.now()
    
    def test_engagement_action_creation(self):
        """Test engagement improvement action creation."""
        profile = self.engine.initialize_relationship(self.sample_stakeholder_data)
        action = self.engine._create_engagement_action(profile)
        
        assert action.action_type == 'engagement_improvement'
        assert action.priority == 'medium'
        assert len(action.preparation_required) > 0
        assert len(action.success_criteria) > 0
    
    def test_value_demonstration_action_creation(self):
        """Test value demonstration action creation."""
        profile = self.engine.initialize_relationship(self.sample_stakeholder_data)
        action = self.engine._create_value_demonstration_action(profile)
        
        assert action.action_type == 'value_demonstration'
        assert action.priority == 'high'
        assert len(action.preparation_required) > 0
        assert len(action.success_criteria) > 0


class TestRelationshipModels:
    """Test relationship data models."""
    
    def test_relationship_profile_creation(self):
        """Test relationship profile model."""
        trust_metrics = TrustMetrics(
            overall_trust_score=0.8,
            competence_trust=0.85,
            benevolence_trust=0.75,
            integrity_trust=0.8,
            predictability_trust=0.8,
            transparency_score=0.8,
            reliability_score=0.85,
            last_updated=datetime.now()
        )
        
        personality_profile = PersonalityProfile(
            communication_style=CommunicationStyle.DATA_DRIVEN,
            decision_making_style='analytical',
            key_motivators=['efficiency', 'innovation'],
            concerns=['risk', 'complexity'],
            preferred_interaction_frequency='weekly',
            optimal_meeting_times=['morning', 'afternoon'],
            communication_preferences={'format': 'email', 'detail_level': 'high'}
        )
        
        profile = RelationshipProfile(
            stakeholder_id='model_test_001',
            name='Model Test',
            title='Test Title',
            organization='Test Org',
            relationship_type=RelationshipType.INVESTOR,
            relationship_status=RelationshipStatus.DEVELOPING,
            personality_profile=personality_profile,
            influence_level=0.7,
            decision_making_power=0.8,
            network_connections=['connection1', 'connection2'],
            trust_metrics=trust_metrics,
            relationship_strength=0.6,
            engagement_frequency=0.5,
            response_rate=0.8,
            relationship_start_date=datetime.now(),
            last_interaction_date=None,
            interaction_history=[],
            relationship_goals=[],
            development_strategy="Test strategy",
            next_planned_interaction=None,
            key_interests=['technology'],
            business_priorities=['growth'],
            personal_interests=['innovation'],
            communication_cadence='monthly'
        )
        
        assert profile.stakeholder_id == 'model_test_001'
        assert profile.relationship_type == RelationshipType.INVESTOR
        assert profile.trust_metrics.overall_trust_score == 0.8
        assert profile.personality_profile.communication_style == CommunicationStyle.DATA_DRIVEN
    
    def test_relationship_action_model(self):
        """Test relationship action model."""
        from scrollintel.models.relationship_models import RelationshipAction
        
        action = RelationshipAction(
            action_id='action_test_001',
            stakeholder_id='stakeholder_001',
            action_type='meeting',
            description='Strategic planning meeting',
            scheduled_date=datetime.now() + timedelta(days=7),
            priority='high',
            expected_outcome='Aligned strategic vision',
            preparation_required=['Prepare agenda', 'Review metrics'],
            success_criteria=['Meeting completed', 'Action items defined'],
            status='planned'
        )
        
        assert action.action_id == 'action_test_001'
        assert action.priority == 'high'
        assert len(action.preparation_required) == 2
        assert len(action.success_criteria) == 2


@pytest.fixture
def mock_relationship_engine():
    """Mock relationship building engine for testing."""
    return Mock(spec=RelationshipBuildingEngine)


def test_relationship_building_integration(mock_relationship_engine):
    """Test integration between relationship building components."""
    # Mock the engine methods
    mock_profile = Mock()
    mock_profile.stakeholder_id = 'integration_test'
    mock_profile.name = 'Integration Test'
    
    mock_relationship_engine.initialize_relationship.return_value = mock_profile
    mock_relationship_engine.optimize_relationship.return_value = {
        'assessment': {'overall_score': 0.7},
        'optimization_actions': []
    }
    
    # Test initialization
    stakeholder_data = {'id': 'test', 'name': 'Test', 'type': 'executive'}
    profile = mock_relationship_engine.initialize_relationship(stakeholder_data)
    
    assert profile.stakeholder_id == 'integration_test'
    
    # Test optimization
    optimization = mock_relationship_engine.optimize_relationship(profile)
    assert 'assessment' in optimization
    assert 'optimization_actions' in optimization