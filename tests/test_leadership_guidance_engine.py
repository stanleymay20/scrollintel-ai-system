"""
Tests for Crisis Leadership Guidance Engine
"""
import pytest
from datetime import datetime
from scrollintel.engines.leadership_guidance_engine import LeadershipGuidanceEngine
from scrollintel.models.leadership_guidance_models import (
    CrisisType, DecisionUrgency, DecisionContext, LeadershipStyle
)

class TestLeadershipGuidanceEngine:
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = LeadershipGuidanceEngine()
        self.sample_context = DecisionContext(
            crisis_id="crisis_001",
            crisis_type=CrisisType.TECHNICAL_OUTAGE,
            severity_level=7,
            stakeholders_affected=["customers", "technical_team", "executives"],
            time_pressure=DecisionUrgency.IMMEDIATE,
            available_information={"system_status": "down", "affected_users": 1000},
            resource_constraints=["limited_staff", "weekend_hours"],
            regulatory_considerations=["data_protection", "service_availability"]
        )
    
    def test_get_leadership_guidance_success(self):
        """Test successful leadership guidance generation"""
        recommendation = self.engine.get_leadership_guidance(self.sample_context)
        
        assert recommendation is not None
        assert recommendation.context == self.sample_context
        assert recommendation.recommended_style in LeadershipStyle
        assert len(recommendation.key_actions) > 0
        assert recommendation.communication_strategy is not None
        assert len(recommendation.stakeholder_priorities) > 0
        assert len(recommendation.risk_mitigation_steps) > 0
        assert len(recommendation.success_metrics) > 0
        assert 0.0 <= recommendation.confidence_score <= 1.0
        assert recommendation.rationale is not None
    
    def test_leadership_style_determination(self):
        """Test leadership style determination logic"""
        # Test immediate technical crisis -> Directive
        immediate_tech_context = DecisionContext(
            crisis_id="crisis_002",
            crisis_type=CrisisType.TECHNICAL_OUTAGE,
            severity_level=8,
            stakeholders_affected=["customers"],
            time_pressure=DecisionUrgency.IMMEDIATE,
            available_information={},
            resource_constraints=[],
            regulatory_considerations=[]
        )
        
        style = self.engine._determine_leadership_style(immediate_tech_context)
        assert style == LeadershipStyle.DIRECTIVE
        
        # Test many stakeholders -> Collaborative
        collaborative_context = DecisionContext(
            crisis_id="crisis_003",
            crisis_type=CrisisType.REGULATORY_ISSUE,
            severity_level=5,
            stakeholders_affected=["customers", "regulators", "legal", "executives", "employees", "partners"],
            time_pressure=DecisionUrgency.MODERATE,
            available_information={},
            resource_constraints=[],
            regulatory_considerations=[]
        )
        
        style = self.engine._determine_leadership_style(collaborative_context)
        assert style == LeadershipStyle.COLLABORATIVE
    
    def test_assess_leadership_effectiveness(self):
        """Test leadership effectiveness assessment"""
        performance_data = {
            "decision_quality": 0.8,
            "communication_effectiveness": 0.75,
            "stakeholder_confidence": 0.7,
            "team_morale": 0.85,
            "resolution_speed": 0.9
        }
        
        assessment = self.engine.assess_leadership_effectiveness(
            "leader_001", "crisis_001", performance_data
        )
        
        assert assessment is not None
        assert assessment.leader_id == "leader_001"
        assert assessment.crisis_id == "crisis_001"
        assert isinstance(assessment.assessment_time, datetime)
        assert 0.0 <= assessment.overall_effectiveness <= 1.0
        assert len(assessment.strengths) > 0
        assert len(assessment.improvement_areas) >= 0
        assert len(assessment.coaching_recommendations) >= 0
    
    def test_provide_coaching_guidance(self):
        """Test coaching guidance provision"""
        # Create assessment with improvement areas
        performance_data = {
            "decision_quality": 0.6,
            "communication_effectiveness": 0.65,
            "stakeholder_confidence": 0.55,
            "team_morale": 0.75,
            "resolution_speed": 0.6
        }
        
        assessment = self.engine.assess_leadership_effectiveness(
            "leader_001", "crisis_001", performance_data
        )
        
        coaching_guidance = self.engine.provide_coaching_guidance(assessment)
        
        assert isinstance(coaching_guidance, list)
        assert len(coaching_guidance) > 0
        
        for guidance in coaching_guidance:
            assert guidance.focus_area is not None
            assert 0.0 <= guidance.current_performance <= 1.0
            assert 0.0 <= guidance.target_performance <= 1.0
            assert len(guidance.improvement_strategies) > 0
            assert len(guidance.practice_exercises) > 0
            assert len(guidance.success_indicators) > 0
            assert guidance.timeline is not None
            assert len(guidance.resources) > 0
    
    def test_get_relevant_practices(self):
        """Test getting relevant best practices"""
        practices = self.engine._get_relevant_practices(CrisisType.TECHNICAL_OUTAGE)
        
        assert isinstance(practices, list)
        for practice in practices:
            assert practice.crisis_type == CrisisType.TECHNICAL_OUTAGE
            assert practice.practice_name is not None
            assert len(practice.implementation_steps) > 0
            assert len(practice.success_indicators) > 0
    
    def test_generate_key_actions(self):
        """Test key actions generation"""
        practices = self.engine._get_relevant_practices(CrisisType.TECHNICAL_OUTAGE)
        actions = self.engine._generate_key_actions(self.sample_context, practices)
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert len(actions) <= 8  # Should limit to 8 actions
        
        # Should include immediate response actions for urgent situations
        action_text = " ".join(actions).lower()
        assert any(keyword in action_text for keyword in ["activate", "establish", "initiate"])
    
    def test_develop_communication_strategy(self):
        """Test communication strategy development"""
        strategy = self.engine._develop_communication_strategy(self.sample_context)
        
        assert isinstance(strategy, str)
        assert len(strategy) > 0
        
        # Should be relevant to technical outage
        strategy_lower = strategy.lower()
        assert any(keyword in strategy_lower for keyword in ["technical", "update", "timeline"])
    
    def test_prioritize_stakeholders(self):
        """Test stakeholder prioritization"""
        priorities = self.engine._prioritize_stakeholders(self.sample_context)
        
        assert isinstance(priorities, list)
        assert len(priorities) > 0
        
        # For technical outage, customers should be high priority
        assert "customers" in priorities
        assert "technical_team" in priorities or "executives" in priorities
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation"""
        # Test with good information availability
        high_info_context = DecisionContext(
            crisis_id="crisis_004",
            crisis_type=CrisisType.TECHNICAL_OUTAGE,
            severity_level=3,  # Low severity
            stakeholders_affected=["customers"],
            time_pressure=DecisionUrgency.MODERATE,
            available_information={f"info_{i}": f"value_{i}" for i in range(10)},  # Lots of info
            resource_constraints=[],
            regulatory_considerations=[]
        )
        
        high_confidence = self.engine._calculate_confidence_score(high_info_context)
        
        # Test with limited information
        low_info_context = DecisionContext(
            crisis_id="crisis_005",
            crisis_type=CrisisType.TECHNICAL_OUTAGE,
            severity_level=9,  # High severity
            stakeholders_affected=["customers"],
            time_pressure=DecisionUrgency.IMMEDIATE,
            available_information={},  # No info
            resource_constraints=[],
            regulatory_considerations=[]
        )
        
        low_confidence = self.engine._calculate_confidence_score(low_info_context)
        
        assert 0.0 <= high_confidence <= 1.0
        assert 0.0 <= low_confidence <= 1.0
        assert high_confidence > low_confidence
    
    def test_error_handling(self):
        """Test error handling in guidance generation"""
        # Test with invalid context (this should still work but test error handling paths)
        try:
            recommendation = self.engine.get_leadership_guidance(self.sample_context)
            assert recommendation is not None
        except Exception as e:
            # If there's an exception, it should be properly formatted
            assert "Failed to generate leadership guidance" in str(e)
    
    def test_assessment_scoring_logic(self):
        """Test assessment scoring logic"""
        # Test high performance data
        high_performance = {
            "decision_quality": 0.9,
            "communication_effectiveness": 0.85,
            "stakeholder_confidence": 0.8,
            "team_morale": 0.9,
            "resolution_speed": 0.85
        }
        
        high_assessment = self.engine.assess_leadership_effectiveness(
            "leader_high", "crisis_001", high_performance
        )
        
        # Test low performance data
        low_performance = {
            "decision_quality": 0.5,
            "communication_effectiveness": 0.4,
            "stakeholder_confidence": 0.3,
            "team_morale": 0.6,
            "resolution_speed": 0.4
        }
        
        low_assessment = self.engine.assess_leadership_effectiveness(
            "leader_low", "crisis_001", low_performance
        )
        
        # High performance should score better
        assert high_assessment.overall_effectiveness > low_assessment.overall_effectiveness
        assert len(high_assessment.strengths) >= len(low_assessment.strengths)
        assert len(high_assessment.improvement_areas) <= len(low_assessment.improvement_areas)
    
    def test_coaching_guidance_specificity(self):
        """Test that coaching guidance is specific to improvement areas"""
        performance_data = {
            "decision_quality": 0.5,  # Low - should trigger decision-making coaching
            "communication_effectiveness": 0.4,  # Low - should trigger communication coaching
            "stakeholder_confidence": 0.8,  # High - should not trigger coaching
            "team_morale": 0.7,
            "resolution_speed": 0.6
        }
        
        assessment = self.engine.assess_leadership_effectiveness(
            "leader_001", "crisis_001", performance_data
        )
        
        coaching_guidance = self.engine.provide_coaching_guidance(assessment)
        
        # Should have coaching for low-scoring areas
        focus_areas = [guidance.focus_area for guidance in coaching_guidance]
        
        # Should include decision-making and communication guidance
        assert any("decision" in area.lower() for area in focus_areas)
        assert any("communication" in area.lower() for area in focus_areas)
        
        # Each guidance should have specific strategies and exercises
        for guidance in coaching_guidance:
            assert len(guidance.improvement_strategies) > 0
            assert len(guidance.practice_exercises) > 0
            assert guidance.target_performance > guidance.current_performance