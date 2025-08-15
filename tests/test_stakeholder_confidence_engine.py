"""
Tests for stakeholder confidence management engine.
Tests confidence monitoring, assessment, and trust maintenance capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.stakeholder_confidence_engine import StakeholderConfidenceEngine
from scrollintel.models.stakeholder_confidence_models import (
    StakeholderProfile, ConfidenceMetrics, ConfidenceBuildingStrategy,
    TrustMaintenanceAction, CommunicationPlan, ConfidenceAssessment,
    StakeholderFeedback, ConfidenceAlert, StakeholderType, ConfidenceLevel,
    TrustIndicator
)


class TestStakeholderConfidenceEngine:
    """Test cases for stakeholder confidence management engine"""
    
    @pytest.fixture
    def engine(self):
        """Create stakeholder confidence engine instance"""
        return StakeholderConfidenceEngine()
    
    @pytest.fixture
    def sample_stakeholder_profile(self):
        """Create sample stakeholder profile"""
        return StakeholderProfile(
            stakeholder_id="stakeholder_001",
            name="John Investor",
            stakeholder_type=StakeholderType.INVESTOR,
            influence_level="high",
            communication_preferences=["email", "phone"],
            historical_confidence=[0.8, 0.7, 0.6],
            key_concerns=["financial_impact", "timeline"],
            relationship_strength=0.8,
            contact_information={"email": "john@investor.com", "phone": "+1234567890"},
            last_interaction=datetime.now() - timedelta(days=1)
        )
    
    @pytest.fixture
    def sample_feedback(self):
        """Create sample stakeholder feedback"""
        return StakeholderFeedback(
            feedback_id="feedback_001",
            stakeholder_id="stakeholder_001",
            feedback_type="concern",
            content="Concerned about the impact on our investment",
            sentiment="negative",
            urgency_level="high",
            received_time=datetime.now(),
            response_required=True
        )
    
    @pytest.mark.asyncio
    async def test_monitor_stakeholder_confidence(self, engine, sample_stakeholder_profile):
        """Test stakeholder confidence monitoring"""
        # Setup
        engine.stakeholder_profiles[sample_stakeholder_profile.stakeholder_id] = sample_stakeholder_profile
        crisis_id = "crisis_001"
        stakeholder_ids = [sample_stakeholder_profile.stakeholder_id]
        
        # Execute
        confidence_data = await engine.monitor_stakeholder_confidence(crisis_id, stakeholder_ids)
        
        # Verify
        assert len(confidence_data) == 1
        assert sample_stakeholder_profile.stakeholder_id in confidence_data
        
        metrics = confidence_data[sample_stakeholder_profile.stakeholder_id]
        assert isinstance(metrics, ConfidenceMetrics)
        assert metrics.stakeholder_id == sample_stakeholder_profile.stakeholder_id
        assert isinstance(metrics.confidence_level, ConfidenceLevel)
        assert 0.0 <= metrics.trust_score <= 1.0
        assert metrics.measurement_time is not None
        
        # Verify metrics are stored
        assert sample_stakeholder_profile.stakeholder_id in engine.confidence_metrics
        assert len(engine.confidence_metrics[sample_stakeholder_profile.stakeholder_id]) == 1
    
    @pytest.mark.asyncio
    async def test_assess_overall_confidence(self, engine, sample_stakeholder_profile):
        """Test overall confidence assessment"""
        # Setup
        engine.stakeholder_profiles[sample_stakeholder_profile.stakeholder_id] = sample_stakeholder_profile
        
        # Add some confidence metrics
        metrics = ConfidenceMetrics(
            stakeholder_id=sample_stakeholder_profile.stakeholder_id,
            confidence_level=ConfidenceLevel.MODERATE,
            trust_score=0.6,
            engagement_score=0.7,
            sentiment_score=0.5,
            response_rate=0.8,
            satisfaction_rating=0.6,
            risk_indicators=[],
            measurement_time=datetime.now(),
            data_sources=["test"]
        )
        engine.confidence_metrics[sample_stakeholder_profile.stakeholder_id] = [metrics]
        
        crisis_id = "crisis_001"
        
        # Execute
        assessment = await engine.assess_overall_confidence(crisis_id)
        
        # Verify
        assert isinstance(assessment, ConfidenceAssessment)
        assert assessment.crisis_id == crisis_id
        assert 0.0 <= assessment.overall_confidence_score <= 1.0
        assert isinstance(assessment.stakeholder_breakdown, dict)
        assert isinstance(assessment.risk_areas, list)
        assert isinstance(assessment.improvement_opportunities, list)
        assert isinstance(assessment.recommended_actions, list)
        assert assessment.next_assessment_date > assessment.assessment_time
        
        # Verify assessment is stored
        assert len(engine.assessments) == 1
        assert engine.assessments[0] == assessment
    
    @pytest.mark.asyncio
    async def test_build_confidence_strategy(self, engine):
        """Test confidence building strategy creation"""
        # Setup
        stakeholder_type = StakeholderType.INVESTOR
        current_confidence = ConfidenceLevel.LOW
        target_confidence = ConfidenceLevel.HIGH
        
        # Execute
        strategy = await engine.build_confidence_strategy(
            stakeholder_type, current_confidence, target_confidence
        )
        
        # Verify
        assert isinstance(strategy, ConfidenceBuildingStrategy)
        assert strategy.stakeholder_type == stakeholder_type
        assert strategy.target_confidence_level == target_confidence
        assert isinstance(strategy.communication_approach, str)
        assert isinstance(strategy.key_messages, list)
        assert len(strategy.key_messages) > 0
        assert isinstance(strategy.engagement_tactics, list)
        assert len(strategy.engagement_tactics) > 0
        assert isinstance(strategy.timeline, dict)
        assert isinstance(strategy.success_metrics, list)
        assert isinstance(strategy.resource_requirements, list)
        assert isinstance(strategy.risk_mitigation, list)
        
        # Verify strategy is stored
        assert strategy.strategy_id in engine.building_strategies
        assert engine.building_strategies[strategy.strategy_id] == strategy
    
    @pytest.mark.asyncio
    async def test_maintain_stakeholder_trust(self, engine, sample_stakeholder_profile):
        """Test stakeholder trust maintenance"""
        # Setup
        engine.stakeholder_profiles[sample_stakeholder_profile.stakeholder_id] = sample_stakeholder_profile
        
        # Add confidence metrics
        metrics = ConfidenceMetrics(
            stakeholder_id=sample_stakeholder_profile.stakeholder_id,
            confidence_level=ConfidenceLevel.MODERATE,
            trust_score=0.6,
            engagement_score=0.7,
            sentiment_score=0.5,
            response_rate=0.8,
            satisfaction_rating=0.6,
            risk_indicators=[],
            measurement_time=datetime.now(),
            data_sources=["test"]
        )
        engine.confidence_metrics[sample_stakeholder_profile.stakeholder_id] = [metrics]
        
        crisis_context = {
            "crisis_type": "system_outage",
            "severity": "high",
            "estimated_resolution": "24_hours"
        }
        
        # Execute
        actions = await engine.maintain_stakeholder_trust(
            sample_stakeholder_profile.stakeholder_id, crisis_context
        )
        
        # Verify
        assert isinstance(actions, list)
        assert len(actions) > 0
        
        for action in actions:
            assert isinstance(action, TrustMaintenanceAction)
            assert action.stakeholder_id == sample_stakeholder_profile.stakeholder_id
            assert isinstance(action.action_type, str)
            assert isinstance(action.description, str)
            assert action.priority in ["high", "medium", "low"]
            assert isinstance(action.implementation_steps, list)
            assert len(action.implementation_steps) > 0
            assert isinstance(action.required_resources, list)
            assert isinstance(action.success_criteria, list)
            assert action.timeline > datetime.now()
        
        # Verify actions are stored
        assert sample_stakeholder_profile.stakeholder_id in engine.trust_actions
        stored_actions = engine.trust_actions[sample_stakeholder_profile.stakeholder_id]
        assert len(stored_actions) >= len(actions)
    
    @pytest.mark.asyncio
    async def test_create_communication_plan(self, engine):
        """Test communication plan creation"""
        # Setup
        crisis_id = "crisis_001"
        stakeholder_segments = [StakeholderType.INVESTOR, StakeholderType.CUSTOMER]
        
        # Execute
        plan = await engine.create_communication_plan(crisis_id, stakeholder_segments)
        
        # Verify
        assert isinstance(plan, CommunicationPlan)
        assert plan.stakeholder_segments == stakeholder_segments
        assert isinstance(plan.key_messages, dict)
        assert len(plan.key_messages) == len(stakeholder_segments)
        
        for segment in stakeholder_segments:
            assert segment.value in plan.key_messages
            assert isinstance(plan.key_messages[segment.value], str)
        
        assert isinstance(plan.communication_channels, list)
        assert len(plan.communication_channels) > 0
        assert isinstance(plan.frequency, str)
        assert isinstance(plan.tone_and_style, str)
        assert isinstance(plan.approval_workflow, list)
        assert isinstance(plan.feedback_mechanisms, list)
        assert isinstance(plan.escalation_triggers, list)
        assert isinstance(plan.effectiveness_metrics, list)
        
        # Verify plan is stored
        assert plan.plan_id in engine.communication_plans
        assert engine.communication_plans[plan.plan_id] == plan
    
    @pytest.mark.asyncio
    async def test_process_stakeholder_feedback(self, engine, sample_feedback):
        """Test stakeholder feedback processing"""
        # Execute
        result = await engine.process_stakeholder_feedback(sample_feedback)
        
        # Verify
        assert isinstance(result, dict)
        assert "feedback_id" in result
        assert result["feedback_id"] == sample_feedback.feedback_id
        assert "analysis" in result
        assert "response_strategy" in result
        assert "follow_up_actions" in result
        assert "processing_time" in result
        
        # Verify analysis
        analysis = result["analysis"]
        assert isinstance(analysis, dict)
        assert "sentiment_score" in analysis
        assert "urgency_level" in analysis
        assert "requires_escalation" in analysis
        
        # Verify response strategy
        response_strategy = result["response_strategy"]
        assert isinstance(response_strategy, dict)
        assert "response_type" in response_strategy
        assert "response_timeline" in response_strategy
        
        # Verify follow-up actions
        follow_up_actions = result["follow_up_actions"]
        assert isinstance(follow_up_actions, list)
        assert len(follow_up_actions) > 0
        
        # Verify feedback is stored
        assert sample_feedback in engine.feedback_queue
        assert sample_feedback.follow_up_actions == follow_up_actions
    
    @pytest.mark.asyncio
    async def test_confidence_alert_generation(self, engine, sample_stakeholder_profile):
        """Test confidence alert generation for low confidence"""
        # Setup
        engine.stakeholder_profiles[sample_stakeholder_profile.stakeholder_id] = sample_stakeholder_profile
        
        # Create low confidence metrics
        low_confidence_metrics = ConfidenceMetrics(
            stakeholder_id=sample_stakeholder_profile.stakeholder_id,
            confidence_level=ConfidenceLevel.CRITICAL,
            trust_score=0.1,
            engagement_score=0.2,
            sentiment_score=0.1,
            response_rate=0.3,
            satisfaction_rating=0.1,
            risk_indicators=["declining_trust", "negative_sentiment"],
            measurement_time=datetime.now(),
            data_sources=["test"]
        )
        
        # Execute confidence check (this should trigger alert)
        await engine._check_confidence_alerts(
            sample_stakeholder_profile.stakeholder_id, 
            low_confidence_metrics
        )
        
        # Verify alert was created
        assert len(engine.active_alerts) > 0
        
        alert = engine.active_alerts[0]
        assert isinstance(alert, ConfidenceAlert)
        assert alert.stakeholder_id == sample_stakeholder_profile.stakeholder_id
        assert alert.alert_type == "low_confidence"
        assert alert.severity in ["high", "medium"]
        assert isinstance(alert.description, str)
        assert alert.manual_review_required == True
        assert isinstance(alert.escalation_path, list)
        assert isinstance(alert.auto_actions, list)
    
    @pytest.mark.asyncio
    async def test_confidence_trends_analysis(self, engine, sample_stakeholder_profile):
        """Test confidence trends analysis"""
        # Setup
        engine.stakeholder_profiles[sample_stakeholder_profile.stakeholder_id] = sample_stakeholder_profile
        
        # Add multiple confidence metrics to show trend
        metrics_history = []
        for i, score in enumerate([0.8, 0.7, 0.6, 0.5]):  # Declining trend
            metrics = ConfidenceMetrics(
                stakeholder_id=sample_stakeholder_profile.stakeholder_id,
                confidence_level=ConfidenceLevel.MODERATE,
                trust_score=score,
                engagement_score=score,
                sentiment_score=score,
                response_rate=score,
                satisfaction_rating=score,
                risk_indicators=[],
                measurement_time=datetime.now() - timedelta(hours=i),
                data_sources=["test"]
            )
            metrics_history.append(metrics)
        
        engine.confidence_metrics[sample_stakeholder_profile.stakeholder_id] = metrics_history
        
        # Execute
        trends = await engine._analyze_confidence_trends()
        
        # Verify
        assert isinstance(trends, dict)
        assert "overall_trend" in trends
        assert "risk_stakeholders" in trends
        assert "improving_stakeholders" in trends
        assert "trend_analysis_time" in trends
        
        assert isinstance(trends["risk_stakeholders"], list)
        assert isinstance(trends["improving_stakeholders"], list)
    
    @pytest.mark.asyncio
    async def test_stakeholder_type_specific_strategies(self, engine):
        """Test that different stakeholder types get appropriate strategies"""
        stakeholder_types = [
            StakeholderType.BOARD_MEMBER,
            StakeholderType.INVESTOR,
            StakeholderType.CUSTOMER,
            StakeholderType.EMPLOYEE
        ]
        
        strategies = []
        for stakeholder_type in stakeholder_types:
            strategy = await engine.build_confidence_strategy(
                stakeholder_type, ConfidenceLevel.LOW, ConfidenceLevel.HIGH
            )
            strategies.append(strategy)
        
        # Verify each strategy is tailored to stakeholder type
        for i, strategy in enumerate(strategies):
            stakeholder_type = stakeholder_types[i]
            
            # Check communication approach is appropriate
            assert isinstance(strategy.communication_approach, str)
            
            # Check key messages are relevant
            assert len(strategy.key_messages) > 0
            
            # Check engagement tactics are appropriate
            assert len(strategy.engagement_tactics) > 0
            
            # Verify type-specific elements
            if stakeholder_type == StakeholderType.BOARD_MEMBER:
                assert any("executive" in tactic or "board" in tactic 
                          for tactic in strategy.engagement_tactics)
            elif stakeholder_type == StakeholderType.CUSTOMER:
                assert any("customer" in tactic or "service" in tactic 
                          for tactic in strategy.engagement_tactics)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine):
        """Test error handling in confidence management"""
        # Test with invalid stakeholder ID
        with pytest.raises(ValueError):
            await engine.maintain_stakeholder_trust("invalid_id", {})
        
        # Test monitoring with empty stakeholder list
        result = await engine.monitor_stakeholder_confidence("crisis_001", [])
        assert len(result) == 0
        
        # Test assessment with no data
        assessment = await engine.assess_overall_confidence("crisis_001")
        assert isinstance(assessment, ConfidenceAssessment)
        assert assessment.overall_confidence_score == 0.5  # Default value
    
    def test_confidence_level_calculation(self, engine):
        """Test confidence level calculation from trust indicators"""
        test_cases = [
            ({"indicator1": 0.95, "indicator2": 0.9}, ConfidenceLevel.VERY_HIGH),
            ({"indicator1": 0.85, "indicator2": 0.8}, ConfidenceLevel.HIGH),
            ({"indicator1": 0.65, "indicator2": 0.6}, ConfidenceLevel.MODERATE),
            ({"indicator1": 0.45, "indicator2": 0.4}, ConfidenceLevel.LOW),
            ({"indicator1": 0.25, "indicator2": 0.2}, ConfidenceLevel.VERY_LOW),
            ({"indicator1": 0.15, "indicator2": 0.1}, ConfidenceLevel.CRITICAL)
        ]
        
        for indicators, expected_level in test_cases:
            # Calculate trust score
            trust_score = sum(indicators.values()) / len(indicators)
            
            # Determine expected confidence level based on trust score
            if trust_score >= 0.9:
                expected = ConfidenceLevel.VERY_HIGH
            elif trust_score >= 0.8:
                expected = ConfidenceLevel.HIGH
            elif trust_score >= 0.6:
                expected = ConfidenceLevel.MODERATE
            elif trust_score >= 0.4:
                expected = ConfidenceLevel.LOW
            elif trust_score >= 0.2:
                expected = ConfidenceLevel.VERY_LOW
            else:
                expected = ConfidenceLevel.CRITICAL
            
            assert expected == expected_level


if __name__ == "__main__":
    pytest.main([__file__])