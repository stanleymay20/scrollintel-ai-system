"""
Tests for Crisis Response Effectiveness Testing Engine

Comprehensive tests for crisis response effectiveness measurement and validation.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from scrollintel.engines.crisis_response_effectiveness_testing import (
    CrisisResponseEffectivenessTesting,
    EffectivenessMetric,
    TestingPhase,
    EffectivenessScore,
    CrisisResponseTest
)

@pytest.fixture
def effectiveness_testing():
    """Create a CrisisResponseEffectivenessTesting instance for testing"""
    return CrisisResponseEffectivenessTesting()

@pytest.fixture
def sample_crisis_scenario():
    """Sample crisis scenario for testing"""
    return "Major system outage affecting 80% of users with potential data loss"

@pytest.fixture
def sample_decisions():
    """Sample decisions made during crisis"""
    return [
        {
            "id": "decision_1",
            "type": "immediate_response",
            "description": "Activate emergency response team",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "decision_2",
            "type": "communication",
            "description": "Send customer notification",
            "timestamp": datetime.now().isoformat()
        }
    ]

@pytest.fixture
def sample_decision_outcomes():
    """Sample decision outcomes for testing"""
    return [
        {
            "information_completeness": 0.8,
            "stakeholder_consideration": 0.9,
            "risk_assessment_accuracy": 0.7,
            "implementation_feasibility": 0.85,
            "outcome_effectiveness": 0.8
        },
        {
            "information_completeness": 0.7,
            "stakeholder_consideration": 0.8,
            "risk_assessment_accuracy": 0.75,
            "implementation_feasibility": 0.9,
            "outcome_effectiveness": 0.85
        }
    ]

@pytest.fixture
def sample_communications():
    """Sample communications sent during crisis"""
    return [
        {
            "id": "comm_1",
            "channel": "email",
            "audience": "customers",
            "message": "We are experiencing technical difficulties",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "comm_2",
            "channel": "slack",
            "audience": "internal_team",
            "message": "Emergency response activated",
            "timestamp": datetime.now().isoformat()
        }
    ]

@pytest.fixture
def sample_stakeholder_feedback():
    """Sample stakeholder feedback for testing"""
    return [
        {
            "communication_id": "comm_1",
            "clarity_rating": 0.8,
            "timeliness_rating": 0.9,
            "completeness_rating": 0.7,
            "appropriateness_rating": 0.85
        },
        {
            "communication_id": "comm_2",
            "clarity_rating": 0.9,
            "timeliness_rating": 0.95,
            "completeness_rating": 0.85,
            "appropriateness_rating": 0.9
        }
    ]

class TestCrisisResponseEffectivenessTesting:
    """Test cases for CrisisResponseEffectivenessTesting"""
    
    @pytest.mark.asyncio
    async def test_start_effectiveness_test(self, effectiveness_testing, sample_crisis_scenario):
        """Test starting a new effectiveness test"""
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario,
            test_type="comprehensive"
        )
        
        assert test_id is not None
        assert test_id in effectiveness_testing.active_tests
        
        test = effectiveness_testing.active_tests[test_id]
        assert test.crisis_scenario == sample_crisis_scenario
        assert test.test_type == "comprehensive"
        assert test.start_time is not None
        assert test.end_time is None
    
    @pytest.mark.asyncio
    async def test_measure_response_speed(self, effectiveness_testing, sample_crisis_scenario):
        """Test measuring crisis response speed"""
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario
        )
        
        detection_time = datetime.now()
        first_response_time = detection_time + timedelta(minutes=3)
        full_response_time = detection_time + timedelta(minutes=15)
        
        score = await effectiveness_testing.measure_response_speed(
            test_id=test_id,
            detection_time=detection_time,
            first_response_time=first_response_time,
            full_response_time=full_response_time
        )
        
        assert score.metric == EffectivenessMetric.RESPONSE_SPEED
        assert 0.0 <= score.score <= 1.0
        assert score.confidence_level == 0.95
        assert "detection_to_first_response_seconds" in score.details
        assert "detection_to_full_response_seconds" in score.details
        
        # Verify score is added to test
        test = effectiveness_testing.active_tests[test_id]
        assert len(test.effectiveness_scores) == 1
        assert test.effectiveness_scores[0] == score
    
    @pytest.mark.asyncio
    async def test_measure_decision_quality(
        self,
        effectiveness_testing,
        sample_crisis_scenario,
        sample_decisions,
        sample_decision_outcomes
    ):
        """Test measuring decision quality"""
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario
        )
        
        score = await effectiveness_testing.measure_decision_quality(
            test_id=test_id,
            decisions_made=sample_decisions,
            decision_outcomes=sample_decision_outcomes
        )
        
        assert score.metric == EffectivenessMetric.DECISION_QUALITY
        assert 0.0 <= score.score <= 1.0
        assert score.confidence_level == 0.9
        assert "decisions_evaluated" in score.details
        assert score.details["decisions_evaluated"] == len(sample_decisions)
        assert "individual_scores" in score.details
        assert "decision_details" in score.details
    
    @pytest.mark.asyncio
    async def test_measure_communication_effectiveness(
        self,
        effectiveness_testing,
        sample_crisis_scenario,
        sample_communications,
        sample_stakeholder_feedback
    ):
        """Test measuring communication effectiveness"""
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario
        )
        
        score = await effectiveness_testing.measure_communication_effectiveness(
            test_id=test_id,
            communications_sent=sample_communications,
            stakeholder_feedback=sample_stakeholder_feedback
        )
        
        assert score.metric == EffectivenessMetric.COMMUNICATION_CLARITY
        assert 0.0 <= score.score <= 1.0
        assert score.confidence_level == 0.85
        assert "communications_evaluated" in score.details
        assert score.details["communications_evaluated"] == len(sample_communications)
        assert "communication_details" in score.details
    
    @pytest.mark.asyncio
    async def test_measure_outcome_success(self, effectiveness_testing, sample_crisis_scenario):
        """Test measuring outcome success"""
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario
        )
        
        crisis_objectives = [
            "Restore system functionality within 2 hours",
            "Maintain customer communication throughout incident",
            "Prevent data loss"
        ]
        
        achieved_outcomes = [
            {
                "completion_rate": 0.9,
                "quality_rating": 0.8,
                "stakeholder_satisfaction": 0.75,
                "long_term_impact_score": 0.85
            },
            {
                "completion_rate": 0.95,
                "quality_rating": 0.9,
                "stakeholder_satisfaction": 0.85,
                "long_term_impact_score": 0.8
            },
            {
                "completion_rate": 1.0,
                "quality_rating": 0.95,
                "stakeholder_satisfaction": 0.9,
                "long_term_impact_score": 0.9
            }
        ]
        
        score = await effectiveness_testing.measure_outcome_success(
            test_id=test_id,
            crisis_objectives=crisis_objectives,
            achieved_outcomes=achieved_outcomes
        )
        
        assert score.metric == EffectivenessMetric.OUTCOME_SUCCESS
        assert 0.0 <= score.score <= 1.0
        assert score.confidence_level == 0.9
        assert "objectives_evaluated" in score.details
        assert score.details["objectives_evaluated"] == len(crisis_objectives)
        assert "outcome_details" in score.details
    
    @pytest.mark.asyncio
    async def test_measure_leadership_effectiveness(self, effectiveness_testing, sample_crisis_scenario):
        """Test measuring leadership effectiveness"""
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario
        )
        
        leadership_actions = [
            {
                "type": "decision_making",
                "description": "Made rapid decision to activate response team",
                "effectiveness_rating": 0.9
            },
            {
                "type": "communication",
                "description": "Communicated clearly with stakeholders",
                "effectiveness_rating": 0.85
            },
            {
                "type": "team_coordination",
                "description": "Coordinated response team effectively",
                "effectiveness_rating": 0.8
            }
        ]
        
        team_feedback = [
            {
                "leadership_clarity": 0.9,
                "decision_confidence": 0.85,
                "communication_effectiveness": 0.8
            },
            {
                "leadership_clarity": 0.85,
                "decision_confidence": 0.9,
                "communication_effectiveness": 0.85
            }
        ]
        
        stakeholder_confidence = {
            "board": 0.8,
            "customers": 0.75,
            "employees": 0.85
        }
        
        score = await effectiveness_testing.measure_leadership_effectiveness(
            test_id=test_id,
            leadership_actions=leadership_actions,
            team_feedback=team_feedback,
            stakeholder_confidence=stakeholder_confidence
        )
        
        assert score.metric == EffectivenessMetric.LEADERSHIP_EFFECTIVENESS
        assert 0.0 <= score.score <= 1.0
        assert score.confidence_level == 0.85
        assert "leadership_dimensions" in score.details
        assert "team_ratings" in score.details
        assert "stakeholder_confidence" in score.details
    
    @pytest.mark.asyncio
    async def test_complete_effectiveness_test(self, effectiveness_testing, sample_crisis_scenario):
        """Test completing an effectiveness test"""
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario
        )
        
        # Add some scores
        detection_time = datetime.now()
        first_response_time = detection_time + timedelta(minutes=3)
        full_response_time = detection_time + timedelta(minutes=15)
        
        await effectiveness_testing.measure_response_speed(
            test_id=test_id,
            detection_time=detection_time,
            first_response_time=first_response_time,
            full_response_time=full_response_time
        )
        
        # Complete the test
        completed_test = await effectiveness_testing.complete_effectiveness_test(test_id)
        
        assert completed_test.test_id == test_id
        assert completed_test.end_time is not None
        assert completed_test.overall_score is not None
        assert 0.0 <= completed_test.overall_score <= 1.0
        assert isinstance(completed_test.recommendations, list)
        
        # Verify test moved to history
        assert test_id not in effectiveness_testing.active_tests
        assert completed_test in effectiveness_testing.test_history
    
    @pytest.mark.asyncio
    async def test_get_effectiveness_trends(self, effectiveness_testing):
        """Test getting effectiveness trends"""
        # Create some test history
        for i in range(3):
            test = CrisisResponseTest(
                test_id=f"test_{i}",
                crisis_scenario=f"Crisis scenario {i}",
                test_type="comprehensive",
                start_time=datetime.now() - timedelta(days=i*10),
                end_time=datetime.now() - timedelta(days=i*10-1),
                overall_score=0.7 + i*0.1
            )
            test.effectiveness_scores = [
                EffectivenessScore(
                    metric=EffectivenessMetric.RESPONSE_SPEED,
                    score=0.8 + i*0.05,
                    details={},
                    measurement_time=datetime.now(),
                    confidence_level=0.9
                )
            ]
            effectiveness_testing.test_history.append(test)
        
        # Test overall trends
        trends = await effectiveness_testing.get_effectiveness_trends()
        assert "overall_scores" in trends
        assert "average_score" in trends
        assert "tests_analyzed" in trends
        assert trends["tests_analyzed"] == 3
        
        # Test metric-specific trends
        speed_trends = await effectiveness_testing.get_effectiveness_trends(
            metric=EffectivenessMetric.RESPONSE_SPEED
        )
        assert speed_trends["metric"] == "response_speed"
        assert "scores" in speed_trends
        assert len(speed_trends["scores"]) == 3
    
    @pytest.mark.asyncio
    async def test_benchmark_against_baseline(self, effectiveness_testing, sample_crisis_scenario):
        """Test benchmarking against baseline metrics"""
        # Set up baselines
        baselines = {
            EffectivenessMetric.RESPONSE_SPEED: 0.7,
            EffectivenessMetric.DECISION_QUALITY: 0.75
        }
        effectiveness_testing.update_baseline_metrics(baselines)
        
        # Create and complete a test
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario
        )
        
        detection_time = datetime.now()
        first_response_time = detection_time + timedelta(minutes=2)
        full_response_time = detection_time + timedelta(minutes=10)
        
        await effectiveness_testing.measure_response_speed(
            test_id=test_id,
            detection_time=detection_time,
            first_response_time=first_response_time,
            full_response_time=full_response_time
        )
        
        completed_test = await effectiveness_testing.complete_effectiveness_test(test_id)
        
        # Benchmark the test
        comparison = await effectiveness_testing.benchmark_against_baseline(test_id)
        
        assert "test_id" in comparison
        assert "metric_comparisons" in comparison
        assert "overall_score" in comparison
        
        # Check that response speed comparison exists
        assert "response_speed" in comparison["metric_comparisons"]
        speed_comparison = comparison["metric_comparisons"]["response_speed"]
        assert "current_score" in speed_comparison
        assert "baseline_score" in speed_comparison
        assert "improvement" in speed_comparison
        assert "performance_level" in speed_comparison
    
    @pytest.mark.asyncio
    async def test_export_test_results(self, effectiveness_testing, sample_crisis_scenario):
        """Test exporting test results"""
        test_id = await effectiveness_testing.start_effectiveness_test(
            crisis_scenario=sample_crisis_scenario
        )
        
        # Add a score
        detection_time = datetime.now()
        first_response_time = detection_time + timedelta(minutes=3)
        full_response_time = detection_time + timedelta(minutes=15)
        
        await effectiveness_testing.measure_response_speed(
            test_id=test_id,
            detection_time=detection_time,
            first_response_time=first_response_time,
            full_response_time=full_response_time
        )
        
        # Complete and export
        await effectiveness_testing.complete_effectiveness_test(test_id)
        results = await effectiveness_testing.export_test_results(test_id)
        
        assert "test_id" in results
        assert "crisis_scenario" in results
        assert "test_type" in results
        assert "start_time" in results
        assert "end_time" in results
        assert "duration_seconds" in results
        assert "effectiveness_scores" in results
        assert "overall_score" in results
        assert "recommendations" in results
        
        # Verify effectiveness scores format
        assert len(results["effectiveness_scores"]) == 1
        score_data = results["effectiveness_scores"][0]
        assert "metric" in score_data
        assert "score" in score_data
        assert "details" in score_data
        assert "confidence_level" in score_data
    
    def test_performance_thresholds(self, effectiveness_testing):
        """Test performance threshold evaluation"""
        metric = EffectivenessMetric.RESPONSE_SPEED
        
        # Test excellent performance
        level = effectiveness_testing._get_performance_level(metric, 0.95)
        assert level == "excellent"
        
        # Test good performance
        level = effectiveness_testing._get_performance_level(metric, 0.8)
        assert level == "good"
        
        # Test acceptable performance
        level = effectiveness_testing._get_performance_level(metric, 0.6)
        assert level == "acceptable"
        
        # Test poor performance
        level = effectiveness_testing._get_performance_level(metric, 0.2)
        assert level == "poor"
    
    def test_recommendation_generation(self, effectiveness_testing):
        """Test recommendation generation based on scores"""
        test = CrisisResponseTest(
            test_id="test_recommendations",
            crisis_scenario="Test scenario",
            test_type="comprehensive",
            start_time=datetime.now()
        )
        
        # Add low scores to trigger recommendations
        test.effectiveness_scores = [
            EffectivenessScore(
                metric=EffectivenessMetric.RESPONSE_SPEED,
                score=0.3,  # Below acceptable threshold
                details={},
                measurement_time=datetime.now(),
                confidence_level=0.9
            ),
            EffectivenessScore(
                metric=EffectivenessMetric.DECISION_QUALITY,
                score=0.4,  # Below acceptable threshold
                details={},
                measurement_time=datetime.now(),
                confidence_level=0.9
            )
        ]
        
        recommendations = effectiveness_testing._generate_recommendations(test)
        
        assert len(recommendations) > 0
        assert any("crisis detection" in rec.lower() for rec in recommendations)
        assert any("decision-making" in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_invalid_test_id_handling(self, effectiveness_testing):
        """Test handling of invalid test IDs"""
        invalid_test_id = "nonexistent_test"
        
        with pytest.raises(ValueError, match="Test .* not found"):
            await effectiveness_testing.measure_response_speed(
                test_id=invalid_test_id,
                detection_time=datetime.now(),
                first_response_time=datetime.now(),
                full_response_time=datetime.now()
            )
        
        with pytest.raises(ValueError, match="Test .* not found"):
            await effectiveness_testing.complete_effectiveness_test(invalid_test_id)
    
    def test_baseline_metrics_update(self, effectiveness_testing):
        """Test updating baseline metrics"""
        new_baselines = {
            EffectivenessMetric.RESPONSE_SPEED: 0.8,
            EffectivenessMetric.DECISION_QUALITY: 0.75,
            EffectivenessMetric.COMMUNICATION_CLARITY: 0.7
        }
        
        effectiveness_testing.update_baseline_metrics(new_baselines)
        
        for metric, expected_value in new_baselines.items():
            assert effectiveness_testing.baseline_metrics[metric] == expected_value

if __name__ == "__main__":
    pytest.main([__file__])