"""
Tests for Recovery Planning Engine

This module contains comprehensive tests for the recovery planning engine
functionality including strategy development, progress tracking, and optimization.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.recovery_planning_engine import RecoveryPlanningEngine
from scrollintel.models.recovery_planning_models import (
    RecoveryStrategy, RecoveryMilestone, RecoveryProgress, RecoveryOptimization,
    RecoveryPhase, RecoveryStatus, RecoveryPriority
)
from scrollintel.models.crisis_detection_models import CrisisModel


class TestRecoveryPlanningEngine:
    """Test cases for RecoveryPlanningEngine"""
    
    @pytest.fixture
    def recovery_engine(self):
        """Create a recovery planning engine instance for testing"""
        return RecoveryPlanningEngine()
    
    @pytest.fixture
    def sample_crisis(self):
        """Create a sample crisis for testing"""
        return CrisisModel(
            id="test_crisis_001",
            crisis_type="system_failure",
            severity_level="high",
            status="resolved",
            start_time=datetime.now() - timedelta(days=1),
            affected_areas=["production", "customer_portal"],
            stakeholders_impacted=["customers", "employees"],
            resolution_time=datetime.now()
        )
    
    @pytest.fixture
    def recovery_objectives(self):
        """Create sample recovery objectives"""
        return [
            "Restore full operational capacity",
            "Rebuild stakeholder confidence",
            "Implement preventive measures"
        ]
    
    def test_develop_recovery_strategy(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test recovery strategy development"""
        # Develop recovery strategy
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify strategy creation
        assert isinstance(strategy, RecoveryStrategy)
        assert strategy.crisis_id == sample_crisis.id
        assert strategy.recovery_objectives == recovery_objectives
        assert len(strategy.milestones) > 0
        assert len(strategy.timeline) == 4  # Four recovery phases
        assert strategy.id in recovery_engine.recovery_strategies
        
        # Verify milestones are properly distributed across phases
        phases_with_milestones = set(milestone.phase for milestone in strategy.milestones)
        assert RecoveryPhase.IMMEDIATE in phases_with_milestones
        assert RecoveryPhase.SHORT_TERM in phases_with_milestones
        
        # Verify resource allocation is defined
        assert "personnel" in strategy.resource_allocation
        assert "budget" in strategy.resource_allocation
        assert "technology" in strategy.resource_allocation
        
        # Verify success metrics are defined
        assert len(strategy.success_metrics) > 0
        assert "milestone_completion_rate" in strategy.success_metrics
    
    def test_track_recovery_progress(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test recovery progress tracking"""
        # First develop a strategy
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Simulate some progress
        strategy.milestones[0].progress_percentage = 100.0
        strategy.milestones[0].status = RecoveryStatus.COMPLETED
        strategy.milestones[0].completion_date = datetime.now()
        
        strategy.milestones[1].progress_percentage = 50.0
        strategy.milestones[1].status = RecoveryStatus.IN_PROGRESS
        
        # Track progress
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        # Verify progress tracking
        assert isinstance(progress, RecoveryProgress)
        assert progress.strategy_id == strategy.id
        assert progress.overall_progress > 0
        assert progress.milestone_completion_rate > 0
        assert len(progress.phase_progress) == 4  # Four phases
        assert progress.timeline_adherence >= 0
        assert len(progress.resource_utilization) > 0
        assert len(progress.success_metric_achievement) > 0
        
        # Verify progress is stored
        assert strategy.id in recovery_engine.recovery_progress
    
    def test_track_progress_nonexistent_strategy(self, recovery_engine):
        """Test tracking progress for non-existent strategy"""
        with pytest.raises(ValueError, match="Recovery strategy not found"):
            recovery_engine.track_recovery_progress("nonexistent_strategy")
    
    def test_optimize_recovery_strategy(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test recovery strategy optimization"""
        # First develop a strategy and track progress
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        # Generate optimizations
        optimizations = recovery_engine.optimize_recovery_strategy(strategy.id)
        
        # Verify optimizations
        assert isinstance(optimizations, list)
        assert len(optimizations) > 0
        
        for optimization in optimizations:
            assert isinstance(optimization, RecoveryOptimization)
            assert optimization.strategy_id == strategy.id
            assert len(optimization.recommended_actions) > 0
            assert optimization.priority_score > 0
            assert optimization.implementation_effort in ["low", "medium", "high"]
        
        # Verify optimizations are stored
        assert strategy.id in recovery_engine.optimization_recommendations
    
    def test_optimize_nonexistent_strategy(self, recovery_engine):
        """Test optimizing non-existent strategy"""
        with pytest.raises(ValueError, match="Recovery strategy not found"):
            recovery_engine.optimize_recovery_strategy("nonexistent_strategy")
    
    def test_milestone_generation(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test milestone generation logic"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify milestones have required attributes
        for milestone in strategy.milestones:
            assert isinstance(milestone, RecoveryMilestone)
            assert milestone.id is not None
            assert milestone.name is not None
            assert milestone.description is not None
            assert isinstance(milestone.phase, RecoveryPhase)
            assert isinstance(milestone.priority, RecoveryPriority)
            assert isinstance(milestone.target_date, datetime)
            assert milestone.status == RecoveryStatus.PLANNED
            assert milestone.progress_percentage == 0.0
            assert len(milestone.success_criteria) > 0
    
    def test_timeline_calculation(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test recovery timeline calculation"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify timeline structure
        assert len(strategy.timeline) == 4
        assert RecoveryPhase.IMMEDIATE in strategy.timeline
        assert RecoveryPhase.SHORT_TERM in strategy.timeline
        assert RecoveryPhase.MEDIUM_TERM in strategy.timeline
        assert RecoveryPhase.LONG_TERM in strategy.timeline
        
        # Verify timeline progression (each phase should be longer than previous)
        immediate_duration = strategy.timeline[RecoveryPhase.IMMEDIATE]
        short_term_duration = strategy.timeline[RecoveryPhase.SHORT_TERM]
        assert short_term_duration > immediate_duration
    
    def test_resource_allocation_determination(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test resource allocation determination"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify resource allocation structure
        assert "personnel" in strategy.resource_allocation
        assert "budget" in strategy.resource_allocation
        assert "technology" in strategy.resource_allocation
        
        # Verify personnel allocation
        personnel = strategy.resource_allocation["personnel"]
        assert isinstance(personnel, dict)
        assert len(personnel) > 0
        
        # Verify budget allocation
        budget = strategy.resource_allocation["budget"]
        assert isinstance(budget, dict)
        assert len(budget) > 0
        
        # Verify technology allocation
        technology = strategy.resource_allocation["technology"]
        assert isinstance(technology, dict)
        assert len(technology) > 0
    
    def test_communication_plan_creation(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test stakeholder communication plan creation"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify communication plan structure
        comm_plan = strategy.stakeholder_communication_plan
        assert "stakeholder_groups" in comm_plan
        assert "communication_milestones" in comm_plan
        
        # Verify stakeholder groups
        stakeholder_groups = comm_plan["stakeholder_groups"]
        assert "customers" in stakeholder_groups
        assert "employees" in stakeholder_groups
        assert "investors" in stakeholder_groups
        
        # Verify each group has required attributes
        for group, details in stakeholder_groups.items():
            assert "frequency" in details
            assert "channels" in details
            assert "key_messages" in details
            assert len(details["channels"]) > 0
            assert len(details["key_messages"]) > 0
    
    def test_risk_mitigation_identification(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test risk mitigation measures identification"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify risk mitigation measures
        assert len(strategy.risk_mitigation_measures) > 0
        
        for measure in strategy.risk_mitigation_measures:
            assert isinstance(measure, str)
            assert len(measure) > 0
    
    def test_contingency_plans_development(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test contingency plans development"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify contingency plans structure
        contingency_plans = strategy.contingency_plans
        assert len(contingency_plans) > 0
        
        for scenario, plan in contingency_plans.items():
            assert "triggers" in plan
            assert "actions" in plan
            assert "escalation" in plan
            assert len(plan["triggers"]) > 0
            assert len(plan["actions"]) > 0
    
    def test_success_metrics_definition(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test success metrics definition"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify success metrics
        success_metrics = strategy.success_metrics
        assert len(success_metrics) > 0
        
        # Verify key metrics are present
        assert "milestone_completion_rate" in success_metrics
        assert "timeline_adherence" in success_metrics
        assert "stakeholder_satisfaction" in success_metrics
        
        # Verify metric values are reasonable
        for metric, value in success_metrics.items():
            assert isinstance(value, (int, float))
            assert 0 <= value <= 100
    
    def test_overall_progress_calculation(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test overall progress calculation"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Set specific progress values
        strategy.milestones[0].progress_percentage = 100.0
        strategy.milestones[1].progress_percentage = 50.0
        if len(strategy.milestones) > 2:
            strategy.milestones[2].progress_percentage = 25.0
        
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        # Verify overall progress calculation
        expected_progress = sum(m.progress_percentage for m in strategy.milestones) / len(strategy.milestones)
        assert abs(progress.overall_progress - expected_progress) < 0.1
    
    def test_phase_progress_calculation(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test phase-specific progress calculation"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Set progress for immediate phase milestones
        immediate_milestones = [m for m in strategy.milestones if m.phase == RecoveryPhase.IMMEDIATE]
        for milestone in immediate_milestones:
            milestone.progress_percentage = 75.0
        
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        # Verify phase progress calculation
        if immediate_milestones:
            assert progress.phase_progress[RecoveryPhase.IMMEDIATE] == 75.0
    
    def test_milestone_completion_rate_calculation(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test milestone completion rate calculation"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Mark some milestones as completed
        completed_count = min(2, len(strategy.milestones))
        for i in range(completed_count):
            strategy.milestones[i].status = RecoveryStatus.COMPLETED
        
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        # Verify completion rate calculation
        expected_rate = (completed_count / len(strategy.milestones)) * 100
        assert abs(progress.milestone_completion_rate - expected_rate) < 0.1
    
    def test_timeline_adherence_assessment(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test timeline adherence assessment"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Set some milestones as overdue
        current_time = datetime.now()
        for milestone in strategy.milestones[:2]:
            milestone.target_date = current_time - timedelta(days=1)  # Make it overdue
            milestone.status = RecoveryStatus.IN_PROGRESS  # Not completed
        
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        # Verify timeline adherence is calculated
        assert isinstance(progress.timeline_adherence, (int, float))
        assert 0 <= progress.timeline_adherence <= 100
    
    def test_recovery_issues_identification(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test identification of recovery issues"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Create some issues by making milestones overdue
        current_time = datetime.now()
        strategy.milestones[0].target_date = current_time - timedelta(days=1)
        strategy.milestones[0].status = RecoveryStatus.IN_PROGRESS
        
        strategy.milestones[1].status = RecoveryStatus.BLOCKED
        
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        # Verify issues are identified
        assert len(progress.identified_issues) > 0
        
        # Check for specific issue types
        issues_text = " ".join(progress.identified_issues)
        assert "Delayed milestones" in issues_text or "Blocked milestones" in issues_text
    
    def test_adjustment_recommendations(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test adjustment recommendations generation"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Create issues to trigger recommendations
        strategy.milestones[0].status = RecoveryStatus.BLOCKED
        
        progress = recovery_engine.track_recovery_progress(strategy.id)
        
        # Verify recommendations are generated
        assert len(progress.recommended_adjustments) > 0
        
        for recommendation in progress.recommended_adjustments:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
    
    def test_optimization_types(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test different types of optimizations"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        recovery_engine.track_recovery_progress(strategy.id)
        
        optimizations = recovery_engine.optimize_recovery_strategy(strategy.id)
        
        # Verify different optimization types are generated
        optimization_types = [opt.optimization_type for opt in optimizations]
        
        # Should include various optimization types
        expected_types = ["timeline", "resource_allocation", "milestone_sequencing", "communication", "risk_mitigation"]
        for opt_type in optimization_types:
            assert opt_type in expected_types
    
    def test_optimization_priority_scoring(self, recovery_engine, sample_crisis, recovery_objectives):
        """Test optimization priority scoring"""
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        recovery_engine.track_recovery_progress(strategy.id)
        
        optimizations = recovery_engine.optimize_recovery_strategy(strategy.id)
        
        # Verify priority scores are reasonable
        for optimization in optimizations:
            assert 0 <= optimization.priority_score <= 10
            assert isinstance(optimization.priority_score, (int, float))
    
    def test_error_handling_invalid_crisis(self, recovery_engine):
        """Test error handling for invalid crisis data"""
        # This would depend on how the engine handles invalid input
        # For now, we'll test with None
        with pytest.raises(Exception):
            recovery_engine.develop_recovery_strategy(None, ["objective1"])
    
    def test_error_handling_empty_objectives(self, recovery_engine, sample_crisis):
        """Test error handling for empty objectives"""
        # Should handle empty objectives gracefully
        strategy = recovery_engine.develop_recovery_strategy(sample_crisis, [])
        assert isinstance(strategy, RecoveryStrategy)
        assert strategy.recovery_objectives == []
    
    @patch('scrollintel.engines.recovery_planning_engine.logger')
    def test_logging(self, mock_logger, recovery_engine, sample_crisis, recovery_objectives):
        """Test that appropriate logging occurs"""
        # Develop strategy
        recovery_engine.develop_recovery_strategy(sample_crisis, recovery_objectives)
        
        # Verify logging calls
        mock_logger.info.assert_called()
        
        # Check that log messages contain expected content
        log_calls = [call.args[0] for call in mock_logger.info.call_args_list]
        assert any("Recovery strategy developed" in call for call in log_calls)