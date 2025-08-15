"""
Tests for Learning Optimization System

This module contains comprehensive tests for the learning optimization system,
including continuous learning optimization, effectiveness measurement, and adaptive learning.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from scrollintel.engines.learning_optimization_system import LearningOptimizationSystem
from scrollintel.models.knowledge_integration_models import (
    LearningMetric, LearningOptimization, ConfidenceLevel
)


class TestLearningOptimizationSystem:
    """Test cases for Learning Optimization System"""
    
    @pytest.fixture
    def system(self):
        """Create a learning optimization system instance"""
        return LearningOptimizationSystem()
    
    @pytest.fixture
    def sample_learning_metrics(self):
        """Sample learning metrics for testing"""
        return [
            LearningMetric(
                metric_name="learning_speed",
                value=0.75,
                timestamp=datetime.now() - timedelta(days=1),
                context={"activity": "deep_learning_course", "difficulty": "medium"},
                improvement_rate=0.05
            ),
            LearningMetric(
                metric_name="knowledge_retention",
                value=0.82,
                timestamp=datetime.now() - timedelta(hours=12),
                context={"activity": "practical_exercises", "difficulty": "high"},
                improvement_rate=0.03
            ),
            LearningMetric(
                metric_name="application_success",
                value=0.68,
                timestamp=datetime.now() - timedelta(hours=6),
                context={"activity": "project_implementation", "difficulty": "high"},
                improvement_rate=-0.02
            ),
            LearningMetric(
                metric_name="learning_speed",
                value=0.78,
                timestamp=datetime.now(),
                context={"activity": "advanced_topics", "difficulty": "high"},
                improvement_rate=0.04
            )
        ]
    
    @pytest.fixture
    def sample_learning_activities(self):
        """Sample learning activities for testing"""
        return [
            {
                "name": "machine_learning_fundamentals",
                "type": "hands_on",
                "duration_hours": 40,
                "feedback_available": True,
                "interactive": True,
                "complexity": "medium",
                "reinforcement": True,
                "practical_application": True
            },
            {
                "name": "advanced_algorithms",
                "type": "theoretical",
                "duration_hours": 30,
                "feedback_available": False,
                "interactive": False,
                "complexity": "high",
                "reinforcement": False,
                "practical_application": False
            },
            {
                "name": "project_based_learning",
                "type": "hands_on",
                "duration_hours": 60,
                "feedback_available": True,
                "interactive": True,
                "complexity": "high",
                "reinforcement": True,
                "practical_application": True,
                "practical_focus": True,
                "mentorship": True
            }
        ]
    
    @pytest.mark.asyncio
    async def test_optimize_continuous_learning(self, system, sample_learning_metrics):
        """Test continuous learning optimization"""
        learning_context = {
            "domain": "artificial_intelligence",
            "learner_level": "intermediate",
            "available_time_hours": 20,
            "preferred_learning_style": "hands_on"
        }
        
        optimization_targets = ["learning_speed", "knowledge_retention"]
        
        # Optimize continuous learning
        optimization = await system.optimize_continuous_learning(
            learning_context, sample_learning_metrics, optimization_targets
        )
        
        # Verify optimization result
        assert isinstance(optimization, LearningOptimization)
        assert optimization.id is not None
        assert optimization.optimization_target == "learning_speed, knowledge_retention"
        assert optimization.optimization_strategy in ["aggressive_optimization", "balanced_optimization", "fine_tuning", "maintenance"]
        assert isinstance(optimization.parameters, dict)
        assert 0.0 <= optimization.effectiveness_score <= 1.0
        assert len(optimization.current_metrics) == len(sample_learning_metrics)
        
        # Verify optimization is stored
        assert optimization.id in system.learning_optimizations
        
        # Verify metrics history is updated
        assert "learning_speed" in system.learning_metrics_history
        assert "knowledge_retention" in system.learning_metrics_history
    
    @pytest.mark.asyncio
    async def test_measure_learning_effectiveness(self, system, sample_learning_activities):
        """Test learning effectiveness measurement"""
        time_window = timedelta(days=30)
        
        # Measure learning effectiveness
        effectiveness = await system.measure_learning_effectiveness(
            sample_learning_activities, time_window
        )
        
        # Verify effectiveness measurement structure
        assert "overall_effectiveness" in effectiveness
        assert "activity_effectiveness" in effectiveness
        assert "improvement_trends" in effectiveness
        assert "learning_velocity" in effectiveness
        assert "knowledge_retention" in effectiveness
        assert "application_success" in effectiveness
        assert "measurement_timestamp" in effectiveness
        
        # Verify effectiveness values are within valid range
        assert 0.0 <= effectiveness["overall_effectiveness"] <= 1.0
        assert 0.0 <= effectiveness["learning_velocity"] <= 10.0
        assert 0.0 <= effectiveness["knowledge_retention"] <= 1.0
        assert 0.0 <= effectiveness["application_success"] <= 1.0
        
        # Verify activity-specific effectiveness
        assert len(effectiveness["activity_effectiveness"]) == len(sample_learning_activities)
        
        for activity_name, activity_eff in effectiveness["activity_effectiveness"].items():
            assert "effectiveness_score" in activity_eff
            assert 0.0 <= activity_eff["effectiveness_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_implement_adaptive_learning(self, system):
        """Test adaptive learning implementation"""
        learning_context = {
            "current_performance": 0.65,
            "learning_goals": ["improve_accuracy", "increase_speed"],
            "constraints": {"time_budget": 40, "resource_budget": 5000}
        }
        
        performance_feedback = [
            {
                "feedback_type": "performance_review",
                "sentiment": "negative",
                "issue": "slow_learning_progress",
                "timestamp": datetime.now() - timedelta(days=2)
            },
            {
                "feedback_type": "peer_review",
                "sentiment": "positive",
                "strength": "good_theoretical_understanding",
                "timestamp": datetime.now() - timedelta(days=1)
            },
            {
                "feedback_type": "self_assessment",
                "sentiment": "neutral",
                "issue": "difficulty_with_practical_application",
                "timestamp": datetime.now()
            }
        ]
        
        adaptation_goals = ["improve_practical_skills", "increase_learning_speed"]
        
        # Implement adaptive learning
        adaptive_result = await system.implement_adaptive_learning(
            learning_context, performance_feedback, adaptation_goals
        )
        
        # Verify adaptive learning result structure
        assert "adaptation_id" in adaptive_result
        assert "adaptation_needs" in adaptive_result
        assert "adaptive_strategies" in adaptive_result
        assert "implementation_results" in adaptive_result
        assert "adaptation_monitoring" in adaptive_result
        assert "effectiveness_improvement" in adaptive_result
        assert "adaptation_timestamp" in adaptive_result
        
        # Verify adaptation needs are identified
        assert len(adaptive_result["adaptation_needs"]) > 0
        
        # Verify adaptive strategies are generated
        assert len(adaptive_result["adaptive_strategies"]) > 0
        
        # Verify implementation results
        impl_results = adaptive_result["implementation_results"]
        assert "implemented_strategies" in impl_results
        assert "implementation_success" in impl_results
        assert "adaptation_metrics" in impl_results
    
    @pytest.mark.asyncio
    async def test_enhance_innovation_processes(self, system):
        """Test innovation process enhancement"""
        process_data = [
            {
                "name": "idea_generation",
                "efficiency": 0.75,
                "bottlenecks": ["limited_creativity_tools"],
                "duration_days": 5
            },
            {
                "name": "prototype_development",
                "efficiency": 0.45,  # Low efficiency - should be identified as bottleneck
                "bottlenecks": ["resource_constraints", "skill_gaps"],
                "duration_days": 15
            },
            {
                "name": "validation_testing",
                "efficiency": 0.85,  # High efficiency - should be identified as strength
                "bottlenecks": [],
                "duration_days": 8
            }
        ]
        
        enhancement_objectives = [
            "reduce_development_time",
            "improve_prototype_quality",
            "increase_innovation_throughput"
        ]
        
        # Enhance innovation processes
        enhancement_result = await system.enhance_innovation_processes(
            process_data, enhancement_objectives
        )
        
        # Verify enhancement result structure
        assert "enhancement_id" in enhancement_result
        assert "process_analysis" in enhancement_result
        assert "enhancement_opportunities" in enhancement_result
        assert "prioritized_enhancements" in enhancement_result
        assert "implementation_plan" in enhancement_result
        assert "impact_estimation" in enhancement_result
        assert "enhancement_timestamp" in enhancement_result
        
        # Verify process analysis
        analysis = enhancement_result["process_analysis"]
        assert "process_efficiency" in analysis
        assert "bottlenecks" in analysis
        assert "strengths" in analysis
        
        # Should identify prototype_development as bottleneck
        bottlenecks = analysis["bottlenecks"]
        assert any(b["process"] == "prototype_development" for b in bottlenecks)
        
        # Should identify validation_testing as strength
        strengths = analysis["strengths"]
        assert any(s["process"] == "validation_testing" for s in strengths)
        
        # Verify enhancement opportunities
        opportunities = enhancement_result["enhancement_opportunities"]
        assert len(opportunities) > 0
        
        # Verify implementation plan
        plan = enhancement_result["implementation_plan"]
        assert "phases" in plan
        assert "timeline" in plan
    
    @pytest.mark.asyncio
    async def test_optimize_learning_parameters(self, system, sample_learning_metrics):
        """Test learning parameter optimization"""
        # First create an optimization
        learning_context = {"domain": "test"}
        optimization_targets = ["learning_speed"]
        
        optimization = await system.optimize_continuous_learning(
            learning_context, sample_learning_metrics, optimization_targets
        )
        
        # Prepare performance data
        performance_data = [
            {"performance_score": 0.65, "timestamp": datetime.now() - timedelta(days=2)},
            {"performance_score": 0.68, "timestamp": datetime.now() - timedelta(days=1)},
            {"performance_score": 0.70, "timestamp": datetime.now()}
        ]
        
        # Optimize parameters
        optimization_result = await system.optimize_learning_parameters(
            optimization.id, performance_data
        )
        
        # Verify optimization result
        assert "optimization_id" in optimization_result
        assert "parameter_adjustments" in optimization_result
        assert "optimized_parameters" in optimization_result
        assert "improvement_metrics" in optimization_result
        assert "new_effectiveness_score" in optimization_result
        
        # Verify optimization is updated
        updated_optimization = system.learning_optimizations[optimization.id]
        assert len(updated_optimization.improvements) > 0
        assert updated_optimization.last_updated > optimization.created_at
    
    @pytest.mark.asyncio
    async def test_parameter_optimization_with_invalid_id(self, system):
        """Test parameter optimization with invalid optimization ID"""
        performance_data = [{"performance_score": 0.5}]
        
        with pytest.raises(ValueError, match="Learning optimization .* not found"):
            await system.optimize_learning_parameters("invalid_id", performance_data)
    
    @pytest.mark.asyncio
    async def test_empty_metrics_handling(self, system):
        """Test handling of empty metrics"""
        learning_context = {"domain": "test"}
        optimization_targets = ["test_metric"]
        
        # Test with empty metrics
        optimization = await system.optimize_continuous_learning(
            learning_context, [], optimization_targets
        )
        
        # Should still create optimization
        assert isinstance(optimization, LearningOptimization)
        assert len(optimization.current_metrics) == 0
    
    @pytest.mark.asyncio
    async def test_empty_activities_handling(self, system):
        """Test handling of empty learning activities"""
        # Test with empty activities
        effectiveness = await system.measure_learning_effectiveness([], timedelta(days=30))
        
        # Should return valid structure with zero values
        assert effectiveness["overall_effectiveness"] == 0.0
        assert effectiveness["learning_velocity"] == 0.0
        assert len(effectiveness["activity_effectiveness"]) == 0
    
    @pytest.mark.asyncio
    async def test_metrics_history_tracking(self, system, sample_learning_metrics):
        """Test metrics history tracking"""
        learning_context = {"domain": "test"}
        optimization_targets = ["learning_speed", "knowledge_retention"]
        
        # Create optimization (should update metrics history)
        await system.optimize_continuous_learning(
            learning_context, sample_learning_metrics, optimization_targets
        )
        
        # Verify metrics history is populated
        assert "learning_speed" in system.learning_metrics_history
        assert "knowledge_retention" in system.learning_metrics_history
        assert "application_success" in system.learning_metrics_history
        
        # Verify correct number of metrics for each type
        learning_speed_metrics = list(system.learning_metrics_history["learning_speed"])
        assert len(learning_speed_metrics) == 2  # Two learning_speed metrics in sample data
        
        knowledge_retention_metrics = list(system.learning_metrics_history["knowledge_retention"])
        assert len(knowledge_retention_metrics) == 1
    
    @pytest.mark.asyncio
    async def test_optimization_strategy_selection(self, system):
        """Test optimization strategy selection based on metrics"""
        learning_context = {"domain": "test"}
        
        # Test with low performance metrics (should trigger aggressive optimization)
        low_performance_metrics = [
            LearningMetric("test_metric", 0.3, datetime.now(), {}, 0.0)
        ]
        
        optimization = await system.optimize_continuous_learning(
            learning_context, low_performance_metrics, ["test_metric"]
        )
        
        # Should select aggressive optimization for low performance
        assert optimization.optimization_strategy in ["aggressive_optimization", "balanced_optimization"]
        
        # Test with high performance metrics (should trigger fine tuning)
        high_performance_metrics = [
            LearningMetric("test_metric", 0.9, datetime.now(), {}, 0.0)
        ]
        
        optimization2 = await system.optimize_continuous_learning(
            learning_context, high_performance_metrics, ["test_metric"]
        )
        
        # Should select fine tuning or maintenance for high performance
        assert optimization2.optimization_strategy in ["fine_tuning", "maintenance", "balanced_optimization"]
    
    @pytest.mark.asyncio
    async def test_adaptive_parameters_update(self, system):
        """Test adaptive parameters update"""
        learning_context = {"domain": "test"}
        performance_feedback = [
            {"sentiment": "positive", "strength": "good_progress"}
        ]
        adaptation_goals = ["improve_efficiency"]
        
        # Implement adaptive learning
        await system.implement_adaptive_learning(
            learning_context, performance_feedback, adaptation_goals
        )
        
        # Verify adaptive parameters are updated
        assert len(system.adaptive_parameters) > 0
        
        # Get latest parameters
        latest_timestamp = max(system.adaptive_parameters.keys())
        latest_params = system.adaptive_parameters[latest_timestamp]
        
        # Verify parameter structure
        expected_params = [
            "adaptation_sensitivity",
            "learning_aggressiveness", 
            "stability_preference",
            "exploration_tendency"
        ]
        
        for param in expected_params:
            assert param in latest_params
            assert 0.0 <= latest_params[param] <= 1.0
    
    @pytest.mark.asyncio
    async def test_effectiveness_measurement_components(self, system, sample_learning_activities):
        """Test individual components of effectiveness measurement"""
        time_window = timedelta(days=30)
        
        effectiveness = await system.measure_learning_effectiveness(
            sample_learning_activities, time_window
        )
        
        # Test improvement trends
        trends = effectiveness["improvement_trends"]
        assert isinstance(trends, dict)
        
        for activity_name in [activity["name"] for activity in sample_learning_activities]:
            if activity_name in trends:
                trend = trends[activity_name]
                assert "direction" in trend
                assert trend["direction"] in ["improving", "stable", "declining"]
                assert "strength" in trend
                assert 0.0 <= trend["strength"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_process_enhancement_prioritization(self, system):
        """Test process enhancement prioritization"""
        process_data = [
            {"name": "high_priority_process", "efficiency": 0.3},  # Very low efficiency
            {"name": "medium_priority_process", "efficiency": 0.6},  # Medium efficiency
            {"name": "low_priority_process", "efficiency": 0.8}   # High efficiency
        ]
        
        enhancement_objectives = ["improve_efficiency"]
        
        enhancement_result = await system.enhance_innovation_processes(
            process_data, enhancement_objectives
        )
        
        # Verify prioritization
        prioritized = enhancement_result["prioritized_enhancements"]
        
        # Should prioritize based on improvement potential and priority
        if prioritized:
            # First enhancement should be high priority (addressing low efficiency)
            first_enhancement = prioritized[0]
            assert first_enhancement.get("priority") in ["high", "medium"]
    
    @pytest.mark.asyncio
    async def test_learning_velocity_calculation(self, system):
        """Test learning velocity calculation"""
        # Test with different activity complexities
        activities_simple = [
            {"name": "simple_task", "complexity": "low"},
            {"name": "simple_task_2", "complexity": "low"}
        ]
        
        activities_complex = [
            {"name": "complex_task", "complexity": "high"},
            {"name": "complex_task_2", "complexity": "high"}
        ]
        
        time_window = timedelta(days=2)
        
        # Measure effectiveness for both
        effectiveness_simple = await system.measure_learning_effectiveness(activities_simple, time_window)
        effectiveness_complex = await system.measure_learning_effectiveness(activities_complex, time_window)
        
        # Both should have valid learning velocities
        assert effectiveness_simple["learning_velocity"] >= 0.0
        assert effectiveness_complex["learning_velocity"] >= 0.0
        
        # Complex activities might have different velocity characteristics
        # (exact relationship depends on implementation details)
    
    def test_initialization(self, system):
        """Test system initialization"""
        assert isinstance(system.learning_optimizations, dict)
        assert isinstance(system.learning_metrics_history, dict)
        assert isinstance(system.optimization_strategies, dict)
        assert isinstance(system.adaptive_parameters, dict)
        assert isinstance(system.performance_baselines, dict)
        
        # All stores should be empty initially
        assert len(system.learning_optimizations) == 0
        assert len(system.learning_metrics_history) == 0
        assert len(system.optimization_strategies) == 0
        assert len(system.adaptive_parameters) == 0
        assert len(system.performance_baselines) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, system):
        """Test error handling in various scenarios"""
        # Test with invalid learning context
        try:
            await system.optimize_continuous_learning({}, [], [])
            # Should not raise exception for empty inputs
        except Exception as e:
            pytest.fail(f"Should handle empty inputs gracefully: {str(e)}")
        
        # Test with invalid performance data
        try:
            await system.measure_learning_effectiveness([], timedelta(days=0))
            # Should handle zero time window
        except Exception as e:
            pytest.fail(f"Should handle zero time window gracefully: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])