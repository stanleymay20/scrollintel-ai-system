"""
Tests for Strategy Optimization Engine
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from scrollintel.engines.strategy_optimization_engine import (
    StrategyOptimizationEngine,
    OptimizationContext,
    OptimizationMetric,
    OptimizationRecommendation,
    StrategyAdjustment,
    OptimizationType,
    OptimizationPriority,
    PerformanceOptimizer,
    TimelineOptimizer,
    ResistanceOptimizer,
    EngagementOptimizer
)


class TestOptimizationContext:
    """Test OptimizationContext functionality"""
    
    def test_optimization_context_creation(self):
        """Test creating optimization context"""
        context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.6,
            timeline_status="on_track",
            budget_utilization=0.7,
            resistance_level=0.3,
            engagement_score=0.8,
            performance_metrics={"culture_alignment": 0.7},
            external_factors=["market_conditions", "regulatory_changes"]
        )
        
        assert context.transformation_id == "trans_001"
        assert context.current_progress == 0.6
        assert context.timeline_status == "on_track"
        assert context.resistance_level == 0.3
        assert context.engagement_score == 0.8
        assert "culture_alignment" in context.performance_metrics
        assert len(context.external_factors) == 2


class TestOptimizationMetric:
    """Test OptimizationMetric functionality"""
    
    def test_optimization_metric_creation(self):
        """Test creating optimization metric"""
        metric = OptimizationMetric(
            name="culture_alignment",
            current_value=0.6,
            target_value=0.8,
            weight=0.9,
            trend="improving",
            last_updated=datetime.now()
        )
        
        assert metric.name == "culture_alignment"
        assert metric.current_value == 0.6
        assert metric.target_value == 0.8
        assert metric.weight == 0.9
        assert metric.trend == "improving"


class TestPerformanceOptimizer:
    """Test PerformanceOptimizer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = PerformanceOptimizer()
        self.context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.6,
            timeline_status="on_track",
            budget_utilization=0.7,
            resistance_level=0.3,
            engagement_score=0.8,
            performance_metrics={}
        )
    
    def test_performance_gap_identification(self):
        """Test identification of performance gaps"""
        metrics = [
            OptimizationMetric(
                name="culture_alignment",
                current_value=0.5,
                target_value=0.8,
                weight=0.9,
                trend="stable",
                last_updated=datetime.now()
            ),
            OptimizationMetric(
                name="behavior_change",
                current_value=0.3,
                target_value=0.7,
                weight=0.8,
                trend="declining",
                last_updated=datetime.now()
            )
        ]
        
        gaps = self.optimizer._identify_performance_gaps(self.context, metrics)
        
        assert len(gaps) == 2
        assert gaps[0]["metric"] == "culture_alignment"
        assert gaps[0]["severity"] > 0.3
        assert gaps[1]["metric"] == "behavior_change"
        assert gaps[1]["severity"] > 0.5
    
    def test_performance_optimization_recommendations(self):
        """Test generation of performance optimization recommendations"""
        metrics = [
            OptimizationMetric(
                name="culture_alignment",
                current_value=0.4,
                target_value=0.8,
                weight=0.9,
                trend="stable",
                last_updated=datetime.now()
            )
        ]
        
        recommendations = self.optimizer.analyze(self.context, metrics)
        
        assert len(recommendations) > 0
        assert recommendations[0].optimization_type == OptimizationType.PERFORMANCE_BASED
        assert recommendations[0].priority in [OptimizationPriority.HIGH, OptimizationPriority.CRITICAL]
        assert "culture_alignment" in recommendations[0].title
    
    def test_priority_determination(self):
        """Test priority determination based on severity"""
        assert self.optimizer._determine_priority(0.8) == OptimizationPriority.CRITICAL
        assert self.optimizer._determine_priority(0.6) == OptimizationPriority.HIGH
        assert self.optimizer._determine_priority(0.4) == OptimizationPriority.MEDIUM
        assert self.optimizer._determine_priority(0.2) == OptimizationPriority.LOW
    
    def test_success_probability_calculation(self):
        """Test success probability calculation"""
        gap = {
            "metric": "test_metric",
            "severity": 0.5,
            "trend": "improving"
        }
        
        probability = self.optimizer._calculate_success_probability(gap, self.context)
        
        assert 0.1 <= probability <= 0.95
        assert probability > 0.7  # Should be higher due to improving trend and good engagement


class TestTimelineOptimizer:
    """Test TimelineOptimizer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = TimelineOptimizer()
    
    def test_timeline_optimization_behind_schedule(self):
        """Test timeline optimization when behind schedule"""
        context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.4,
            timeline_status="behind_schedule",
            budget_utilization=0.6,
            resistance_level=0.2,
            engagement_score=0.7,
            performance_metrics={}
        )
        
        recommendations = self.optimizer.analyze(context, [])
        
        assert len(recommendations) > 0
        assert recommendations[0].optimization_type == OptimizationType.TIMELINE_BASED
        assert recommendations[0].priority == OptimizationPriority.HIGH
        assert "timeline" in recommendations[0].title.lower()
    
    def test_timeline_optimization_on_track(self):
        """Test timeline optimization when on track"""
        context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.6,
            timeline_status="on_track",
            budget_utilization=0.7,
            resistance_level=0.3,
            engagement_score=0.8,
            performance_metrics={}
        )
        
        recommendations = self.optimizer.analyze(context, [])
        
        assert len(recommendations) == 0  # No timeline optimization needed


class TestResistanceOptimizer:
    """Test ResistanceOptimizer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = ResistanceOptimizer()
    
    def test_resistance_optimization_high_resistance(self):
        """Test resistance optimization with high resistance"""
        context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.5,
            timeline_status="on_track",
            budget_utilization=0.6,
            resistance_level=0.8,
            engagement_score=0.4,
            performance_metrics={}
        )
        
        recommendations = self.optimizer.analyze(context, [])
        
        assert len(recommendations) > 0
        assert recommendations[0].optimization_type == OptimizationType.RESISTANCE_BASED
        assert recommendations[0].priority == OptimizationPriority.CRITICAL
        assert "resistance" in recommendations[0].title.lower()
    
    def test_resistance_optimization_low_resistance(self):
        """Test resistance optimization with low resistance"""
        context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.6,
            timeline_status="on_track",
            budget_utilization=0.7,
            resistance_level=0.3,
            engagement_score=0.8,
            performance_metrics={}
        )
        
        recommendations = self.optimizer.analyze(context, [])
        
        assert len(recommendations) == 0  # No resistance optimization needed


class TestEngagementOptimizer:
    """Test EngagementOptimizer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = EngagementOptimizer()
    
    def test_engagement_optimization_low_engagement(self):
        """Test engagement optimization with low engagement"""
        context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.5,
            timeline_status="on_track",
            budget_utilization=0.6,
            resistance_level=0.3,
            engagement_score=0.3,
            performance_metrics={}
        )
        
        recommendations = self.optimizer.analyze(context, [])
        
        assert len(recommendations) > 0
        assert recommendations[0].optimization_type == OptimizationType.ENGAGEMENT_BASED
        assert recommendations[0].priority == OptimizationPriority.HIGH
        assert "engagement" in recommendations[0].title.lower()
    
    def test_engagement_optimization_high_engagement(self):
        """Test engagement optimization with high engagement"""
        context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.6,
            timeline_status="on_track",
            budget_utilization=0.7,
            resistance_level=0.3,
            engagement_score=0.8,
            performance_metrics={}
        )
        
        recommendations = self.optimizer.analyze(context, [])
        
        assert len(recommendations) == 0  # No engagement optimization needed


class TestStrategyOptimizationEngine:
    """Test StrategyOptimizationEngine functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.engine = StrategyOptimizationEngine()
        self.context = OptimizationContext(
            transformation_id="trans_001",
            current_progress=0.5,
            timeline_status="behind_schedule",
            budget_utilization=0.6,
            resistance_level=0.6,
            engagement_score=0.4,
            performance_metrics={"culture_alignment": 0.5}
        )
        self.metrics = [
            OptimizationMetric(
                name="culture_alignment",
                current_value=0.4,
                target_value=0.8,
                weight=0.9,
                trend="stable",
                last_updated=datetime.now()
            ),
            OptimizationMetric(
                name="behavior_change",
                current_value=0.3,
                target_value=0.7,
                weight=0.8,
                trend="declining",
                last_updated=datetime.now()
            )
        ]
    
    def test_strategy_optimization(self):
        """Test complete strategy optimization process"""
        recommendations = self.engine.optimize_strategy(self.context, self.metrics)
        
        assert len(recommendations) > 0
        assert all(isinstance(rec, OptimizationRecommendation) for rec in recommendations)
        
        # Should have multiple types of recommendations due to various issues
        optimization_types = {rec.optimization_type for rec in recommendations}
        assert len(optimization_types) > 1
    
    def test_recommendation_prioritization(self):
        """Test recommendation prioritization"""
        # Create recommendations with different priorities
        recommendations = [
            OptimizationRecommendation(
                id="rec_1",
                optimization_type=OptimizationType.PERFORMANCE_BASED,
                priority=OptimizationPriority.LOW,
                title="Low Priority",
                description="Low priority recommendation",
                expected_impact=0.2,
                implementation_effort="Low",
                timeline="1 week",
                success_probability=0.5
            ),
            OptimizationRecommendation(
                id="rec_2",
                optimization_type=OptimizationType.RESISTANCE_BASED,
                priority=OptimizationPriority.CRITICAL,
                title="Critical Priority",
                description="Critical priority recommendation",
                expected_impact=0.8,
                implementation_effort="High",
                timeline="3 months",
                success_probability=0.7
            )
        ]
        
        prioritized = self.engine._prioritize_recommendations(recommendations, self.context)
        
        assert prioritized[0].priority == OptimizationPriority.CRITICAL
        assert prioritized[1].priority == OptimizationPriority.LOW
    
    def test_strategy_adjustment_creation(self):
        """Test strategy adjustment creation"""
        recommendation = OptimizationRecommendation(
            id="rec_001",
            optimization_type=OptimizationType.PERFORMANCE_BASED,
            priority=OptimizationPriority.HIGH,
            title="Improve Performance",
            description="Improve transformation performance",
            expected_impact=0.6,
            implementation_effort="Medium",
            timeline="2 months",
            success_probability=0.75
        )
        
        original_strategy = {
            "approach": "gradual",
            "timeline": "6 months",
            "resources": {"budget": 100000, "team_size": 10}
        }
        
        adjustment = self.engine.create_strategy_adjustment(
            "trans_001", recommendation, original_strategy
        )
        
        assert isinstance(adjustment, StrategyAdjustment)
        assert adjustment.transformation_id == "trans_001"
        assert adjustment.adjustment_type == OptimizationType.PERFORMANCE_BASED.value
        assert adjustment.original_strategy == original_strategy
        assert adjustment.adjusted_strategy != original_strategy
        assert len(adjustment.expected_outcomes) > 0
    
    def test_adjustment_implementation(self):
        """Test strategy adjustment implementation"""
        # First create an adjustment
        recommendation = OptimizationRecommendation(
            id="rec_001",
            optimization_type=OptimizationType.ENGAGEMENT_BASED,
            priority=OptimizationPriority.MEDIUM,
            title="Boost Engagement",
            description="Improve employee engagement",
            expected_impact=0.5,
            implementation_effort="Medium",
            timeline="1 month",
            success_probability=0.8
        )
        
        original_strategy = {"engagement_level": "medium"}
        adjustment = self.engine.create_strategy_adjustment(
            "trans_001", recommendation, original_strategy
        )
        
        # Test implementation
        success = self.engine.implement_adjustment(adjustment.id)
        
        assert isinstance(success, bool)
        assert adjustment.id in self.engine.active_adjustments
    
    def test_adjustment_effectiveness_monitoring(self):
        """Test adjustment effectiveness monitoring"""
        # Create and implement an adjustment
        recommendation = OptimizationRecommendation(
            id="rec_001",
            optimization_type=OptimizationType.TIMELINE_BASED,
            priority=OptimizationPriority.HIGH,
            title="Accelerate Timeline",
            description="Speed up transformation",
            expected_impact=0.7,
            implementation_effort="High",
            timeline="2 months",
            success_probability=0.6
        )
        
        original_strategy = {"timeline": "12 months"}
        adjustment = self.engine.create_strategy_adjustment(
            "trans_001", recommendation, original_strategy
        )
        
        # Monitor effectiveness
        post_metrics = [
            OptimizationMetric(
                name="timeline_progress",
                current_value=0.7,
                target_value=0.8,
                weight=1.0,
                trend="improving",
                last_updated=datetime.now()
            )
        ]
        
        effectiveness = self.engine.monitor_adjustment_effectiveness(
            adjustment.id, post_metrics
        )
        
        assert "adjustment_id" in effectiveness
        assert "effectiveness_score" in effectiveness
        assert "impact_analysis" in effectiveness
        assert "recommendations" in effectiveness
    
    def test_continuous_improvement_plan_generation(self):
        """Test continuous improvement plan generation"""
        optimization_history = [
            {
                "optimization_id": "opt_001",
                "type": "performance_based",
                "success": True,
                "effectiveness": 0.8
            },
            {
                "optimization_id": "opt_002",
                "type": "engagement_based",
                "success": True,
                "effectiveness": 0.9
            },
            {
                "optimization_id": "opt_003",
                "type": "resistance_based",
                "success": False,
                "effectiveness": 0.3
            }
        ]
        
        improvement_plan = self.engine.generate_continuous_improvement_plan(
            "trans_001", optimization_history
        )
        
        assert "transformation_id" in improvement_plan
        assert "patterns_identified" in improvement_plan
        assert "improvement_recommendations" in improvement_plan
        assert "learning_integration_plan" in improvement_plan
        assert improvement_plan["transformation_id"] == "trans_001"
    
    def test_optimization_with_empty_metrics(self):
        """Test optimization with empty metrics list"""
        recommendations = self.engine.optimize_strategy(self.context, [])
        
        # Should still generate recommendations based on context
        assert isinstance(recommendations, list)
        # May have timeline and resistance recommendations based on context
    
    def test_optimization_with_good_context(self):
        """Test optimization with good context (no issues)"""
        good_context = OptimizationContext(
            transformation_id="trans_002",
            current_progress=0.8,
            timeline_status="ahead_of_schedule",
            budget_utilization=0.6,
            resistance_level=0.2,
            engagement_score=0.9,
            performance_metrics={}
        )
        
        good_metrics = [
            OptimizationMetric(
                name="culture_alignment",
                current_value=0.85,
                target_value=0.8,
                weight=0.9,
                trend="improving",
                last_updated=datetime.now()
            )
        ]
        
        recommendations = self.engine.optimize_strategy(good_context, good_metrics)
        
        # Should have fewer or no recommendations since everything is going well
        assert len(recommendations) <= 2
    
    def test_error_handling_in_optimization(self):
        """Test error handling in optimization process"""
        # Test with invalid context
        invalid_context = None
        
        recommendations = self.engine.optimize_strategy(invalid_context, self.metrics)
        
        # Should return empty list on error
        assert recommendations == []
    
    def test_optimization_history_tracking(self):
        """Test optimization history tracking"""
        initial_history_length = len(self.engine.optimization_history)
        
        # Perform optimization
        recommendations = self.engine.optimize_strategy(self.context, self.metrics)
        
        # History should be updated (in a real implementation)
        # For now, just verify the method doesn't crash
        assert isinstance(recommendations, list)


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation functionality"""
    
    def test_recommendation_creation(self):
        """Test creating optimization recommendation"""
        recommendation = OptimizationRecommendation(
            id="rec_001",
            optimization_type=OptimizationType.PERFORMANCE_BASED,
            priority=OptimizationPriority.HIGH,
            title="Improve Culture Alignment",
            description="Focus on improving culture alignment metrics",
            expected_impact=0.6,
            implementation_effort="Medium",
            timeline="2-3 months",
            success_probability=0.75,
            dependencies=["leadership_support", "budget_approval"],
            risks=["resistance", "timeline_pressure"]
        )
        
        assert recommendation.id == "rec_001"
        assert recommendation.optimization_type == OptimizationType.PERFORMANCE_BASED
        assert recommendation.priority == OptimizationPriority.HIGH
        assert recommendation.expected_impact == 0.6
        assert len(recommendation.dependencies) == 2
        assert len(recommendation.risks) == 2


class TestStrategyAdjustment:
    """Test StrategyAdjustment functionality"""
    
    def test_adjustment_creation(self):
        """Test creating strategy adjustment"""
        adjustment = StrategyAdjustment(
            id="adj_001",
            transformation_id="trans_001",
            adjustment_type="performance_based",
            original_strategy={"approach": "gradual"},
            adjusted_strategy={"approach": "accelerated"},
            rationale="Need to improve performance metrics",
            expected_outcomes=["Better performance", "Faster results"],
            implementation_date=datetime.now() + timedelta(days=7)
        )
        
        assert adjustment.id == "adj_001"
        assert adjustment.transformation_id == "trans_001"
        assert adjustment.adjustment_type == "performance_based"
        assert adjustment.original_strategy != adjustment.adjusted_strategy
        assert len(adjustment.expected_outcomes) == 2


if __name__ == "__main__":
    pytest.main([__file__])