"""
Tests for Strategic Recommendation Engine

Tests for strategic recommendation development, quality assessment,
optimization, and validation functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.strategic_recommendation_engine import (
    StrategicRecommendationEngine,
    RecommendationType,
    PriorityLevel,
    ImpactArea,
    BoardPriority,
    StrategicRecommendation
)

class TestStrategicRecommendationEngine:
    
    @pytest.fixture
    def engine(self):
        """Create a strategic recommendation engine instance"""
        return StrategicRecommendationEngine()
    
    @pytest.fixture
    def sample_board_priority(self):
        """Create a sample board priority"""
        return BoardPriority(
            id="priority_001",
            title="Digital Transformation Initiative",
            description="Accelerate digital transformation across all business units",
            priority_level=PriorityLevel.HIGH,
            impact_areas=[ImpactArea.EFFICIENCY, ImpactArea.COMPETITIVE_ADVANTAGE],
            target_timeline="12 months",
            success_metrics=["Operational efficiency increase by 25%", "Customer satisfaction score > 4.5"],
            stakeholders=["CTO", "COO", "Board Chair"]
        )
    
    @pytest.fixture
    def sample_strategic_context(self):
        """Create sample strategic context"""
        return {
            'challenges': 'legacy system limitations and manual processes',
            'value_creation': 'automation and improved customer experience',
            'outcomes': 'increased efficiency and market competitiveness',
            'base_revenue': 5000000,
            'base_cost': 4000000,
            'budget': 1000000,
            'team_size': 12
        }
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert len(engine.board_priorities) == 0
        assert len(engine.recommendations) == 0
        assert engine.quality_thresholds['minimum_score'] == 0.7
    
    def test_add_board_priority(self, engine, sample_board_priority):
        """Test adding board priority"""
        engine.add_board_priority(sample_board_priority)
        
        assert len(engine.board_priorities) == 1
        assert engine.board_priorities[0].id == "priority_001"
        assert engine.board_priorities[0].title == "Digital Transformation Initiative"
    
    def test_create_strategic_recommendation(self, engine, sample_board_priority, sample_strategic_context):
        """Test creating strategic recommendation"""
        # Add board priority first
        engine.add_board_priority(sample_board_priority)
        
        # Create recommendation
        recommendation = engine.create_strategic_recommendation(
            title="Cloud Migration Initiative",
            recommendation_type=RecommendationType.TECHNOLOGY_INVESTMENT,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        assert recommendation is not None
        assert recommendation.title == "Cloud Migration Initiative"
        assert recommendation.recommendation_type == RecommendationType.TECHNOLOGY_INVESTMENT
        assert "priority_001" in recommendation.board_priorities
        assert recommendation.quality_score > 0
        assert recommendation.validation_status == "pending"
        assert len(engine.recommendations) == 1
    
    def test_strategic_rationale_generation(self, engine, sample_board_priority, sample_strategic_context):
        """Test strategic rationale generation"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation = engine.create_strategic_recommendation(
            title="AI Implementation Strategy",
            recommendation_type=RecommendationType.STRATEGIC_INITIATIVE,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        assert "Digital Transformation Initiative" in recommendation.strategic_rationale
        assert "legacy system limitations" in recommendation.strategic_rationale
        assert "automation and improved customer experience" in recommendation.strategic_rationale
    
    def test_financial_impact_assessment(self, engine, sample_board_priority, sample_strategic_context):
        """Test financial impact assessment"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation = engine.create_strategic_recommendation(
            title="Technology Investment",
            recommendation_type=RecommendationType.TECHNOLOGY_INVESTMENT,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        financial_impact = recommendation.financial_impact
        assert financial_impact.revenue_impact > 0
        assert financial_impact.cost_impact > 0
        assert financial_impact.roi_projection > 0
        assert financial_impact.payback_period > 0
        assert financial_impact.confidence_level == 0.8
        assert len(financial_impact.assumptions) > 0
    
    def test_risk_assessment(self, engine, sample_board_priority, sample_strategic_context):
        """Test risk assessment"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation = engine.create_strategic_recommendation(
            title="Market Expansion",
            recommendation_type=RecommendationType.MARKET_EXPANSION,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        risk_assessment = recommendation.risk_assessment
        assert risk_assessment.risk_level == "high"
        assert len(risk_assessment.key_risks) > 0
        assert len(risk_assessment.mitigation_strategies) > 0
        assert 0 < risk_assessment.success_probability <= 1
        assert len(risk_assessment.contingency_plans) > 0
    
    def test_implementation_plan_creation(self, engine, sample_board_priority, sample_strategic_context):
        """Test implementation plan creation"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation = engine.create_strategic_recommendation(
            title="Operational Improvement",
            recommendation_type=RecommendationType.OPERATIONAL_IMPROVEMENT,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        impl_plan = recommendation.implementation_plan
        assert len(impl_plan.phases) == 3
        assert impl_plan.timeline == "9 months"
        assert 'budget' in impl_plan.resource_requirements
        assert len(impl_plan.dependencies) > 0
        assert len(impl_plan.milestones) > 0
        assert len(impl_plan.success_criteria) > 0
    
    def test_impact_prediction(self, engine, sample_board_priority, sample_strategic_context):
        """Test impact prediction"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation = engine.create_strategic_recommendation(
            title="Strategic Initiative",
            recommendation_type=RecommendationType.STRATEGIC_INITIATIVE,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        impact_prediction = recommendation.impact_prediction
        assert 'financial' in impact_prediction
        assert 'strategic_alignment' in impact_prediction
        assert 'operational' in impact_prediction
        assert 'market' in impact_prediction
        assert 'risk_reduction' in impact_prediction
        
        # All impact scores should be between 0 and 1
        for score in impact_prediction.values():
            assert 0 <= score <= 1
    
    def test_quality_assessment(self, engine, sample_board_priority, sample_strategic_context):
        """Test recommendation quality assessment"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation = engine.create_strategic_recommendation(
            title="High Quality Recommendation",
            recommendation_type=RecommendationType.TECHNOLOGY_INVESTMENT,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        # Quality score should be calculated
        assert recommendation.quality_score > 0
        assert recommendation.quality_score <= 1
        
        # Should meet minimum threshold for well-aligned recommendation
        assert recommendation.quality_score >= engine.quality_thresholds['minimum_score']
    
    def test_recommendation_optimization(self, engine, sample_board_priority, sample_strategic_context):
        """Test recommendation optimization"""
        engine.add_board_priority(sample_board_priority)
        
        # Create a recommendation with lower quality
        recommendation = engine.create_strategic_recommendation(
            title="Optimization Test",
            recommendation_type=RecommendationType.COST_OPTIMIZATION,
            strategic_context={'base_revenue': 100000, 'base_cost': 90000},  # Lower impact context
            target_priorities=[]  # No priorities for lower alignment
        )
        
        original_score = recommendation.quality_score
        
        # Optimize the recommendation
        optimized = engine.optimize_recommendation(recommendation.id)
        
        # Quality should improve
        assert optimized.quality_score >= original_score
        assert optimized.updated_at > recommendation.created_at
    
    def test_recommendation_validation(self, engine, sample_board_priority, sample_strategic_context):
        """Test recommendation validation"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation = engine.create_strategic_recommendation(
            title="Validation Test",
            recommendation_type=RecommendationType.STRATEGIC_INITIATIVE,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        validation_results = engine.validate_recommendation(recommendation.id)
        
        assert validation_results['recommendation_id'] == recommendation.id
        assert 'validation_status' in validation_results
        assert 'quality_score' in validation_results
        assert 'meets_threshold' in validation_results
        assert 'validation_details' in validation_results
        assert 'recommendations_for_improvement' in validation_results
        
        # Validation details should include key areas
        details = validation_results['validation_details']
        assert 'strategic_alignment' in details
        assert 'financial_viability' in details
        assert 'implementation_feasibility' in details
    
    def test_get_recommendations_by_priority(self, engine, sample_board_priority, sample_strategic_context):
        """Test getting recommendations by priority"""
        engine.add_board_priority(sample_board_priority)
        
        # Create multiple recommendations
        rec1 = engine.create_strategic_recommendation(
            title="Recommendation 1",
            recommendation_type=RecommendationType.STRATEGIC_INITIATIVE,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        rec2 = engine.create_strategic_recommendation(
            title="Recommendation 2",
            recommendation_type=RecommendationType.TECHNOLOGY_INVESTMENT,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        rec3 = engine.create_strategic_recommendation(
            title="Recommendation 3",
            recommendation_type=RecommendationType.MARKET_EXPANSION,
            strategic_context=sample_strategic_context,
            target_priorities=[]  # No priority alignment
        )
        
        # Get recommendations by priority
        priority_recs = engine.get_recommendations_by_priority("priority_001")
        
        assert len(priority_recs) == 2
        assert rec1 in priority_recs
        assert rec2 in priority_recs
        assert rec3 not in priority_recs
    
    def test_get_high_quality_recommendations(self, engine, sample_board_priority, sample_strategic_context):
        """Test getting high quality recommendations"""
        engine.add_board_priority(sample_board_priority)
        
        # Create recommendations with different quality levels
        high_quality_rec = engine.create_strategic_recommendation(
            title="High Quality",
            recommendation_type=RecommendationType.STRATEGIC_INITIATIVE,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        low_quality_rec = engine.create_strategic_recommendation(
            title="Low Quality",
            recommendation_type=RecommendationType.COST_OPTIMIZATION,
            strategic_context={'base_revenue': 10000, 'base_cost': 9500},
            target_priorities=[]
        )
        
        # Get high quality recommendations
        high_quality_recs = engine.get_high_quality_recommendations(0.7)
        
        # Should include high quality recommendation
        assert high_quality_rec in high_quality_recs
        
        # May or may not include low quality recommendation depending on actual score
        if low_quality_rec.quality_score < 0.7:
            assert low_quality_rec not in high_quality_recs
    
    def test_generate_recommendation_summary(self, engine, sample_board_priority, sample_strategic_context):
        """Test generating recommendation summary"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation = engine.create_strategic_recommendation(
            title="Summary Test",
            recommendation_type=RecommendationType.TECHNOLOGY_INVESTMENT,
            strategic_context=sample_strategic_context,
            target_priorities=["priority_001"]
        )
        
        summary = engine.generate_recommendation_summary(recommendation.id)
        
        assert summary['title'] == "Summary Test"
        assert summary['type'] == "technology_investment"
        assert 'quality_score' in summary
        assert 'strategic_alignment' in summary
        assert 'financial_impact' in summary
        assert 'risk_assessment' in summary
        assert 'implementation_timeline' in summary
        assert 'validation_status' in summary
        assert 'key_benefits' in summary
        assert 'next_steps' in summary
        
        # Financial impact should be formatted
        financial = summary['financial_impact']
        assert 'roi_projection' in financial
        assert 'payback_period' in financial
        assert 'revenue_impact' in financial
    
    def test_invalid_recommendation_id(self, engine):
        """Test handling of invalid recommendation ID"""
        with pytest.raises(ValueError, match="Recommendation invalid_id not found"):
            engine.optimize_recommendation("invalid_id")
        
        with pytest.raises(ValueError, match="Recommendation invalid_id not found"):
            engine.validate_recommendation("invalid_id")
        
        with pytest.raises(ValueError, match="Recommendation invalid_id not found"):
            engine.generate_recommendation_summary("invalid_id")
    
    def test_priority_validation(self, engine, sample_strategic_context):
        """Test priority validation during recommendation creation"""
        # Create recommendation with non-existent priority
        recommendation = engine.create_strategic_recommendation(
            title="Invalid Priority Test",
            recommendation_type=RecommendationType.STRATEGIC_INITIATIVE,
            strategic_context=sample_strategic_context,
            target_priorities=["non_existent_priority"]
        )
        
        # Should still create recommendation but with empty priority alignment
        assert recommendation is not None
        assert len(recommendation.board_priorities) == 1  # Still includes the invalid ID
        # Quality score should be lower due to poor alignment
        assert recommendation.quality_score < 0.8
    
    def test_different_recommendation_types(self, engine, sample_board_priority, sample_strategic_context):
        """Test creating different types of recommendations"""
        engine.add_board_priority(sample_board_priority)
        
        recommendation_types = [
            RecommendationType.STRATEGIC_INITIATIVE,
            RecommendationType.OPERATIONAL_IMPROVEMENT,
            RecommendationType.TECHNOLOGY_INVESTMENT,
            RecommendationType.MARKET_EXPANSION,
            RecommendationType.RISK_MITIGATION,
            RecommendationType.COST_OPTIMIZATION,
            RecommendationType.PARTNERSHIP,
            RecommendationType.ACQUISITION
        ]
        
        for rec_type in recommendation_types:
            recommendation = engine.create_strategic_recommendation(
                title=f"Test {rec_type.value}",
                recommendation_type=rec_type,
                strategic_context=sample_strategic_context,
                target_priorities=["priority_001"]
            )
            
            assert recommendation.recommendation_type == rec_type
            assert recommendation.quality_score > 0
            
            # Different types should have different risk profiles
            if rec_type == RecommendationType.MARKET_EXPANSION:
                assert recommendation.risk_assessment.risk_level == "high"
            elif rec_type == RecommendationType.TECHNOLOGY_INVESTMENT:
                assert recommendation.risk_assessment.risk_level == "medium-high"
    
    def test_optimization_strategies(self, engine, sample_board_priority, sample_strategic_context):
        """Test different optimization strategies"""
        engine.add_board_priority(sample_board_priority)
        
        # Create additional priority for alignment testing
        priority2 = BoardPriority(
            id="priority_002",
            title="Cost Reduction",
            description="Reduce operational costs",
            priority_level=PriorityLevel.MEDIUM,
            impact_areas=[ImpactArea.COST, ImpactArea.EFFICIENCY],
            target_timeline="6 months",
            success_metrics=["Cost reduction of 15%"],
            stakeholders=["CFO"]
        )
        engine.add_board_priority(priority2)
        
        # Create recommendation with room for improvement
        recommendation = engine.create_strategic_recommendation(
            title="Optimization Test",
            recommendation_type=RecommendationType.COST_OPTIMIZATION,
            strategic_context={'base_revenue': 200000, 'base_cost': 180000},
            target_priorities=["priority_001"]  # Only one priority initially
        )
        
        original_score = recommendation.quality_score
        original_priorities = len(recommendation.board_priorities)
        original_roi = recommendation.financial_impact.roi_projection
        
        # Optimize
        optimized = engine.optimize_recommendation(recommendation.id)
        
        # Should improve in multiple areas
        assert optimized.quality_score >= original_score
        
        # May have additional priority alignment if optimization found relevant connections
        if len(optimized.board_priorities) > original_priorities:
            assert "priority_002" in optimized.board_priorities  # Cost reduction priority should be relevant
        
        # Financial metrics should improve
        assert optimized.financial_impact.roi_projection >= original_roi
    
    @pytest.mark.asyncio
    async def test_concurrent_recommendation_creation(self, engine, sample_board_priority, sample_strategic_context):
        """Test concurrent recommendation creation"""
        import asyncio
        
        engine.add_board_priority(sample_board_priority)
        
        async def create_recommendation(title_suffix):
            return engine.create_strategic_recommendation(
                title=f"Concurrent Test {title_suffix}",
                recommendation_type=RecommendationType.STRATEGIC_INITIATIVE,
                strategic_context=sample_strategic_context,
                target_priorities=["priority_001"]
            )
        
        # Create multiple recommendations concurrently
        tasks = [create_recommendation(i) for i in range(5)]
        recommendations = await asyncio.gather(*tasks)
        
        # All should be created successfully
        assert len(recommendations) == 5
        assert len(engine.recommendations) == 5
        
        # Each should have unique ID
        ids = [r.id for r in recommendations]
        assert len(set(ids)) == 5