"""
Tests for Impact Assessment Framework.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.impact_assessment_framework import ImpactAssessmentFramework
from scrollintel.models.validation_models import (
    Innovation, ImpactAssessment, ImpactLevel
)


@pytest.fixture
async def impact_framework():
    """Create and initialize impact assessment framework for testing."""
    framework = ImpactAssessmentFramework()
    await framework.start()
    return framework


@pytest.fixture
def sample_innovation():
    """Create sample innovation for testing."""
    return Innovation(
        title="AI-Powered Healthcare Assistant",
        description="An AI system that assists healthcare professionals with diagnosis and treatment recommendations",
        category="Healthcare Technology",
        domain="Healthcare",
        technology_stack=["Python", "TensorFlow", "React", "PostgreSQL", "AWS"],
        target_market="Healthcare providers",
        problem_statement="Healthcare professionals need better tools for accurate diagnosis",
        proposed_solution="AI-powered assistant that analyzes patient data and provides recommendations",
        unique_value_proposition="Reduces diagnostic errors by 40% and improves treatment outcomes",
        competitive_advantages=["Advanced AI algorithms", "Integration with existing systems"],
        estimated_timeline="18 months",
        estimated_cost=2000000.0,
        potential_revenue=10000000.0,
        risk_factors=["Regulatory approval", "Data privacy concerns"],
        success_metrics=["Diagnostic accuracy", "User adoption rate", "Patient outcomes"]
    )


@pytest.fixture
def high_impact_innovation():
    """Create high-impact innovation for testing."""
    return Innovation(
        title="Quantum Computing Platform",
        description="Revolutionary quantum computing platform for enterprise applications",
        category="Quantum Technology",
        domain="Technology",
        technology_stack=["Quantum circuits", "Python", "Kubernetes", "AWS"],
        target_market="Global enterprise customers",
        problem_statement="Limited access to quantum computing resources",
        proposed_solution="Revolutionary quantum cloud platform providing quantum computing as a service",
        unique_value_proposition="First commercially viable quantum cloud platform with breakthrough performance",
        competitive_advantages=["Proprietary quantum algorithms", "Patent protection", "Scalable architecture"],
        estimated_timeline="36 months",
        estimated_cost=50000000.0,
        potential_revenue=5000000000.0,  # $5B revenue potential
        risk_factors=["Technology maturity", "High development costs"],
        success_metrics=["Platform adoption", "Quantum advantage demonstrations"]
    )


class TestImpactAssessmentFramework:
    """Test cases for ImpactAssessmentFramework class."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, impact_framework):
        """Test impact assessment framework initialization."""
        assert impact_framework.engine_id == "impact_assessment_framework"
        assert impact_framework.name == "Innovation Impact Assessment Framework"
        assert len(impact_framework.impact_models) > 0
        assert len(impact_framework.market_data) > 0
        assert len(impact_framework.industry_benchmarks) > 0
        assert impact_framework.assessment_context is not None
    
    @pytest.mark.asyncio
    async def test_assess_innovation_impact(self, impact_framework, sample_innovation):
        """Test comprehensive innovation impact assessment."""
        # Assess innovation impact
        assessment = await impact_framework.assess_innovation_impact(sample_innovation)
        
        # Verify assessment structure
        assert isinstance(assessment, ImpactAssessment)
        assert assessment.innovation_id == sample_innovation.id
        assert assessment.market_impact in ImpactLevel
        assert assessment.technical_impact in ImpactLevel
        assert assessment.business_impact in ImpactLevel
        assert assessment.social_impact in ImpactLevel
        assert assessment.environmental_impact in ImpactLevel
        assert assessment.economic_impact in ImpactLevel
        
        # Verify quantitative metrics
        assert assessment.market_size > 0
        assert assessment.addressable_market > 0
        assert assessment.revenue_potential > 0
        assert 0.0 <= assessment.market_penetration_potential <= 1.0
        assert 0.0 <= assessment.disruption_potential <= 1.0
        assert 0.0 <= assessment.scalability_factor <= 1.0
        assert assessment.time_to_market > 0
        assert assessment.competitive_advantage_duration > 0
        
        # Verify additional data
        assert len(assessment.quantitative_metrics) > 0
        assert len(assessment.qualitative_factors) > 0
        assert len(assessment.stakeholder_impact) > 0
        assert assessment.created_at is not None
    
    @pytest.mark.asyncio
    async def test_assess_market_impact(self, impact_framework, sample_innovation):
        """Test market impact assessment."""
        market_impact = await impact_framework.assess_market_impact(sample_innovation)
        
        assert isinstance(market_impact, dict)
        assert "market_size" in market_impact
        assert "addressable_market" in market_impact
        assert "penetration_potential" in market_impact
        assert "revenue_potential" in market_impact
        assert "disruption_potential" in market_impact
        assert "time_to_market" in market_impact
        assert "competitive_advantage_duration" in market_impact
        
        # Verify values are reasonable
        assert market_impact["market_size"] > 0
        assert market_impact["addressable_market"] > 0
        assert 0.0 <= market_impact["penetration_potential"] <= 1.0
        assert market_impact["revenue_potential"] > 0
        assert 0.0 <= market_impact["disruption_potential"] <= 1.0
        assert market_impact["time_to_market"] > 0
    
    @pytest.mark.asyncio
    async def test_assess_technical_impact(self, impact_framework, sample_innovation):
        """Test technical impact assessment."""
        technical_impact = await impact_framework.assess_technical_impact(sample_innovation)
        
        assert isinstance(technical_impact, dict)
        assert "advancement_level" in technical_impact
        assert "complexity_score" in technical_impact
        assert "scalability_potential" in technical_impact
        assert "integration_potential" in technical_impact
        assert "novelty_score" in technical_impact
        
        # Verify values are in expected ranges
        assert 0.0 <= technical_impact["advancement_level"] <= 1.0
        assert technical_impact["complexity_score"] >= 0.0
        assert 0.0 <= technical_impact["scalability_potential"] <= 1.0
        assert 0.0 <= technical_impact["novelty_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_assess_business_impact(self, impact_framework, sample_innovation):
        """Test business impact assessment."""
        business_impact = await impact_framework.assess_business_impact(sample_innovation)
        
        assert isinstance(business_impact, dict)
        assert "roi_potential" in business_impact
        assert "cost_savings_potential" in business_impact
        assert "business_model_impact" in business_impact
        assert "operational_impact" in business_impact
        assert "strategic_value" in business_impact
        
        # Verify ROI calculation
        assert business_impact["roi_potential"] > 0
        assert business_impact["cost_savings_potential"] >= 0
    
    @pytest.mark.asyncio
    async def test_assess_social_impact(self, impact_framework, sample_innovation):
        """Test social impact assessment."""
        social_impact = await impact_framework.assess_social_impact(sample_innovation)
        
        assert isinstance(social_impact, dict)
        assert "job_creation_potential" in social_impact
        assert "job_displacement_risk" in social_impact
        assert "social_benefits" in social_impact
        assert "accessibility_impact" in social_impact
        assert "quality_of_life_impact" in social_impact
        
        # Verify job impact calculations
        assert social_impact["job_creation_potential"] >= 0
        assert social_impact["job_displacement_risk"] >= 0
    
    @pytest.mark.asyncio
    async def test_assess_environmental_impact(self, impact_framework, sample_innovation):
        """Test environmental impact assessment."""
        environmental_impact = await impact_framework.assess_environmental_impact(sample_innovation)
        
        assert isinstance(environmental_impact, dict)
        assert "carbon_footprint" in environmental_impact
        assert "resource_consumption" in environmental_impact
        assert "sustainability_score" in environmental_impact
        assert "environmental_benefits" in environmental_impact
        
        # Verify sustainability score
        assert 0.0 <= environmental_impact["sustainability_score"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_quantify_impact_metrics(self, impact_framework, sample_innovation):
        """Test impact metrics quantification."""
        metrics = await impact_framework.quantify_impact_metrics(sample_innovation)
        
        assert isinstance(metrics, dict)
        
        # Verify market metrics
        assert "market_size_usd" in metrics
        assert "addressable_market_usd" in metrics
        assert "revenue_potential_usd" in metrics
        assert "market_penetration_percent" in metrics
        
        # Verify technical metrics
        assert "technical_advancement_score" in metrics
        assert "complexity_score" in metrics
        assert "scalability_score" in metrics
        assert "novelty_score" in metrics
        
        # Verify business metrics
        assert "roi_percent" in metrics
        assert "cost_savings_usd" in metrics
        assert "strategic_value_score" in metrics
        
        # Verify social metrics
        assert "job_creation_count" in metrics
        assert "social_benefit_score" in metrics
        
        # Verify environmental metrics
        assert "sustainability_score" in metrics
        
        # Verify value ranges
        assert metrics["market_size_usd"] > 0
        assert metrics["revenue_potential_usd"] > 0
        assert 0.0 <= metrics["market_penetration_percent"] <= 100.0
    
    @pytest.mark.asyncio
    async def test_high_impact_innovation_assessment(self, impact_framework, high_impact_innovation):
        """Test assessment of high-impact innovation."""
        assessment = await impact_framework.assess_innovation_impact(high_impact_innovation)
        
        # High-impact innovation should have higher impact levels
        assert assessment.market_impact in [ImpactLevel.HIGH, ImpactLevel.CRITICAL, ImpactLevel.TRANSFORMATIONAL]
        assert assessment.revenue_potential >= 1000000000  # At least $1B
        assert assessment.disruption_potential > 0.5  # High disruption potential
    
    @pytest.mark.asyncio
    async def test_different_domains_impact(self, impact_framework):
        """Test impact assessment for different domains."""
        domains = ["Healthcare", "Technology", "Energy", "Transportation", "Finance"]
        
        for domain in domains:
            innovation = Innovation(
                title=f"{domain} Innovation",
                domain=domain,
                estimated_cost=1000000.0,
                potential_revenue=5000000.0
            )
            
            assessment = await impact_framework.assess_innovation_impact(innovation)
            
            assert isinstance(assessment, ImpactAssessment)
            assert assessment.innovation_id == innovation.id
            assert assessment.market_size > 0
            # Different domains should have different market sizes
            # This is tested implicitly through the market data loading
    
    @pytest.mark.asyncio
    async def test_process_method(self, impact_framework, sample_innovation):
        """Test the process method."""
        result = await impact_framework.process(sample_innovation)
        
        assert isinstance(result, ImpactAssessment)
        assert result.innovation_id == sample_innovation.id
    
    @pytest.mark.asyncio
    async def test_process_invalid_input(self, impact_framework):
        """Test process method with invalid input."""
        with pytest.raises(ValueError, match="Input must be Innovation object"):
            await impact_framework.process("invalid_input")
    
    @pytest.mark.asyncio
    async def test_historical_assessments_tracking(self, impact_framework, sample_innovation):
        """Test that assessments are tracked in history."""
        initial_count = len(impact_framework.historical_assessments)
        
        # Perform assessment
        await impact_framework.assess_innovation_impact(sample_innovation)
        
        # Check that assessment was added to history
        assert len(impact_framework.historical_assessments) == initial_count + 1
    
    @pytest.mark.asyncio
    async def test_get_status(self, impact_framework):
        """Test getting framework status."""
        status = impact_framework.get_status()
        
        assert isinstance(status, dict)
        assert "healthy" in status
        assert "impact_models_loaded" in status
        assert "market_data_loaded" in status
        assert "industry_benchmarks" in status
        assert "historical_assessments" in status
        assert "context_initialized" in status
        assert status["healthy"] is True
    
    @pytest.mark.asyncio
    async def test_health_check(self, impact_framework):
        """Test framework health check."""
        is_healthy = await impact_framework.health_check()
        assert is_healthy is True
    
    @pytest.mark.asyncio
    async def test_cleanup(self, impact_framework, sample_innovation):
        """Test framework cleanup."""
        # Add some data
        await impact_framework.assess_innovation_impact(sample_innovation)
        
        # Verify data exists
        assert len(impact_framework.historical_assessments) > 0
        
        # Cleanup
        await impact_framework.cleanup()
        
        # Verify cleanup
        assert len(impact_framework.impact_models) == 0
        assert len(impact_framework.market_data) == 0
        assert len(impact_framework.industry_benchmarks) == 0
        assert len(impact_framework.historical_assessments) == 0


class TestImpactLevelDetermination:
    """Test cases for impact level determination logic."""
    
    @pytest.mark.asyncio
    async def test_revenue_impact_levels(self, impact_framework):
        """Test revenue-based impact level determination."""
        # Test different revenue levels
        test_cases = [
            (500000, ImpactLevel.LOW),           # $500K
            (5000000, ImpactLevel.MEDIUM),       # $5M
            (50000000, ImpactLevel.HIGH),        # $50M
            (500000000, ImpactLevel.CRITICAL),   # $500M
            (5000000000, ImpactLevel.TRANSFORMATIONAL)  # $5B
        ]
        
        for revenue, expected_level in test_cases:
            level = impact_framework._determine_impact_level(revenue, "revenue")
            assert level == expected_level
    
    @pytest.mark.asyncio
    async def test_score_impact_levels(self, impact_framework):
        """Test score-based impact level determination."""
        # Test different score levels
        test_cases = [
            (0.2, ImpactLevel.LOW),
            (0.4, ImpactLevel.MEDIUM),
            (0.6, ImpactLevel.HIGH),
            (0.8, ImpactLevel.CRITICAL),
            (0.95, ImpactLevel.TRANSFORMATIONAL)
        ]
        
        for score, expected_level in test_cases:
            level = impact_framework._determine_impact_level(score, "technical")
            assert level == expected_level


class TestImpactAssessmentIntegration:
    """Integration tests for impact assessment framework."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_assessment_flow(self, impact_framework, sample_innovation):
        """Test complete assessment flow from innovation to impact assessment."""
        # Step 1: Assess innovation
        assessment = await impact_framework.assess_innovation_impact(sample_innovation)
        
        # Step 2: Verify comprehensive assessment
        assert isinstance(assessment, ImpactAssessment)
        assert assessment.innovation_id == sample_innovation.id
        
        # Step 3: Verify all impact dimensions assessed
        assert assessment.market_impact in ImpactLevel
        assert assessment.technical_impact in ImpactLevel
        assert assessment.business_impact in ImpactLevel
        assert assessment.social_impact in ImpactLevel
        assert assessment.environmental_impact in ImpactLevel
        assert assessment.economic_impact in ImpactLevel
        
        # Step 4: Verify quantitative metrics
        assert len(assessment.quantitative_metrics) > 0
        assert assessment.market_size > 0
        assert assessment.revenue_potential > 0
        
        # Step 5: Verify assessment is in history
        assert assessment in impact_framework.historical_assessments
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_assessments(self, impact_framework):
        """Test handling multiple concurrent assessments."""
        innovations = [
            Innovation(title=f"Innovation {i}", domain="Technology", estimated_cost=1000000.0, potential_revenue=5000000.0)
            for i in range(3)
        ]
        
        # Start concurrent assessments
        tasks = [
            impact_framework.assess_innovation_impact(innovation)
            for innovation in innovations
        ]
        
        # Wait for all to complete
        assessments = await asyncio.gather(*tasks)
        
        # Verify all completed successfully
        assert len(assessments) == 3
        for assessment in assessments:
            assert isinstance(assessment, ImpactAssessment)
            assert assessment.created_at is not None
    
    @pytest.mark.asyncio
    async def test_assessment_with_minimal_data(self, impact_framework):
        """Test assessment with minimal innovation data."""
        minimal_innovation = Innovation(
            title="Minimal Innovation",
            domain="Technology"
            # Missing many fields
        )
        
        # Should still complete assessment
        assessment = await impact_framework.assess_innovation_impact(minimal_innovation)
        
        assert isinstance(assessment, ImpactAssessment)
        assert assessment.innovation_id == minimal_innovation.id
        # Should have default/calculated values
        assert assessment.market_size > 0
        assert assessment.revenue_potential >= 0


if __name__ == "__main__":
    pytest.main([__file__])