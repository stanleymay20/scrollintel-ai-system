"""
Tests for Breakthrough Innovation Integration System
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.breakthrough_innovation_integration import BreakthroughInnovationIntegration
from scrollintel.models.breakthrough_innovation_integration_models import (
    BreakthroughInnovation, InnovationType, BreakthroughPotential,
    InnovationSynergy, SynergyType, CrossPollinationOpportunity,
    InnovationAccelerationPlan, BreakthroughValidationResult
)

@pytest.fixture
def integration_engine():
    """Create a breakthrough innovation integration engine for testing"""
    return BreakthroughInnovationIntegration()

@pytest.fixture
def sample_breakthrough_innovation():
    """Create a sample breakthrough innovation for testing"""
    return BreakthroughInnovation(
        id="innovation-001",
        title="Quantum-Enhanced AI Processing",
        description="Revolutionary AI processing using quantum computing principles",
        innovation_type=InnovationType.PARADIGM_SHIFT,
        breakthrough_potential=BreakthroughPotential.REVOLUTIONARY,
        domains_involved=["quantum_computing", "artificial_intelligence", "optimization"],
        key_concepts=["quantum_superposition", "neural_networks", "optimization"],
        feasibility_score=0.8,
        impact_score=0.95,
        novelty_score=0.9,
        implementation_complexity=0.7,
        resource_requirements={
            "computational_resources": "high",
            "research_hours": 500,
            "budget": 100000
        },
        timeline_estimate=180,
        success_probability=0.75,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

@pytest.fixture
def sample_lab_components():
    """Create sample lab components for testing"""
    return [
        "automated_research_engine",
        "experimental_design_system",
        "prototype_development_framework",
        "innovation_validation_engine",
        "knowledge_integration_system"
    ]

class TestBreakthroughInnovationIntegration:
    """Test cases for breakthrough innovation integration"""
    
    @pytest.mark.asyncio
    async def test_identify_innovation_synergies(self, integration_engine, sample_lab_components, sample_breakthrough_innovation):
        """Test identification of innovation synergies"""
        
        synergies = await integration_engine.identify_innovation_synergies(
            sample_lab_components, [sample_breakthrough_innovation]
        )
        
        assert len(synergies) > 0
        assert all(isinstance(synergy, InnovationSynergy) for synergy in synergies)
        assert all(synergy.synergy_strength > 0 for synergy in synergies)
        assert all(synergy.enhancement_potential > 0 for synergy in synergies)
        
        # Check that synergies are stored in the engine
        assert len(integration_engine.active_synergies) == len(synergies)
        
        # Verify synergy properties
        for synergy in synergies:
            assert synergy.innovation_lab_component in sample_lab_components
            assert synergy.breakthrough_innovation_id == sample_breakthrough_innovation.id
            assert isinstance(synergy.synergy_type, SynergyType)
            assert 0 <= synergy.synergy_strength <= 1
            assert 0 <= synergy.enhancement_potential <= 1
            assert len(synergy.expected_benefits) > 0
            assert len(synergy.integration_requirements) > 0
    
    @pytest.mark.asyncio
    async def test_implement_cross_pollination(self, integration_engine, sample_breakthrough_innovation):
        """Test implementation of cross-pollination"""
        
        target_research_areas = [
            "machine_learning_optimization",
            "distributed_computing",
            "cognitive_architectures"
        ]
        
        opportunities = await integration_engine.implement_cross_pollination(
            sample_breakthrough_innovation, target_research_areas
        )
        
        assert len(opportunities) > 0
        assert all(isinstance(opp, CrossPollinationOpportunity) for opp in opportunities)
        assert all(opp.feasibility_score > 0.7 for opp in opportunities)
        assert all(opp.enhancement_potential > 0 for opp in opportunities)
        
        # Check that opportunities are stored in the engine
        assert len(integration_engine.cross_pollination_opportunities) == len(opportunities)
        
        # Verify opportunity properties
        for opportunity in opportunities:
            assert opportunity.source_innovation == sample_breakthrough_innovation.id
            assert opportunity.target_research_area in target_research_areas
            assert opportunity.feasibility_score > 0
            assert opportunity.enhancement_potential > 0
            assert len(opportunity.expected_outcomes) > 0
            assert len(opportunity.integration_pathway) > 0
            assert opportunity.timeline_estimate > 0
    
    @pytest.mark.asyncio
    async def test_identify_synergy_exploitation(self, integration_engine, sample_lab_components, sample_breakthrough_innovation):
        """Test identification of synergy exploitation opportunities"""
        
        # First create some synergies
        synergies = await integration_engine.identify_innovation_synergies(
            sample_lab_components, [sample_breakthrough_innovation]
        )
        
        # Filter for high-strength synergies and set them to high strength for testing
        high_strength_synergies = []
        for synergy in synergies:
            synergy.synergy_strength = 0.85  # Set to high strength for testing
            high_strength_synergies.append(synergy)
        
        acceleration_plans = await integration_engine.identify_synergy_exploitation(high_strength_synergies)
        
        assert len(acceleration_plans) > 0
        assert all(isinstance(plan, InnovationAccelerationPlan) for plan in acceleration_plans)
        
        # Check that plans are stored in the engine
        assert len(integration_engine.acceleration_plans) == len(acceleration_plans)
        
        # Verify plan properties
        for plan in acceleration_plans:
            assert plan.target_innovation == sample_breakthrough_innovation.id
            assert len(plan.acceleration_strategies) > 0
            assert isinstance(plan.resource_optimization, dict)
            assert 0 <= plan.timeline_compression <= 1
            assert len(plan.risk_mitigation) > 0
            assert len(plan.implementation_steps) > 0
    
    @pytest.mark.asyncio
    async def test_validate_breakthrough_integration(self, integration_engine, sample_breakthrough_innovation):
        """Test validation of breakthrough innovation integration"""
        
        integration_context = {
            "lab_capabilities": ["research", "experimentation", "validation"],
            "resource_availability": "high",
            "timeline_constraints": "flexible"
        }
        
        validation_result = await integration_engine.validate_breakthrough_integration(
            sample_breakthrough_innovation, integration_context
        )
        
        assert isinstance(validation_result, BreakthroughValidationResult)
        assert validation_result.innovation_id == sample_breakthrough_innovation.id
        assert 0 <= validation_result.validation_score <= 1
        assert isinstance(validation_result.feasibility_assessment, dict)
        assert isinstance(validation_result.impact_prediction, dict)
        assert isinstance(validation_result.risk_analysis, dict)
        assert len(validation_result.implementation_pathway) > 0
        assert len(validation_result.recommendations) > 0
        assert 0 <= validation_result.success_probability <= 1
        
        # Check that validation result is stored
        assert validation_result.innovation_id in integration_engine.validation_results
    
    @pytest.mark.asyncio
    async def test_optimize_integration_performance(self, integration_engine, sample_lab_components, sample_breakthrough_innovation):
        """Test optimization of integration performance"""
        
        # First create some synergies and opportunities
        await integration_engine.identify_innovation_synergies(
            sample_lab_components, [sample_breakthrough_innovation]
        )
        
        await integration_engine.implement_cross_pollination(
            sample_breakthrough_innovation, ["machine_learning", "optimization"]
        )
        
        optimization_results = await integration_engine.optimize_integration_performance()
        
        assert isinstance(optimization_results, dict)
        assert 'synergy_optimization' in optimization_results
        assert 'cross_pollination_optimization' in optimization_results
        assert 'acceleration_optimization' in optimization_results
        assert 'resource_optimization' in optimization_results
        assert 'performance_metrics' in optimization_results
        
        # Verify performance metrics
        performance_metrics = optimization_results['performance_metrics']
        assert 'integration_success_rate' in performance_metrics
        assert 'innovation_velocity_improvement' in performance_metrics
        assert 'resource_efficiency_gain' in performance_metrics
        assert all(0 <= value <= 1 for value in performance_metrics.values() if isinstance(value, float))
    
    def test_get_integration_status(self, integration_engine):
        """Test getting integration status"""
        
        status = integration_engine.get_integration_status()
        
        assert isinstance(status, dict)
        assert 'active_synergies' in status
        assert 'cross_pollination_opportunities' in status
        assert 'acceleration_plans' in status
        assert 'breakthrough_validations' in status
        assert 'integration_metrics' in status
        
        # Verify integration metrics
        metrics = status['integration_metrics']
        assert 'total_synergies_identified' in metrics
        assert 'active_cross_pollinations' in metrics
        assert 'acceleration_projects' in metrics
        assert 'breakthrough_validations' in metrics
        assert 'integration_success_rate' in metrics
        assert 'last_updated' in metrics
    
    @pytest.mark.asyncio
    async def test_component_innovation_synergy_analysis(self, integration_engine, sample_breakthrough_innovation):
        """Test component-innovation synergy analysis"""
        
        component = "automated_research_engine"
        
        synergy_analysis = await integration_engine._analyze_component_innovation_synergy(
            component, sample_breakthrough_innovation
        )
        
        assert isinstance(synergy_analysis, dict)
        assert 'synergy_type' in synergy_analysis
        assert 'synergy_strength' in synergy_analysis
        assert 'enhancement_potential' in synergy_analysis
        assert 'implementation_effort' in synergy_analysis
        assert 'expected_benefits' in synergy_analysis
        assert 'integration_requirements' in synergy_analysis
        assert 'validation_metrics' in synergy_analysis
        
        assert isinstance(synergy_analysis['synergy_type'], SynergyType)
        assert 0 <= synergy_analysis['synergy_strength'] <= 1
        assert 0 <= synergy_analysis['enhancement_potential'] <= 1
        assert 0 <= synergy_analysis['implementation_effort'] <= 1
        assert len(synergy_analysis['expected_benefits']) > 0
        assert len(synergy_analysis['integration_requirements']) > 0
    
    @pytest.mark.asyncio
    async def test_cross_pollination_potential_analysis(self, integration_engine, sample_breakthrough_innovation):
        """Test cross-pollination potential analysis"""
        
        research_area = "machine_learning_optimization"
        
        pollination_analysis = await integration_engine._analyze_cross_pollination_potential(
            sample_breakthrough_innovation, research_area
        )
        
        assert isinstance(pollination_analysis, dict)
        assert 'pollination_type' in pollination_analysis
        assert 'enhancement_potential' in pollination_analysis
        assert 'feasibility_score' in pollination_analysis
        assert 'expected_outcomes' in pollination_analysis
        assert 'integration_pathway' in pollination_analysis
        assert 'resource_requirements' in pollination_analysis
        assert 'timeline_estimate' in pollination_analysis
        assert 'success_indicators' in pollination_analysis
        
        assert 0 <= pollination_analysis['feasibility_score'] <= 1
        assert 0 <= pollination_analysis['enhancement_potential'] <= 1
        assert len(pollination_analysis['expected_outcomes']) > 0
        assert len(pollination_analysis['integration_pathway']) > 0
        assert pollination_analysis['timeline_estimate'] > 0
    
    @pytest.mark.asyncio
    async def test_breakthrough_validation_analysis(self, integration_engine, sample_breakthrough_innovation):
        """Test breakthrough validation analysis"""
        
        context = {
            "lab_capabilities": ["research", "experimentation"],
            "resource_availability": "high"
        }
        
        validation_analysis = await integration_engine._perform_breakthrough_validation(
            sample_breakthrough_innovation, context
        )
        
        assert isinstance(validation_analysis, dict)
        assert 'validation_score' in validation_analysis
        assert 'feasibility_assessment' in validation_analysis
        assert 'impact_prediction' in validation_analysis
        assert 'risk_analysis' in validation_analysis
        assert 'implementation_pathway' in validation_analysis
        assert 'resource_requirements' in validation_analysis
        assert 'success_probability' in validation_analysis
        assert 'recommendations' in validation_analysis
        
        assert 0 <= validation_analysis['validation_score'] <= 1
        assert 0 <= validation_analysis['success_probability'] <= 1
        assert len(validation_analysis['implementation_pathway']) > 0
        assert len(validation_analysis['recommendations']) > 0
    
    @pytest.mark.asyncio
    async def test_integration_with_multiple_innovations(self, integration_engine, sample_lab_components):
        """Test integration with multiple breakthrough innovations"""
        
        innovations = []
        for i in range(3):
            innovation = BreakthroughInnovation(
                id=f"innovation-{i:03d}",
                title=f"Innovation {i}",
                description=f"Test innovation {i}",
                innovation_type=InnovationType.DISRUPTIVE_TECHNOLOGY,
                breakthrough_potential=BreakthroughPotential.TRANSFORMATIVE,
                domains_involved=["ai", "computing"],
                key_concepts=["innovation", "technology"],
                feasibility_score=0.7 + i * 0.1,
                impact_score=0.8 + i * 0.05,
                novelty_score=0.75 + i * 0.05,
                implementation_complexity=0.6,
                resource_requirements={"budget": 50000},
                timeline_estimate=120,
                success_probability=0.7,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            innovations.append(innovation)
        
        synergies = await integration_engine.identify_innovation_synergies(
            sample_lab_components, innovations
        )
        
        assert len(synergies) > 0
        assert len(synergies) <= len(sample_lab_components) * len(innovations)
        
        # Verify that synergies are created for different innovations
        innovation_ids = {synergy.breakthrough_innovation_id for synergy in synergies}
        assert len(innovation_ids) > 1  # Should have synergies for multiple innovations
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_innovation(self, integration_engine):
        """Test error handling with invalid innovation data"""
        
        # Test with empty innovations list
        synergies = await integration_engine.identify_innovation_synergies(
            ["test_component"], []
        )
        assert len(synergies) == 0
        
        # Test with empty components list
        sample_innovation = BreakthroughInnovation(
            id="test-001",
            title="Test Innovation",
            description="Test",
            innovation_type=InnovationType.INNOVATIVE,
            breakthrough_potential=BreakthroughPotential.INCREMENTAL,
            domains_involved=["test"],
            key_concepts=["test"],
            feasibility_score=0.5,
            impact_score=0.5,
            novelty_score=0.5,
            implementation_complexity=0.5,
            resource_requirements={},
            timeline_estimate=30,
            success_probability=0.5,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        synergies = await integration_engine.identify_innovation_synergies(
            [], [sample_innovation]
        )
        assert len(synergies) == 0

if __name__ == "__main__":
    pytest.main([__file__])