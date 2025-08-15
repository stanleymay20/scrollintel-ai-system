"""
Tests for Quantum AI Research Integration System
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from scrollintel.engines.quantum_ai_integration import QuantumAIIntegration
from scrollintel.models.quantum_ai_integration_models import (
    QuantumAlgorithm, QuantumAlgorithmType, QuantumHardwarePlatform,
    QuantumEnhancedInnovation, QuantumAdvantageType, QuantumResearchAcceleration,
    QuantumClassicalHybrid, QuantumInnovationOpportunity, QuantumValidationResult,
    QuantumIntegrationPlan, IntegrationType
)

@pytest.fixture
def quantum_integration_engine():
    """Create a quantum AI integration engine for testing"""
    return QuantumAIIntegration()

@pytest.fixture
def sample_quantum_algorithm():
    """Create a sample quantum algorithm for testing"""
    return QuantumAlgorithm(
        id="quantum-alg-001",
        name="Quantum Optimization Algorithm",
        algorithm_type=QuantumAlgorithmType.QUANTUM_OPTIMIZATION,
        description="Advanced quantum optimization for complex problems",
        quantum_circuit_depth=50,
        qubit_requirements=20,
        gate_count=500,
        classical_preprocessing=["data_encoding", "parameter_initialization"],
        classical_postprocessing=["result_decoding", "optimization_analysis"],
        hardware_requirements={
            "platform": "IBM_Quantum",
            "connectivity": "all_to_all",
            "error_rate": 0.01
        },
        performance_metrics={
            "execution_time": 10.5,
            "accuracy": 0.95,
            "success_rate": 0.88
        },
        quantum_advantage_factor=3.5,
        error_tolerance=0.05,
        execution_time_estimate=15.0,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )

@pytest.fixture
def sample_innovation_data():
    """Create sample innovation data for testing"""
    return {
        "id": "innovation-001",
        "title": "AI-Powered Optimization System",
        "description": "Advanced optimization system using machine learning",
        "complexity": 0.7,
        "impact_potential": 0.8,
        "domains": ["optimization", "machine_learning"],
        "requirements": {
            "computational_power": "high",
            "accuracy": 0.9,
            "speed": "real_time"
        }
    }

@pytest.fixture
def sample_quantum_capabilities():
    """Create sample quantum capabilities for testing"""
    return [
        "quantum_annealing",
        "variational_quantum_algorithms",
        "quantum_machine_learning",
        "quantum_optimization",
        "hybrid_classical_quantum"
    ]

class TestQuantumAIIntegration:
    """Test cases for quantum AI integration"""
    
    @pytest.mark.asyncio
    async def test_create_quantum_enhanced_innovation(self, quantum_integration_engine, sample_innovation_data, sample_quantum_capabilities):
        """Test creation of quantum-enhanced innovation"""
        
        quantum_enhanced_innovation = await quantum_integration_engine.create_quantum_enhanced_innovation(
            sample_innovation_data, sample_quantum_capabilities
        )
        
        assert isinstance(quantum_enhanced_innovation, QuantumEnhancedInnovation)
        assert quantum_enhanced_innovation.innovation_id == sample_innovation_data["id"]
        assert quantum_enhanced_innovation.speedup_factor > 1.0
        assert quantum_enhanced_innovation.accuracy_improvement > 0
        assert quantum_enhanced_innovation.resource_efficiency_gain > 0
        assert len(quantum_enhanced_innovation.quantum_features) > 0
        assert quantum_enhanced_innovation.classical_fallback is True
        assert quantum_enhanced_innovation.validation_status == 'pending'
        
        # Check that innovation is stored in the engine
        assert quantum_enhanced_innovation.id in quantum_integration_engine.quantum_enhanced_innovations
        assert quantum_integration_engine.performance_metrics.quantum_enhanced_innovations == 1
    
    @pytest.mark.asyncio
    async def test_accelerate_quantum_research(self, quantum_integration_engine):
        """Test quantum research acceleration"""
        
        research_areas = [
            "quantum_optimization",
            "quantum_machine_learning",
            "quantum_simulation"
        ]
        
        quantum_resources = {
            "computational_power": 2.0,
            "qubit_count": 50,
            "error_rate": 0.01,
            "connectivity": "high"
        }
        
        accelerations = await quantum_integration_engine.accelerate_quantum_research(
            research_areas, quantum_resources
        )
        
        assert len(accelerations) > 0
        assert all(isinstance(acc, QuantumResearchAcceleration) for acc in accelerations)
        assert all(acc.acceleration_factor > 1.0 for acc in accelerations)
        assert all(acc.discovery_potential > 0 for acc in accelerations)
        assert all(acc.breakthrough_probability > 0 for acc in accelerations)
        
        # Check that accelerations are stored in the engine
        assert len(quantum_integration_engine.research_accelerations) == len(accelerations)
        
        # Verify acceleration properties
        for acceleration in accelerations:
            assert acceleration.research_area in research_areas
            assert len(acceleration.quantum_algorithms) > 0
            assert isinstance(acceleration.computational_advantage, dict)
            assert isinstance(acceleration.resource_optimization, dict)
            assert len(acceleration.integration_requirements) > 0
            assert len(acceleration.success_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_optimize_quantum_innovation(self, quantum_integration_engine):
        """Test quantum innovation optimization"""
        
        innovation_opportunities = [
            {
                "type": "optimization_enhancement",
                "complexity": 0.8,
                "impact_potential": 0.9,
                "description": "Quantum-enhanced optimization system"
            },
            {
                "type": "machine_learning_acceleration",
                "complexity": 0.6,
                "impact_potential": 0.7,
                "description": "Quantum machine learning acceleration"
            }
        ]
        
        optimized_opportunities = await quantum_integration_engine.optimize_quantum_innovation(
            innovation_opportunities
        )
        
        assert len(optimized_opportunities) > 0
        assert all(isinstance(opp, QuantumInnovationOpportunity) for opp in optimized_opportunities)
        assert all(opp.feasibility_score > 0.6 for opp in optimized_opportunities)
        assert all(opp.innovation_potential > 0 for opp in optimized_opportunities)
        
        # Check that opportunities are stored in the engine
        assert len(quantum_integration_engine.innovation_opportunities) == len(optimized_opportunities)
        
        # Verify opportunity properties
        for opportunity in optimized_opportunities:
            assert opportunity.quantum_capability is not None
            assert opportunity.timeline_estimate > 0
            assert isinstance(opportunity.resource_requirements, dict)
            assert isinstance(opportunity.risk_assessment, dict)
            assert len(opportunity.expected_outcomes) > 0
            assert len(opportunity.quantum_advantage_areas) > 0
            assert len(opportunity.integration_pathway) > 0
    
    @pytest.mark.asyncio
    async def test_create_quantum_classical_hybrid(self, quantum_integration_engine, sample_quantum_capabilities):
        """Test creation of quantum-classical hybrid system"""
        
        innovation_requirements = {
            "performance_target": "high",
            "accuracy_requirement": 0.95,
            "scalability": "medium",
            "fault_tolerance": 0.9
        }
        
        classical_capabilities = [
            "classical_optimization",
            "machine_learning",
            "data_processing",
            "result_analysis"
        ]
        
        hybrid_system = await quantum_integration_engine.create_quantum_classical_hybrid(
            innovation_requirements, sample_quantum_capabilities, classical_capabilities
        )
        
        assert isinstance(hybrid_system, QuantumClassicalHybrid)
        assert len(hybrid_system.quantum_components) > 0
        assert len(hybrid_system.classical_components) > 0
        assert hybrid_system.integration_strategy is not None
        assert isinstance(hybrid_system.data_flow_optimization, dict)
        assert isinstance(hybrid_system.resource_allocation, dict)
        assert hybrid_system.fault_tolerance_level > 0
        assert hybrid_system.scalability_factor > 1.0
        assert isinstance(hybrid_system.efficiency_metrics, dict)
        
        # Check that hybrid system is stored in the engine
        assert hybrid_system.id in quantum_integration_engine.hybrid_systems
    
    @pytest.mark.asyncio
    async def test_validate_quantum_advantage(self, quantum_integration_engine, sample_quantum_algorithm):
        """Test quantum advantage validation"""
        
        classical_baseline = {
            "performance_factor": 1.0,
            "accuracy": 0.85,
            "execution_time": 50.0,
            "resource_usage": 1.0
        }
        
        validation_result = await quantum_integration_engine.validate_quantum_advantage(
            sample_quantum_algorithm, classical_baseline
        )
        
        assert isinstance(validation_result, QuantumValidationResult)
        assert validation_result.algorithm_id == sample_quantum_algorithm.id
        assert isinstance(validation_result.quantum_advantage_validated, bool)
        assert isinstance(validation_result.performance_comparison, dict)
        assert isinstance(validation_result.accuracy_metrics, dict)
        assert isinstance(validation_result.efficiency_analysis, dict)
        assert isinstance(validation_result.scalability_assessment, dict)
        assert isinstance(validation_result.error_analysis, dict)
        assert isinstance(validation_result.hardware_compatibility, dict)
        assert len(validation_result.recommendations) > 0
        
        # Check that validation result is stored
        assert validation_result.algorithm_id in quantum_integration_engine.validation_results
        
        # Verify performance comparison metrics
        perf_comparison = validation_result.performance_comparison
        assert 'quantum_speedup' in perf_comparison
        assert 'accuracy_improvement' in perf_comparison
        assert 'resource_efficiency' in perf_comparison
        assert 'scalability_factor' in perf_comparison
    
    @pytest.mark.asyncio
    async def test_create_integration_plan(self, quantum_integration_engine):
        """Test creation of quantum integration plan"""
        
        integration_type = IntegrationType.RESEARCH_ACCELERATION
        target_system = "autonomous_innovation_lab"
        quantum_requirements = {
            "complexity": 0.7,
            "performance_target": "high",
            "timeline": "6_months",
            "resources": "medium"
        }
        
        plan = await quantum_integration_engine.create_integration_plan(
            integration_type, target_system, quantum_requirements
        )
        
        assert isinstance(plan, QuantumIntegrationPlan)
        assert plan.integration_type == integration_type
        assert plan.target_system == target_system
        assert len(plan.quantum_components) > 0
        assert plan.integration_strategy is not None
        assert len(plan.implementation_phases) > 0
        assert isinstance(plan.resource_allocation, dict)
        assert isinstance(plan.timeline_milestones, dict)
        assert len(plan.risk_mitigation) > 0
        assert isinstance(plan.success_criteria, dict)
        assert len(plan.monitoring_metrics) > 0
        assert isinstance(plan.optimization_targets, dict)
        
        # Check that plan is stored in the engine
        assert plan.id in quantum_integration_engine.integration_plans
        assert quantum_integration_engine.performance_metrics.active_quantum_integrations == 1
    
    @pytest.mark.asyncio
    async def test_optimize_quantum_performance(self, quantum_integration_engine, sample_innovation_data, sample_quantum_capabilities):
        """Test quantum performance optimization"""
        
        # First create some quantum components to optimize
        await quantum_integration_engine.create_quantum_enhanced_innovation(
            sample_innovation_data, sample_quantum_capabilities
        )
        
        await quantum_integration_engine.create_quantum_classical_hybrid(
            {"performance": "high"}, sample_quantum_capabilities, ["classical_processing"]
        )
        
        optimization_results = await quantum_integration_engine.optimize_quantum_performance()
        
        assert isinstance(optimization_results, dict)
        assert 'algorithm_optimization' in optimization_results
        assert 'hybrid_optimization' in optimization_results
        assert 'resource_optimization' in optimization_results
        assert 'integration_optimization' in optimization_results
        assert 'performance_metrics' in optimization_results
        
        # Verify performance metrics
        performance_metrics = optimization_results['performance_metrics']
        assert 'average_speedup_factor' in performance_metrics
        assert 'integration_efficiency' in performance_metrics
        assert 'discovery_acceleration_factor' in performance_metrics
        assert 'innovation_enhancement_score' in performance_metrics
        assert 'resource_optimization_gain' in performance_metrics
    
    def test_get_quantum_integration_status(self, quantum_integration_engine):
        """Test getting quantum integration status"""
        
        status = quantum_integration_engine.get_quantum_integration_status()
        
        assert isinstance(status, dict)
        assert 'quantum_algorithms' in status
        assert 'quantum_enhanced_innovations' in status
        assert 'research_accelerations' in status
        assert 'hybrid_systems' in status
        assert 'innovation_opportunities' in status
        assert 'validation_results' in status
        assert 'integration_plans' in status
        assert 'performance_metrics' in status
        
        # Verify performance metrics
        metrics = status['performance_metrics']
        assert 'total_quantum_algorithms' in metrics
        assert 'active_quantum_integrations' in metrics
        assert 'quantum_enhanced_innovations' in metrics
        assert 'average_speedup_factor' in metrics
        assert 'quantum_advantage_success_rate' in metrics
        assert 'integration_efficiency' in metrics
        assert 'last_updated' in metrics
    
    @pytest.mark.asyncio
    async def test_quantum_algorithm_selection(self, quantum_integration_engine, sample_innovation_data, sample_quantum_capabilities):
        """Test quantum algorithm selection logic"""
        
        # Test with optimization-focused innovation
        optimization_innovation = {
            **sample_innovation_data,
            "description": "Advanced optimization system for complex problems",
            "complexity": 0.9
        }
        
        algorithm_selection = await quantum_integration_engine._select_optimal_quantum_algorithm(
            optimization_innovation, sample_quantum_capabilities
        )
        
        assert isinstance(algorithm_selection, dict)
        assert 'id' in algorithm_selection
        assert 'algorithm_type' in algorithm_selection
        assert 'quantum_advantage_factor' in algorithm_selection
        assert 'qubit_requirements' in algorithm_selection
        assert 'gate_count' in algorithm_selection
        
        assert algorithm_selection['quantum_advantage_factor'] > 1.0
        assert algorithm_selection['qubit_requirements'] > 0
        assert algorithm_selection['gate_count'] > 0
    
    @pytest.mark.asyncio
    async def test_quantum_enhancement_analysis(self, quantum_integration_engine, sample_innovation_data):
        """Test quantum enhancement potential analysis"""
        
        quantum_algorithm = {
            'id': 'test-alg-001',
            'algorithm_type': QuantumAlgorithmType.QUANTUM_OPTIMIZATION,
            'quantum_advantage_factor': 3.0
        }
        
        enhancement_analysis = await quantum_integration_engine._analyze_quantum_enhancement_potential(
            sample_innovation_data, quantum_algorithm
        )
        
        assert isinstance(enhancement_analysis, dict)
        assert 'enhancement_type' in enhancement_analysis
        assert 'quantum_advantage_type' in enhancement_analysis
        assert 'speedup_factor' in enhancement_analysis
        assert 'accuracy_improvement' in enhancement_analysis
        assert 'resource_efficiency_gain' in enhancement_analysis
        assert 'complexity_reduction' in enhancement_analysis
        assert 'quantum_features' in enhancement_analysis
        assert 'classical_fallback' in enhancement_analysis
        assert 'integration_complexity' in enhancement_analysis
        
        assert enhancement_analysis['speedup_factor'] > 1.0
        assert enhancement_analysis['accuracy_improvement'] > 0
        assert enhancement_analysis['resource_efficiency_gain'] > 0
        assert len(enhancement_analysis['quantum_features']) > 0
        assert isinstance(enhancement_analysis['classical_fallback'], bool)
    
    @pytest.mark.asyncio
    async def test_research_acceleration_analysis(self, quantum_integration_engine):
        """Test research acceleration potential analysis"""
        
        research_area = "quantum_machine_learning"
        algorithms = ["QML", "Quantum_SVM", "Quantum_Neural_Networks"]
        quantum_resources = {
            "computational_power": 2.5,
            "qubit_count": 100,
            "error_rate": 0.005
        }
        
        acceleration_analysis = await quantum_integration_engine._analyze_research_acceleration_potential(
            research_area, algorithms, quantum_resources
        )
        
        assert isinstance(acceleration_analysis, dict)
        assert 'acceleration_factor' in acceleration_analysis
        assert 'discovery_potential' in acceleration_analysis
        assert 'computational_advantage' in acceleration_analysis
        assert 'resource_optimization' in acceleration_analysis
        assert 'timeline_compression' in acceleration_analysis
        assert 'breakthrough_probability' in acceleration_analysis
        assert 'integration_requirements' in acceleration_analysis
        assert 'success_metrics' in acceleration_analysis
        assert 'monitoring_framework' in acceleration_analysis
        
        assert acceleration_analysis['acceleration_factor'] > 1.0
        assert acceleration_analysis['discovery_potential'] > 0
        assert isinstance(acceleration_analysis['computational_advantage'], dict)
        assert isinstance(acceleration_analysis['resource_optimization'], dict)
        assert len(acceleration_analysis['integration_requirements']) > 0
    
    @pytest.mark.asyncio
    async def test_multiple_quantum_enhancements(self, quantum_integration_engine, sample_quantum_capabilities):
        """Test creation of multiple quantum enhancements"""
        
        innovations = []
        for i in range(3):
            innovation_data = {
                "id": f"innovation-{i:03d}",
                "title": f"Innovation {i}",
                "description": f"Test innovation {i}",
                "complexity": 0.5 + i * 0.1,
                "impact_potential": 0.6 + i * 0.1,
                "domains": ["optimization", "ai"]
            }
            innovations.append(innovation_data)
        
        quantum_enhancements = []
        for innovation_data in innovations:
            enhancement = await quantum_integration_engine.create_quantum_enhanced_innovation(
                innovation_data, sample_quantum_capabilities
            )
            quantum_enhancements.append(enhancement)
        
        assert len(quantum_enhancements) == 3
        assert len(quantum_integration_engine.quantum_enhanced_innovations) == 3
        
        # Verify that enhancements have different characteristics based on complexity
        speedup_factors = [enh.speedup_factor for enh in quantum_enhancements]
        assert len(set(speedup_factors)) > 1  # Should have different speedup factors
        
        # Verify performance metrics are updated
        assert quantum_integration_engine.performance_metrics.quantum_enhanced_innovations == 3
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_data(self, quantum_integration_engine):
        """Test error handling with invalid data"""
        
        # Test with empty innovation data
        with pytest.raises(Exception):
            await quantum_integration_engine.create_quantum_enhanced_innovation(
                {}, []
            )
        
        # Test with empty research areas
        accelerations = await quantum_integration_engine.accelerate_quantum_research(
            [], {"computational_power": 1.0}
        )
        assert len(accelerations) == 0
        
        # Test with empty innovation opportunities
        opportunities = await quantum_integration_engine.optimize_quantum_innovation([])
        assert len(opportunities) == 0

if __name__ == "__main__":
    pytest.main([__file__])