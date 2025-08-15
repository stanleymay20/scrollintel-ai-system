"""
Comprehensive tests for meta-learning and adaptation engines.
Tests learning-to-learn algorithms, rapid skill acquisition, and environmental adaptation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from scrollintel.engines.meta_learning_engine import MetaLearningEngine
from scrollintel.engines.adaptation_engine import AdaptationEngine
from scrollintel.models.meta_learning_models import (
    Task, LearningExperience, MetaKnowledge, SkillAcquisition,
    TransferLearningMap, SelfImprovementPlan, EnvironmentalChallenge,
    AdaptationState, LearningStrategy, AdaptationType
)


class TestMetaLearningEngine:
    """Test suite for MetaLearningEngine."""
    
    @pytest.fixture
    def meta_learning_engine(self):
        """Create MetaLearningEngine instance for testing."""
        return MetaLearningEngine()
    
    @pytest.mark.asyncio
    async def test_rapid_skill_acquisition(self, meta_learning_engine):
        """Test rapid skill acquisition functionality."""
        skill_name = "advanced_reasoning"
        domain = "analytical"
        target_performance = 0.85
        
        skill_acquisition = await meta_learning_engine.rapid_skill_acquisition(
            skill_name, domain, target_performance
        )
        
        assert isinstance(skill_acquisition, SkillAcquisition)
        assert skill_acquisition.skill_name == skill_name
        assert skill_acquisition.domain == domain
        assert skill_acquisition.mastery_level >= 0.0
        assert len(skill_acquisition.learning_curve) > 0
        assert skill_acquisition.acquisition_time > 0
        assert skill_acquisition.retention_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_transfer_learning_across_domains(self, meta_learning_engine):
        """Test transfer learning between different domains."""
        source_domain = "technical"
        target_domain = "analytical"
        knowledge_type = "problem_solving"
        
        transfer_map = await meta_learning_engine.transfer_learning_across_domains(
            source_domain, target_domain, knowledge_type
        )
        
        assert isinstance(transfer_map, TransferLearningMap)
        assert transfer_map.source_domain == source_domain
        assert transfer_map.target_domain == target_domain
        assert transfer_map.transfer_efficiency >= 0.0
        assert transfer_map.success_probability >= 0.0
        assert len(transfer_map.transferable_features) >= 0
        assert len(transfer_map.transfer_history) > 0
    
    @pytest.mark.asyncio
    async def test_self_improving_algorithms(self, meta_learning_engine):
        """Test self-improving algorithm capabilities."""
        target_capability = "reasoning_speed"
        improvement_factor = 1.3
        
        improvement_plan = await meta_learning_engine.self_improving_algorithms(
            target_capability, improvement_factor
        )
        
        assert isinstance(improvement_plan, SelfImprovementPlan)
        assert improvement_plan.improvement_target == target_capability
        assert improvement_plan.target_capability > improvement_plan.current_capability
        assert isinstance(improvement_plan.improvement_strategy, dict)
        assert isinstance(improvement_plan.resource_requirements, dict)
        assert len(improvement_plan.success_metrics) > 0
    
    @pytest.mark.asyncio
    async def test_adapt_to_new_environments(self, meta_learning_engine):
        """Test adaptation to new environments."""
        environment_description = {
            "type": "high_complexity",
            "variables": ["uncertainty", "time_pressure", "resource_constraints"],
            "complexity": 0.8,
            "volatility": 0.6
        }
        
        adaptation_result = await meta_learning_engine.adapt_to_new_environments(
            environment_description, "fast"
        )
        
        assert "adaptation_results" in adaptation_result
        assert "success" in adaptation_result
        assert "new_capabilities" in adaptation_result
        assert isinstance(adaptation_result["success"], bool)
    
    @pytest.mark.asyncio
    async def test_learning_strategy_selection(self, meta_learning_engine):
        """Test optimal learning strategy selection."""
        # Test different domain strategy selections
        technical_strategy = await meta_learning_engine._select_optimal_strategy(
            "programming", "technical"
        )
        creative_strategy = await meta_learning_engine._select_optimal_strategy(
            "design", "creative"
        )
        
        assert isinstance(technical_strategy, LearningStrategy)
        assert isinstance(creative_strategy, LearningStrategy)
        assert technical_strategy == LearningStrategy.GRADIENT_BASED
        assert creative_strategy == LearningStrategy.EVOLUTIONARY
    
    @pytest.mark.asyncio
    async def test_meta_learning_algorithms(self, meta_learning_engine):
        """Test different meta-learning algorithms."""
        skill_acquisition = SkillAcquisition(
            skill_name="test_skill",
            domain="test_domain",
            acquisition_strategy=LearningStrategy.MODEL_AGNOSTIC,
            learning_curve=[],
            milestones=[],
            transfer_sources=[],
            mastery_level=0.0,
            acquisition_time=0.0,
            retention_score=0.0
        )
        
        # Test MAML algorithm
        maml_gain = await meta_learning_engine._model_agnostic_meta_learning(
            skill_acquisition, 10
        )
        assert 0.0 <= maml_gain <= 0.2
        
        # Test Reptile algorithm
        reptile_gain = await meta_learning_engine._reptile_algorithm(
            skill_acquisition, 10
        )
        assert 0.0 <= reptile_gain <= 0.15
        
        # Test memory-augmented learning
        memory_gain = await meta_learning_engine._memory_augmented_learning(
            skill_acquisition, 10
        )
        assert 0.0 <= memory_gain <= 0.12
    
    @pytest.mark.asyncio
    async def test_transfer_source_identification(self, meta_learning_engine):
        """Test identification of transfer learning sources."""
        # Add some skills to the inventory
        meta_learning_engine.state.skill_inventory = [
            SkillAcquisition(
                skill_name="data_analysis",
                domain="analytical",
                acquisition_strategy=LearningStrategy.GRADIENT_BASED,
                learning_curve=[0.1, 0.3, 0.7, 0.9],
                milestones=[],
                transfer_sources=[],
                mastery_level=0.9,
                acquisition_time=5.0,
                retention_score=0.8
            ),
            SkillAcquisition(
                skill_name="pattern_recognition",
                domain="technical",
                acquisition_strategy=LearningStrategy.MODEL_AGNOSTIC,
                learning_curve=[0.2, 0.5, 0.8],
                milestones=[],
                transfer_sources=[],
                mastery_level=0.8,
                acquisition_time=3.0,
                retention_score=0.7
            )
        ]
        
        transfer_sources = await meta_learning_engine._identify_transfer_sources("strategic")
        
        assert isinstance(transfer_sources, list)
        assert len(transfer_sources) <= 5
    
    @pytest.mark.asyncio
    async def test_learning_insights_extraction(self, meta_learning_engine):
        """Test extraction of learning insights."""
        skill_acquisition = SkillAcquisition(
            skill_name="test_skill",
            domain="test_domain",
            acquisition_strategy=LearningStrategy.GRADIENT_BASED,
            learning_curve=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            milestones=[],
            transfer_sources=[],
            mastery_level=0.9,
            acquisition_time=10.0,
            retention_score=0.8
        )
        
        insights = await meta_learning_engine._extract_learning_insights(skill_acquisition)
        
        assert isinstance(insights, dict)
        assert "learning_rate" in insights
        assert "stability" in insights
        assert "acceleration" in insights
        assert "efficiency" in insights
    
    def test_learning_statistics(self, meta_learning_engine):
        """Test learning statistics calculation."""
        # Add some test data
        meta_learning_engine.state.skill_inventory = [
            SkillAcquisition(
                skill_name="skill1",
                domain="domain1",
                acquisition_strategy=LearningStrategy.GRADIENT_BASED,
                learning_curve=[],
                milestones=[],
                transfer_sources=[],
                mastery_level=0.8,
                acquisition_time=5.0,
                retention_score=0.7
            ),
            SkillAcquisition(
                skill_name="skill2",
                domain="domain2",
                acquisition_strategy=LearningStrategy.EVOLUTIONARY,
                learning_curve=[],
                milestones=[],
                transfer_sources=[],
                mastery_level=0.9,
                acquisition_time=3.0,
                retention_score=0.8
            )
        ]
        
        stats = meta_learning_engine.get_learning_statistics()
        
        assert isinstance(stats, dict)
        assert stats["total_skills"] == 2
        assert stats["average_mastery"] == 0.85
        assert stats["domains_covered"] == 2
        assert "learning_efficiency" in stats


class TestAdaptationEngine:
    """Test suite for AdaptationEngine."""
    
    @pytest.fixture
    def adaptation_engine(self):
        """Create AdaptationEngine instance for testing."""
        return AdaptationEngine()
    
    @pytest.mark.asyncio
    async def test_adapt_to_environment(self, adaptation_engine):
        """Test environment adaptation functionality."""
        environment_description = {
            "type": "dynamic_market",
            "complexity": 0.7,
            "volatility": 0.6,
            "variables": ["demand", "competition", "regulation"],
            "constraints": ["budget", "time"],
            "opportunities": ["new_market", "technology_advancement"]
        }
        
        adaptation_result = await adaptation_engine.adapt_to_environment(
            environment_description, "high"
        )
        
        assert isinstance(adaptation_result, dict)
        assert "adaptation_results" in adaptation_result
        assert "effectiveness" in adaptation_result
        assert "performance_improvement" in adaptation_result
        assert "adaptation_time" in adaptation_result
        assert "confidence" in adaptation_result
        
        assert 0.0 <= adaptation_result["effectiveness"] <= 1.0
        assert adaptation_result["confidence"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_handle_environmental_challenge(self, adaptation_engine):
        """Test handling of specific environmental challenges."""
        challenge = EnvironmentalChallenge(
            challenge_id="market_disruption",
            environment_type="competitive_market",
            challenge_description="New competitor with disruptive technology",
            difficulty_level=0.8,
            required_adaptations=[
                AdaptationType.STRATEGY_ADAPTATION,
                AdaptationType.PARAMETER_ADAPTATION
            ],
            success_criteria={"market_share": 0.7, "response_time": 0.9},
            time_constraints=5.0,
            resource_constraints={"budget": 0.8, "personnel": 0.6}
        )
        
        resolution_result = await adaptation_engine.handle_environmental_challenge(challenge)
        
        assert isinstance(resolution_result, dict)
        assert "resolution_results" in resolution_result
        assert "success_evaluation" in resolution_result
        assert "lessons_learned" in resolution_result
        assert "capability_improvements" in resolution_result
        
        success_eval = resolution_result["success_evaluation"]
        assert "overall_success" in success_eval
        assert 0.0 <= success_eval["overall_success"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_continuous_self_improvement(self, adaptation_engine):
        """Test continuous self-improvement mechanisms."""
        improvement_domains = ["reasoning", "learning", "adaptation"]
        
        improvement_result = await adaptation_engine.continuous_self_improvement(
            improvement_domains
        )
        
        assert isinstance(improvement_result, dict)
        assert "domain_improvements" in improvement_result
        assert "integration_results" in improvement_result
        assert "overall_improvement" in improvement_result
        assert "new_capabilities" in improvement_result
        assert "next_improvement_targets" in improvement_result
        
        domain_improvements = improvement_result["domain_improvements"]
        assert len(domain_improvements) == len(improvement_domains)
        for domain in improvement_domains:
            assert domain in domain_improvements
    
    @pytest.mark.asyncio
    async def test_adaptive_performance_optimization(self, adaptation_engine):
        """Test adaptive performance optimization."""
        performance_metrics = {
            "accuracy": 0.85,
            "speed": 0.70,
            "efficiency": 0.75,
            "robustness": 0.80
        }
        optimization_targets = ["speed", "efficiency"]
        
        optimization_result = await adaptation_engine.adaptive_performance_optimization(
            performance_metrics, optimization_targets
        )
        
        assert isinstance(optimization_result, dict)
        assert "optimization_results" in optimization_result
        assert "effectiveness" in optimization_result
        assert "performance_gains" in optimization_result
        assert "optimization_insights" in optimization_result
        
        assert 0.0 <= optimization_result["effectiveness"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_adaptation_strategies(self, adaptation_engine):
        """Test different adaptation strategies."""
        context = {
            "urgency": "high",
            "complexity": 0.7,
            "resources": {"computational": 0.8, "memory": 0.6}
        }
        
        # Test parameter adaptation
        param_result = await adaptation_engine._parameter_adaptation(context)
        assert param_result["type"] == "parameter_adaptation"
        assert "adaptations" in param_result
        assert "effectiveness" in param_result
        
        # Test architecture adaptation
        arch_result = await adaptation_engine._architecture_adaptation(context)
        assert arch_result["type"] == "architecture_adaptation"
        assert "adaptations" in arch_result
        assert "effectiveness" in arch_result
        
        # Test strategy adaptation
        strategy_result = await adaptation_engine._strategy_adaptation(context)
        assert strategy_result["type"] == "strategy_adaptation"
        assert "adaptations" in strategy_result
        assert "effectiveness" in strategy_result
        
        # Test environment adaptation
        env_result = await adaptation_engine._environment_adaptation(context)
        assert env_result["type"] == "environment_adaptation"
        assert "adaptations" in env_result
        assert "effectiveness" in env_result
    
    @pytest.mark.asyncio
    async def test_environment_analysis(self, adaptation_engine):
        """Test environment characteristic analysis."""
        environment_description = {
            "variables": ["var1", "var2", "var3"] * 10,  # 30 variables
            "uncertainty_level": 0.7,
            "constraints": ["constraint1", "constraint2"] * 5,  # 10 constraints
            "interaction_complexity": 0.8,
            "change_frequency": 0.6,
            "unpredictability": 0.5,
            "external_influences": 0.4
        }
        
        characteristics = await adaptation_engine._analyze_environment_characteristics(
            environment_description
        )
        
        assert isinstance(characteristics, dict)
        assert "complexity" in characteristics
        assert "volatility" in characteristics
        assert "resource_availability" in characteristics
        assert "constraints" in characteristics
        assert "opportunities" in characteristics
        
        assert 0.0 <= characteristics["complexity"] <= 1.0
        assert 0.0 <= characteristics["volatility"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_challenge_analysis(self, adaptation_engine):
        """Test environmental challenge analysis."""
        challenge = EnvironmentalChallenge(
            challenge_id="test_challenge",
            environment_type="test_env",
            challenge_description="Test challenge",
            difficulty_level=0.8,
            required_adaptations=[AdaptationType.PARAMETER_ADAPTATION],
            success_criteria={"metric1": 0.9, "metric2": 0.8},
            time_constraints=10.0,
            resource_constraints={"cpu": 0.8, "memory": 0.6, "storage": 0.7}
        )
        
        complexity_analysis = await adaptation_engine._analyze_challenge_complexity(challenge)
        
        assert isinstance(complexity_analysis, dict)
        assert "difficulty_score" in complexity_analysis
        assert "adaptation_types_required" in complexity_analysis
        assert "time_pressure" in complexity_analysis
        assert "resource_constraints" in complexity_analysis
        assert "success_criteria_complexity" in complexity_analysis
        
        assert complexity_analysis["difficulty_score"] == 0.8
        assert complexity_analysis["adaptation_types_required"] == 1
        assert complexity_analysis["resource_constraints"] == 3
        assert complexity_analysis["success_criteria_complexity"] == 2
    
    def test_adaptation_status(self, adaptation_engine):
        """Test adaptation status reporting."""
        # Set up some test state
        adaptation_engine.adaptation_state.current_environment = {"test": "env"}
        adaptation_engine.adaptation_state.active_adaptations = [AdaptationType.PARAMETER_ADAPTATION]
        adaptation_engine.adaptation_state.adaptation_confidence = 0.85
        adaptation_engine.adaptation_state.performance_trend = [0.7, 0.8, 0.85, 0.9]
        
        status = adaptation_engine.get_adaptation_status()
        
        assert isinstance(status, dict)
        assert "current_environment" in status
        assert "active_adaptations" in status
        assert "adaptation_confidence" in status
        assert "performance_trend" in status
        assert "adaptation_history_count" in status
        
        assert status["adaptation_confidence"] == 0.85
        assert len(status["performance_trend"]) == 4
        assert "parameter_adaptation" in status["active_adaptations"]


class TestMetaLearningAdaptationIntegration:
    """Test integration between meta-learning and adaptation engines."""
    
    @pytest.fixture
    def engines(self):
        """Create both engines for integration testing."""
        return {
            "meta_learning": MetaLearningEngine(),
            "adaptation": AdaptationEngine()
        }
    
    @pytest.mark.asyncio
    async def test_skill_acquisition_with_adaptation(self, engines):
        """Test skill acquisition that requires environmental adaptation."""
        meta_engine = engines["meta_learning"]
        adapt_engine = engines["adaptation"]
        
        # First acquire a skill
        skill = await meta_engine.rapid_skill_acquisition(
            "adaptive_reasoning", "analytical", 0.8
        )
        
        # Then adapt to a new environment that requires this skill
        environment = {
            "type": "analytical_challenge",
            "complexity": 0.9,
            "required_skills": ["adaptive_reasoning"]
        }
        
        adaptation_result = await adapt_engine.adapt_to_environment(environment)
        
        assert skill.mastery_level >= 0.0
        assert adaptation_result["effectiveness"] >= 0.0
        
        # Verify that the skill can be used in adaptation
        assert skill.skill_name in environment["required_skills"]
    
    @pytest.mark.asyncio
    async def test_transfer_learning_for_adaptation(self, engines):
        """Test using transfer learning to improve adaptation capabilities."""
        meta_engine = engines["meta_learning"]
        adapt_engine = engines["adaptation"]
        
        # Create transfer learning between adaptation-related domains
        transfer_map = await meta_engine.transfer_learning_across_domains(
            "parameter_optimization", "environment_adaptation"
        )
        
        # Use the transfer learning to improve adaptation
        challenge = EnvironmentalChallenge(
            challenge_id="adaptation_challenge",
            environment_type="dynamic_system",
            challenge_description="System requiring parameter optimization",
            difficulty_level=0.7,
            required_adaptations=[AdaptationType.PARAMETER_ADAPTATION],
            success_criteria={"optimization_score": 0.8}
        )
        
        resolution_result = await adapt_engine.handle_environmental_challenge(challenge)
        
        assert transfer_map.transfer_efficiency >= 0.0
        assert resolution_result["success_evaluation"]["overall_success"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_self_improvement_through_adaptation(self, engines):
        """Test self-improvement that uses adaptation mechanisms."""
        meta_engine = engines["meta_learning"]
        adapt_engine = engines["adaptation"]
        
        # Create self-improvement plan
        improvement_plan = await meta_engine.self_improving_algorithms(
            "adaptation_speed", 1.5
        )
        
        # Execute adaptation-based improvements
        improvement_result = await adapt_engine.continuous_self_improvement(
            ["adaptation"]
        )
        
        assert improvement_plan.target_capability > improvement_plan.current_capability
        assert "adaptation" in improvement_result["domain_improvements"]
        assert improvement_result["overall_improvement"] >= 0.0


@pytest.mark.asyncio
async def test_capability_assessment_integration():
    """Test integration of capability assessment across engines."""
    meta_engine = MetaLearningEngine()
    adapt_engine = AdaptationEngine()
    
    # Test that both engines can assess related capabilities
    meta_stats = meta_engine.get_learning_statistics()
    adapt_status = adapt_engine.get_adaptation_status()
    
    assert isinstance(meta_stats, dict)
    assert isinstance(adapt_status, dict)
    
    # Verify that both provide complementary information
    assert "learning_efficiency" in meta_stats
    assert "adaptation_confidence" in adapt_status


if __name__ == "__main__":
    pytest.main([__file__])