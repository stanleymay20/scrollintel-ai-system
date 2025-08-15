"""
Tests for intuitive reasoning engine and models.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.intuitive_reasoning_engine import IntuitiveReasoning
from scrollintel.models.intuitive_models import (
    IntuitiveInsight, PatternSynthesis, Pattern, DataPoint, Problem,
    CreativeSolution, Challenge, HolisticInsight, Context, IntuitiveLeap,
    ValidationResult, ConfidenceMetrics, InsightType, PatternComplexity,
    CreativityLevel
)


class TestIntuitiveModels:
    """Test intuitive reasoning models"""
    
    def test_intuitive_insight_creation(self):
        """Test IntuitiveInsight model creation"""
        insight = IntuitiveInsight()
        assert insight.id is not None
        assert insight.insight_type == InsightType.PATTERN_RECOGNITION
        assert insight.confidence == 0.0
        assert insight.novelty_score == 0.0
        assert insight.coherence_score == 0.0
        assert isinstance(insight.timestamp, datetime)
    
    def test_intuitive_insight_overall_score(self):
        """Test overall score calculation"""
        insight = IntuitiveInsight()
        insight.confidence = 0.8
        insight.novelty_score = 0.7
        insight.coherence_score = 0.9
        
        expected_score = 0.8 * 0.4 + 0.7 * 0.3 + 0.9 * 0.3
        assert abs(insight.calculate_overall_score() - expected_score) < 0.001
    
    def test_pattern_creation(self):
        """Test Pattern model creation"""
        pattern = Pattern()
        assert pattern.id is not None
        assert pattern.complexity == PatternComplexity.SIMPLE
        assert pattern.confidence == 0.0
        assert pattern.predictive_power == 0.0
        assert isinstance(pattern.timestamp, datetime)
    
    def test_data_point_creation(self):
        """Test DataPoint model creation"""
        data_point = DataPoint()
        data_point.value = "test_value"
        data_point.domain = "test_domain"
        
        assert data_point.id is not None
        assert data_point.value == "test_value"
        assert data_point.domain == "test_domain"
        assert isinstance(data_point.timestamp, datetime)
    
    def test_creative_solution_quality(self):
        """Test creative solution quality calculation"""
        solution = CreativeSolution()
        solution.feasibility_score = 0.8
        solution.innovation_score = 0.9
        solution.elegance_score = 0.7
        
        expected_quality = 0.8 * 0.3 + 0.9 * 0.4 + 0.7 * 0.3
        assert abs(solution.calculate_solution_quality() - expected_quality) < 0.001
    
    def test_confidence_metrics_weighted_calculation(self):
        """Test confidence metrics weighted calculation"""
        metrics = ConfidenceMetrics()
        metrics.pattern_confidence = 0.8
        metrics.synthesis_confidence = 0.7
        metrics.creativity_confidence = 0.9
        metrics.validation_confidence = 0.6
        
        # Test default weights
        expected_weighted = (0.8 + 0.7 + 0.9 + 0.6) / 4
        assert abs(metrics.calculate_weighted_confidence() - expected_weighted) < 0.001
        
        # Test custom weights
        custom_weights = {'pattern': 0.4, 'synthesis': 0.3, 'creativity': 0.2, 'validation': 0.1}
        expected_custom = 0.8 * 0.4 + 0.7 * 0.3 + 0.9 * 0.2 + 0.6 * 0.1
        assert abs(metrics.calculate_weighted_confidence(custom_weights) - expected_custom) < 0.001


class TestIntuitiveReasoningEngine:
    """Test intuitive reasoning engine"""
    
    @pytest.fixture
    def reasoning_engine(self):
        """Create reasoning engine fixture"""
        return IntuitiveReasoning()
    
    @pytest.fixture
    def sample_problem(self):
        """Create sample problem fixture"""
        problem = Problem()
        problem.description = "Complex optimization challenge"
        problem.domain = "optimization"
        problem.complexity_level = 0.8
        problem.constraints = ["time_limit", "resource_constraint"]
        problem.objectives = ["maximize_efficiency", "minimize_cost"]
        problem.context = {"urgency": "high", "stakeholders": ["team_a", "team_b"]}
        return problem
    
    @pytest.fixture
    def sample_data_points(self):
        """Create sample data points fixture"""
        data_points = []
        for i in range(10):
            point = DataPoint()
            point.value = f"value_{i}"
            point.domain = "test_domain" if i % 2 == 0 else "other_domain"
            point.context = {"index": i}
            if i > 5:
                point.relationships = [f"rel_{i-1}", f"rel_{i+1}"]
            data_points.append(point)
        return data_points
    
    @pytest.fixture
    def sample_challenge(self):
        """Create sample challenge fixture"""
        challenge = Challenge()
        challenge.title = "Innovation Challenge"
        challenge.description = "Develop breakthrough solution"
        challenge.challenge_type = "innovation"
        challenge.difficulty_level = 0.9
        challenge.resource_constraints = ["limited_budget", "tight_timeline"]
        challenge.success_metrics = ["novelty", "feasibility", "impact"]
        challenge.context_factors = {"market_pressure": "high", "competition": "intense"}
        return challenge
    
    @pytest.fixture
    def sample_context(self):
        """Create sample context fixture"""
        context = Context()
        context.situation = "Complex system analysis"
        context.domain = "systems_engineering"
        context.environmental_factors = {"complexity": "high", "uncertainty": "medium"}
        context.constraints = ["regulatory", "technical", "financial"]
        context.opportunities = ["innovation", "efficiency", "scalability"]
        context.uncertainty_level = 0.6
        context.ambiguity_level = 0.4
        return context
    
    @pytest.mark.asyncio
    async def test_generate_intuitive_leap(self, reasoning_engine, sample_problem):
        """Test intuitive leap generation"""
        insight = await reasoning_engine.generate_intuitive_leap(sample_problem)
        
        assert isinstance(insight, IntuitiveInsight)
        assert insight.insight_type == InsightType.CREATIVE_LEAP
        assert insight.confidence > 0.0
        assert insight.novelty_score > 0.0
        assert insight.coherence_score > 0.0
        assert len(insight.validation_criteria) > 0
        assert len(insight.potential_applications) > 0
        assert insight in reasoning_engine.insight_history
    
    @pytest.mark.asyncio
    async def test_synthesize_patterns(self, reasoning_engine, sample_data_points):
        """Test pattern synthesis"""
        synthesis = await reasoning_engine.synthesize_patterns(sample_data_points)
        
        assert isinstance(synthesis, PatternSynthesis)
        assert synthesis.synthesis_method == "holistic_emergence_synthesis"
        assert len(synthesis.input_patterns) > 0
        assert synthesis.synthesized_pattern is not None
        assert synthesis.synthesis_confidence > 0.0
        assert len(synthesis.emergence_properties) > 0
        assert synthesis in reasoning_engine.synthesis_history
    
    @pytest.mark.asyncio
    async def test_creative_problem_solving(self, reasoning_engine, sample_challenge):
        """Test creative problem solving"""
        solution = await reasoning_engine.creative_problem_solving(sample_challenge)
        
        assert isinstance(solution, CreativeSolution)
        assert solution.problem_id == sample_challenge.id
        assert solution.solution_description != ""
        assert isinstance(solution.creativity_level, CreativityLevel)
        assert solution.feasibility_score > 0.0
        assert solution.innovation_score > 0.0
        assert solution.elegance_score > 0.0
        assert len(solution.implementation_steps) > 0
        assert len(solution.required_resources) > 0
        assert len(solution.expected_outcomes) > 0
    
    @pytest.mark.asyncio
    async def test_holistic_understanding(self, reasoning_engine, sample_context):
        """Test holistic understanding generation"""
        insight = await reasoning_engine.holistic_understanding(sample_context)
        
        assert isinstance(insight, HolisticInsight)
        assert insight.system_description == sample_context.situation
        assert len(insight.emergent_properties) > 0
        assert len(insight.system_dynamics) > 0
        assert len(insight.interconnections) > 0
        assert len(insight.leverage_points) > 0
        assert len(insight.system_archetypes) > 0
        assert len(insight.feedback_loops) > 0
        assert len(insight.boundary_conditions) > 0
        assert insight.holistic_understanding_score > 0.0
    
    @pytest.mark.asyncio
    async def test_validate_intuition(self, reasoning_engine):
        """Test intuition validation"""
        # Create a sample insight
        insight = IntuitiveInsight()
        insight.description = "Test insight for validation"
        insight.confidence = 0.8
        insight.novelty_score = 0.7
        insight.coherence_score = 0.9
        
        validation = await reasoning_engine.validate_intuition(insight)
        
        assert isinstance(validation, ValidationResult)
        assert validation.insight_id == insight.id
        assert validation.validation_method == "multi_criteria_validation"
        assert validation.validation_score > 0.0
        assert validation.consistency_score > 0.0
        assert validation.evidence_strength > 0.0
        assert validation.predictive_accuracy > 0.0
        assert len(validation.peer_validation) > 0
        assert len(validation.empirical_support) > 0
        assert len(validation.theoretical_grounding) > 0
        assert len(validation.limitations_identified) > 0
        assert len(validation.confidence_intervals) > 0
        
        # Test caching
        validation2 = await reasoning_engine.validate_intuition(insight)
        assert validation2 is validation  # Should return cached result
    
    @pytest.mark.asyncio
    async def test_calculate_confidence_score(self, reasoning_engine):
        """Test confidence score calculation"""
        insight = IntuitiveInsight()
        insight.description = "Test insight for confidence calculation"
        insight.confidence = 0.8
        insight.novelty_score = 0.7
        insight.coherence_score = 0.9
        
        metrics = await reasoning_engine.calculate_confidence_score(insight)
        
        assert isinstance(metrics, ConfidenceMetrics)
        assert metrics.overall_confidence > 0.0
        assert metrics.pattern_confidence > 0.0
        assert metrics.synthesis_confidence > 0.0
        assert metrics.creativity_confidence > 0.0
        assert metrics.validation_confidence > 0.0
        assert len(metrics.uncertainty_quantification) > 0
        assert len(metrics.confidence_sources) > 0
        assert len(metrics.confidence_degradation_factors) > 0
    
    @pytest.mark.asyncio
    async def test_pattern_discovery_with_empty_data(self, reasoning_engine):
        """Test pattern synthesis with empty data"""
        empty_data = []
        synthesis = await reasoning_engine.synthesize_patterns(empty_data)
        
        assert isinstance(synthesis, PatternSynthesis)
        assert len(synthesis.input_patterns) == 0
        assert synthesis.synthesized_pattern is not None  # Should create default pattern
        assert synthesis.synthesis_confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_cross_domain_pattern_discovery(self, reasoning_engine):
        """Test cross-domain pattern discovery"""
        # Create data points from different domains
        data_points = []
        domains = ["domain_a", "domain_b", "domain_c"]
        
        for i, domain in enumerate(domains * 3):  # 9 points across 3 domains
            point = DataPoint()
            point.value = f"value_{i}"
            point.domain = domain
            point.relationships = [f"rel_{j}" for j in range(i % 3)]
            data_points.append(point)
        
        synthesis = await reasoning_engine.synthesize_patterns(data_points)
        
        assert len(synthesis.input_patterns) > 0
        assert len(synthesis.cross_domain_bridges) > 0
        
        # Check that cross-domain patterns were found
        cross_domain_patterns = [p for p in synthesis.input_patterns if len(p.domains) > 1]
        assert len(cross_domain_patterns) > 0 or len(synthesis.cross_domain_bridges) > 0
    
    @pytest.mark.asyncio
    async def test_emergence_property_identification(self, reasoning_engine, sample_data_points):
        """Test emergence property identification"""
        synthesis = await reasoning_engine.synthesize_patterns(sample_data_points)
        
        # Should identify emergence properties
        assert len(synthesis.emergence_properties) > 0
        
        # Common emergence properties should be present
        emergence_types = synthesis.emergence_properties
        possible_emergences = [
            "cross_domain_emergence", "complexity_emergence", 
            "confidence_emergence", "predictive_emergence", "scale_emergence"
        ]
        
        # At least one emergence property should be identified
        assert any(prop in possible_emergences for prop in emergence_types)
    
    @pytest.mark.asyncio
    async def test_creative_solution_quality_metrics(self, reasoning_engine, sample_challenge):
        """Test creative solution quality metrics"""
        solution = await reasoning_engine.creative_problem_solving(sample_challenge)
        
        # Test quality calculation
        quality_score = solution.calculate_solution_quality()
        assert 0.0 <= quality_score <= 1.0
        
        # Quality should be reasonable for a creative solution
        assert quality_score > 0.5  # Should be above average
        
        # Individual scores should be reasonable
        assert 0.0 <= solution.feasibility_score <= 1.0
        assert 0.0 <= solution.innovation_score <= 1.0
        assert 0.0 <= solution.elegance_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_holistic_insight_completeness(self, reasoning_engine, sample_context):
        """Test completeness of holistic insights"""
        insight = await reasoning_engine.holistic_understanding(sample_context)
        
        # All required components should be present
        assert insight.system_description != ""
        assert len(insight.emergent_properties) > 0
        assert len(insight.system_dynamics) > 0
        assert len(insight.interconnections) > 0
        assert len(insight.leverage_points) > 0
        assert len(insight.system_archetypes) > 0
        assert len(insight.feedback_loops) > 0
        assert len(insight.boundary_conditions) > 0
        
        # Holistic understanding score should be reasonable
        assert 0.0 <= insight.holistic_understanding_score <= 1.0
        assert insight.holistic_understanding_score > 0.3  # Should have some understanding
    
    @pytest.mark.asyncio
    async def test_validation_result_completeness(self, reasoning_engine):
        """Test validation result completeness"""
        insight = IntuitiveInsight()
        insight.description = "Comprehensive test insight"
        insight.confidence = 0.85
        insight.novelty_score = 0.75
        insight.coherence_score = 0.90
        
        validation = await reasoning_engine.validate_intuition(insight)
        
        # All validation components should be present
        assert validation.validation_score > 0.0
        assert validation.consistency_score > 0.0
        assert validation.evidence_strength > 0.0
        assert validation.predictive_accuracy > 0.0
        assert len(validation.peer_validation) > 0
        assert len(validation.empirical_support) > 0
        assert len(validation.theoretical_grounding) > 0
        assert len(validation.limitations_identified) > 0
        assert len(validation.confidence_intervals) > 0
        
        # Confidence intervals should be reasonable
        for metric, (lower, upper) in validation.confidence_intervals.items():
            assert lower <= upper
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0
    
    def test_engine_initialization(self, reasoning_engine):
        """Test engine initialization"""
        assert len(reasoning_engine.insight_history) == 0
        assert len(reasoning_engine.pattern_database) == 0
        assert len(reasoning_engine.synthesis_history) == 0
        assert len(reasoning_engine.neural_architectures) == 0
        assert len(reasoning_engine.validation_cache) == 0
        assert len(reasoning_engine.cross_domain_connections) == 0
        assert reasoning_engine.emergence_threshold == 0.7
        assert reasoning_engine.creativity_boost_factor == 1.2
    
    @pytest.mark.asyncio
    async def test_insight_history_tracking(self, reasoning_engine, sample_problem):
        """Test that insights are properly tracked in history"""
        initial_count = len(reasoning_engine.insight_history)
        
        # Generate multiple insights
        insight1 = await reasoning_engine.generate_intuitive_leap(sample_problem)
        insight2 = await reasoning_engine.generate_intuitive_leap(sample_problem)
        
        assert len(reasoning_engine.insight_history) == initial_count + 2
        assert insight1 in reasoning_engine.insight_history
        assert insight2 in reasoning_engine.insight_history
        assert insight1.id != insight2.id  # Should have unique IDs
    
    @pytest.mark.asyncio
    async def test_synthesis_history_tracking(self, reasoning_engine, sample_data_points):
        """Test that syntheses are properly tracked in history"""
        initial_count = len(reasoning_engine.synthesis_history)
        
        # Generate multiple syntheses
        synthesis1 = await reasoning_engine.synthesize_patterns(sample_data_points[:5])
        synthesis2 = await reasoning_engine.synthesize_patterns(sample_data_points[5:])
        
        assert len(reasoning_engine.synthesis_history) == initial_count + 2
        assert synthesis1 in reasoning_engine.synthesis_history
        assert synthesis2 in reasoning_engine.synthesis_history
        assert synthesis1.id != synthesis2.id  # Should have unique IDs


class TestIntuitiveReasoningIntegration:
    """Integration tests for intuitive reasoning components"""
    
    @pytest.fixture
    def reasoning_engine(self):
        return IntuitiveReasoning()
    
    @pytest.mark.asyncio
    async def test_end_to_end_creative_problem_solving(self, reasoning_engine):
        """Test end-to-end creative problem solving workflow"""
        # Create a complex challenge
        challenge = Challenge()
        challenge.title = "AI Safety Challenge"
        challenge.description = "Develop safe AGI alignment mechanisms"
        challenge.challenge_type = "safety_critical"
        challenge.difficulty_level = 0.95
        challenge.resource_constraints = ["computational_limits", "theoretical_gaps"]
        challenge.success_metrics = ["safety_guarantee", "alignment_verification", "scalability"]
        challenge.context_factors = {
            "urgency": "critical",
            "stakes": "existential",
            "complexity": "unprecedented"
        }
        
        # Solve the challenge
        solution = await reasoning_engine.creative_problem_solving(challenge)
        
        # Validate the solution meets requirements
        assert isinstance(solution, CreativeSolution)
        assert solution.problem_id == challenge.id
        assert solution.creativity_level in [CreativityLevel.BREAKTHROUGH, CreativityLevel.REVOLUTIONARY, CreativityLevel.INNOVATIVE]
        assert solution.calculate_solution_quality() > 0.6  # Should be high quality
        
        # Solution should address the challenge complexity
        assert len(solution.implementation_steps) >= 3
        assert len(solution.required_resources) >= 2
        assert len(solution.expected_outcomes) >= 2
    
    @pytest.mark.asyncio
    async def test_pattern_synthesis_to_insight_generation(self, reasoning_engine):
        """Test workflow from pattern synthesis to insight generation"""
        # Create diverse data points
        data_points = []
        domains = ["neuroscience", "computer_science", "philosophy", "physics"]
        
        for i in range(20):
            point = DataPoint()
            point.value = f"insight_{i}"
            point.domain = domains[i % len(domains)]
            point.context = {"complexity": i / 20.0, "novelty": (i * 7) % 10 / 10.0}
            point.relationships = [f"rel_{j}" for j in range(i % 4)]
            data_points.append(point)
        
        # Synthesize patterns
        synthesis = await reasoning_engine.synthesize_patterns(data_points)
        
        # Create a problem based on the synthesized patterns
        problem = Problem()
        problem.description = "Apply synthesized patterns to novel problem"
        problem.domain = "interdisciplinary"
        problem.complexity_level = 0.8
        problem.context = {"synthesis_id": synthesis.id}
        
        # Generate insight based on the problem
        insight = await reasoning_engine.generate_intuitive_leap(problem)
        
        # Validate the insight
        validation = await reasoning_engine.validate_intuition(insight)
        
        # Calculate confidence
        confidence = await reasoning_engine.calculate_confidence_score(insight)
        
        # Verify the workflow produced meaningful results
        assert synthesis.synthesis_confidence > 0.5
        assert insight.confidence > 0.4
        assert validation.validation_score > 0.4
        assert confidence.overall_confidence > 0.4
        
        # Verify cross-domain connections were made
        assert len(synthesis.cross_domain_bridges) > 0
        assert len(insight.cross_domain_connections) > 0
    
    @pytest.mark.asyncio
    async def test_holistic_understanding_to_creative_solution(self, reasoning_engine):
        """Test workflow from holistic understanding to creative solution"""
        # Create complex context
        context = Context()
        context.situation = "Global climate change mitigation system"
        context.domain = "environmental_systems"
        context.environmental_factors = {
            "temperature_rise": 1.5,
            "co2_levels": 420,
            "ecosystem_stress": 0.8
        }
        context.constraints = [
            "economic_feasibility", "political_acceptance", 
            "technological_readiness", "time_urgency"
        ]
        context.opportunities = [
            "renewable_energy", "carbon_capture", 
            "behavioral_change", "policy_innovation"
        ]
        context.uncertainty_level = 0.7
        context.ambiguity_level = 0.6
        
        # Develop holistic understanding
        holistic_insight = await reasoning_engine.holistic_understanding(context)
        
        # Create challenge based on holistic understanding
        challenge = Challenge()
        challenge.title = "Climate Solution Challenge"
        challenge.description = "Develop comprehensive climate mitigation strategy"
        challenge.challenge_type = "systems_intervention"
        challenge.difficulty_level = 0.9
        challenge.context_factors = {
            "holistic_insight_id": holistic_insight.id,
            "leverage_points": holistic_insight.leverage_points,
            "system_archetypes": holistic_insight.system_archetypes
        }
        
        # Generate creative solution
        solution = await reasoning_engine.creative_problem_solving(challenge)
        
        # Verify the solution leverages holistic understanding
        assert holistic_insight.holistic_understanding_score > 0.5
        assert len(holistic_insight.leverage_points) > 0
        assert len(holistic_insight.feedback_loops) > 0
        
        assert solution.creativity_level in [CreativityLevel.INNOVATIVE, CreativityLevel.BREAKTHROUGH]
        assert solution.calculate_solution_quality() > 0.6
        assert len(solution.implementation_steps) >= 4  # Complex solution needs multiple steps
    
    @pytest.mark.asyncio
    async def test_validation_feedback_loop(self, reasoning_engine):
        """Test validation feedback improving insight quality"""
        # Generate initial insight
        problem = Problem()
        problem.description = "Quantum computing optimization"
        problem.domain = "quantum_computing"
        problem.complexity_level = 0.9
        
        insight = await reasoning_engine.generate_intuitive_leap(problem)
        initial_confidence = insight.confidence
        
        # Validate insight
        validation = await reasoning_engine.validate_intuition(insight)
        
        # Calculate comprehensive confidence
        confidence_metrics = await reasoning_engine.calculate_confidence_score(insight)
        
        # Verify validation provides meaningful feedback
        assert validation.validation_score > 0.0
        assert len(validation.limitations_identified) > 0
        assert len(validation.peer_validation) > 0
        
        # Confidence metrics should incorporate validation
        assert confidence_metrics.validation_confidence == validation.validation_score
        assert confidence_metrics.overall_confidence > 0.0
        
        # Overall confidence should be reasonable combination of factors
        expected_range = (0.3, 0.95)
        assert expected_range[0] <= confidence_metrics.overall_confidence <= expected_range[1]


if __name__ == "__main__":
    pytest.main([__file__])