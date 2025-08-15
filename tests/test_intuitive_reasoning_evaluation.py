"""
Evaluation metrics and benchmarks for intuitive reasoning engine.
Tests the quality, creativity, and effectiveness of intuitive insights.
"""

import pytest
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

from scrollintel.engines.intuitive_reasoning_engine import IntuitiveReasoning
from scrollintel.models.intuitive_models import (
    IntuitiveInsight, PatternSynthesis, Pattern, DataPoint, Problem,
    CreativeSolution, Challenge, HolisticInsight, Context,
    InsightType, PatternComplexity, CreativityLevel
)


class IntuitiveReasoningEvaluator:
    """Evaluator for intuitive reasoning capabilities"""
    
    def __init__(self, reasoning_engine: IntuitiveReasoning):
        self.reasoning_engine = reasoning_engine
        self.evaluation_history: List[Dict[str, Any]] = []
    
    async def evaluate_insight_quality(self, insights: List[IntuitiveInsight]) -> Dict[str, float]:
        """Evaluate the quality of generated insights"""
        if not insights:
            return {
                "overall_quality": 0.0, 
                "confidence": 0.0, 
                "novelty": 0.0, 
                "coherence": 0.0,
                "confidence_std": 0.0,
                "novelty_std": 0.0,
                "coherence_std": 0.0,
                "count": 0
            }
        
        confidences = [insight.confidence for insight in insights]
        novelties = [insight.novelty_score for insight in insights]
        coherences = [insight.coherence_score for insight in insights]
        overall_scores = [insight.calculate_overall_score() for insight in insights]
        
        return {
            "overall_quality": statistics.mean(overall_scores),
            "confidence": statistics.mean(confidences),
            "novelty": statistics.mean(novelties),
            "coherence": statistics.mean(coherences),
            "confidence_std": statistics.stdev(confidences) if len(confidences) > 1 else 0.0,
            "novelty_std": statistics.stdev(novelties) if len(novelties) > 1 else 0.0,
            "coherence_std": statistics.stdev(coherences) if len(coherences) > 1 else 0.0,
            "count": len(insights)
        }
    
    async def evaluate_creativity_levels(self, solutions: List[CreativeSolution]) -> Dict[str, Any]:
        """Evaluate creativity levels of solutions"""
        if not solutions:
            return {
                "creativity_distribution": {}, 
                "average_creativity": 0.0,
                "average_innovation": 0.0,
                "average_elegance": 0.0,
                "average_quality": 0.0,
                "innovation_std": 0.0,
                "count": 0
            }
        
        creativity_counts = {}
        innovation_scores = []
        elegance_scores = []
        quality_scores = []
        
        for solution in solutions:
            level = solution.creativity_level.value
            creativity_counts[level] = creativity_counts.get(level, 0) + 1
            innovation_scores.append(solution.innovation_score)
            elegance_scores.append(solution.elegance_score)
            quality_scores.append(solution.calculate_solution_quality())
        
        # Calculate creativity level distribution
        total_solutions = len(solutions)
        creativity_distribution = {
            level: count / total_solutions 
            for level, count in creativity_counts.items()
        }
        
        # Calculate average creativity score based on levels
        creativity_weights = {
            "conventional": 0.2,
            "adaptive": 0.4,
            "innovative": 0.6,
            "breakthrough": 0.8,
            "revolutionary": 1.0
        }
        
        weighted_creativity = sum(
            creativity_weights.get(level, 0.5) * count 
            for level, count in creativity_counts.items()
        ) / total_solutions
        
        return {
            "creativity_distribution": creativity_distribution,
            "average_creativity": weighted_creativity,
            "average_innovation": statistics.mean(innovation_scores),
            "average_elegance": statistics.mean(elegance_scores),
            "average_quality": statistics.mean(quality_scores),
            "innovation_std": statistics.stdev(innovation_scores) if len(innovation_scores) > 1 else 0.0,
            "count": total_solutions
        }
    
    async def evaluate_pattern_synthesis_effectiveness(self, syntheses: List[PatternSynthesis]) -> Dict[str, float]:
        """Evaluate effectiveness of pattern synthesis"""
        if not syntheses:
            return {
                "synthesis_effectiveness": 0.0,
                "average_emergence_properties": 0.0,
                "average_cross_domain_bridges": 0.0,
                "average_input_patterns": 0.0,
                "average_complexity_score": 0.0,
                "synthesis_confidence_std": 0.0,
                "count": 0
            }
        
        synthesis_confidences = [s.synthesis_confidence for s in syntheses]
        emergence_counts = [len(s.emergence_properties) for s in syntheses]
        bridge_counts = [len(s.cross_domain_bridges) for s in syntheses]
        input_pattern_counts = [len(s.input_patterns) for s in syntheses]
        
        # Calculate complexity scores
        complexity_scores = []
        for synthesis in syntheses:
            if synthesis.synthesized_pattern:
                if synthesis.synthesized_pattern.complexity == PatternComplexity.EMERGENT:
                    complexity_scores.append(1.0)
                elif synthesis.synthesized_pattern.complexity == PatternComplexity.HIGHLY_COMPLEX:
                    complexity_scores.append(0.8)
                elif synthesis.synthesized_pattern.complexity == PatternComplexity.COMPLEX:
                    complexity_scores.append(0.6)
                else:
                    complexity_scores.append(0.4)
            else:
                complexity_scores.append(0.0)
        
        return {
            "synthesis_effectiveness": statistics.mean(synthesis_confidences),
            "average_emergence_properties": statistics.mean(emergence_counts),
            "average_cross_domain_bridges": statistics.mean(bridge_counts),
            "average_input_patterns": statistics.mean(input_pattern_counts),
            "average_complexity_score": statistics.mean(complexity_scores) if complexity_scores else 0.0,
            "synthesis_confidence_std": statistics.stdev(synthesis_confidences) if len(synthesis_confidences) > 1 else 0.0,
            "count": len(syntheses)
        }
    
    async def evaluate_holistic_understanding_depth(self, insights: List[HolisticInsight]) -> Dict[str, float]:
        """Evaluate depth of holistic understanding"""
        if not insights:
            return {
                "holistic_depth": 0.0,
                "average_emergent_properties": 0.0,
                "average_leverage_points": 0.0,
                "average_feedback_loops": 0.0,
                "average_interconnections": 0.0,
                "average_system_archetypes": 0.0,
                "understanding_score_std": 0.0,
                "count": 0
            }
        
        understanding_scores = [insight.holistic_understanding_score for insight in insights]
        emergent_property_counts = [len(insight.emergent_properties) for insight in insights]
        leverage_point_counts = [len(insight.leverage_points) for insight in insights]
        feedback_loop_counts = [len(insight.feedback_loops) for insight in insights]
        interconnection_counts = [len(insight.interconnections) for insight in insights]
        archetype_counts = [len(insight.system_archetypes) for insight in insights]
        
        return {
            "holistic_depth": statistics.mean(understanding_scores),
            "average_emergent_properties": statistics.mean(emergent_property_counts),
            "average_leverage_points": statistics.mean(leverage_point_counts),
            "average_feedback_loops": statistics.mean(feedback_loop_counts),
            "average_interconnections": statistics.mean(interconnection_counts),
            "average_system_archetypes": statistics.mean(archetype_counts),
            "understanding_score_std": statistics.stdev(understanding_scores) if len(understanding_scores) > 1 else 0.0,
            "count": len(insights)
        }
    
    async def benchmark_insight_generation_speed(self, problem_count: int = 10) -> Dict[str, float]:
        """Benchmark insight generation speed"""
        problems = []
        for i in range(problem_count):
            problem = Problem()
            problem.description = f"Benchmark problem {i}"
            problem.domain = f"domain_{i % 3}"
            problem.complexity_level = 0.5 + (i / problem_count) * 0.4  # Varying complexity
            problems.append(problem)
        
        start_time = datetime.now()
        insights = []
        
        for problem in problems:
            insight = await self.reasoning_engine.generate_intuitive_leap(problem)
            insights.append(insight)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        return {
            "total_time_seconds": total_time,
            "average_time_per_insight": total_time / problem_count,
            "insights_per_second": problem_count / total_time,
            "problem_count": problem_count,
            "average_insight_quality": statistics.mean([i.calculate_overall_score() for i in insights])
        }
    
    async def benchmark_pattern_synthesis_scalability(self, data_point_counts: List[int]) -> Dict[str, Any]:
        """Benchmark pattern synthesis scalability with varying data sizes"""
        results = {}
        
        for count in data_point_counts:
            # Generate data points
            data_points = []
            for i in range(count):
                point = DataPoint()
                point.value = f"data_{i}"
                point.domain = f"domain_{i % 5}"  # 5 different domains
                point.context = {"index": i, "complexity": i / count}
                if i > count // 2:  # Add relationships to some points
                    point.relationships = [f"rel_{j}" for j in range(i % 3)]
                data_points.append(point)
            
            # Measure synthesis time and quality
            start_time = datetime.now()
            synthesis = await self.reasoning_engine.synthesize_patterns(data_points)
            end_time = datetime.now()
            
            processing_time = (end_time - start_time).total_seconds()
            
            results[count] = {
                "processing_time": processing_time,
                "synthesis_confidence": synthesis.synthesis_confidence,
                "input_patterns_found": len(synthesis.input_patterns),
                "emergence_properties": len(synthesis.emergence_properties),
                "cross_domain_bridges": len(synthesis.cross_domain_bridges),
                "data_points_per_second": count / processing_time if processing_time > 0 else float('inf')
            }
        
        return results
    
    async def evaluate_cross_domain_connectivity(self, insights: List[IntuitiveInsight]) -> Dict[str, float]:
        """Evaluate cross-domain connectivity in insights"""
        if not insights:
            return {"cross_domain_score": 0.0}
        
        cross_domain_counts = [len(insight.cross_domain_connections) for insight in insights]
        insights_with_connections = sum(1 for count in cross_domain_counts if count > 0)
        
        # Analyze connection diversity
        all_connections = []
        for insight in insights:
            all_connections.extend(insight.cross_domain_connections)
        
        unique_connections = len(set(all_connections))
        total_connections = len(all_connections)
        
        return {
            "cross_domain_score": statistics.mean(cross_domain_counts),
            "connection_coverage": insights_with_connections / len(insights),
            "connection_diversity": unique_connections / max(1, total_connections),
            "average_connections_per_insight": statistics.mean(cross_domain_counts),
            "max_connections": max(cross_domain_counts) if cross_domain_counts else 0,
            "unique_connection_types": unique_connections
        }
    
    async def comprehensive_evaluation(self, test_problems: List[Problem], 
                                     test_data: List[DataPoint],
                                     test_challenges: List[Challenge],
                                     test_contexts: List[Context]) -> Dict[str, Any]:
        """Perform comprehensive evaluation of all capabilities"""
        evaluation_start = datetime.now()
        
        # Generate insights
        insights = []
        for problem in test_problems:
            insight = await self.reasoning_engine.generate_intuitive_leap(problem)
            insights.append(insight)
        
        # Generate pattern syntheses
        syntheses = []
        # Split data into chunks for multiple syntheses
        chunk_size = max(1, len(test_data) // max(1, len(test_problems)))
        for i in range(0, len(test_data), chunk_size):
            chunk = test_data[i:i + chunk_size]
            if chunk:  # Only process non-empty chunks
                synthesis = await self.reasoning_engine.synthesize_patterns(chunk)
                syntheses.append(synthesis)
        
        # Generate creative solutions
        solutions = []
        for challenge in test_challenges:
            solution = await self.reasoning_engine.creative_problem_solving(challenge)
            solutions.append(solution)
        
        # Generate holistic insights
        holistic_insights = []
        for context in test_contexts:
            holistic_insight = await self.reasoning_engine.holistic_understanding(context)
            holistic_insights.append(holistic_insight)
        
        evaluation_end = datetime.now()
        total_evaluation_time = (evaluation_end - evaluation_start).total_seconds()
        
        # Evaluate all components
        insight_quality = await self.evaluate_insight_quality(insights)
        creativity_evaluation = await self.evaluate_creativity_levels(solutions)
        synthesis_effectiveness = await self.evaluate_pattern_synthesis_effectiveness(syntheses)
        holistic_depth = await self.evaluate_holistic_understanding_depth(holistic_insights)
        cross_domain_connectivity = await self.evaluate_cross_domain_connectivity(insights)
        
        # Calculate overall performance score
        overall_score = (
            insight_quality["overall_quality"] * 0.25 +
            creativity_evaluation["average_creativity"] * 0.25 +
            synthesis_effectiveness["synthesis_effectiveness"] * 0.25 +
            holistic_depth["holistic_depth"] * 0.25
        )
        
        comprehensive_results = {
            "overall_performance_score": overall_score,
            "evaluation_time_seconds": total_evaluation_time,
            "insight_quality": insight_quality,
            "creativity_evaluation": creativity_evaluation,
            "synthesis_effectiveness": synthesis_effectiveness,
            "holistic_understanding_depth": holistic_depth,
            "cross_domain_connectivity": cross_domain_connectivity,
            "test_counts": {
                "problems": len(test_problems),
                "data_points": len(test_data),
                "challenges": len(test_challenges),
                "contexts": len(test_contexts)
            },
            "generated_counts": {
                "insights": len(insights),
                "syntheses": len(syntheses),
                "solutions": len(solutions),
                "holistic_insights": len(holistic_insights)
            }
        }
        
        # Store evaluation in history
        self.evaluation_history.append({
            "timestamp": evaluation_end,
            "results": comprehensive_results
        })
        
        return comprehensive_results


class TestIntuitiveReasoningEvaluation:
    """Test suite for intuitive reasoning evaluation"""
    
    @pytest.fixture
    def reasoning_engine(self):
        return IntuitiveReasoning()
    
    @pytest.fixture
    def evaluator(self, reasoning_engine):
        return IntuitiveReasoningEvaluator(reasoning_engine)
    
    @pytest.fixture
    def test_problems(self):
        """Generate test problems for evaluation"""
        problems = []
        domains = ["mathematics", "physics", "biology", "computer_science", "philosophy"]
        
        for i, domain in enumerate(domains):
            problem = Problem()
            problem.description = f"Complex {domain} optimization problem {i}"
            problem.domain = domain
            problem.complexity_level = 0.6 + (i * 0.08)  # Varying complexity
            problem.constraints = [f"constraint_{j}" for j in range(i % 3 + 1)]
            problem.objectives = [f"objective_{j}" for j in range(i % 2 + 1)]
            problem.context = {"test_id": i, "domain_specific": True}
            problems.append(problem)
        
        return problems
    
    @pytest.fixture
    def test_data_points(self):
        """Generate test data points for evaluation"""
        data_points = []
        domains = ["neuroscience", "economics", "psychology", "engineering"]
        
        for i in range(50):  # Larger dataset for pattern synthesis
            point = DataPoint()
            point.value = f"test_value_{i}"
            point.domain = domains[i % len(domains)]
            point.context = {
                "complexity": i / 50.0,
                "importance": (i * 7) % 10 / 10.0,
                "novelty": (i * 3) % 8 / 8.0
            }
            if i > 25:  # Add relationships to later points
                point.relationships = [f"rel_{j}" for j in range(i % 4)]
            data_points.append(point)
        
        return data_points
    
    @pytest.fixture
    def test_challenges(self):
        """Generate test challenges for evaluation"""
        challenges = []
        challenge_types = ["innovation", "optimization", "design", "strategy", "research"]
        
        for i, challenge_type in enumerate(challenge_types):
            challenge = Challenge()
            challenge.title = f"{challenge_type.title()} Challenge {i}"
            challenge.description = f"Solve complex {challenge_type} problem"
            challenge.challenge_type = challenge_type
            challenge.difficulty_level = 0.7 + (i * 0.05)
            challenge.resource_constraints = [f"resource_{j}" for j in range(i % 3 + 1)]
            challenge.success_metrics = [f"metric_{j}" for j in range(i % 2 + 2)]
            challenge.context_factors = {"test_challenge": True, "complexity": i}
            challenges.append(challenge)
        
        return challenges
    
    @pytest.fixture
    def test_contexts(self):
        """Generate test contexts for evaluation"""
        contexts = []
        situations = [
            "Global supply chain optimization",
            "Climate change adaptation system",
            "Urban transportation network",
            "Healthcare delivery system",
            "Educational technology ecosystem"
        ]
        
        for i, situation in enumerate(situations):
            context = Context()
            context.situation = situation
            context.domain = f"systems_{i}"
            context.environmental_factors = {
                "complexity": 0.6 + i * 0.08,
                "uncertainty": 0.5 + i * 0.06,
                "dynamism": 0.4 + i * 0.1
            }
            context.constraints = [f"constraint_{j}" for j in range(i % 4 + 2)]
            context.opportunities = [f"opportunity_{j}" for j in range(i % 3 + 1)]
            context.uncertainty_level = 0.5 + i * 0.1
            context.ambiguity_level = 0.4 + i * 0.08
            contexts.append(context)
        
        return contexts
    
    @pytest.mark.asyncio
    async def test_insight_quality_evaluation(self, evaluator, test_problems):
        """Test insight quality evaluation"""
        # Generate insights
        insights = []
        for problem in test_problems[:3]:  # Use subset for faster testing
            insight = await evaluator.reasoning_engine.generate_intuitive_leap(problem)
            insights.append(insight)
        
        # Evaluate quality
        quality_metrics = await evaluator.evaluate_insight_quality(insights)
        
        # Verify metrics
        assert "overall_quality" in quality_metrics
        assert "confidence" in quality_metrics
        assert "novelty" in quality_metrics
        assert "coherence" in quality_metrics
        assert quality_metrics["count"] == len(insights)
        
        # Quality scores should be reasonable
        assert 0.0 <= quality_metrics["overall_quality"] <= 1.0
        assert 0.0 <= quality_metrics["confidence"] <= 1.0
        assert 0.0 <= quality_metrics["novelty"] <= 1.0
        assert 0.0 <= quality_metrics["coherence"] <= 1.0
        
        # Should have some quality (not all zeros)
        assert quality_metrics["overall_quality"] > 0.2
    
    @pytest.mark.asyncio
    async def test_creativity_evaluation(self, evaluator, test_challenges):
        """Test creativity level evaluation"""
        # Generate solutions
        solutions = []
        for challenge in test_challenges[:3]:  # Use subset for faster testing
            solution = await evaluator.reasoning_engine.creative_problem_solving(challenge)
            solutions.append(solution)
        
        # Evaluate creativity
        creativity_metrics = await evaluator.evaluate_creativity_levels(solutions)
        
        # Verify metrics
        assert "creativity_distribution" in creativity_metrics
        assert "average_creativity" in creativity_metrics
        assert "average_innovation" in creativity_metrics
        assert "average_elegance" in creativity_metrics
        assert "average_quality" in creativity_metrics
        assert creativity_metrics["count"] == len(solutions)
        
        # Creativity scores should be reasonable
        assert 0.0 <= creativity_metrics["average_creativity"] <= 1.0
        assert 0.0 <= creativity_metrics["average_innovation"] <= 1.0
        assert 0.0 <= creativity_metrics["average_elegance"] <= 1.0
        assert 0.0 <= creativity_metrics["average_quality"] <= 1.0
        
        # Should show some creativity
        assert creativity_metrics["average_creativity"] > 0.3
    
    @pytest.mark.asyncio
    async def test_pattern_synthesis_evaluation(self, evaluator, test_data_points):
        """Test pattern synthesis effectiveness evaluation"""
        # Generate syntheses
        syntheses = []
        chunk_size = len(test_data_points) // 3
        for i in range(0, len(test_data_points), chunk_size):
            chunk = test_data_points[i:i + chunk_size]
            if chunk:
                synthesis = await evaluator.reasoning_engine.synthesize_patterns(chunk)
                syntheses.append(synthesis)
        
        # Evaluate synthesis effectiveness
        synthesis_metrics = await evaluator.evaluate_pattern_synthesis_effectiveness(syntheses)
        
        # Verify metrics
        assert "synthesis_effectiveness" in synthesis_metrics
        assert "average_emergence_properties" in synthesis_metrics
        assert "average_cross_domain_bridges" in synthesis_metrics
        assert "average_input_patterns" in synthesis_metrics
        assert synthesis_metrics["count"] == len(syntheses)
        
        # Effectiveness should be reasonable
        assert 0.0 <= synthesis_metrics["synthesis_effectiveness"] <= 1.0
        assert synthesis_metrics["average_emergence_properties"] >= 0
        assert synthesis_metrics["average_cross_domain_bridges"] >= 0
        assert synthesis_metrics["average_input_patterns"] >= 0
        
        # Should show some synthesis capability
        assert synthesis_metrics["synthesis_effectiveness"] > 0.3
    
    @pytest.mark.asyncio
    async def test_holistic_understanding_evaluation(self, evaluator, test_contexts):
        """Test holistic understanding depth evaluation"""
        # Generate holistic insights
        holistic_insights = []
        for context in test_contexts[:3]:  # Use subset for faster testing
            insight = await evaluator.reasoning_engine.holistic_understanding(context)
            holistic_insights.append(insight)
        
        # Evaluate holistic depth
        holistic_metrics = await evaluator.evaluate_holistic_understanding_depth(holistic_insights)
        
        # Verify metrics
        assert "holistic_depth" in holistic_metrics
        assert "average_emergent_properties" in holistic_metrics
        assert "average_leverage_points" in holistic_metrics
        assert "average_feedback_loops" in holistic_metrics
        assert holistic_metrics["count"] == len(holistic_insights)
        
        # Depth should be reasonable
        assert 0.0 <= holistic_metrics["holistic_depth"] <= 1.0
        assert holistic_metrics["average_emergent_properties"] >= 0
        assert holistic_metrics["average_leverage_points"] >= 0
        assert holistic_metrics["average_feedback_loops"] >= 0
        
        # Should show some holistic understanding
        assert holistic_metrics["holistic_depth"] > 0.3
    
    @pytest.mark.asyncio
    async def test_insight_generation_speed_benchmark(self, evaluator):
        """Test insight generation speed benchmark"""
        speed_metrics = await evaluator.benchmark_insight_generation_speed(problem_count=5)
        
        # Verify metrics
        assert "total_time_seconds" in speed_metrics
        assert "average_time_per_insight" in speed_metrics
        assert "insights_per_second" in speed_metrics
        assert "problem_count" in speed_metrics
        assert "average_insight_quality" in speed_metrics
        
        # Performance should be reasonable
        assert speed_metrics["total_time_seconds"] > 0
        assert speed_metrics["average_time_per_insight"] > 0
        assert speed_metrics["insights_per_second"] > 0
        assert speed_metrics["problem_count"] == 5
        assert 0.0 <= speed_metrics["average_insight_quality"] <= 1.0
        
        # Should be reasonably fast (less than 10 seconds per insight for testing)
        assert speed_metrics["average_time_per_insight"] < 10.0
    
    @pytest.mark.asyncio
    async def test_pattern_synthesis_scalability_benchmark(self, evaluator):
        """Test pattern synthesis scalability benchmark"""
        data_counts = [10, 25, 50]  # Test with different data sizes
        scalability_results = await evaluator.benchmark_pattern_synthesis_scalability(data_counts)
        
        # Verify results for each data count
        for count in data_counts:
            assert count in scalability_results
            result = scalability_results[count]
            
            assert "processing_time" in result
            assert "synthesis_confidence" in result
            assert "input_patterns_found" in result
            assert "data_points_per_second" in result
            
            # Performance should be reasonable
            assert result["processing_time"] > 0
            assert 0.0 <= result["synthesis_confidence"] <= 1.0
            assert result["input_patterns_found"] >= 0
            assert result["data_points_per_second"] > 0
        
        # Processing time should generally increase with data size
        times = [scalability_results[count]["processing_time"] for count in data_counts]
        # Allow for some variation but expect general trend
        assert times[-1] >= times[0] * 0.5  # Last should be at least half of first (allowing for optimization)
    
    @pytest.mark.asyncio
    async def test_cross_domain_connectivity_evaluation(self, evaluator, test_problems):
        """Test cross-domain connectivity evaluation"""
        # Generate insights with cross-domain potential
        insights = []
        for problem in test_problems[:4]:  # Use subset for faster testing
            insight = await evaluator.reasoning_engine.generate_intuitive_leap(problem)
            insights.append(insight)
        
        # Evaluate cross-domain connectivity
        connectivity_metrics = await evaluator.evaluate_cross_domain_connectivity(insights)
        
        # Verify metrics
        assert "cross_domain_score" in connectivity_metrics
        assert "connection_coverage" in connectivity_metrics
        assert "connection_diversity" in connectivity_metrics
        assert "average_connections_per_insight" in connectivity_metrics
        assert "unique_connection_types" in connectivity_metrics
        
        # Connectivity should be reasonable
        assert connectivity_metrics["cross_domain_score"] >= 0
        assert 0.0 <= connectivity_metrics["connection_coverage"] <= 1.0
        assert 0.0 <= connectivity_metrics["connection_diversity"] <= 1.0
        assert connectivity_metrics["average_connections_per_insight"] >= 0
        assert connectivity_metrics["unique_connection_types"] >= 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation(self, evaluator, test_problems, test_data_points, 
                                          test_challenges, test_contexts):
        """Test comprehensive evaluation of all capabilities"""
        # Use smaller subsets for faster testing
        problems_subset = test_problems[:2]
        data_subset = test_data_points[:20]
        challenges_subset = test_challenges[:2]
        contexts_subset = test_contexts[:2]
        
        # Perform comprehensive evaluation
        comprehensive_results = await evaluator.comprehensive_evaluation(
            problems_subset, data_subset, challenges_subset, contexts_subset
        )
        
        # Verify comprehensive results structure
        assert "overall_performance_score" in comprehensive_results
        assert "evaluation_time_seconds" in comprehensive_results
        assert "insight_quality" in comprehensive_results
        assert "creativity_evaluation" in comprehensive_results
        assert "synthesis_effectiveness" in comprehensive_results
        assert "holistic_understanding_depth" in comprehensive_results
        assert "cross_domain_connectivity" in comprehensive_results
        assert "test_counts" in comprehensive_results
        assert "generated_counts" in comprehensive_results
        
        # Verify overall performance score
        assert 0.0 <= comprehensive_results["overall_performance_score"] <= 1.0
        assert comprehensive_results["overall_performance_score"] > 0.2  # Should show some capability
        
        # Verify evaluation completed in reasonable time
        assert comprehensive_results["evaluation_time_seconds"] > 0
        assert comprehensive_results["evaluation_time_seconds"] < 300  # Should complete within 5 minutes
        
        # Verify test counts match inputs
        test_counts = comprehensive_results["test_counts"]
        assert test_counts["problems"] == len(problems_subset)
        assert test_counts["data_points"] == len(data_subset)
        assert test_counts["challenges"] == len(challenges_subset)
        assert test_counts["contexts"] == len(contexts_subset)
        
        # Verify generated counts are reasonable
        generated_counts = comprehensive_results["generated_counts"]
        assert generated_counts["insights"] == len(problems_subset)
        assert generated_counts["solutions"] == len(challenges_subset)
        assert generated_counts["holistic_insights"] == len(contexts_subset)
        assert generated_counts["syntheses"] > 0  # Should generate at least one synthesis
        
        # Verify evaluation was stored in history
        assert len(evaluator.evaluation_history) > 0
        latest_evaluation = evaluator.evaluation_history[-1]
        assert "timestamp" in latest_evaluation
        assert "results" in latest_evaluation
        assert latest_evaluation["results"] == comprehensive_results
    
    @pytest.mark.asyncio
    async def test_evaluation_with_empty_inputs(self, evaluator):
        """Test evaluation handles empty inputs gracefully"""
        # Test with empty lists
        empty_insights = []
        empty_solutions = []
        empty_syntheses = []
        empty_holistic = []
        
        # Should handle empty inputs without errors
        insight_quality = await evaluator.evaluate_insight_quality(empty_insights)
        creativity_eval = await evaluator.evaluate_creativity_levels(empty_solutions)
        synthesis_eval = await evaluator.evaluate_pattern_synthesis_effectiveness(empty_syntheses)
        holistic_eval = await evaluator.evaluate_holistic_understanding_depth(empty_holistic)
        connectivity_eval = await evaluator.evaluate_cross_domain_connectivity(empty_insights)
        
        # Should return default values
        assert insight_quality["overall_quality"] == 0.0
        assert insight_quality["count"] == 0
        
        assert creativity_eval["average_creativity"] == 0.0
        assert creativity_eval["count"] == 0
        
        assert synthesis_eval["synthesis_effectiveness"] == 0.0
        assert synthesis_eval["count"] == 0
        
        assert holistic_eval["holistic_depth"] == 0.0
        assert holistic_eval["count"] == 0
        
        assert connectivity_eval["cross_domain_score"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__])