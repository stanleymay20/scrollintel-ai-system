"""
Benchmark tests for meta-learning and adaptation capabilities.
Tests performance, scalability, and effectiveness of learning algorithms.
"""

import pytest
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

from scrollintel.engines.meta_learning_engine import MetaLearningEngine
from scrollintel.engines.adaptation_engine import AdaptationEngine
from scrollintel.models.meta_learning_models import (
    Task, EnvironmentalChallenge, AdaptationType, LearningStrategy
)


class TestMetaLearningBenchmarks:
    """Benchmark tests for meta-learning capabilities."""
    
    @pytest.fixture
    def meta_learning_engine(self):
        """Create MetaLearningEngine for benchmarking."""
        return MetaLearningEngine()
    
    @pytest.mark.asyncio
    async def test_rapid_skill_acquisition_speed(self, meta_learning_engine):
        """Benchmark rapid skill acquisition speed."""
        skills_to_acquire = [
            ("pattern_recognition", "analytical"),
            ("creative_problem_solving", "creative"),
            ("strategic_planning", "strategic"),
            ("technical_optimization", "technical"),
            ("social_intelligence", "social")
        ]
        
        acquisition_times = []
        mastery_levels = []
        
        for skill_name, domain in skills_to_acquire:
            start_time = time.time()
            
            skill_acquisition = await meta_learning_engine.rapid_skill_acquisition(
                skill_name, domain, target_performance=0.8
            )
            
            end_time = time.time()
            acquisition_times.append(end_time - start_time)
            mastery_levels.append(skill_acquisition.mastery_level)
        
        # Performance benchmarks
        avg_acquisition_time = np.mean(acquisition_times)
        avg_mastery_level = np.mean(mastery_levels)
        
        # Assertions for performance standards
        assert avg_acquisition_time < 1.0, f"Average acquisition time too slow: {avg_acquisition_time:.3f}s"
        assert avg_mastery_level >= 0.7, f"Average mastery level too low: {avg_mastery_level:.3f}"
        assert min(mastery_levels) >= 0.5, "Some skills had very low mastery"
        
        print(f"Skill Acquisition Benchmark Results:")
        print(f"  Average acquisition time: {avg_acquisition_time:.3f}s")
        print(f"  Average mastery level: {avg_mastery_level:.3f}")
        print(f"  Skills acquired: {len(skills_to_acquire)}")
    
    @pytest.mark.asyncio
    async def test_transfer_learning_efficiency(self, meta_learning_engine):
        """Benchmark transfer learning efficiency across domains."""
        domain_pairs = [
            ("technical", "analytical"),
            ("creative", "strategic"),
            ("analytical", "technical"),
            ("strategic", "social"),
            ("social", "creative")
        ]
        
        transfer_efficiencies = []
        success_probabilities = []
        
        for source_domain, target_domain in domain_pairs:
            transfer_map = await meta_learning_engine.transfer_learning_across_domains(
                source_domain, target_domain, "problem_solving"
            )
            
            transfer_efficiencies.append(transfer_map.transfer_efficiency)
            success_probabilities.append(transfer_map.success_probability)
        
        # Performance benchmarks
        avg_transfer_efficiency = np.mean(transfer_efficiencies)
        avg_success_probability = np.mean(success_probabilities)
        
        # Assertions for transfer learning standards
        assert avg_transfer_efficiency >= 0.6, f"Transfer efficiency too low: {avg_transfer_efficiency:.3f}"
        assert avg_success_probability >= 0.5, f"Success probability too low: {avg_success_probability:.3f}"
        assert min(transfer_efficiencies) >= 0.3, "Some transfers had very low efficiency"
        
        print(f"Transfer Learning Benchmark Results:")
        print(f"  Average transfer efficiency: {avg_transfer_efficiency:.3f}")
        print(f"  Average success probability: {avg_success_probability:.3f}")
        print(f"  Domain pairs tested: {len(domain_pairs)}")
    
    @pytest.mark.asyncio
    async def test_learning_algorithm_comparison(self, meta_learning_engine):
        """Benchmark different meta-learning algorithms."""
        algorithms = [
            LearningStrategy.MODEL_AGNOSTIC,
            LearningStrategy.GRADIENT_BASED,
            LearningStrategy.MEMORY_AUGMENTED,
            LearningStrategy.EVOLUTIONARY,
            LearningStrategy.REINFORCEMENT
        ]
        
        algorithm_performance = {}
        
        for algorithm in algorithms:
            # Test algorithm with multiple skills
            performances = []
            
            for i in range(3):  # Test each algorithm 3 times
                skill_acquisition = await meta_learning_engine.rapid_skill_acquisition(
                    f"test_skill_{algorithm.value}_{i}", "test_domain", 0.8
                )
                performances.append(skill_acquisition.mastery_level)
            
            algorithm_performance[algorithm.value] = {
                "avg_performance": np.mean(performances),
                "std_performance": np.std(performances),
                "min_performance": min(performances),
                "max_performance": max(performances)
            }
        
        # Find best performing algorithm
        best_algorithm = max(
            algorithm_performance.keys(),
            key=lambda k: algorithm_performance[k]["avg_performance"]
        )
        
        best_performance = algorithm_performance[best_algorithm]["avg_performance"]
        
        # Performance assertions
        assert best_performance >= 0.7, f"Best algorithm performance too low: {best_performance:.3f}"
        
        print(f"Learning Algorithm Comparison Results:")
        for algo, perf in algorithm_performance.items():
            print(f"  {algo}: avg={perf['avg_performance']:.3f}, std={perf['std_performance']:.3f}")
        print(f"  Best algorithm: {best_algorithm} ({best_performance:.3f})")
    
    @pytest.mark.asyncio
    async def test_self_improvement_effectiveness(self, meta_learning_engine):
        """Benchmark self-improvement algorithm effectiveness."""
        capabilities_to_improve = [
            "reasoning_speed",
            "learning_efficiency",
            "memory_capacity",
            "decision_accuracy",
            "adaptation_speed"
        ]
        
        improvement_results = []
        
        for capability in capabilities_to_improve:
            improvement_plan = await meta_learning_engine.self_improving_algorithms(
                capability, improvement_factor=1.4
            )
            
            improvement_ratio = (
                improvement_plan.target_capability / improvement_plan.current_capability
            )
            improvement_results.append(improvement_ratio)
        
        # Performance benchmarks
        avg_improvement_ratio = np.mean(improvement_results)
        min_improvement_ratio = min(improvement_results)
        
        # Assertions for self-improvement standards
        assert avg_improvement_ratio >= 1.3, f"Average improvement too low: {avg_improvement_ratio:.3f}"
        assert min_improvement_ratio >= 1.2, f"Minimum improvement too low: {min_improvement_ratio:.3f}"
        
        print(f"Self-Improvement Benchmark Results:")
        print(f"  Average improvement ratio: {avg_improvement_ratio:.3f}")
        print(f"  Minimum improvement ratio: {min_improvement_ratio:.3f}")
        print(f"  Capabilities improved: {len(capabilities_to_improve)}")
    
    @pytest.mark.asyncio
    async def test_learning_scalability(self, meta_learning_engine):
        """Test scalability of learning with increasing number of skills."""
        skill_counts = [5, 10, 20, 30]
        scalability_results = {}
        
        for skill_count in skill_counts:
            start_time = time.time()
            
            # Acquire multiple skills
            for i in range(skill_count):
                await meta_learning_engine.rapid_skill_acquisition(
                    f"scalability_skill_{i}", f"domain_{i % 5}", 0.7
                )
            
            end_time = time.time()
            total_time = end_time - start_time
            time_per_skill = total_time / skill_count
            
            scalability_results[skill_count] = {
                "total_time": total_time,
                "time_per_skill": time_per_skill,
                "skills_per_second": skill_count / total_time
            }
        
        # Check scalability - time per skill shouldn't increase dramatically
        time_per_skill_values = [result["time_per_skill"] for result in scalability_results.values()]
        scalability_factor = max(time_per_skill_values) / min(time_per_skill_values)
        
        assert scalability_factor < 3.0, f"Poor scalability: {scalability_factor:.2f}x slowdown"
        
        print(f"Learning Scalability Benchmark Results:")
        for count, result in scalability_results.items():
            print(f"  {count} skills: {result['time_per_skill']:.3f}s per skill, "
                  f"{result['skills_per_second']:.2f} skills/s")
        print(f"  Scalability factor: {scalability_factor:.2f}x")


class TestAdaptationBenchmarks:
    """Benchmark tests for adaptation capabilities."""
    
    @pytest.fixture
    def adaptation_engine(self):
        """Create AdaptationEngine for benchmarking."""
        return AdaptationEngine()
    
    @pytest.mark.asyncio
    async def test_environment_adaptation_speed(self, adaptation_engine):
        """Benchmark environment adaptation speed."""
        environments = [
            {
                "type": "low_complexity",
                "complexity": 0.3,
                "volatility": 0.2,
                "variables": ["var1", "var2"]
            },
            {
                "type": "medium_complexity",
                "complexity": 0.6,
                "volatility": 0.5,
                "variables": ["var1", "var2", "var3", "var4"]
            },
            {
                "type": "high_complexity",
                "complexity": 0.9,
                "volatility": 0.8,
                "variables": ["var1", "var2", "var3", "var4", "var5", "var6"]
            }
        ]
        
        adaptation_times = []
        effectiveness_scores = []
        
        for env in environments:
            start_time = time.time()
            
            adaptation_result = await adaptation_engine.adapt_to_environment(env, "normal")
            
            end_time = time.time()
            adaptation_times.append(end_time - start_time)
            effectiveness_scores.append(adaptation_result["effectiveness"])
        
        # Performance benchmarks
        avg_adaptation_time = np.mean(adaptation_times)
        avg_effectiveness = np.mean(effectiveness_scores)
        
        # Assertions for adaptation speed standards
        assert avg_adaptation_time < 0.5, f"Average adaptation time too slow: {avg_adaptation_time:.3f}s"
        assert avg_effectiveness >= 0.6, f"Average effectiveness too low: {avg_effectiveness:.3f}"
        
        print(f"Environment Adaptation Benchmark Results:")
        print(f"  Average adaptation time: {avg_adaptation_time:.3f}s")
        print(f"  Average effectiveness: {avg_effectiveness:.3f}")
        print(f"  Environments tested: {len(environments)}")
    
    @pytest.mark.asyncio
    async def test_challenge_resolution_effectiveness(self, adaptation_engine):
        """Benchmark challenge resolution effectiveness."""
        challenges = [
            EnvironmentalChallenge(
                challenge_id="easy_challenge",
                environment_type="stable",
                challenge_description="Low difficulty challenge",
                difficulty_level=0.3,
                required_adaptations=[AdaptationType.PARAMETER_ADAPTATION],
                success_criteria={"metric1": 0.7}
            ),
            EnvironmentalChallenge(
                challenge_id="medium_challenge",
                environment_type="dynamic",
                challenge_description="Medium difficulty challenge",
                difficulty_level=0.6,
                required_adaptations=[
                    AdaptationType.PARAMETER_ADAPTATION,
                    AdaptationType.STRATEGY_ADAPTATION
                ],
                success_criteria={"metric1": 0.8, "metric2": 0.7}
            ),
            EnvironmentalChallenge(
                challenge_id="hard_challenge",
                environment_type="volatile",
                challenge_description="High difficulty challenge",
                difficulty_level=0.9,
                required_adaptations=[
                    AdaptationType.PARAMETER_ADAPTATION,
                    AdaptationType.ARCHITECTURE_ADAPTATION,
                    AdaptationType.STRATEGY_ADAPTATION
                ],
                success_criteria={"metric1": 0.9, "metric2": 0.8, "metric3": 0.7}
            )
        ]
        
        resolution_results = []
        
        for challenge in challenges:
            result = await adaptation_engine.handle_environmental_challenge(challenge)
            success_score = result["success_evaluation"]["overall_success"]
            resolution_results.append({
                "difficulty": challenge.difficulty_level,
                "success": success_score,
                "adaptations_required": len(challenge.required_adaptations)
            })
        
        # Performance analysis
        avg_success = np.mean([r["success"] for r in resolution_results])
        success_by_difficulty = {
            "easy": [r["success"] for r in resolution_results if r["difficulty"] < 0.5],
            "medium": [r["success"] for r in resolution_results if 0.5 <= r["difficulty"] < 0.8],
            "hard": [r["success"] for r in resolution_results if r["difficulty"] >= 0.8]
        }
        
        # Assertions for challenge resolution standards
        assert avg_success >= 0.6, f"Average success rate too low: {avg_success:.3f}"
        
        if success_by_difficulty["easy"]:
            assert np.mean(success_by_difficulty["easy"]) >= 0.8, "Easy challenges should have high success rate"
        
        print(f"Challenge Resolution Benchmark Results:")
        print(f"  Average success rate: {avg_success:.3f}")
        for difficulty, successes in success_by_difficulty.items():
            if successes:
                print(f"  {difficulty.capitalize()} challenges: {np.mean(successes):.3f} success rate")
    
    @pytest.mark.asyncio
    async def test_adaptation_type_effectiveness(self, adaptation_engine):
        """Benchmark effectiveness of different adaptation types."""
        adaptation_types = [
            AdaptationType.PARAMETER_ADAPTATION,
            AdaptationType.ARCHITECTURE_ADAPTATION,
            AdaptationType.STRATEGY_ADAPTATION,
            AdaptationType.ENVIRONMENT_ADAPTATION
        ]
        
        adaptation_effectiveness = {}
        
        for adaptation_type in adaptation_types:
            context = {
                "adaptation_type": adaptation_type,
                "urgency": "normal",
                "complexity": 0.6
            }
            
            # Test adaptation type multiple times
            effectiveness_scores = []
            
            for _ in range(3):
                if adaptation_type == AdaptationType.PARAMETER_ADAPTATION:
                    result = await adaptation_engine._parameter_adaptation(context)
                elif adaptation_type == AdaptationType.ARCHITECTURE_ADAPTATION:
                    result = await adaptation_engine._architecture_adaptation(context)
                elif adaptation_type == AdaptationType.STRATEGY_ADAPTATION:
                    result = await adaptation_engine._strategy_adaptation(context)
                elif adaptation_type == AdaptationType.ENVIRONMENT_ADAPTATION:
                    result = await adaptation_engine._environment_adaptation(context)
                
                effectiveness_scores.append(result.get("effectiveness", 0.5))
            
            adaptation_effectiveness[adaptation_type.value] = {
                "avg_effectiveness": np.mean(effectiveness_scores),
                "std_effectiveness": np.std(effectiveness_scores),
                "min_effectiveness": min(effectiveness_scores)
            }
        
        # Find most effective adaptation type
        best_adaptation = max(
            adaptation_effectiveness.keys(),
            key=lambda k: adaptation_effectiveness[k]["avg_effectiveness"]
        )
        
        best_effectiveness = adaptation_effectiveness[best_adaptation]["avg_effectiveness"]
        
        # Performance assertions
        assert best_effectiveness >= 0.7, f"Best adaptation effectiveness too low: {best_effectiveness:.3f}"
        
        print(f"Adaptation Type Effectiveness Benchmark Results:")
        for adapt_type, effectiveness in adaptation_effectiveness.items():
            print(f"  {adapt_type}: avg={effectiveness['avg_effectiveness']:.3f}, "
                  f"std={effectiveness['std_effectiveness']:.3f}")
        print(f"  Most effective: {best_adaptation} ({best_effectiveness:.3f})")
    
    @pytest.mark.asyncio
    async def test_continuous_improvement_rate(self, adaptation_engine):
        """Benchmark continuous self-improvement rate."""
        improvement_domains = ["reasoning", "learning", "adaptation", "memory"]
        
        improvement_rates = []
        
        for domain in improvement_domains:
            # Measure improvement over time
            initial_capability = await adaptation_engine._assess_domain_capability(domain)
            
            # Execute improvement
            improvement_result = await adaptation_engine.continuous_self_improvement([domain])
            
            # Calculate improvement rate
            if domain in improvement_result["domain_improvements"]:
                # Simplified improvement rate calculation
                improvement_rate = improvement_result["overall_improvement"]
                improvement_rates.append(improvement_rate)
        
        # Performance benchmarks
        avg_improvement_rate = np.mean(improvement_rates) if improvement_rates else 0.0
        min_improvement_rate = min(improvement_rates) if improvement_rates else 0.0
        
        # Assertions for improvement rate standards
        assert avg_improvement_rate >= 0.1, f"Average improvement rate too low: {avg_improvement_rate:.3f}"
        assert min_improvement_rate >= 0.05, f"Minimum improvement rate too low: {min_improvement_rate:.3f}"
        
        print(f"Continuous Improvement Benchmark Results:")
        print(f"  Average improvement rate: {avg_improvement_rate:.3f}")
        print(f"  Minimum improvement rate: {min_improvement_rate:.3f}")
        print(f"  Domains improved: {len(improvement_rates)}")


class TestIntegratedBenchmarks:
    """Benchmark tests for integrated meta-learning and adaptation."""
    
    @pytest.fixture
    def engines(self):
        """Create both engines for integrated benchmarking."""
        return {
            "meta_learning": MetaLearningEngine(),
            "adaptation": AdaptationEngine()
        }
    
    @pytest.mark.asyncio
    async def test_integrated_learning_adaptation_cycle(self, engines):
        """Benchmark complete learning-adaptation cycle."""
        meta_engine = engines["meta_learning"]
        adapt_engine = engines["adaptation"]
        
        cycle_results = []
        
        # Test multiple learning-adaptation cycles
        for cycle in range(3):
            cycle_start = time.time()
            
            # Phase 1: Learn new skill
            skill = await meta_engine.rapid_skill_acquisition(
                f"cycle_skill_{cycle}", "analytical", 0.8
            )
            
            # Phase 2: Adapt to environment requiring this skill
            environment = {
                "type": f"analytical_env_{cycle}",
                "complexity": 0.6 + cycle * 0.1,
                "required_skills": [skill.skill_name]
            }
            
            adaptation_result = await adapt_engine.adapt_to_environment(environment)
            
            # Phase 3: Self-improve based on experience
            improvement_result = await adapt_engine.continuous_self_improvement(
                ["learning", "adaptation"]
            )
            
            cycle_end = time.time()
            cycle_time = cycle_end - cycle_start
            
            cycle_results.append({
                "cycle": cycle,
                "cycle_time": cycle_time,
                "skill_mastery": skill.mastery_level,
                "adaptation_effectiveness": adaptation_result["effectiveness"],
                "improvement_gain": improvement_result["overall_improvement"]
            })
        
        # Performance analysis
        avg_cycle_time = np.mean([r["cycle_time"] for r in cycle_results])
        avg_skill_mastery = np.mean([r["skill_mastery"] for r in cycle_results])
        avg_adaptation_effectiveness = np.mean([r["adaptation_effectiveness"] for r in cycle_results])
        avg_improvement_gain = np.mean([r["improvement_gain"] for r in cycle_results])
        
        # Performance assertions
        assert avg_cycle_time < 2.0, f"Average cycle time too slow: {avg_cycle_time:.3f}s"
        assert avg_skill_mastery >= 0.7, f"Average skill mastery too low: {avg_skill_mastery:.3f}"
        assert avg_adaptation_effectiveness >= 0.6, f"Average adaptation effectiveness too low: {avg_adaptation_effectiveness:.3f}"
        assert avg_improvement_gain >= 0.1, f"Average improvement gain too low: {avg_improvement_gain:.3f}"
        
        print(f"Integrated Learning-Adaptation Cycle Benchmark Results:")
        print(f"  Average cycle time: {avg_cycle_time:.3f}s")
        print(f"  Average skill mastery: {avg_skill_mastery:.3f}")
        print(f"  Average adaptation effectiveness: {avg_adaptation_effectiveness:.3f}")
        print(f"  Average improvement gain: {avg_improvement_gain:.3f}")
        print(f"  Cycles completed: {len(cycle_results)}")
    
    @pytest.mark.asyncio
    async def test_system_performance_under_load(self, engines):
        """Benchmark system performance under high load."""
        meta_engine = engines["meta_learning"]
        adapt_engine = engines["adaptation"]
        
        # Simulate high load scenario
        concurrent_tasks = 10
        load_results = []
        
        start_time = time.time()
        
        # Execute multiple concurrent operations
        tasks = []
        for i in range(concurrent_tasks):
            # Mix of different operations
            if i % 3 == 0:
                task = meta_engine.rapid_skill_acquisition(f"load_skill_{i}", "technical", 0.7)
            elif i % 3 == 1:
                env = {"type": f"load_env_{i}", "complexity": 0.5}
                task = adapt_engine.adapt_to_environment(env)
            else:
                task = adapt_engine.continuous_self_improvement(["adaptation"])
            
            tasks.append(task)
        
        # Wait for all tasks to complete (simulated concurrent execution)
        for task in tasks:
            result = await task
            load_results.append(result)
        
        end_time = time.time()
        total_load_time = end_time - start_time
        
        # Performance analysis
        throughput = concurrent_tasks / total_load_time
        
        # Performance assertions
        assert total_load_time < 5.0, f"Load test took too long: {total_load_time:.3f}s"
        assert throughput >= 2.0, f"Throughput too low: {throughput:.2f} ops/s"
        assert len(load_results) == concurrent_tasks, "Not all tasks completed"
        
        print(f"System Load Benchmark Results:")
        print(f"  Total load time: {total_load_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} operations/s")
        print(f"  Concurrent tasks: {concurrent_tasks}")
        print(f"  Tasks completed: {len(load_results)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])