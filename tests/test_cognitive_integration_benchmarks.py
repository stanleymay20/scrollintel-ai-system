"""
Benchmarks and Performance Tests for Cognitive Integration System
"""

import pytest
import pytest_asyncio
import asyncio
import time
import statistics
from typing import List, Dict, Any

from scrollintel.engines.cognitive_integrator import (
    CognitiveIntegrator, AttentionFocus, CognitiveLoadLevel
)


class TestCognitiveIntegrationBenchmarks:
    """Benchmark tests for cognitive integration system performance"""
    
    @pytest_asyncio.fixture
    async def integrator(self):
        """Create a cognitive integrator for benchmarking"""
        integrator = CognitiveIntegrator()
        await integrator.start()
        yield integrator
        await integrator.stop()
    
    @pytest.mark.asyncio
    async def test_decision_making_speed_benchmark(self, integrator):
        """Benchmark decision-making speed across different complexity levels"""
        test_situations = [
            ("Simple task completion", {"complexity": 0.2}),
            ("Moderate problem solving", {"complexity": 0.5}),
            ("Complex strategic planning", {"complexity": 0.8}),
            ("Critical system design", {"complexity": 0.95})
        ]
        
        results = {}
        
        for situation, context in test_situations:
            times = []
            
            # Run multiple iterations for statistical significance
            for _ in range(5):
                start_time = time.time()
                decision = await integrator.process_complex_situation(situation, context)
                end_time = time.time()
                
                processing_time = end_time - start_time
                times.append(processing_time)
                
                # Verify decision quality
                assert decision.confidence > 0.0
                assert decision.integration_quality > 0.0
            
            results[situation] = {
                "mean_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
                "min_time": min(times),
                "max_time": max(times)
            }
        
        # Verify performance characteristics
        simple_time = results["Simple task completion"]["mean_time"]
        complex_time = results["Critical system design"]["mean_time"]
        
        # Complex decisions should take longer but not excessively
        assert complex_time > simple_time
        assert complex_time < simple_time * 5  # Should not be more than 5x slower
        
        # All decisions should complete within reasonable time
        for situation_results in results.values():
            assert situation_results["max_time"] < 10.0  # 10 seconds max
        
        print("\nDecision Making Speed Benchmark Results:")
        for situation, metrics in results.items():
            print(f"{situation}: {metrics['mean_time']:.3f}s ± {metrics['std_dev']:.3f}s")
    
    @pytest.mark.asyncio
    async def test_attention_management_performance(self, integrator):
        """Benchmark attention management and reallocation performance"""
        attention_focuses = [
            AttentionFocus.CONSCIOUSNESS,
            AttentionFocus.REASONING,
            AttentionFocus.MEMORY,
            AttentionFocus.LEARNING,
            AttentionFocus.EMOTION,
            AttentionFocus.INTEGRATION
        ]
        
        reallocation_times = []
        
        for focus in attention_focuses:
            for intensity in [0.3, 0.6, 0.9]:
                start_time = time.time()
                allocation = await integrator.manage_attention(focus, intensity)
                end_time = time.time()
                
                reallocation_time = end_time - start_time
                reallocation_times.append(reallocation_time)
                
                # Verify allocation is valid
                total_attention = (
                    allocation.consciousness_attention + allocation.reasoning_attention +
                    allocation.memory_attention + allocation.learning_attention +
                    allocation.emotion_attention + allocation.integration_attention
                )
                assert abs(total_attention - 1.0) < 0.01
        
        # Analyze performance
        mean_time = statistics.mean(reallocation_times)
        max_time = max(reallocation_times)
        
        # Attention reallocation should be very fast
        assert mean_time < 0.1  # Less than 100ms on average
        assert max_time < 0.5   # Less than 500ms maximum
        
        print(f"\nAttention Management Performance:")
        print(f"Mean reallocation time: {mean_time:.4f}s")
        print(f"Max reallocation time: {max_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_cognitive_load_balancing_efficiency(self, integrator):
        """Benchmark cognitive load balancing efficiency"""
        load_balancing_times = []
        
        # Test load balancing under different conditions
        for _ in range(10):
            start_time = time.time()
            system_loads = await integrator.balance_cognitive_load()
            end_time = time.time()
            
            balancing_time = end_time - start_time
            load_balancing_times.append(balancing_time)
            
            # Verify load balancing results
            assert isinstance(system_loads, dict)
            assert len(system_loads) > 0
            
            # All loads should be reasonable
            for system, load in system_loads.items():
                assert 0.0 <= load <= 1.0
        
        # Analyze performance
        mean_time = statistics.mean(load_balancing_times)
        max_time = max(load_balancing_times)
        
        # Load balancing should be efficient
        assert mean_time < 0.2  # Less than 200ms on average
        assert max_time < 1.0   # Less than 1 second maximum
        
        print(f"\nCognitive Load Balancing Performance:")
        print(f"Mean balancing time: {mean_time:.4f}s")
        print(f"Max balancing time: {max_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_system_integration_scalability(self, integrator):
        """Test system scalability with multiple concurrent operations"""
        concurrent_operations = [1, 2, 5, 10]
        scalability_results = {}
        
        for num_operations in concurrent_operations:
            situations = [f"Concurrent situation {i}" for i in range(num_operations)]
            
            start_time = time.time()
            
            # Run operations concurrently
            tasks = [
                integrator.process_complex_situation(situation, {"complexity": 0.6})
                for situation in situations
            ]
            
            decisions = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            time_per_operation = total_time / num_operations
            
            scalability_results[num_operations] = {
                "total_time": total_time,
                "time_per_operation": time_per_operation,
                "decisions_made": len(decisions)
            }
            
            # Verify all decisions were made successfully
            assert len(decisions) == num_operations
            for decision in decisions:
                assert decision.confidence > 0.0
                assert decision.integration_quality > 0.0
        
        # Analyze scalability
        single_op_time = scalability_results[1]["time_per_operation"]
        ten_op_time = scalability_results[10]["time_per_operation"]
        
        # Time per operation should not increase dramatically with concurrency
        scalability_factor = ten_op_time / single_op_time
        assert scalability_factor < 3.0  # Should not be more than 3x slower per operation
        
        print(f"\nScalability Benchmark Results:")
        for ops, results in scalability_results.items():
            print(f"{ops} operations: {results['time_per_operation']:.3f}s per operation")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_benchmark(self, integrator):
        """Benchmark memory usage and efficiency"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many decisions to test memory efficiency
        decisions = []
        for i in range(50):
            decision = await integrator.process_complex_situation(
                f"Memory test situation {i}",
                {"complexity": 0.5}
            )
            decisions.append(decision)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Verify decisions were made
        assert len(decisions) == 50
        assert len(integrator.decision_history) >= 50
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
        
        # Test memory cleanup by clearing history
        integrator.decision_history.clear()
        
        print(f"\nMemory Efficiency Benchmark:")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
    
    @pytest.mark.asyncio
    async def test_decision_quality_consistency(self, integrator):
        """Benchmark decision quality consistency across multiple runs"""
        test_situation = "Evaluate the best approach for implementing a new feature with limited resources"
        test_context = {"complexity": 0.7, "resources": ["time", "budget"], "constraints": ["deadline"]}
        
        decision_qualities = []
        confidence_scores = []
        
        # Run same decision multiple times
        for _ in range(20):
            decision = await integrator.process_complex_situation(test_situation, test_context)
            decision_qualities.append(decision.integration_quality)
            confidence_scores.append(decision.confidence)
        
        # Analyze consistency
        quality_mean = statistics.mean(decision_qualities)
        quality_std = statistics.stdev(decision_qualities)
        confidence_mean = statistics.mean(confidence_scores)
        confidence_std = statistics.stdev(confidence_scores)
        
        # Quality should be consistently high
        assert quality_mean > 0.5
        assert quality_std < 0.3  # Low variance indicates consistency
        
        # Confidence should be reasonable and consistent
        assert confidence_mean > 0.4
        assert confidence_std < 0.3
        
        print(f"\nDecision Quality Consistency:")
        print(f"Quality: {quality_mean:.3f} ± {quality_std:.3f}")
        print(f"Confidence: {confidence_mean:.3f} ± {confidence_std:.3f}")
    
    @pytest.mark.asyncio
    async def test_system_health_monitoring_performance(self, integrator):
        """Benchmark system health monitoring performance"""
        monitoring_times = []
        
        for _ in range(10):
            start_time = time.time()
            health_report = await integrator.monitor_cognitive_health()
            end_time = time.time()
            
            monitoring_time = end_time - start_time
            monitoring_times.append(monitoring_time)
            
            # Verify health report completeness
            assert "system_health" in health_report
            assert "integration_health" in health_report
            assert "overall_score" in health_report
            assert 0.0 <= health_report["overall_score"] <= 1.0
        
        # Analyze monitoring performance
        mean_time = statistics.mean(monitoring_times)
        max_time = max(monitoring_times)
        
        # Health monitoring should be fast
        assert mean_time < 0.5  # Less than 500ms on average
        assert max_time < 2.0   # Less than 2 seconds maximum
        
        print(f"\nHealth Monitoring Performance:")
        print(f"Mean monitoring time: {mean_time:.4f}s")
        print(f"Max monitoring time: {max_time:.4f}s")
    
    @pytest.mark.asyncio
    async def test_self_regulation_effectiveness(self, integrator):
        """Benchmark self-regulation effectiveness and performance"""
        regulation_times = []
        effectiveness_scores = []
        
        for _ in range(10):
            start_time = time.time()
            regulation_result = await integrator.self_regulate()
            end_time = time.time()
            
            regulation_time = end_time - start_time
            regulation_times.append(regulation_time)
            effectiveness_scores.append(regulation_result["effectiveness"])
            
            # Verify regulation result structure
            assert "regulation_needs" in regulation_result
            assert "regulation_applied" in regulation_result
            assert "effectiveness" in regulation_result
            assert 0.0 <= regulation_result["effectiveness"] <= 1.0
        
        # Analyze regulation performance
        mean_time = statistics.mean(regulation_times)
        mean_effectiveness = statistics.mean(effectiveness_scores)
        
        # Self-regulation should be efficient and effective
        assert mean_time < 1.0  # Less than 1 second on average
        assert mean_effectiveness > 0.3  # Reasonably effective
        
        print(f"\nSelf-Regulation Performance:")
        print(f"Mean regulation time: {mean_time:.4f}s")
        print(f"Mean effectiveness: {mean_effectiveness:.3f}")
    
    @pytest.mark.asyncio
    async def test_integration_coherence_benchmark(self, integrator):
        """Benchmark integration coherence across different scenarios"""
        scenarios = [
            ("Technical problem solving", {"complexity": 0.8, "domain": "technical"}),
            ("Strategic business decision", {"complexity": 0.7, "domain": "business"}),
            ("Creative design challenge", {"complexity": 0.6, "domain": "creative"}),
            ("Social conflict resolution", {"complexity": 0.5, "domain": "social"}),
            ("Ethical dilemma analysis", {"complexity": 0.9, "domain": "ethical"})
        ]
        
        coherence_results = {}
        
        for scenario, context in scenarios:
            coherence_scores = []
            
            # Test each scenario multiple times
            for _ in range(5):
                decision = await integrator.process_complex_situation(scenario, context)
                coherence_scores.append(decision.integration_quality)
            
            mean_coherence = statistics.mean(coherence_scores)
            std_coherence = statistics.stdev(coherence_scores) if len(coherence_scores) > 1 else 0
            
            coherence_results[scenario] = {
                "mean_coherence": mean_coherence,
                "std_coherence": std_coherence,
                "min_coherence": min(coherence_scores),
                "max_coherence": max(coherence_scores)
            }
            
            # Verify coherence is reasonable
            assert mean_coherence > 0.3
            assert std_coherence < 0.4  # Reasonable consistency
        
        # Overall coherence should be good across all scenarios
        overall_mean = statistics.mean([r["mean_coherence"] for r in coherence_results.values()])
        assert overall_mean > 0.5
        
        print(f"\nIntegration Coherence Benchmark:")
        for scenario, results in coherence_results.items():
            print(f"{scenario}: {results['mean_coherence']:.3f} ± {results['std_coherence']:.3f}")
        print(f"Overall mean coherence: {overall_mean:.3f}")
    
    @pytest.mark.asyncio
    async def test_system_startup_shutdown_performance(self):
        """Benchmark system startup and shutdown performance"""
        startup_times = []
        shutdown_times = []
        
        for _ in range(5):
            # Test startup
            integrator = CognitiveIntegrator()
            start_time = time.time()
            await integrator.start()
            startup_time = time.time() - start_time
            startup_times.append(startup_time)
            
            # Test shutdown
            start_time = time.time()
            await integrator.stop()
            shutdown_time = time.time() - start_time
            shutdown_times.append(shutdown_time)
        
        # Analyze startup/shutdown performance
        mean_startup = statistics.mean(startup_times)
        mean_shutdown = statistics.mean(shutdown_times)
        
        # Startup and shutdown should be reasonably fast
        assert mean_startup < 5.0   # Less than 5 seconds
        assert mean_shutdown < 2.0  # Less than 2 seconds
        
        print(f"\nStartup/Shutdown Performance:")
        print(f"Mean startup time: {mean_startup:.3f}s")
        print(f"Mean shutdown time: {mean_shutdown:.3f}s")


class TestCognitiveIntegrationStressTests:
    """Stress tests for cognitive integration system"""
    
    @pytest_asyncio.fixture
    async def integrator(self):
        """Create a cognitive integrator for stress testing"""
        integrator = CognitiveIntegrator()
        await integrator.start()
        yield integrator
        await integrator.stop()
    
    @pytest.mark.asyncio
    async def test_high_load_stress_test(self, integrator):
        """Stress test with high cognitive load"""
        # Create many complex situations simultaneously
        complex_situations = [
            f"Complex strategic decision {i} requiring deep analysis and multiple stakeholder consideration"
            for i in range(20)
        ]
        
        contexts = [
            {
                "complexity": 0.9,
                "time_pressure": 0.8,
                "stakeholders": ["team1", "team2", "management"],
                "constraints": ["budget", "timeline", "resources"]
            }
            for _ in range(20)
        ]
        
        start_time = time.time()
        
        # Process all situations concurrently
        tasks = [
            integrator.process_complex_situation(situation, context)
            for situation, context in zip(complex_situations, contexts)
        ]
        
        decisions = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Verify all decisions were made successfully
        assert len(decisions) == 20
        for decision in decisions:
            assert decision.confidence > 0.0
            assert decision.integration_quality > 0.0
            assert len(decision.reasoning_path) > 0
        
        # System should handle high load reasonably
        assert total_time < 60.0  # Should complete within 1 minute
        
        print(f"\nHigh Load Stress Test:")
        print(f"Processed 20 complex decisions in {total_time:.2f}s")
        print(f"Average time per decision: {total_time/20:.3f}s")
    
    @pytest.mark.asyncio
    async def test_rapid_attention_switching_stress(self, integrator):
        """Stress test rapid attention switching"""
        attention_focuses = list(AttentionFocus)
        
        # Rapidly switch attention focus
        for _ in range(100):
            focus = attention_focuses[_ % len(attention_focuses)]
            intensity = 0.5 + (_ % 5) * 0.1  # Vary intensity
            
            allocation = await integrator.manage_attention(focus, intensity)
            
            # Verify allocation is always valid
            total = (allocation.consciousness_attention + allocation.reasoning_attention +
                    allocation.memory_attention + allocation.learning_attention +
                    allocation.emotion_attention + allocation.integration_attention)
            assert abs(total - 1.0) < 0.01
        
        # System should remain stable after rapid switching
        final_status = integrator.get_system_status()
        assert final_status["running"]
        assert isinstance(final_status["cognitive_state"].attention_focus, AttentionFocus)
    
    @pytest.mark.asyncio
    async def test_memory_pressure_stress_test(self, integrator):
        """Stress test with high memory pressure"""
        # Generate many decisions to fill memory
        decisions = []
        
        for i in range(200):
            decision = await integrator.process_complex_situation(
                f"Memory pressure test situation {i} with detailed context and multiple considerations",
                {
                    "complexity": 0.6,
                    "detailed_context": f"Very detailed context for situation {i} " * 10,
                    "stakeholders": [f"stakeholder_{j}" for j in range(10)],
                    "constraints": [f"constraint_{j}" for j in range(5)]
                }
            )
            decisions.append(decision)
        
        # Verify system remains functional
        assert len(decisions) == 200
        assert len(integrator.decision_history) > 0  # Some decisions should be stored
        
        # System should still be able to make new decisions
        final_decision = await integrator.process_complex_situation(
            "Final test decision after memory pressure",
            {"complexity": 0.5}
        )
        
        assert final_decision.confidence > 0.0
        assert final_decision.integration_quality > 0.0
        
        print(f"\nMemory Pressure Stress Test:")
        print(f"Successfully processed 200 decisions")
        print(f"Decision history length: {len(integrator.decision_history)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])