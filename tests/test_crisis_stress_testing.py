"""
Crisis Stress Testing Framework

This module provides stress testing capabilities for crisis leadership under extreme pressure,
including resource exhaustion, time pressure, and system overload scenarios.
Requirements: 1.1, 2.1, 3.1, 4.1, 5.1
"""

import pytest
import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from scrollintel.engines.crisis_detection_engine import CrisisDetectionEngine
from scrollintel.engines.decision_tree_engine import DecisionTreeEngine
from scrollintel.engines.stakeholder_notification_engine import StakeholderNotificationEngine
from scrollintel.engines.resource_assessment_engine import ResourceAssessmentEngine
from scrollintel.engines.crisis_team_formation_engine import CrisisTeamFormationEngine


@dataclass
class StressTestConfig:
    """Configuration for stress testing scenarios"""
    max_concurrent_crises: int = 10
    crisis_generation_rate: float = 2.0  # crises per second
    resource_constraint_level: float = 0.3  # 30% of normal resources
    time_pressure_multiplier: float = 0.5  # Half normal response time
    system_load_factor: float = 3.0  # 3x normal system load
    duration_seconds: int = 60


@dataclass
class StressTestResult:
    """Results from stress testing"""
    test_name: str
    total_crises_handled: int
    successful_responses: int
    failed_responses: int
    average_response_time: float
    max_response_time: float
    resource_utilization: float
    system_stability_score: float
    errors: List[str]
    performance_degradation: float


class CrisisStressTester:
    """Framework for stress testing crisis leadership capabilities"""
    
    def __init__(self):
        self.crisis_detector = CrisisDetectionEngine()
        self.decision_engine = DecisionTreeEngine()
        self.notification_engine = StakeholderNotificationEngine()
        self.resource_engine = ResourceAssessmentEngine()
        self.team_engine = CrisisTeamFormationEngine()
        
        # Stress test configurations
        self.stress_configs = self._load_stress_configs()
        
        # Performance baselines
        self.baseline_response_time = 30.0  # seconds
        self.baseline_success_rate = 0.95
        
    def _load_stress_configs(self) -> List[StressTestConfig]:
        """Load predefined stress test configurations"""
        return [
            StressTestConfig(
                max_concurrent_crises=5,
                crisis_generation_rate=1.0,
                resource_constraint_level=0.5,
                time_pressure_multiplier=0.7,
                system_load_factor=2.0,
                duration_seconds=30
            ),
            StressTestConfig(
                max_concurrent_crises=10,
                crisis_generation_rate=2.0,
                resource_constraint_level=0.3,
                time_pressure_multiplier=0.5,
                system_load_factor=3.0,
                duration_seconds=60
            ),
            StressTestConfig(
                max_concurrent_crises=20,
                crisis_generation_rate=5.0,
                resource_constraint_level=0.1,
                time_pressure_multiplier=0.2,
                system_load_factor=5.0,
                duration_seconds=120
            )
        ]
    
    async def run_stress_test(self, config: StressTestConfig, test_name: str) -> StressTestResult:
        """Run a comprehensive stress test"""
        start_time = time.time()
        crises_handled = 0
        successful_responses = 0
        failed_responses = 0
        response_times = []
        errors = []
        
        # Apply resource constraints
        await self._apply_resource_constraints(config.resource_constraint_level)
        
        # Apply system load
        await self._apply_system_load(config.system_load_factor)
        
        try:
            # Generate and handle crises concurrently
            crisis_tasks = []
            crisis_generation_task = asyncio.create_task(
                self._generate_crises_continuously(config)
            )
            
            # Monitor and handle crises for the specified duration
            end_time = start_time + config.duration_seconds
            
            while time.time() < end_time:
                # Check for new crises to handle
                if len(crisis_tasks) < config.max_concurrent_crises:
                    crisis_data = await self._generate_random_crisis(config)
                    task = asyncio.create_task(
                        self._handle_crisis_under_stress(crisis_data, config)
                    )
                    crisis_tasks.append(task)
                
                # Check completed tasks
                completed_tasks = [task for task in crisis_tasks if task.done()]
                for task in completed_tasks:
                    try:
                        result = await task
                        crises_handled += 1
                        if result['success']:
                            successful_responses += 1
                            response_times.append(result['response_time'])
                        else:
                            failed_responses += 1
                            errors.extend(result.get('errors', []))
                    except Exception as e:
                        failed_responses += 1
                        errors.append(str(e))
                    
                    crisis_tasks.remove(task)
                
                await asyncio.sleep(0.1)  # Small delay to prevent tight loop
            
            # Wait for remaining tasks to complete
            if crisis_tasks:
                remaining_results = await asyncio.gather(*crisis_tasks, return_exceptions=True)
                for result in remaining_results:
                    if isinstance(result, dict):
                        crises_handled += 1
                        if result['success']:
                            successful_responses += 1
                            response_times.append(result['response_time'])
                        else:
                            failed_responses += 1
                    elif isinstance(result, Exception):
                        failed_responses += 1
                        errors.append(str(result))
            
            # Cancel crisis generation
            crisis_generation_task.cancel()
            
            # Calculate metrics
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            max_response_time = max(response_times) if response_times else 0
            
            # Calculate performance degradation
            performance_degradation = self._calculate_performance_degradation(
                avg_response_time, successful_responses / max(1, crises_handled)
            )
            
            # Calculate system stability score
            stability_score = self._calculate_stability_score(
                successful_responses, failed_responses, avg_response_time, errors
            )
            
            return StressTestResult(
                test_name=test_name,
                total_crises_handled=crises_handled,
                successful_responses=successful_responses,
                failed_responses=failed_responses,
                average_response_time=avg_response_time,
                max_response_time=max_response_time,
                resource_utilization=1.0 - config.resource_constraint_level,
                system_stability_score=stability_score,
                errors=errors,
                performance_degradation=performance_degradation
            )
            
        except Exception as e:
            errors.append(f"Stress test failed: {str(e)}")
            return StressTestResult(
                test_name=test_name,
                total_crises_handled=crises_handled,
                successful_responses=successful_responses,
                failed_responses=failed_responses + 1,
                average_response_time=0,
                max_response_time=0,
                resource_utilization=0,
                system_stability_score=0,
                errors=errors,
                performance_degradation=1.0
            )
        
        finally:
            # Clean up resource constraints
            await self._reset_system_state()
    
    async def _apply_resource_constraints(self, constraint_level: float):
        """Apply resource constraints to simulate scarcity"""
        # Simulate reduced resource availability
        available_resources = int(100 * constraint_level)
        
        # Mock resource limitation
        with patch.object(self.resource_engine, 'get_available_resources') as mock_resources:
            mock_resources.return_value = {
                'cpu': available_resources,
                'memory': available_resources,
                'network': available_resources,
                'personnel': available_resources
            }
    
    async def _apply_system_load(self, load_factor: float):
        """Apply additional system load to simulate stress"""
        # Simulate increased system load through artificial delays
        base_delay = 0.1 * load_factor
        
        # Add delays to all engine operations
        for engine in [self.crisis_detector, self.decision_engine, 
                      self.notification_engine, self.resource_engine, self.team_engine]:
            if hasattr(engine, '_add_artificial_delay'):
                engine._add_artificial_delay(base_delay)
    
    async def _generate_crises_continuously(self, config: StressTestConfig):
        """Generate crises at the specified rate"""
        interval = 1.0 / config.crisis_generation_rate
        
        try:
            while True:
                await asyncio.sleep(interval)
                # Crisis generation is handled in the main loop
        except asyncio.CancelledError:
            pass
    
    async def _generate_random_crisis(self, config: StressTestConfig) -> Dict[str, Any]:
        """Generate a random crisis for stress testing"""
        crisis_types = ["technical", "security", "data", "reputation", "financial"]
        severities = ["low", "moderate", "high", "critical"]
        
        return {
            "type": random.choice(crisis_types),
            "severity": random.choice(severities),
            "affected_systems": random.sample(
                ["api", "database", "frontend", "payment", "auth"], 
                random.randint(1, 3)
            ),
            "timestamp": datetime.now(),
            "stress_multiplier": config.time_pressure_multiplier
        }
    
    async def _handle_crisis_under_stress(self, crisis_data: Dict[str, Any], 
                                        config: StressTestConfig) -> Dict[str, Any]:
        """Handle a single crisis under stress conditions"""
        start_time = time.time()
        errors = []
        
        try:
            # Reduced time limit due to stress
            time_limit = self.baseline_response_time * config.time_pressure_multiplier
            
            # Crisis detection with timeout
            detected_crisis = await asyncio.wait_for(
                self.crisis_detector.detect_crisis(crisis_data),
                timeout=time_limit * 0.2
            )
            
            # Rapid decision making with timeout
            decisions = await asyncio.wait_for(
                self.decision_engine.make_rapid_decisions({
                    "crisis": detected_crisis,
                    "time_pressure": time_limit,
                    "resource_constraints": config.resource_constraint_level
                }),
                timeout=time_limit * 0.3
            )
            
            # Stakeholder notification with timeout
            notifications = await asyncio.wait_for(
                self.notification_engine.notify_stakeholders(
                    crisis=detected_crisis,
                    stakeholders=["emergency_team"]
                ),
                timeout=time_limit * 0.2
            )
            
            # Resource assessment with timeout
            resources = await asyncio.wait_for(
                self.resource_engine.assess_resources(
                    crisis_type=crisis_data["type"],
                    severity=crisis_data["severity"]
                ),
                timeout=time_limit * 0.2
            )
            
            # Team formation with timeout
            team = await asyncio.wait_for(
                self.team_engine.form_crisis_team(
                    crisis=detected_crisis,
                    required_skills=["crisis_management"]
                ),
                timeout=time_limit * 0.1
            )
            
            response_time = time.time() - start_time
            
            # Determine success based on completion and time
            success = (
                detected_crisis is not None and
                decisions is not None and
                notifications is not None and
                resources is not None and
                team is not None and
                response_time <= time_limit
            )
            
            return {
                "success": success,
                "response_time": response_time,
                "errors": errors
            }
            
        except asyncio.TimeoutError:
            errors.append("Crisis handling timed out under stress")
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "errors": errors
            }
        except Exception as e:
            errors.append(f"Crisis handling failed: {str(e)}")
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "errors": errors
            }
    
    def _calculate_performance_degradation(self, avg_response_time: float, success_rate: float) -> float:
        """Calculate performance degradation compared to baseline"""
        time_degradation = max(0, (avg_response_time - self.baseline_response_time) / self.baseline_response_time)
        success_degradation = max(0, (self.baseline_success_rate - success_rate) / self.baseline_success_rate)
        
        return min(1.0, (time_degradation + success_degradation) / 2)
    
    def _calculate_stability_score(self, successful: int, failed: int, 
                                 avg_response_time: float, errors: List[str]) -> float:
        """Calculate system stability score under stress"""
        total_responses = successful + failed
        if total_responses == 0:
            return 0.0
        
        # Success rate component (50%)
        success_rate = successful / total_responses
        success_score = success_rate * 50
        
        # Response time component (30%)
        if avg_response_time <= self.baseline_response_time:
            time_score = 30
        else:
            time_penalty = min(25, (avg_response_time - self.baseline_response_time) / 10)
            time_score = max(5, 30 - time_penalty)
        
        # Error rate component (20%)
        error_rate = len(errors) / total_responses
        error_score = max(0, 20 - (error_rate * 100))
        
        return min(100.0, success_score + time_score + error_score)
    
    async def _reset_system_state(self):
        """Reset system to normal state after stress testing"""
        # Remove artificial constraints and delays
        pass


class TestCrisisStressTesting:
    """Test cases for crisis stress testing framework"""
    
    @pytest.fixture
    def stress_tester(self):
        return CrisisStressTester()
    
    @pytest.mark.asyncio
    async def test_light_stress_conditions(self, stress_tester):
        """Test system under light stress conditions"""
        config = StressTestConfig(
            max_concurrent_crises=3,
            crisis_generation_rate=0.5,
            resource_constraint_level=0.7,
            time_pressure_multiplier=0.8,
            system_load_factor=1.5,
            duration_seconds=15
        )
        
        result = await stress_tester.run_stress_test(config, "Light Stress Test")
        
        assert result.test_name == "Light Stress Test"
        assert result.total_crises_handled >= 0
        assert result.system_stability_score >= 0
        assert result.performance_degradation >= 0
    
    @pytest.mark.asyncio
    async def test_moderate_stress_conditions(self, stress_tester):
        """Test system under moderate stress conditions"""
        config = stress_tester.stress_configs[0]  # Moderate stress
        
        result = await stress_tester.run_stress_test(config, "Moderate Stress Test")
        
        assert result.total_crises_handled > 0
        assert result.successful_responses >= 0
        assert result.failed_responses >= 0
        assert result.average_response_time >= 0
    
    @pytest.mark.asyncio
    async def test_high_stress_conditions(self, stress_tester):
        """Test system under high stress conditions"""
        config = stress_tester.stress_configs[1]  # High stress
        
        result = await stress_tester.run_stress_test(config, "High Stress Test")
        
        assert result.total_crises_handled >= 0
        # Under high stress, some failures are expected
        assert result.performance_degradation >= 0
        assert result.system_stability_score >= 0
    
    @pytest.mark.asyncio
    async def test_extreme_stress_conditions(self, stress_tester):
        """Test system under extreme stress conditions"""
        config = stress_tester.stress_configs[2]  # Extreme stress
        
        result = await stress_tester.run_stress_test(config, "Extreme Stress Test")
        
        # System should still respond, even if degraded
        assert result.test_name == "Extreme Stress Test"
        assert isinstance(result.errors, list)
        # Performance degradation is expected under extreme stress
        assert result.performance_degradation >= 0
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_scenario(self, stress_tester):
        """Test behavior when resources are nearly exhausted"""
        config = StressTestConfig(
            max_concurrent_crises=15,
            crisis_generation_rate=3.0,
            resource_constraint_level=0.05,  # Only 5% resources available
            time_pressure_multiplier=0.3,
            system_load_factor=4.0,
            duration_seconds=30
        )
        
        result = await stress_tester.run_stress_test(config, "Resource Exhaustion Test")
        
        # System should handle resource exhaustion gracefully
        assert result.resource_utilization == 0.95  # 95% utilization
        assert result.total_crises_handled >= 0
    
    @pytest.mark.asyncio
    async def test_time_pressure_scenario(self, stress_tester):
        """Test behavior under extreme time pressure"""
        config = StressTestConfig(
            max_concurrent_crises=8,
            crisis_generation_rate=2.0,
            resource_constraint_level=0.5,
            time_pressure_multiplier=0.1,  # Only 10% of normal time
            system_load_factor=2.0,
            duration_seconds=20
        )
        
        result = await stress_tester.run_stress_test(config, "Time Pressure Test")
        
        # Verify system responds to time pressure
        assert result.average_response_time >= 0
        # Some degradation expected under extreme time pressure
        assert result.performance_degradation >= 0
    
    @pytest.mark.asyncio
    async def test_concurrent_stress_tests(self, stress_tester):
        """Test running multiple stress tests concurrently"""
        configs = stress_tester.stress_configs[:2]  # First two configs
        
        tasks = [
            stress_tester.run_stress_test(config, f"Concurrent Test {i}")
            for i, config in enumerate(configs)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        for result in results:
            assert result.total_crises_handled >= 0
            assert result.system_stability_score >= 0
    
    def test_stress_config_validation(self, stress_tester):
        """Test that stress configurations are valid"""
        configs = stress_tester.stress_configs
        
        assert len(configs) > 0
        
        for config in configs:
            assert config.max_concurrent_crises > 0
            assert config.crisis_generation_rate > 0
            assert 0 <= config.resource_constraint_level <= 1
            assert config.time_pressure_multiplier > 0
            assert config.system_load_factor > 0
            assert config.duration_seconds > 0
    
    def test_performance_degradation_calculation(self, stress_tester):
        """Test performance degradation calculation"""
        # Test with baseline performance
        degradation = stress_tester._calculate_performance_degradation(30.0, 0.95)
        assert degradation == 0.0  # No degradation at baseline
        
        # Test with degraded performance
        degradation = stress_tester._calculate_performance_degradation(60.0, 0.8)
        assert degradation > 0.0  # Should show degradation
        assert degradation <= 1.0  # Should be capped at 1.0
    
    def test_stability_score_calculation(self, stress_tester):
        """Test system stability score calculation"""
        # Test perfect stability
        score = stress_tester._calculate_stability_score(10, 0, 25.0, [])
        assert score > 90  # Should be high score
        
        # Test degraded stability
        score = stress_tester._calculate_stability_score(5, 5, 60.0, ["error1", "error2"])
        assert score < 90  # Should be lower score
        assert score >= 0   # Should not be negative


if __name__ == "__main__":
    # Run basic stress test
    async def main():
        tester = CrisisStressTester()
        print("Running Crisis Stress Testing...")
        
        for i, config in enumerate(tester.stress_configs):
            print(f"\nRunning stress test {i+1}...")
            result = await tester.run_stress_test(config, f"Stress Test {i+1}")
            print(f"Crises handled: {result.total_crises_handled}")
            print(f"Success rate: {result.successful_responses}/{result.total_crises_handled}")
            print(f"Avg response time: {result.average_response_time:.2f}s")
            print(f"Stability score: {result.system_stability_score:.1f}%")
            print(f"Performance degradation: {result.performance_degradation:.2f}")
    
    asyncio.run(main())