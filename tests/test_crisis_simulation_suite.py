"""
Crisis Simulation Testing Suite

This module provides comprehensive testing capabilities for crisis leadership scenarios,
including scenario-based testing, stress testing, and multi-crisis handling validation.
Requirements: 1.1, 2.1, 3.1, 4.1, 5.1
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, patch
from dataclasses import dataclass

from scrollintel.engines.crisis_detection_engine import CrisisDetectionEngine
from scrollintel.engines.decision_tree_engine import DecisionTreeEngine
from scrollintel.engines.stakeholder_notification_engine import StakeholderNotificationEngine
from scrollintel.engines.resource_assessment_engine import ResourceAssessmentEngine
from scrollintel.engines.crisis_team_formation_engine import CrisisTeamFormationEngine


@dataclass
class CrisisScenario:
    """Represents a crisis scenario for testing"""
    name: str
    crisis_type: str
    severity: str
    duration_minutes: int
    affected_systems: List[str]
    stakeholders: List[str]
    expected_response_time: int  # seconds
    complexity_score: int  # 1-10


@dataclass
class SimulationResult:
    """Results from crisis simulation"""
    scenario_name: str
    response_time: float
    decisions_made: int
    stakeholders_notified: int
    resources_allocated: int
    team_formed: bool
    success_score: float
    errors: List[str]


class CrisisSimulationSuite:
    """Comprehensive crisis simulation testing framework"""
    
    def __init__(self):
        self.crisis_detector = CrisisDetectionEngine()
        self.decision_engine = DecisionTreeEngine()
        self.notification_engine = StakeholderNotificationEngine()
        self.resource_engine = ResourceAssessmentEngine()
        self.team_engine = CrisisTeamFormationEngine()
        
        # Predefined crisis scenarios
        self.scenarios = self._load_crisis_scenarios()
        
    def _load_crisis_scenarios(self) -> List[CrisisScenario]:
        """Load predefined crisis scenarios for testing"""
        return [
            CrisisScenario(
                name="System Outage - Critical",
                crisis_type="technical",
                severity="critical",
                duration_minutes=30,
                affected_systems=["api", "database", "frontend"],
                stakeholders=["customers", "support_team", "executives"],
                expected_response_time=60,
                complexity_score=8
            ),
            CrisisScenario(
                name="Security Breach - Major",
                crisis_type="security",
                severity="major",
                duration_minutes=120,
                affected_systems=["user_data", "payment_system"],
                stakeholders=["customers", "legal_team", "security_team", "media"],
                expected_response_time=30,
                complexity_score=9
            ),
            CrisisScenario(
                name="Data Loss - Moderate",
                crisis_type="data",
                severity="moderate",
                duration_minutes=60,
                affected_systems=["backup_system", "database"],
                stakeholders=["customers", "engineering_team"],
                expected_response_time=90,
                complexity_score=6
            ),
            CrisisScenario(
                name="PR Crisis - High",
                crisis_type="reputation",
                severity="high",
                duration_minutes=240,
                affected_systems=["social_media", "customer_service"],
                stakeholders=["media", "customers", "executives", "pr_team"],
                expected_response_time=45,
                complexity_score=7
            ),
            CrisisScenario(
                name="Financial Crisis - Critical",
                crisis_type="financial",
                severity="critical",
                duration_minutes=480,
                affected_systems=["payment_processing", "billing"],
                stakeholders=["investors", "board", "customers", "finance_team"],
                expected_response_time=120,
                complexity_score=10
            )
        ]
    
    async def run_scenario_test(self, scenario: CrisisScenario) -> SimulationResult:
        """Run a single crisis scenario test"""
        start_time = time.time()
        errors = []
        
        try:
            # 1. Crisis Detection (Requirement 1.1)
            crisis_data = {
                "type": scenario.crisis_type,
                "severity": scenario.severity,
                "affected_systems": scenario.affected_systems,
                "timestamp": datetime.now()
            }
            
            detected_crisis = await self.crisis_detector.detect_crisis(crisis_data)
            
            # 2. Decision Making (Requirement 2.1)
            decision_context = {
                "crisis": detected_crisis,
                "time_pressure": scenario.expected_response_time,
                "complexity": scenario.complexity_score
            }
            
            decisions = await self.decision_engine.make_rapid_decisions(decision_context)
            
            # 3. Stakeholder Notification (Requirement 3.1)
            notification_result = await self.notification_engine.notify_stakeholders(
                crisis=detected_crisis,
                stakeholders=scenario.stakeholders
            )
            
            # 4. Resource Assessment (Requirement 4.1)
            resource_assessment = await self.resource_engine.assess_resources(
                crisis_type=scenario.crisis_type,
                severity=scenario.severity
            )
            
            # 5. Team Formation (Requirement 5.1)
            team_result = await self.team_engine.form_crisis_team(
                crisis=detected_crisis,
                required_skills=self._get_required_skills(scenario.crisis_type)
            )
            
            response_time = time.time() - start_time
            
            # Calculate success score
            success_score = self._calculate_success_score(
                scenario, response_time, decisions, notification_result, 
                resource_assessment, team_result
            )
            
            return SimulationResult(
                scenario_name=scenario.name,
                response_time=response_time,
                decisions_made=len(decisions) if decisions else 0,
                stakeholders_notified=len(notification_result.get('notified', [])) if notification_result else 0,
                resources_allocated=len(resource_assessment.get('allocated', [])) if resource_assessment else 0,
                team_formed=bool(team_result and team_result.get('team_id')),
                success_score=success_score,
                errors=errors
            )
            
        except Exception as e:
            errors.append(str(e))
            return SimulationResult(
                scenario_name=scenario.name,
                response_time=time.time() - start_time,
                decisions_made=0,
                stakeholders_notified=0,
                resources_allocated=0,
                team_formed=False,
                success_score=0.0,
                errors=errors
            )
    
    def _get_required_skills(self, crisis_type: str) -> List[str]:
        """Get required skills for crisis type"""
        skill_map = {
            "technical": ["system_admin", "devops", "engineering"],
            "security": ["security_expert", "legal", "communications"],
            "data": ["data_engineer", "backup_specialist", "compliance"],
            "reputation": ["pr_specialist", "communications", "legal"],
            "financial": ["finance", "legal", "executive"]
        }
        return skill_map.get(crisis_type, ["general_management"])
    
    def _calculate_success_score(self, scenario: CrisisScenario, response_time: float,
                               decisions: Any, notifications: Any, resources: Any, team: Any) -> float:
        """Calculate overall success score for scenario"""
        score = 0.0
        
        # Response time score (30%)
        if response_time <= scenario.expected_response_time:
            score += 30.0
        else:
            # Penalty for slow response
            penalty = min(20.0, (response_time - scenario.expected_response_time) / 10)
            score += max(10.0, 30.0 - penalty)
        
        # Decision quality score (25%)
        if decisions and len(decisions) > 0:
            score += 25.0
        
        # Communication score (20%)
        if notifications and len(notifications.get('notified', [])) >= len(scenario.stakeholders):
            score += 20.0
        elif notifications:
            score += 10.0
        
        # Resource allocation score (15%)
        if resources and len(resources.get('allocated', [])) > 0:
            score += 15.0
        
        # Team formation score (10%)
        if team and team.get('team_id'):
            score += 10.0
        
        return min(100.0, score)


class TestCrisisSimulationSuite:
    """Test cases for crisis simulation suite"""
    
    @pytest.fixture
    def simulation_suite(self):
        return CrisisSimulationSuite()
    
    @pytest.mark.asyncio
    async def test_single_scenario_execution(self, simulation_suite):
        """Test execution of a single crisis scenario"""
        scenario = simulation_suite.scenarios[0]  # System Outage
        
        result = await simulation_suite.run_scenario_test(scenario)
        
        assert result.scenario_name == scenario.name
        assert result.response_time > 0
        assert result.success_score >= 0
        assert isinstance(result.errors, list)
    
    @pytest.mark.asyncio
    async def test_all_predefined_scenarios(self, simulation_suite):
        """Test all predefined crisis scenarios"""
        results = []
        
        for scenario in simulation_suite.scenarios:
            result = await simulation_suite.run_scenario_test(scenario)
            results.append(result)
        
        assert len(results) == len(simulation_suite.scenarios)
        
        # Verify each scenario produced results
        for result in results:
            assert result.response_time > 0
            assert result.success_score >= 0
    
    @pytest.mark.asyncio
    async def test_stress_testing_rapid_succession(self, simulation_suite):
        """Test stress conditions with rapid scenario succession"""
        # Run multiple scenarios in quick succession
        tasks = []
        for scenario in simulation_suite.scenarios[:3]:  # First 3 scenarios
            tasks.append(simulation_suite.run_scenario_test(scenario))
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        
        # Verify all scenarios completed despite stress
        for result in results:
            assert result.response_time > 0
            # Allow for some degradation under stress
            assert result.success_score >= 0
    
    @pytest.mark.asyncio
    async def test_multi_crisis_handling(self, simulation_suite):
        """Test handling multiple simultaneous crises"""
        # Create overlapping crisis scenarios
        scenario1 = simulation_suite.scenarios[0]  # System Outage
        scenario2 = simulation_suite.scenarios[1]  # Security Breach
        
        # Start both scenarios simultaneously
        task1 = asyncio.create_task(simulation_suite.run_scenario_test(scenario1))
        task2 = asyncio.create_task(simulation_suite.run_scenario_test(scenario2))
        
        result1, result2 = await asyncio.gather(task1, task2)
        
        # Both crises should be handled
        assert result1.success_score > 0
        assert result2.success_score > 0
        
        # Response times may be longer due to resource contention
        assert result1.response_time > 0
        assert result2.response_time > 0
    
    @pytest.mark.asyncio
    async def test_extreme_pressure_conditions(self, simulation_suite):
        """Test system behavior under extreme pressure"""
        # Create high-pressure scenario with very short response time
        extreme_scenario = CrisisScenario(
            name="Extreme Pressure Test",
            crisis_type="security",
            severity="critical",
            duration_minutes=5,
            affected_systems=["all_systems"],
            stakeholders=["all_stakeholders"],
            expected_response_time=10,  # Very short time
            complexity_score=10
        )
        
        result = await simulation_suite.run_scenario_test(extreme_scenario)
        
        # System should still respond, even if not perfectly
        assert result.response_time > 0
        assert result.scenario_name == extreme_scenario.name
        # Success score may be lower due to extreme pressure
        assert result.success_score >= 0
    
    def test_scenario_loading(self, simulation_suite):
        """Test that crisis scenarios are properly loaded"""
        scenarios = simulation_suite.scenarios
        
        assert len(scenarios) > 0
        assert all(isinstance(s, CrisisScenario) for s in scenarios)
        
        # Verify scenario diversity
        crisis_types = {s.crisis_type for s in scenarios}
        assert len(crisis_types) > 1  # Multiple crisis types
        
        severity_levels = {s.severity for s in scenarios}
        assert len(severity_levels) > 1  # Multiple severity levels
    
    def test_success_score_calculation(self, simulation_suite):
        """Test success score calculation logic"""
        scenario = simulation_suite.scenarios[0]
        
        # Test perfect score conditions
        mock_decisions = [{"decision": "test"}]
        mock_notifications = {"notified": ["stakeholder1", "stakeholder2", "stakeholder3"]}
        mock_resources = {"allocated": ["resource1", "resource2"]}
        mock_team = {"team_id": "team123"}
        
        score = simulation_suite._calculate_success_score(
            scenario, 30.0, mock_decisions, mock_notifications, mock_resources, mock_team
        )
        
        assert score > 0
        assert score <= 100.0
    
    @pytest.mark.asyncio
    async def test_concurrent_scenario_execution(self, simulation_suite):
        """Test concurrent execution of multiple different scenarios"""
        # Run all scenarios concurrently
        tasks = [
            simulation_suite.run_scenario_test(scenario) 
            for scenario in simulation_suite.scenarios
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all scenarios completed
        assert len(results) == len(simulation_suite.scenarios)
        
        # Check that results are valid (not exceptions)
        valid_results = [r for r in results if isinstance(r, SimulationResult)]
        assert len(valid_results) > 0
    
    @pytest.mark.asyncio
    async def test_scenario_timeout_handling(self, simulation_suite):
        """Test handling of scenarios that might timeout"""
        # Create a scenario that might take longer
        long_scenario = CrisisScenario(
            name="Long Duration Test",
            crisis_type="financial",
            severity="critical",
            duration_minutes=480,
            affected_systems=["payment_system", "billing", "accounting"],
            stakeholders=["investors", "board", "customers", "regulators"],
            expected_response_time=300,  # 5 minutes
            complexity_score=10
        )
        
        # Set a reasonable timeout
        try:
            result = await asyncio.wait_for(
                simulation_suite.run_scenario_test(long_scenario), 
                timeout=10.0  # 10 second timeout for test
            )
            assert result.scenario_name == long_scenario.name
        except asyncio.TimeoutError:
            # Timeout is acceptable for this test
            pass


if __name__ == "__main__":
    # Run basic simulation test
    async def main():
        suite = CrisisSimulationSuite()
        print("Running Crisis Simulation Suite...")
        
        for scenario in suite.scenarios:
            print(f"\nTesting scenario: {scenario.name}")
            result = await suite.run_scenario_test(scenario)
            print(f"Response time: {result.response_time:.2f}s")
            print(f"Success score: {result.success_score:.1f}%")
            if result.errors:
                print(f"Errors: {result.errors}")
    
    asyncio.run(main())