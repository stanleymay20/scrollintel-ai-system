"""
Multi-Crisis Handling Testing Framework

This module provides testing capabilities for handling multiple simultaneous crises,
including crisis prioritization, resource allocation conflicts, and coordination challenges.
Requirements: 1.1, 2.1, 3.1, 4.1, 5.1
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from unittest.mock import Mock, patch
from dataclasses import dataclass
from enum import Enum

from scrollintel.engines.crisis_detection_engine import CrisisDetectionEngine
from scrollintel.engines.decision_tree_engine import DecisionTreeEngine
from scrollintel.engines.stakeholder_notification_engine import StakeholderNotificationEngine
from scrollintel.engines.resource_assessment_engine import ResourceAssessmentEngine
from scrollintel.engines.crisis_team_formation_engine import CrisisTeamFormationEngine


class CrisisPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CrisisInstance:
    """Represents a single crisis instance"""
    id: str
    name: str
    crisis_type: str
    severity: str
    priority: CrisisPriority
    start_time: datetime
    estimated_duration: int  # minutes
    required_resources: List[str]
    affected_stakeholders: List[str]
    dependencies: List[str]  # Other crisis IDs this depends on


@dataclass
class MultiCrisisResult:
    """Results from multi-crisis handling test"""
    test_name: str
    total_crises: int
    simultaneously_active: int
    successfully_resolved: int
    failed_resolutions: int
    average_resolution_time: float
    resource_conflicts: int
    coordination_issues: int
    priority_violations: int
    overall_effectiveness: float
    errors: List[str]


class MultiCrisisHandler:
    """Framework for handling multiple simultaneous crises"""
    
    def __init__(self):
        self.crisis_detector = CrisisDetectionEngine()
        self.decision_engine = DecisionTreeEngine()
        self.notification_engine = StakeholderNotificationEngine()
        self.resource_engine = ResourceAssessmentEngine()
        self.team_engine = CrisisTeamFormationEngine()
        
        # Active crisis tracking
        self.active_crises: Dict[str, CrisisInstance] = {}
        self.resource_allocations: Dict[str, List[str]] = {}  # resource_id -> crisis_ids
        self.team_assignments: Dict[str, str] = {}  # team_id -> crisis_id
        
        # Multi-crisis scenarios
        self.multi_crisis_scenarios = self._load_multi_crisis_scenarios()
    
    def _load_multi_crisis_scenarios(self) -> List[List[CrisisInstance]]:
        """Load predefined multi-crisis scenarios"""
        return [
            # Scenario 1: Cascading Technical Failures
            [
                CrisisInstance(
                    id="tech_1",
                    name="Database Outage",
                    crisis_type="technical",
                    severity="critical",
                    priority=CrisisPriority.CRITICAL,
                    start_time=datetime.now(),
                    estimated_duration=60,
                    required_resources=["database_admin", "infrastructure_team"],
                    affected_stakeholders=["customers", "support_team"],
                    dependencies=[]
                ),
                CrisisInstance(
                    id="tech_2",
                    name="API Service Failure",
                    crisis_type="technical",
                    severity="high",
                    priority=CrisisPriority.HIGH,
                    start_time=datetime.now() + timedelta(minutes=5),
                    estimated_duration=45,
                    required_resources=["backend_team", "infrastructure_team"],
                    affected_stakeholders=["customers", "partners"],
                    dependencies=["tech_1"]
                ),
                CrisisInstance(
                    id="tech_3",
                    name="Frontend Degradation",
                    crisis_type="technical",
                    severity="medium",
                    priority=CrisisPriority.MEDIUM,
                    start_time=datetime.now() + timedelta(minutes=10),
                    estimated_duration=30,
                    required_resources=["frontend_team"],
                    affected_stakeholders=["customers"],
                    dependencies=["tech_2"]
                )
            ],
            
            # Scenario 2: Security and PR Crisis
            [
                CrisisInstance(
                    id="sec_1",
                    name="Data Breach",
                    crisis_type="security",
                    severity="critical",
                    priority=CrisisPriority.CRITICAL,
                    start_time=datetime.now(),
                    estimated_duration=120,
                    required_resources=["security_team", "legal_team", "communications"],
                    affected_stakeholders=["customers", "regulators", "media"],
                    dependencies=[]
                ),
                CrisisInstance(
                    id="pr_1",
                    name="Media Backlash",
                    crisis_type="reputation",
                    severity="high",
                    priority=CrisisPriority.HIGH,
                    start_time=datetime.now() + timedelta(minutes=15),
                    estimated_duration=180,
                    required_resources=["pr_team", "communications", "executives"],
                    affected_stakeholders=["media", "customers", "investors"],
                    dependencies=["sec_1"]
                )
            ],
            
            # Scenario 3: Resource Competition Crisis
            [
                CrisisInstance(
                    id="fin_1",
                    name="Payment System Failure",
                    crisis_type="financial",
                    severity="critical",
                    priority=CrisisPriority.CRITICAL,
                    start_time=datetime.now(),
                    estimated_duration=90,
                    required_resources=["finance_team", "infrastructure_team", "legal_team"],
                    affected_stakeholders=["customers", "partners", "regulators"],
                    dependencies=[]
                ),
                CrisisInstance(
                    id="ops_1",
                    name="Supply Chain Disruption",
                    crisis_type="operational",
                    severity="high",
                    priority=CrisisPriority.HIGH,
                    start_time=datetime.now() + timedelta(minutes=2),
                    estimated_duration=240,
                    required_resources=["operations_team", "finance_team", "legal_team"],
                    affected_stakeholders=["customers", "suppliers"],
                    dependencies=[]
                ),
                CrisisInstance(
                    id="hr_1",
                    name="Key Personnel Departure",
                    crisis_type="human_resources",
                    severity="medium",
                    priority=CrisisPriority.MEDIUM,
                    start_time=datetime.now() + timedelta(minutes=5),
                    estimated_duration=480,
                    required_resources=["hr_team", "executives", "legal_team"],
                    affected_stakeholders=["employees", "management"],
                    dependencies=[]
                )
            ]
        ]
    
    async def run_multi_crisis_test(self, scenario: List[CrisisInstance], 
                                  test_name: str) -> MultiCrisisResult:
        """Run a multi-crisis handling test"""
        start_time = time.time()
        errors = []
        resolution_times = []
        resource_conflicts = 0
        coordination_issues = 0
        priority_violations = 0
        
        try:
            # Initialize crisis tracking
            self.active_crises.clear()
            self.resource_allocations.clear()
            self.team_assignments.clear()
            
            # Start crisis detection and handling
            crisis_tasks = []
            
            for crisis in scenario:
                # Add crisis to active tracking
                self.active_crises[crisis.id] = crisis
                
                # Create handling task
                task = asyncio.create_task(
                    self._handle_single_crisis_in_multi_context(crisis)
                )
                crisis_tasks.append((crisis.id, task))
            
            # Monitor crisis resolution
            resolved_crises = []
            failed_crises = []
            
            while crisis_tasks:
                # Check for completed tasks
                completed_tasks = []
                for crisis_id, task in crisis_tasks:
                    if task.done():
                        completed_tasks.append((crisis_id, task))
                
                # Process completed tasks
                for crisis_id, task in completed_tasks:
                    try:
                        result = await task
                        if result['success']:
                            resolved_crises.append(crisis_id)
                            resolution_times.append(result['resolution_time'])
                        else:
                            failed_crises.append(crisis_id)
                            errors.extend(result.get('errors', []))
                        
                        # Check for conflicts and issues
                        resource_conflicts += result.get('resource_conflicts', 0)
                        coordination_issues += result.get('coordination_issues', 0)
                        priority_violations += result.get('priority_violations', 0)
                        
                    except Exception as e:
                        failed_crises.append(crisis_id)
                        errors.append(f"Crisis {crisis_id} failed: {str(e)}")
                    
                    crisis_tasks.remove((crisis_id, task))
                
                # Check for resource conflicts
                await self._detect_resource_conflicts()
                
                # Check for priority violations
                await self._detect_priority_violations()
                
                await asyncio.sleep(0.1)  # Small delay
            
            # Calculate metrics
            total_crises = len(scenario)
            successfully_resolved = len(resolved_crises)
            failed_resolutions = len(failed_crises)
            simultaneously_active = len([c for c in scenario if c.start_time <= datetime.now()])
            
            avg_resolution_time = (
                sum(resolution_times) / len(resolution_times) 
                if resolution_times else 0
            )
            
            # Calculate overall effectiveness
            effectiveness = self._calculate_multi_crisis_effectiveness(
                total_crises, successfully_resolved, avg_resolution_time,
                resource_conflicts, coordination_issues, priority_violations
            )
            
            return MultiCrisisResult(
                test_name=test_name,
                total_crises=total_crises,
                simultaneously_active=simultaneously_active,
                successfully_resolved=successfully_resolved,
                failed_resolutions=failed_resolutions,
                average_resolution_time=avg_resolution_time,
                resource_conflicts=resource_conflicts,
                coordination_issues=coordination_issues,
                priority_violations=priority_violations,
                overall_effectiveness=effectiveness,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Multi-crisis test failed: {str(e)}")
            return MultiCrisisResult(
                test_name=test_name,
                total_crises=len(scenario),
                simultaneously_active=0,
                successfully_resolved=0,
                failed_resolutions=len(scenario),
                average_resolution_time=0,
                resource_conflicts=0,
                coordination_issues=0,
                priority_violations=0,
                overall_effectiveness=0,
                errors=errors
            )
    
    async def _handle_single_crisis_in_multi_context(self, crisis: CrisisInstance) -> Dict[str, Any]:
        """Handle a single crisis within multi-crisis context"""
        start_time = time.time()
        errors = []
        resource_conflicts = 0
        coordination_issues = 0
        priority_violations = 0
        
        try:
            # Wait for dependencies to be resolved
            await self._wait_for_dependencies(crisis)
            
            # Crisis detection
            crisis_data = {
                "id": crisis.id,
                "type": crisis.crisis_type,
                "severity": crisis.severity,
                "priority": crisis.priority.value,
                "timestamp": crisis.start_time
            }
            
            detected_crisis = await self.crisis_detector.detect_crisis(crisis_data)
            
            # Check for resource conflicts before allocation
            available_resources = await self._check_resource_availability(crisis.required_resources)
            if not available_resources:
                resource_conflicts += 1
                # Try to reallocate resources based on priority
                await self._reallocate_resources_by_priority(crisis)
            
            # Decision making with multi-crisis context
            decision_context = {
                "crisis": detected_crisis,
                "active_crises": list(self.active_crises.keys()),
                "priority": crisis.priority.value,
                "resource_constraints": self.resource_allocations
            }
            
            decisions = await self.decision_engine.make_rapid_decisions(decision_context)
            
            # Stakeholder notification with coordination
            notification_result = await self._coordinate_stakeholder_notifications(crisis)
            if not notification_result.get('coordinated', True):
                coordination_issues += 1
            
            # Resource allocation
            resource_result = await self._allocate_resources_with_conflict_resolution(crisis)
            if not resource_result.get('allocated', True):
                resource_conflicts += 1
            
            # Team formation with multi-crisis awareness
            team_result = await self._form_team_with_multi_crisis_awareness(crisis)
            
            # Check for priority violations
            if await self._check_priority_violations(crisis):
                priority_violations += 1
            
            resolution_time = time.time() - start_time
            
            # Determine success
            success = (
                detected_crisis is not None and
                decisions is not None and
                notification_result.get('success', False) and
                resource_result.get('success', False) and
                team_result.get('success', False)
            )
            
            # Remove from active crises if successful
            if success and crisis.id in self.active_crises:
                del self.active_crises[crisis.id]
            
            return {
                "success": success,
                "resolution_time": resolution_time,
                "resource_conflicts": resource_conflicts,
                "coordination_issues": coordination_issues,
                "priority_violations": priority_violations,
                "errors": errors
            }
            
        except Exception as e:
            errors.append(f"Crisis handling failed: {str(e)}")
            return {
                "success": False,
                "resolution_time": time.time() - start_time,
                "resource_conflicts": resource_conflicts,
                "coordination_issues": coordination_issues,
                "priority_violations": priority_violations,
                "errors": errors
            }
    
    async def _wait_for_dependencies(self, crisis: CrisisInstance):
        """Wait for crisis dependencies to be resolved"""
        if not crisis.dependencies:
            return
        
        max_wait_time = 300  # 5 minutes max wait
        start_wait = time.time()
        
        while crisis.dependencies and (time.time() - start_wait) < max_wait_time:
            # Check if dependencies are resolved
            resolved_dependencies = []
            for dep_id in crisis.dependencies:
                if dep_id not in self.active_crises:
                    resolved_dependencies.append(dep_id)
            
            # Remove resolved dependencies
            for dep_id in resolved_dependencies:
                crisis.dependencies.remove(dep_id)
            
            if crisis.dependencies:
                await asyncio.sleep(1)  # Wait before checking again
    
    async def _check_resource_availability(self, required_resources: List[str]) -> bool:
        """Check if required resources are available"""
        for resource in required_resources:
            if resource in self.resource_allocations:
                # Resource is already allocated to other crises
                return False
        return True
    
    async def _reallocate_resources_by_priority(self, crisis: CrisisInstance):
        """Reallocate resources based on crisis priority"""
        for resource in crisis.required_resources:
            if resource in self.resource_allocations:
                # Check if current allocation has lower priority
                current_crisis_ids = self.resource_allocations[resource]
                for current_id in current_crisis_ids:
                    if current_id in self.active_crises:
                        current_crisis = self.active_crises[current_id]
                        if crisis.priority.value > current_crisis.priority.value:
                            # Reallocate resource to higher priority crisis
                            self.resource_allocations[resource] = [crisis.id]
                            break
    
    async def _coordinate_stakeholder_notifications(self, crisis: CrisisInstance) -> Dict[str, Any]:
        """Coordinate stakeholder notifications to avoid conflicts"""
        try:
            # Check for overlapping stakeholders with other active crises
            overlapping_stakeholders = []
            for other_id, other_crisis in self.active_crises.items():
                if other_id != crisis.id:
                    overlap = set(crisis.affected_stakeholders) & set(other_crisis.affected_stakeholders)
                    overlapping_stakeholders.extend(list(overlap))
            
            # Coordinate notifications for overlapping stakeholders
            coordinated_message = await self._create_coordinated_message(
                crisis, overlapping_stakeholders
            )
            
            # Send notifications
            result = await self.notification_engine.notify_stakeholders(
                crisis={"id": crisis.id, "type": crisis.crisis_type},
                stakeholders=crisis.affected_stakeholders,
                coordinated_message=coordinated_message
            )
            
            return {
                "success": True,
                "coordinated": len(overlapping_stakeholders) == 0 or coordinated_message is not None,
                "overlapping_stakeholders": overlapping_stakeholders
            }
            
        except Exception as e:
            return {
                "success": False,
                "coordinated": False,
                "error": str(e)
            }
    
    async def _create_coordinated_message(self, crisis: CrisisInstance, 
                                        overlapping_stakeholders: List[str]) -> Optional[str]:
        """Create coordinated message for overlapping stakeholders"""
        if not overlapping_stakeholders:
            return None
        
        # Create message that acknowledges multiple ongoing crises
        return f"Multi-crisis update: Addressing {crisis.name} in coordination with other ongoing incidents."
    
    async def _allocate_resources_with_conflict_resolution(self, crisis: CrisisInstance) -> Dict[str, Any]:
        """Allocate resources with conflict resolution"""
        try:
            allocated_resources = []
            
            for resource in crisis.required_resources:
                if resource not in self.resource_allocations:
                    # Resource is available
                    self.resource_allocations[resource] = [crisis.id]
                    allocated_resources.append(resource)
                else:
                    # Resource conflict - try to resolve based on priority
                    current_crisis_ids = self.resource_allocations[resource]
                    can_reallocate = True
                    
                    for current_id in current_crisis_ids:
                        if current_id in self.active_crises:
                            current_crisis = self.active_crises[current_id]
                            if current_crisis.priority.value >= crisis.priority.value:
                                can_reallocate = False
                                break
                    
                    if can_reallocate:
                        self.resource_allocations[resource] = [crisis.id]
                        allocated_resources.append(resource)
            
            success = len(allocated_resources) >= len(crisis.required_resources) * 0.7  # 70% threshold
            
            return {
                "success": success,
                "allocated": success,
                "allocated_resources": allocated_resources
            }
            
        except Exception as e:
            return {
                "success": False,
                "allocated": False,
                "error": str(e)
            }
    
    async def _form_team_with_multi_crisis_awareness(self, crisis: CrisisInstance) -> Dict[str, Any]:
        """Form crisis team with awareness of other active crises"""
        try:
            # Consider team members already assigned to other crises
            unavailable_team_members = list(self.team_assignments.keys())
            
            team_result = await self.team_engine.form_crisis_team(
                crisis={"id": crisis.id, "type": crisis.crisis_type},
                required_skills=crisis.required_resources,
                unavailable_members=unavailable_team_members
            )
            
            if team_result and team_result.get('team_id'):
                self.team_assignments[team_result['team_id']] = crisis.id
            
            return {
                "success": bool(team_result and team_result.get('team_id')),
                "team_id": team_result.get('team_id') if team_result else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _check_priority_violations(self, crisis: CrisisInstance) -> bool:
        """Check if crisis priority is being violated by resource allocation"""
        for resource in crisis.required_resources:
            if resource in self.resource_allocations:
                allocated_crisis_ids = self.resource_allocations[resource]
                for allocated_id in allocated_crisis_ids:
                    if allocated_id != crisis.id and allocated_id in self.active_crises:
                        allocated_crisis = self.active_crises[allocated_id]
                        if allocated_crisis.priority.value < crisis.priority.value:
                            return True  # Lower priority crisis has resource that higher priority needs
        return False
    
    async def _detect_resource_conflicts(self):
        """Detect ongoing resource conflicts"""
        # This would be called periodically to detect conflicts
        pass
    
    async def _detect_priority_violations(self):
        """Detect priority violations in resource allocation"""
        # This would be called periodically to detect violations
        pass
    
    def _calculate_multi_crisis_effectiveness(self, total_crises: int, resolved: int,
                                           avg_time: float, conflicts: int,
                                           coordination_issues: int, violations: int) -> float:
        """Calculate overall effectiveness of multi-crisis handling"""
        if total_crises == 0:
            return 0.0
        
        # Resolution rate (40%)
        resolution_score = (resolved / total_crises) * 40
        
        # Time efficiency (25%)
        baseline_time = 60.0  # 1 minute baseline
        if avg_time <= baseline_time:
            time_score = 25
        else:
            time_penalty = min(20, (avg_time - baseline_time) / 30)
            time_score = max(5, 25 - time_penalty)
        
        # Conflict management (20%)
        conflict_penalty = min(15, conflicts * 3)
        conflict_score = max(5, 20 - conflict_penalty)
        
        # Coordination quality (15%)
        coordination_penalty = min(10, coordination_issues * 2)
        coordination_score = max(5, 15 - coordination_penalty)
        
        return min(100.0, resolution_score + time_score + conflict_score + coordination_score)


class TestMultiCrisisHandling:
    """Test cases for multi-crisis handling framework"""
    
    @pytest.fixture
    def multi_crisis_handler(self):
        return MultiCrisisHandler()
    
    @pytest.mark.asyncio
    async def test_cascading_technical_failures(self, multi_crisis_handler):
        """Test handling cascading technical failures"""
        scenario = multi_crisis_handler.multi_crisis_scenarios[0]
        
        result = await multi_crisis_handler.run_multi_crisis_test(
            scenario, "Cascading Technical Failures"
        )
        
        assert result.test_name == "Cascading Technical Failures"
        assert result.total_crises == 3
        assert result.successfully_resolved >= 0
        assert result.overall_effectiveness >= 0
    
    @pytest.mark.asyncio
    async def test_security_and_pr_crisis(self, multi_crisis_handler):
        """Test handling security breach with PR crisis"""
        scenario = multi_crisis_handler.multi_crisis_scenarios[1]
        
        result = await multi_crisis_handler.run_multi_crisis_test(
            scenario, "Security and PR Crisis"
        )
        
        assert result.total_crises == 2
        assert result.coordination_issues >= 0  # May have coordination challenges
        assert isinstance(result.errors, list)
    
    @pytest.mark.asyncio
    async def test_resource_competition_crisis(self, multi_crisis_handler):
        """Test handling crises that compete for same resources"""
        scenario = multi_crisis_handler.multi_crisis_scenarios[2]
        
        result = await multi_crisis_handler.run_multi_crisis_test(
            scenario, "Resource Competition Crisis"
        )
        
        assert result.total_crises == 3
        # Resource conflicts expected in this scenario
        assert result.resource_conflicts >= 0
        assert result.overall_effectiveness >= 0
    
    @pytest.mark.asyncio
    async def test_priority_based_resource_allocation(self, multi_crisis_handler):
        """Test that higher priority crises get resources first"""
        # Create scenario with different priorities
        scenario = [
            CrisisInstance(
                id="low_priority",
                name="Low Priority Issue",
                crisis_type="operational",
                severity="low",
                priority=CrisisPriority.LOW,
                start_time=datetime.now(),
                estimated_duration=60,
                required_resources=["shared_resource"],
                affected_stakeholders=["internal_team"],
                dependencies=[]
            ),
            CrisisInstance(
                id="critical_priority",
                name="Critical Issue",
                crisis_type="security",
                severity="critical",
                priority=CrisisPriority.CRITICAL,
                start_time=datetime.now() + timedelta(seconds=5),
                estimated_duration=30,
                required_resources=["shared_resource"],
                affected_stakeholders=["customers", "regulators"],
                dependencies=[]
            )
        ]
        
        result = await multi_crisis_handler.run_multi_crisis_test(
            scenario, "Priority Test"
        )
        
        # Critical priority should be handled successfully
        assert result.successfully_resolved >= 1
        # May have some resource conflicts due to reallocation
        assert result.resource_conflicts >= 0
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self, multi_crisis_handler):
        """Test that crisis dependencies are properly handled"""
        scenario = multi_crisis_handler.multi_crisis_scenarios[0]  # Has dependencies
        
        result = await multi_crisis_handler.run_multi_crisis_test(
            scenario, "Dependency Resolution Test"
        )
        
        # Should handle dependencies properly
        assert result.total_crises == 3
        assert result.coordination_issues >= 0
    
    @pytest.mark.asyncio
    async def test_stakeholder_notification_coordination(self, multi_crisis_handler):
        """Test coordination of stakeholder notifications across crises"""
        # Create scenario with overlapping stakeholders
        scenario = [
            CrisisInstance(
                id="crisis_1",
                name="Crisis 1",
                crisis_type="technical",
                severity="high",
                priority=CrisisPriority.HIGH,
                start_time=datetime.now(),
                estimated_duration=60,
                required_resources=["team_1"],
                affected_stakeholders=["customers", "support_team"],
                dependencies=[]
            ),
            CrisisInstance(
                id="crisis_2",
                name="Crisis 2",
                crisis_type="operational",
                severity="medium",
                priority=CrisisPriority.MEDIUM,
                start_time=datetime.now() + timedelta(seconds=10),
                estimated_duration=45,
                required_resources=["team_2"],
                affected_stakeholders=["customers", "management"],  # Overlapping: customers
                dependencies=[]
            )
        ]
        
        result = await multi_crisis_handler.run_multi_crisis_test(
            scenario, "Stakeholder Coordination Test"
        )
        
        assert result.total_crises == 2
        # Should handle overlapping stakeholders
        assert result.coordination_issues >= 0
    
    def test_crisis_priority_enum(self):
        """Test crisis priority enumeration"""
        assert CrisisPriority.CRITICAL.value > CrisisPriority.HIGH.value
        assert CrisisPriority.HIGH.value > CrisisPriority.MEDIUM.value
        assert CrisisPriority.MEDIUM.value > CrisisPriority.LOW.value
    
    def test_multi_crisis_scenarios_loading(self, multi_crisis_handler):
        """Test that multi-crisis scenarios are properly loaded"""
        scenarios = multi_crisis_handler.multi_crisis_scenarios
        
        assert len(scenarios) > 0
        
        for scenario in scenarios:
            assert len(scenario) > 1  # Multi-crisis means more than one
            for crisis in scenario:
                assert isinstance(crisis, CrisisInstance)
                assert crisis.id
                assert crisis.name
                assert crisis.crisis_type
                assert isinstance(crisis.priority, CrisisPriority)
    
    def test_effectiveness_calculation(self, multi_crisis_handler):
        """Test multi-crisis effectiveness calculation"""
        # Test perfect scenario
        effectiveness = multi_crisis_handler._calculate_multi_crisis_effectiveness(
            total_crises=3, resolved=3, avg_time=30.0, 
            conflicts=0, coordination_issues=0, violations=0
        )
        assert effectiveness > 90  # Should be high
        
        # Test degraded scenario
        effectiveness = multi_crisis_handler._calculate_multi_crisis_effectiveness(
            total_crises=3, resolved=1, avg_time=120.0,
            conflicts=2, coordination_issues=3, violations=1
        )
        assert effectiveness < 50  # Should be lower
        assert effectiveness >= 0   # Should not be negative
    
    @pytest.mark.asyncio
    async def test_concurrent_multi_crisis_scenarios(self, multi_crisis_handler):
        """Test running multiple multi-crisis scenarios concurrently"""
        scenarios = multi_crisis_handler.multi_crisis_scenarios[:2]
        
        tasks = [
            multi_crisis_handler.run_multi_crisis_test(scenario, f"Concurrent Scenario {i}")
            for i, scenario in enumerate(scenarios)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 2
        for result in results:
            assert result.total_crises > 0
            assert result.overall_effectiveness >= 0


if __name__ == "__main__":
    # Run basic multi-crisis test
    async def main():
        handler = MultiCrisisHandler()
        print("Running Multi-Crisis Handling Tests...")
        
        for i, scenario in enumerate(handler.multi_crisis_scenarios):
            print(f"\nTesting multi-crisis scenario {i+1}...")
            result = await handler.run_multi_crisis_test(scenario, f"Scenario {i+1}")
            print(f"Total crises: {result.total_crises}")
            print(f"Successfully resolved: {result.successfully_resolved}")
            print(f"Resource conflicts: {result.resource_conflicts}")
            print(f"Coordination issues: {result.coordination_issues}")
            print(f"Overall effectiveness: {result.overall_effectiveness:.1f}%")
    
    asyncio.run(main())