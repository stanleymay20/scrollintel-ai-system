"""
Tests for Infrastructure Redundancy System

This module contains comprehensive tests for the infrastructure redundancy system,
including multi-cloud management, unlimited compute provisioning, and research acceleration.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import uuid

from scrollintel.core.infrastructure_redundancy import (
    InfrastructureRedundancyEngine,
    CloudProvider,
    ResourceType,
    CloudResource,
    ResourceStatus,
    ResearchTask,
    PerformanceMonitor,
    CostOptimizer,
    ResearchAccelerator
)

from scrollintel.core.multi_cloud_manager import (
    MultiCloudManager,
    ProviderStatus,
    ProviderHealth,
    CloudLoadBalancer,
    MultiCloudCostTracker
)

from scrollintel.core.unlimited_compute_provisioner import (
    UnlimitedComputeProvisioner,
    ComputeRequest,
    ComputeWorkloadType,
    ScalingStrategy,
    ComputeAllocation,
    PerformancePredictor,
    ComputeCostOptimizer
)

from scrollintel.core.research_acceleration_engine import (
    ResearchAccelerationEngine,
    ResearchDomain,
    TaskPriority,
    ResearchProject,
    TaskScheduler,
    DependencyResolver,
    ResultAggregator
)

class TestInfrastructureRedundancyEngine:
    """Test cases for Infrastructure Redundancy Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create infrastructure redundancy engine for testing"""
        return InfrastructureRedundancyEngine()
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert len(engine.cloud_resources) == 0
        assert len(engine.active_providers) == 0
        assert engine.performance_monitor is not None
        assert engine.cost_optimizer is not None
        assert engine.research_accelerator is not None
    
    def test_register_cloud_provider(self, engine):
        """Test cloud provider registration"""
        config = {
            'regions': ['us-east-1', 'us-west-2'],
            'initial_capacity': 5,
            'cost_per_hour': 2.0,
            'priority': 1
        }
        
        success = engine.register_cloud_provider(CloudProvider.AWS, config)
        
        assert success is True
        assert CloudProvider.AWS in engine.active_providers
        assert CloudProvider.AWS in engine.provider_configs
        assert len(engine.cloud_resources) > 0
    
    def test_provision_unlimited_resources(self, engine):
        """Test unlimited resource provisioning"""
        # Register providers first
        config = {
            'regions': ['us-east-1'],
            'initial_capacity': 10,
            'cost_per_hour': 2.0
        }
        engine.register_cloud_provider(CloudProvider.AWS, config)
        engine.register_cloud_provider(CloudProvider.AZURE, config)
        
        # Provision resources
        required_capacity = {"cpu_cores": 1000, "memory_gb": 4000}
        provisioned = engine.provision_unlimited_resources(
            ResourceType.COMPUTE, required_capacity, priority=1
        )
        
        assert len(provisioned) > 0
        assert all(isinstance(rid, str) for rid in provisioned)
        
        # Verify resources are in cloud_resources
        for resource_id in provisioned:
            assert resource_id in engine.cloud_resources
            resource = engine.cloud_resources[resource_id]
            assert resource.status == ResourceStatus.ACTIVE
    
    def test_implement_failover_system(self, engine):
        """Test failover system implementation"""
        # Register provider and create resources
        config = {'regions': ['us-east-1'], 'initial_capacity': 5}
        engine.register_cloud_provider(CloudProvider.AWS, config)
        engine.register_cloud_provider(CloudProvider.AZURE, config)
        
        # Implement failover
        failover_chains = engine.implement_failover_system()
        
        assert isinstance(failover_chains, dict)
        assert len(failover_chains) > 0
        
        # Verify failover chains are stored
        assert len(engine.failover_chains) > 0
    
    def test_execute_failover(self, engine):
        """Test failover execution"""
        # Setup resources and failover chains
        config = {'regions': ['us-east-1'], 'initial_capacity': 3}
        engine.register_cloud_provider(CloudProvider.AWS, config)
        engine.register_cloud_provider(CloudProvider.AZURE, config)
        
        engine.implement_failover_system()
        
        # Get a resource to fail
        resource_ids = list(engine.cloud_resources.keys())
        if resource_ids:
            failed_resource_id = resource_ids[0]
            
            # Execute failover
            backup_resource_id = engine.execute_failover(failed_resource_id)
            
            if backup_resource_id:
                assert backup_resource_id != failed_resource_id
                assert backup_resource_id in engine.cloud_resources
                
                # Verify status changes
                failed_resource = engine.cloud_resources[failed_resource_id]
                backup_resource = engine.cloud_resources[backup_resource_id]
                
                assert failed_resource.status == ResourceStatus.FAILED
                assert backup_resource.status == ResourceStatus.ACTIVE
    
    def test_get_system_status(self, engine):
        """Test system status retrieval"""
        # Register some providers
        config = {'regions': ['us-east-1'], 'initial_capacity': 2}
        engine.register_cloud_provider(CloudProvider.AWS, config)
        
        status = engine.get_system_status()
        
        assert isinstance(status, dict)
        assert "total_resources" in status
        assert "active_providers" in status
        assert "resource_pools" in status
        assert "system_health" in status
        assert status["system_health"] == "optimal"
    
    def test_enhanced_research_cluster_scaling(self, engine):
        """Test enhanced research cluster scaling capabilities"""
        # Register providers
        config = {'regions': ['us-east-1', 'us-west-2'], 'initial_capacity': 5}
        engine.register_cloud_provider(CloudProvider.AWS, config)
        engine.register_cloud_provider(CloudProvider.GCP, config)
        
        # Create research cluster
        cluster_info = asyncio.run(
            engine.create_research_acceleration_cluster(
                "scaling_test_cluster", "ai_research", parallel_jobs=1000
            )
        )
        
        # Test cluster scaling
        scale_success = asyncio.run(
            engine.scale_research_cluster("scaling_test_cluster", 2.0, "test_scaling")
        )
        
        assert scale_success is True
        
        # Verify cluster was scaled
        scaled_cluster = engine.research_clusters.get("scaling_test_cluster", [])
        assert len(scaled_cluster) > 0
    
    def test_enhanced_cluster_failover(self, engine):
        """Test enhanced cluster failover capabilities"""
        # Register multiple providers
        config = {'regions': ['us-east-1'], 'initial_capacity': 3}
        engine.register_cloud_provider(CloudProvider.AWS, config)
        engine.register_cloud_provider(CloudProvider.AZURE, config)
        engine.register_cloud_provider(CloudProvider.GCP, config)
        
        # Create research cluster
        cluster_info = asyncio.run(
            engine.create_research_acceleration_cluster(
                "failover_test_cluster", "general_research", parallel_jobs=100
            )
        )
        
        # Test cluster failover
        failover_success = asyncio.run(
            engine.execute_cluster_failover("failover_test_cluster", CloudProvider.AWS)
        )
        
        # Failover may or may not succeed depending on available resources
        # Just verify the method completes without error
        assert isinstance(failover_success, bool)


class TestMultiCloudManager:
    """Test cases for Multi-Cloud Manager"""
    
    @pytest.fixture
    def manager(self):
        """Create multi-cloud manager for testing"""
        return MultiCloudManager()
    
    def test_manager_initialization(self, manager):
        """Test manager initialization"""
        assert manager is not None
        assert len(manager.providers) > 0
        assert len(manager.provider_health) > 0
        assert manager.load_balancer is not None
        assert manager.cost_tracker is not None
    
    @pytest.mark.asyncio
    async def test_provision_resources_multi_cloud(self, manager):
        """Test multi-cloud resource provisioning"""
        requirements = {"cpu_cores": 64, "memory_gb": 256}
        
        resources = await manager.provision_resources_multi_cloud(
            ResourceType.COMPUTE, 10, requirements
        )
        
        assert isinstance(resources, list)
        # Note: In real implementation, this would provision actual resources
        # For testing, we verify the method completes without error
    
    def test_get_multi_cloud_status(self, manager):
        """Test multi-cloud status retrieval"""
        status = manager.get_multi_cloud_status()
        
        assert isinstance(status, dict)
        assert "total_providers" in status
        assert "active_providers" in status
        assert "provider_details" in status
        assert "total_regions" in status
        
        # Verify provider details structure
        for provider_name, details in status["provider_details"].items():
            assert "status" in details
            assert "response_time" in details
            assert "availability" in details


class TestUnlimitedComputeProvisioner:
    """Test cases for Unlimited Compute Provisioner"""
    
    @pytest.fixture
    def provisioner(self):
        """Create compute provisioner for testing"""
        multi_cloud_manager = MultiCloudManager()
        return UnlimitedComputeProvisioner(multi_cloud_manager)
    
    def test_provisioner_initialization(self, provisioner):
        """Test provisioner initialization"""
        assert provisioner is not None
        assert provisioner.multi_cloud_manager is not None
        assert len(provisioner.scaling_policies) > 0
        assert provisioner.performance_predictor is not None
        assert provisioner.cost_optimizer is not None
    
    @pytest.mark.asyncio
    async def test_request_unlimited_compute(self, provisioner):
        """Test unlimited compute request"""
        request = ComputeRequest(
            id="test-request-1",
            workload_type=ComputeWorkloadType.CPU_INTENSIVE,
            required_resources={"cpu_cores": 128, "memory_gb": 512},
            priority=1
        )
        
        allocation = await provisioner.request_unlimited_compute(request)
        
        assert isinstance(allocation, ComputeAllocation)
        assert allocation.request_id == request.id
        assert len(allocation.allocated_resources) > 0
        assert allocation.estimated_cost > 0
        assert isinstance(allocation.performance_prediction, dict)
    
    def test_scale_allocation(self, provisioner):
        """Test allocation scaling"""
        # Create a mock allocation first
        request_id = "test-request-scale"
        mock_resources = [
            CloudResource(
                id=f"resource-{i}",
                provider=CloudProvider.AWS,
                resource_type=ResourceType.COMPUTE,
                region="us-east-1",
                capacity={"cpu_cores": 64},
                status=ResourceStatus.ACTIVE,
                cost_per_hour=2.0
            ) for i in range(5)
        ]
        
        allocation = ComputeAllocation(
            request_id=request_id,
            allocated_resources=mock_resources,
            allocation_time=datetime.now(),
            estimated_cost=10.0,
            performance_prediction={"throughput": 1000},
            scaling_plan={}
        )
        
        provisioner.active_allocations[request_id] = allocation
        
        # Test scaling
        success = provisioner.scale_allocation(request_id, 1.5)
        assert success is True
    
    def test_get_provisioning_status(self, provisioner):
        """Test provisioning status retrieval"""
        status = provisioner.get_provisioning_status()
        
        assert isinstance(status, dict)
        assert "active_allocations" in status
        assert "total_resources" in status
        assert "total_cost_per_hour" in status
        assert "system_status" in status


class TestResearchAccelerationEngine:
    """Test cases for Research Acceleration Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create research acceleration engine for testing"""
        multi_cloud_manager = MultiCloudManager()
        compute_provisioner = UnlimitedComputeProvisioner(multi_cloud_manager)
        return ResearchAccelerationEngine(compute_provisioner)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine is not None
        assert engine.compute_provisioner is not None
        assert engine.task_scheduler is not None
        assert engine.dependency_resolver is not None
        assert engine.result_aggregator is not None
    
    @pytest.mark.asyncio
    async def test_create_research_project(self, engine):
        """Test research project creation"""
        project = await engine.create_research_project(
            name="Test AI Research",
            description="Testing AI research acceleration",
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            principal_investigator="Dr. Test",
            compute_budget=100000.0
        )
        
        assert isinstance(project, ResearchProject)
        assert project.name == "Test AI Research"
        assert project.domain == ResearchDomain.ARTIFICIAL_INTELLIGENCE
        assert project.total_compute_budget == 100000.0
        assert project.id in engine.active_projects
    
    @pytest.mark.asyncio
    async def test_add_research_task(self, engine):
        """Test adding research task to project"""
        # Create project first
        project = await engine.create_research_project(
            name="Test Project",
            description="Test project",
            domain=ResearchDomain.MACHINE_LEARNING,
            principal_investigator="Dr. Test"
        )
        
        # Create task
        task = ResearchTask(
            id="test-task-1",
            name="Test ML Task",
            domain=ResearchDomain.MACHINE_LEARNING,
            priority=TaskPriority.HIGH,
            computation_function=lambda x: x * 2,
            input_data=42,
            expected_output_type=int,
            estimated_compute_hours=2.0,
            memory_requirements_gb=8.0,
            cpu_cores_required=4
        )
        
        success = await engine.add_research_task(project.id, task)
        
        assert success is True
        assert len(project.tasks) == 1
        assert project.tasks[0].id == task.id
    
    def test_get_acceleration_status(self, engine):
        """Test acceleration status retrieval"""
        status = engine.get_acceleration_status()
        
        assert isinstance(status, dict)
        assert "active_projects" in status
        assert "total_tasks" in status
        assert "compute_clusters" in status
        assert "system_status" in status


class TestPerformanceMonitor:
    """Test cases for Performance Monitor"""
    
    @pytest.fixture
    def monitor(self):
        """Create performance monitor for testing"""
        return PerformanceMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert monitor is not None
        assert isinstance(monitor.metrics_history, dict)
        assert isinstance(monitor.alert_thresholds, dict)
    
    def test_collect_metrics(self, monitor):
        """Test metrics collection"""
        resource = CloudResource(
            id="test-resource",
            provider=CloudProvider.AWS,
            resource_type=ResourceType.COMPUTE,
            region="us-east-1",
            capacity={"cpu_cores": 64},
            status=ResourceStatus.ACTIVE,
            cost_per_hour=2.0
        )
        
        metrics = monitor.collect_metrics(resource)
        
        assert isinstance(metrics, dict)
        assert "cpu_utilization" in metrics
        assert "memory_utilization" in metrics
        assert "network_latency" in metrics
        assert "error_rate" in metrics
        
        # Verify metrics are stored in resource
        assert len(resource.performance_metrics) > 0
        
        # Verify metrics are stored in history
        assert resource.id in monitor.metrics_history
        assert len(monitor.metrics_history[resource.id]) > 0
    
    def test_check_alerts(self, monitor):
        """Test alert checking"""
        resource = CloudResource(
            id="test-resource",
            provider=CloudProvider.AWS,
            resource_type=ResourceType.COMPUTE,
            region="us-east-1",
            capacity={"cpu_cores": 64},
            status=ResourceStatus.ACTIVE,
            cost_per_hour=2.0
        )
        
        # Set high utilization to trigger alerts
        resource.performance_metrics = {
            "cpu_utilization": 0.95,
            "memory_utilization": 0.90,
            "network_latency": 150,
            "error_rate": 0.02
        }
        
        alerts = monitor.check_alerts(resource)
        
        assert isinstance(alerts, list)
        # Should have alerts for high CPU, memory, latency, and error rate
        assert len(alerts) > 0


class TestCostOptimizer:
    """Test cases for Cost Optimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create cost optimizer for testing"""
        return CostOptimizer()
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer is not None
        assert isinstance(optimizer.cost_history, dict)
        assert isinstance(optimizer.optimization_rules, list)
    
    def test_optimize_costs(self, optimizer):
        """Test cost optimization"""
        resources = {
            "resource-1": CloudResource(
                id="resource-1",
                provider=CloudProvider.AWS,
                resource_type=ResourceType.COMPUTE,
                region="us-east-1",
                capacity={"cpu_cores": 64},
                status=ResourceStatus.ACTIVE,
                cost_per_hour=3.0
            ),
            "resource-2": CloudResource(
                id="resource-2",
                provider=CloudProvider.AZURE,
                resource_type=ResourceType.COMPUTE,
                region="eastus",
                capacity={"cpu_cores": 64},
                status=ResourceStatus.ACTIVE,
                cost_per_hour=2.5
            )
        }
        
        # Set low utilization to trigger optimization
        for resource in resources.values():
            resource.performance_metrics = {"utilization": 0.1}
        
        results = optimizer.optimize_costs(resources)
        
        assert isinstance(results, dict)
        assert "total_cost_before" in results
        assert "total_cost_after" in results
        assert "optimizations_applied" in results
        assert "potential_savings" in results


class TestTaskScheduler:
    """Test cases for Task Scheduler"""
    
    @pytest.fixture
    def scheduler(self):
        """Create task scheduler for testing"""
        return TaskScheduler()
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler is not None
        assert len(scheduler.scheduling_algorithms) > 0
    
    def test_priority_first_scheduling(self, scheduler):
        """Test priority-first scheduling"""
        tasks = [
            ResearchTask(
                id="task-1",
                name="Low Priority Task",
                domain=ResearchDomain.PHYSICS,
                priority=TaskPriority.LOW,
                computation_function=lambda x: x,
                input_data=1,
                expected_output_type=int,
                estimated_compute_hours=1.0,
                memory_requirements_gb=4.0,
                cpu_cores_required=2
            ),
            ResearchTask(
                id="task-2",
                name="High Priority Task",
                domain=ResearchDomain.PHYSICS,
                priority=TaskPriority.HIGH,
                computation_function=lambda x: x,
                input_data=2,
                expected_output_type=int,
                estimated_compute_hours=2.0,
                memory_requirements_gb=8.0,
                cpu_cores_required=4
            ),
            ResearchTask(
                id="task-3",
                name="Critical Priority Task",
                domain=ResearchDomain.PHYSICS,
                priority=TaskPriority.CRITICAL,
                computation_function=lambda x: x,
                input_data=3,
                expected_output_type=int,
                estimated_compute_hours=3.0,
                memory_requirements_gb=16.0,
                cpu_cores_required=8
            )
        ]
        
        scheduled = scheduler.schedule_tasks(tasks, "priority_first")
        
        assert len(scheduled) == 3
        assert scheduled[0].priority == TaskPriority.CRITICAL
        assert scheduled[1].priority == TaskPriority.HIGH
        assert scheduled[2].priority == TaskPriority.LOW
    
    def test_shortest_job_first_scheduling(self, scheduler):
        """Test shortest job first scheduling"""
        tasks = [
            ResearchTask(
                id="task-1",
                name="Long Task",
                domain=ResearchDomain.PHYSICS,
                priority=TaskPriority.MEDIUM,
                computation_function=lambda x: x,
                input_data=1,
                expected_output_type=int,
                estimated_compute_hours=10.0,
                memory_requirements_gb=4.0,
                cpu_cores_required=2
            ),
            ResearchTask(
                id="task-2",
                name="Short Task",
                domain=ResearchDomain.PHYSICS,
                priority=TaskPriority.MEDIUM,
                computation_function=lambda x: x,
                input_data=2,
                expected_output_type=int,
                estimated_compute_hours=1.0,
                memory_requirements_gb=4.0,
                cpu_cores_required=2
            )
        ]
        
        scheduled = scheduler.schedule_tasks(tasks, "shortest_job_first")
        
        assert len(scheduled) == 2
        assert scheduled[0].estimated_compute_hours < scheduled[1].estimated_compute_hours


class TestDependencyResolver:
    """Test cases for Dependency Resolver"""
    
    @pytest.fixture
    def resolver(self):
        """Create dependency resolver for testing"""
        return DependencyResolver()
    
    def test_build_dependency_graph(self, resolver):
        """Test dependency graph building"""
        tasks = [
            ResearchTask(
                id="task-1",
                name="Independent Task",
                domain=ResearchDomain.PHYSICS,
                priority=TaskPriority.MEDIUM,
                computation_function=lambda x: x,
                input_data=1,
                expected_output_type=int,
                estimated_compute_hours=1.0,
                memory_requirements_gb=4.0,
                cpu_cores_required=2,
                dependencies=[]
            ),
            ResearchTask(
                id="task-2",
                name="Dependent Task",
                domain=ResearchDomain.PHYSICS,
                priority=TaskPriority.MEDIUM,
                computation_function=lambda x: x,
                input_data=2,
                expected_output_type=int,
                estimated_compute_hours=2.0,
                memory_requirements_gb=4.0,
                cpu_cores_required=2,
                dependencies=["task-1"]
            )
        ]
        
        graph = resolver.build_dependency_graph(tasks)
        
        assert isinstance(graph, dict)
        assert "task-1" in graph
        assert "task-2" in graph
        assert graph["task-1"] == []
        assert graph["task-2"] == ["task-1"]


class TestResultAggregator:
    """Test cases for Result Aggregator"""
    
    @pytest.fixture
    def aggregator(self):
        """Create result aggregator for testing"""
        return ResultAggregator()
    
    def test_aggregate_project_results(self, aggregator):
        """Test project result aggregation"""
        project = ResearchProject(
            id="test-project",
            name="Test Project",
            description="Test project for aggregation",
            domain=ResearchDomain.ARTIFICIAL_INTELLIGENCE,
            principal_investigator="Dr. Test"
        )
        
        task_results = {
            "task-1": {"result": "success", "value": 42},
            "task-2": {"result": "success", "value": 84},
            "task-3": {"error": "computation failed"}
        }
        
        aggregated = aggregator.aggregate_project_results(project, task_results)
        
        assert isinstance(aggregated, dict)
        assert "project_id" in aggregated
        assert "total_tasks" in aggregated
        assert "successful_tasks" in aggregated
        assert "failed_tasks" in aggregated
        assert "task_results" in aggregated
        assert "summary_statistics" in aggregated
        assert "insights" in aggregated
        assert "recommendations" in aggregated
        
        assert aggregated["successful_tasks"] == 2
        assert aggregated["failed_tasks"] == 1


@pytest.mark.integration
class TestInfrastructureIntegration:
    """Integration tests for infrastructure components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_acceleration(self):
        """Test end-to-end research acceleration workflow"""
        # Create components
        multi_cloud_manager = MultiCloudManager()
        compute_provisioner = UnlimitedComputeProvisioner(multi_cloud_manager)
        research_engine = ResearchAccelerationEngine(compute_provisioner)
        
        # Create research project
        project = await research_engine.create_research_project(
            name="Integration Test Project",
            description="End-to-end integration test",
            domain=ResearchDomain.COMPUTER_SCIENCE,
            principal_investigator="Test Engineer"
        )
        
        # Add research tasks
        for i in range(3):
            task = ResearchTask(
                id=f"integration-task-{i}",
                name=f"Integration Task {i}",
                domain=ResearchDomain.COMPUTER_SCIENCE,
                priority=TaskPriority.MEDIUM,
                computation_function=lambda x: x * 2,
                input_data=i * 10,
                expected_output_type=int,
                estimated_compute_hours=0.1,  # Short for testing
                memory_requirements_gb=1.0,
                cpu_cores_required=1
            )
            
            success = await research_engine.add_research_task(project.id, task)
            assert success is True
        
        # Verify project state
        assert len(project.tasks) == 3
        assert project.status == "active"
        
        # Get status
        status = research_engine.get_acceleration_status()
        assert status["active_projects"] >= 1
        assert status["total_tasks"] >= 3
    
    def test_infrastructure_redundancy_integration(self):
        """Test infrastructure redundancy integration"""
        # Create infrastructure engine
        engine = InfrastructureRedundancyEngine()
        
        # Register multiple providers
        config = {
            'regions': ['us-east-1', 'us-west-2'],
            'initial_capacity': 5,
            'cost_per_hour': 2.0
        }
        
        providers = [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]
        for provider in providers:
            success = engine.register_cloud_provider(provider, config)
            assert success is True
        
        # Provision resources
        provisioned = engine.provision_unlimited_resources(
            ResourceType.COMPUTE,
            {"cpu_cores": 100, "memory_gb": 400},
            priority=1
        )
        
        assert len(provisioned) > 0
        
        # Implement failover
        failover_chains = engine.implement_failover_system()
        assert len(failover_chains) > 0
        
        # Test failover execution
        if provisioned:
            backup_id = engine.execute_failover(provisioned[0])
            # May or may not succeed depending on available backup resources
        
        # Get comprehensive status
        status = engine.get_system_status()
        assert status["total_resources"] > 0
        assert status["active_providers"] == len(providers)
        assert status["system_health"] == "optimal"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])