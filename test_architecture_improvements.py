"""
Test Script for ScrollIntel Architecture Improvements

This script provides comprehensive tests for all the enhanced architecture components.
"""

import asyncio
import pytest
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Import components to test
from scrollintel.core.enhanced_specialized_agent import (
    EnhancedSpecializedAgent, AgentRequest, AgentResponse, 
    AgentCapability, RequestPriority, AgentConfiguration
)
from scrollintel.core.schema_validation import (
    SchemaRegistry, SchemaValidator, ValidationResult, 
    SchemaType, register_custom_schema
)
from scrollintel.core.agent_monitoring import (
    AgentMonitor, MetricsCollector, HealthChecker, 
    MetricType, HealthCheckStatus
)
from scrollintel.core.scroll_conductor import (
    ScrollConductor, WorkflowDefinition, WorkflowStep,
    ExecutionMode, WorkflowStatus
)
from scrollintel.core.workflow_error_handling import (
    WorkflowErrorHandler, ErrorClassifier, CompensationManager,
    ErrorCategory, RecoveryAction
)
from scrollintel.core.agent_lifecycle import (
    AgentLifecycleManager, AgentDiscovery, AgentScaler,
    ScalingPolicy, LifecycleState
)
from scrollintel.core.intelligent_load_balancer import (
    IntelligentLoadBalancer, LoadBalancerConfig, 
    RoutingStrategy, LoadBalancingDecision
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAgent(EnhancedSpecializedAgent):
    """Test agent for unit testing"""
    
    def __init__(self, agent_id: str = "test-agent", fail_requests: bool = False):
        super().__init__(
            agent_id=agent_id,
            name="Test Agent",
            capabilities=[AgentCapability.ANALYSIS],
            config=AgentConfiguration(max_concurrent_requests=5)
        )
        self.fail_requests = fail_requests
        self.processed_requests = []
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process test request"""
        self.processed_requests.append(request)
        
        if self.fail_requests:
            raise Exception("Simulated failure")
        
        await asyncio.sleep(0.1)  # Simulate processing
        
        return AgentResponse(
            request_id=request.id,
            agent_id=self.agent_id,
            status="success",
            data={"result": "test_result", "input": request.payload},
            confidence=0.9,
            processing_time=0.0
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Return test schema"""
        return {
            "type": "object",
            "properties": {
                "test_data": {"type": "string"}
            },
            "required": ["test_data"]
        }

class TestArchitectureImprovements:
    """Test suite for architecture improvements"""
    
    @pytest.mark.asyncio
    async def test_enhanced_specialized_agent(self):
        """Test enhanced specialized agent functionality"""
        logger.info("Testing Enhanced Specialized Agent...")
        
        # Create test agent
        agent = TestAgent("test-agent-001")
        await agent.initialize()
        
        # Test basic request processing
        request = AgentRequest(
            id="test-request-001",
            type="test",
            payload={"test_data": "hello world"},
            capabilities_required=["analysis"]
        )
        
        response = await agent.handle_request(request)
        
        assert response.status == "success"
        assert response.agent_id == "test-agent-001"
        assert response.confidence == 0.9
        assert "result" in response.data
        
        # Test health check
        health = await agent.health_check()
        assert health.status in ["healthy", "degraded"]
        assert len(health.checks) > 0
        
        # Test metrics
        metrics = agent.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        
        await agent.shutdown()
        logger.info("✓ Enhanced Specialized Agent tests passed")
    
    @pytest.mark.asyncio
    async def test_schema_validation(self):
        """Test schema validation framework"""
        logger.info("Testing Schema Validation Framework...")
        
        registry = SchemaRegistry()
        validator = SchemaValidator(registry)
        
        # Test schema registration
        test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        success = registry.register_schema("test_schema", "1.0.0", test_schema)
        assert success
        
        # Test valid data validation
        valid_data = {"name": "John", "age": 30}
        result = validator._validate(valid_data, "test_schema", "1.0.0", SchemaType.REQUEST)
        assert result.valid
        assert len(result.errors) == 0
        
        # Test invalid data validation
        invalid_data = {"age": 30}  # Missing required 'name'
        result = validator._validate(invalid_data, "test_schema", "1.0.0", SchemaType.REQUEST)
        assert not result.valid
        assert len(result.errors) > 0
        
        # Test schema not found
        result = validator._validate(valid_data, "nonexistent", "1.0.0", SchemaType.REQUEST)
        assert not result.valid
        
        logger.info("✓ Schema Validation Framework tests passed")
    
    @pytest.mark.asyncio
    async def test_agent_monitoring(self):
        """Test agent monitoring system"""
        logger.info("Testing Agent Monitoring System...")
        
        monitor = AgentMonitor()
        
        # Test agent registration
        await monitor.add_agent("test-agent-001", "Test Agent")
        
        # Test metrics recording
        monitor.record_request_metric("test-agent-001", True, 0.5)
        monitor.record_request_metric("test-agent-001", True, 0.3)
        monitor.record_request_metric("test-agent-001", False, 1.0)
        
        # Test status retrieval
        status = monitor.get_agent_status("test-agent-001")
        assert status.agent_id == "test-agent-001"
        assert "requests_total" in status.metrics_summary
        
        # Test system overview
        overview = monitor.get_system_overview()
        assert overview["total_agents"] >= 1
        assert "test-agent-001" in overview["agents"]
        
        await monitor.remove_agent("test-agent-001")
        logger.info("✓ Agent Monitoring System tests passed")
    
    @pytest.mark.asyncio
    async def test_scroll_conductor(self):
        """Test ScrollConductor orchestration"""
        logger.info("Testing ScrollConductor Orchestration...")
        
        conductor = ScrollConductor()
        
        # Create test workflow
        workflow = WorkflowDefinition(
            id="test-workflow",
            name="Test Workflow",
            description="Test workflow for unit testing",
            steps=[
                WorkflowStep(
                    id="step1",
                    name="First Step",
                    agent_type="test",
                    capabilities_required=["analysis"],
                    timeout=60
                ),
                WorkflowStep(
                    id="step2",
                    name="Second Step",
                    agent_type="test",
                    capabilities_required=["analysis"],
                    depends_on=["step1"],
                    timeout=60
                )
            ]
        )
        
        # Register workflow
        success = conductor.workflow_registry.register_workflow(workflow)
        assert success
        
        # Test workflow retrieval
        retrieved = conductor.workflow_registry.get_workflow("test-workflow")
        assert retrieved is not None
        assert retrieved.name == "Test Workflow"
        assert len(retrieved.steps) == 2
        
        # Test workflow listing
        workflows = conductor.workflow_registry.list_workflows()
        assert len(workflows) > 0
        
        logger.info("✓ ScrollConductor Orchestration tests passed")
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test workflow error handling"""
        logger.info("Testing Workflow Error Handling...")
        
        error_handler = WorkflowErrorHandler()
        
        # Test error classification
        from scrollintel.core.workflow_error_handling import ErrorContext
        
        error_context = ErrorContext(
            workflow_id="test-workflow",
            execution_id="test-execution",
            step_id="test-step",
            error_message="Connection timeout occurred",
            error_code="timeout"
        )
        
        classification = error_handler.classifier.classify_error(error_context)
        assert classification.category.value in ["timeout", "network", "unknown"]
        assert classification.is_retryable in [True, False]
        
        # Test recovery plan creation
        from scrollintel.core.scroll_conductor import WorkflowExecution, StepExecution
        
        execution = WorkflowExecution(
            id="test-execution",
            workflow_id="test-workflow"
        )
        
        step_execution = StepExecution(step_id="test-step")
        
        recovery_plan = await error_handler.handle_workflow_error(
            execution, step_execution, Exception("Test error")
        )
        
        assert recovery_plan.id is not None
        assert recovery_plan.primary_action in RecoveryAction
        
        # Test error statistics
        stats = error_handler.get_error_statistics()
        assert "error_counts" in stats
        assert "recovery_counts" in stats
        
        logger.info("✓ Workflow Error Handling tests passed")
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle(self):
        """Test agent lifecycle management"""
        logger.info("Testing Agent Lifecycle Management...")
        
        lifecycle_manager = AgentLifecycleManager()
        
        # Test agent discovery
        test_agent = TestAgent("lifecycle-test-agent")
        await test_agent.initialize()
        
        instance = await lifecycle_manager.register_agent(test_agent)
        assert instance.agent_id == "lifecycle-test-agent"
        assert instance.lifecycle_state == LifecycleState.RUNNING
        
        # Test agent discovery
        found_agents = lifecycle_manager.discovery.find_agents_by_capability("analysis")
        assert len(found_agents) > 0
        assert any(agent.agent_id == "lifecycle-test-agent" for agent in found_agents)
        
        # Test scaling policy
        policy = ScalingPolicy(
            agent_type="TestAgent",
            min_instances=1,
            max_instances=5,
            target_cpu_utilization=70.0
        )
        
        lifecycle_manager.scaler.set_scaling_policy(policy)
        retrieved_policy = lifecycle_manager.scaler.get_scaling_policy("TestAgent")
        assert retrieved_policy is not None
        assert retrieved_policy.min_instances == 1
        
        # Test system status
        status = lifecycle_manager.get_system_status()
        assert status["total_agents"] >= 1
        assert status["running_agents"] >= 1
        
        # Cleanup
        await lifecycle_manager.unregister_agent(instance.instance_id)
        await test_agent.shutdown()
        
        logger.info("✓ Agent Lifecycle Management tests passed")
    
    @pytest.mark.asyncio
    async def test_intelligent_load_balancer(self):
        """Test intelligent load balancer"""
        logger.info("Testing Intelligent Load Balancer...")
        
        config = LoadBalancerConfig(
            default_strategy=RoutingStrategy.INTELLIGENT,
            max_queue_size=10
        )
        
        load_balancer = IntelligentLoadBalancer(config)
        await load_balancer.start()
        
        # Create test request
        request = AgentRequest(
            id="lb-test-request",
            type="test",
            payload={"test_data": "load balancer test"},
            capabilities_required=["analysis"],
            priority=RequestPriority.NORMAL
        )
        
        # Test routing (will likely queue since no agents registered)
        decision = await load_balancer.route_request(request)
        assert decision.request_id == "lb-test-request"
        assert decision.decision in LoadBalancingDecision
        
        # Test request result recording
        await load_balancer.record_request_result(
            "test-request", "test-agent", 0.5, True
        )
        
        # Test statistics
        stats = load_balancer.get_load_balancer_stats()
        assert "total_requests" in stats
        assert "success_rate" in stats
        
        await load_balancer.stop()
        logger.info("✓ Intelligent Load Balancer tests passed")
    
    @pytest.mark.asyncio
    async def test_integration(self):
        """Test integration between components"""
        logger.info("Testing Component Integration...")
        
        # Create integrated test scenario
        agent = TestAgent("integration-test-agent")
        await agent.initialize()
        
        # Test request flow through multiple components
        request = AgentRequest(
            id="integration-test",
            type="test",
            payload={"test_data": "integration test"},
            capabilities_required=["analysis"]
        )
        
        # Validate request
        from scrollintel.core.schema_validation import validate_agent_request
        validation = validate_agent_request({
            "id": request.id,
            "type": request.type,
            "payload": request.payload
        })
        assert validation.valid
        
        # Process request
        response = await agent.handle_request(request)
        assert response.status == "success"
        
        # Check health after processing
        health = await agent.health_check()
        assert health.status in ["healthy", "degraded"]
        
        await agent.shutdown()
        logger.info("✓ Component Integration tests passed")

async def run_tests():
    """Run all tests"""
    logger.info("Starting Architecture Improvements Test Suite")
    logger.info("=" * 60)
    
    test_suite = TestArchitectureImprovements()
    
    try:
        await test_suite.test_enhanced_specialized_agent()
        await test_suite.test_schema_validation()
        await test_suite.test_agent_monitoring()
        await test_suite.test_scroll_conductor()
        await test_suite.test_error_handling()
        await test_suite.test_agent_lifecycle()
        await test_suite.test_intelligent_load_balancer()
        await test_suite.test_integration()
        
        logger.info("=" * 60)
        logger.info("All tests passed successfully! ✓")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(run_tests())