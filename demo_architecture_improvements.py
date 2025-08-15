"""
Demo Script for ScrollIntel Architecture Improvements

This script demonstrates the enhanced architecture components including:
- Enhanced Specialized Agents
- Schema Validation Framework
- Agent Monitoring System
- ScrollConductor Orchestration
- Workflow Error Handling
- Agent Lifecycle Management
- Intelligent Load Balancer
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Import our enhanced components
from scrollintel.core.enhanced_specialized_agent import (
    EnhancedSpecializedAgent, AgentRequest, AgentResponse, 
    AgentCapability, RequestPriority, enhanced_agent_registry
)
from scrollintel.core.schema_validation import (
    schema_registry, schema_validator, validate_agent_request, 
    register_custom_schema, SchemaType
)
from scrollintel.core.agent_monitoring import (
    agent_monitor, start_agent_monitoring, MetricType,
    register_agent_for_monitoring, record_agent_request
)
from scrollintel.core.scroll_conductor import (
    scroll_conductor, WorkflowDefinition, WorkflowStep, 
    ExecutionMode, execute_workflow
)
from scrollintel.core.workflow_error_handling import (
    workflow_error_handler, handle_workflow_error, 
    RecoveryAction, ErrorSeverity
)
from scrollintel.core.agent_lifecycle import (
    agent_lifecycle_manager, start_lifecycle_management,
    ScalingPolicy, set_scaling_policy
)
from scrollintel.core.intelligent_load_balancer import (
    intelligent_load_balancer, start_load_balancer,
    route_request, LoadBalancerConfig, RoutingStrategy
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DemoAnalysisAgent(EnhancedSpecializedAgent):
    """Demo analysis agent for testing"""
    
    def __init__(self):
        super().__init__(
            agent_id="demo-analysis-001",
            name="Demo Analysis Agent",
            capabilities=[AgentCapability.ANALYSIS, AgentCapability.PREDICTION]
        )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process analysis request"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        data = request.payload.get("data", [])
        analysis_type = request.payload.get("type", "general")
        
        result = {
            "analysis_type": analysis_type,
            "data_points_analyzed": len(data),
            "insights": [
                f"Pattern detected in {analysis_type} analysis",
                f"Processed {len(data)} data points successfully",
                "Confidence level: High"
            ],
            "confidence_score": 0.87,
            "processing_metadata": {
                "agent_id": self.agent_id,
                "processing_time": 0.5,
                "algorithm_version": "v2.1"
            }
        }
        
        return AgentResponse(
            request_id=request.id,
            agent_id=self.agent_id,
            status="success",
            data=result,
            confidence=0.87,
            processing_time=0.0  # Will be set by framework
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Return schema for analysis requests"""
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["statistical", "predictive", "descriptive"]},
                "data": {"type": "array", "items": {"type": "number"}},
                "parameters": {"type": "object"}
            },
            "required": ["type", "data"]
        }

class DemoGenerationAgent(EnhancedSpecializedAgent):
    """Demo generation agent for testing"""
    
    def __init__(self):
        super().__init__(
            agent_id="demo-generation-001",
            name="Demo Generation Agent",
            capabilities=[AgentCapability.GENERATION, AgentCapability.COMMUNICATION]
        )
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process generation request"""
        await asyncio.sleep(0.3)  # Simulate processing time
        
        content_type = request.payload.get("type", "text")
        prompt = request.payload.get("prompt", "")
        
        result = {
            "content_type": content_type,
            "generated_content": f"Generated {content_type} content based on: '{prompt}'",
            "word_count": 150,
            "quality_metrics": {
                "coherence": 0.92,
                "relevance": 0.89,
                "creativity": 0.85
            },
            "processing_metadata": {
                "agent_id": self.agent_id,
                "model_version": "demo-gen-v1.0",
                "processing_time": 0.3
            }
        }
        
        return AgentResponse(
            request_id=request.id,
            agent_id=self.agent_id,
            status="success",
            data=result,
            confidence=0.89,
            processing_time=0.0  # Will be set by framework
        )
    
    def get_schema(self) -> Dict[str, Any]:
        """Return schema for generation requests"""
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["text", "code", "summary"]},
                "prompt": {"type": "string", "minLength": 1},
                "parameters": {"type": "object"}
            },
            "required": ["type", "prompt"]
        }

async def demo_enhanced_agents():
    """Demonstrate enhanced specialized agents"""
    logger.info("=== Demo: Enhanced Specialized Agents ===")
    
    # Create demo agents
    analysis_agent = DemoAnalysisAgent()
    generation_agent = DemoGenerationAgent()
    
    # Initialize agents
    await analysis_agent.initialize()
    await generation_agent.initialize()
    
    # Register agents
    await enhanced_agent_registry.register_agent(analysis_agent)
    await enhanced_agent_registry.register_agent(generation_agent)
    
    # Create test requests
    analysis_request = AgentRequest(
        id="test-analysis-001",
        type="analysis",
        payload={
            "type": "statistical",
            "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "parameters": {"confidence_level": 0.95}
        },
        capabilities_required=["analysis"],
        priority=RequestPriority.NORMAL
    )
    
    generation_request = AgentRequest(
        id="test-generation-001",
        type="generation",
        payload={
            "type": "text",
            "prompt": "Write a summary of AI advancements",
            "parameters": {"max_length": 200}
        },
        capabilities_required=["generation"],
        priority=RequestPriority.HIGH
    )
    
    # Process requests
    logger.info("Processing analysis request...")
    analysis_response = await analysis_agent.handle_request(analysis_request)
    logger.info(f"Analysis result: {analysis_response.status} (confidence: {analysis_response.confidence})")
    
    logger.info("Processing generation request...")
    generation_response = await generation_agent.handle_request(generation_request)
    logger.info(f"Generation result: {generation_response.status} (confidence: {generation_response.confidence})")
    
    # Check health status
    analysis_health = await analysis_agent.health_check()
    logger.info(f"Analysis agent health: {analysis_health.status}")
    
    generation_health = await generation_agent.health_check()
    logger.info(f"Generation agent health: {generation_health.status}")
    
    return analysis_agent, generation_agent

async def demo_schema_validation():
    """Demonstrate schema validation framework"""
    logger.info("=== Demo: Schema Validation Framework ===")
    
    # Register custom schema
    custom_schema = {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "action": {"type": "string", "enum": ["create", "update", "delete"]},
            "data": {"type": "object"}
        },
        "required": ["user_id", "action"]
    }
    
    success = register_custom_schema("user_action", custom_schema, SchemaType.REQUEST)
    logger.info(f"Custom schema registered: {success}")
    
    # Test valid request
    valid_request_data = {
        "id": "test-001",
        "type": "analysis",
        "payload": {
            "type": "statistical",
            "data": [1, 2, 3, 4, 5]
        }
    }
    
    validation_result = validate_agent_request(valid_request_data)
    logger.info(f"Valid request validation: {validation_result.valid}")
    
    # Test invalid request
    invalid_request_data = {
        "id": "test-002",
        "type": "analysis"
        # Missing required 'payload' field
    }
    
    validation_result = validate_agent_request(invalid_request_data)
    logger.info(f"Invalid request validation: {validation_result.valid}")
    if not validation_result.valid:
        logger.info(f"Validation errors: {[error.message for error in validation_result.errors]}")

async def demo_monitoring_system():
    """Demonstrate agent monitoring system"""
    logger.info("=== Demo: Agent Monitoring System ===")
    
    # Start monitoring system
    await start_agent_monitoring()
    
    # Register agents for monitoring
    register_agent_for_monitoring("demo-analysis-001", "Demo Analysis Agent")
    register_agent_for_monitoring("demo-generation-001", "Demo Generation Agent")
    
    # Simulate some requests and record metrics
    for i in range(5):
        # Simulate successful requests
        record_agent_request("demo-analysis-001", True, 0.5 + i * 0.1)
        record_agent_request("demo-generation-001", True, 0.3 + i * 0.05)
    
    # Simulate some failed requests
    record_agent_request("demo-analysis-001", False, 2.0)
    record_agent_request("demo-generation-001", False, 1.5)
    
    # Get monitoring overview
    overview = agent_monitor.get_system_overview()
    logger.info(f"Monitoring overview: {json.dumps(overview, indent=2)}")
    
    # Get specific agent health
    analysis_health = agent_monitor.get_agent_status("demo-analysis-001")
    logger.info(f"Analysis agent status: {analysis_health.overall_status.value}")

async def demo_workflow_orchestration():
    """Demonstrate ScrollConductor workflow orchestration"""
    logger.info("=== Demo: Workflow Orchestration ===")
    
    # Create a custom workflow
    demo_workflow = WorkflowDefinition(
        id="demo_data_processing",
        name="Demo Data Processing",
        description="Process data through analysis and generation steps",
        steps=[
            WorkflowStep(
                id="validate_data",
                name="Validate Input Data",
                agent_type="validation",
                capabilities_required=["validation"],
                timeout=60
            ),
            WorkflowStep(
                id="analyze_data",
                name="Analyze Data",
                agent_type="analysis",
                capabilities_required=["analysis"],
                depends_on=["validate_data"],
                timeout=120
            ),
            WorkflowStep(
                id="generate_report",
                name="Generate Report",
                agent_type="generation",
                capabilities_required=["generation"],
                depends_on=["analyze_data"],
                timeout=90
            )
        ],
        execution_mode=ExecutionMode.SEQUENTIAL
    )
    
    # Register workflow
    success = scroll_conductor.workflow_registry.register_workflow(demo_workflow)
    logger.info(f"Workflow registered: {success}")
    
    # Execute workflow
    input_data = {
        "dataset": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "analysis_type": "statistical",
        "report_format": "summary"
    }
    
    logger.info("Executing workflow...")
    execution = await execute_workflow("demo_data_processing", input_data)
    
    logger.info(f"Workflow execution status: {execution.status.value}")
    logger.info(f"Execution duration: {execution.duration:.2f} seconds")
    
    if execution.status.value == "completed":
        logger.info("Workflow completed successfully!")
        for step_id, step_exec in execution.steps.items():
            logger.info(f"  Step {step_id}: {step_exec.status.value}")
    else:
        logger.error(f"Workflow failed: {execution.error}")

async def demo_error_handling():
    """Demonstrate workflow error handling"""
    logger.info("=== Demo: Workflow Error Handling ===")
    
    # Create a workflow execution that will fail
    from scrollintel.core.scroll_conductor import WorkflowExecution, StepExecution, WorkflowStatus, StepStatus
    
    failed_execution = WorkflowExecution(
        id="failed-execution-001",
        workflow_id="demo_data_processing",
        status=WorkflowStatus.FAILED,
        error="Simulated failure for demo",
        error_code="demo_error"
    )
    
    failed_step = StepExecution(
        step_id="analyze_data",
        status=StepStatus.FAILED,
        error="Analysis agent timeout",
        error_code="timeout"
    )
    
    # Handle the error
    recovery_plan = await handle_workflow_error(failed_execution, failed_step)
    
    logger.info(f"Recovery plan created: {recovery_plan.id}")
    logger.info(f"Primary recovery action: {recovery_plan.primary_action.value}")
    logger.info(f"Error classification: {recovery_plan.classification.category.value}")
    logger.info(f"Error severity: {recovery_plan.classification.severity.value}")
    
    # Get error statistics
    error_stats = workflow_error_handler.get_error_statistics()
    logger.info(f"Error handling stats: {json.dumps(error_stats, indent=2)}")

async def demo_lifecycle_management():
    """Demonstrate agent lifecycle management"""
    logger.info("=== Demo: Agent Lifecycle Management ===")
    
    # Start lifecycle management
    await start_lifecycle_management()
    
    # Set scaling policies
    analysis_policy = set_scaling_policy(
        "DemoAnalysisAgent",
        min_instances=1,
        max_instances=5,
        target_cpu_utilization=70.0,
        scale_up_threshold=80.0,
        scale_down_threshold=30.0
    )
    
    generation_policy = set_scaling_policy(
        "DemoGenerationAgent",
        min_instances=1,
        max_instances=3,
        target_cpu_utilization=75.0,
        scale_up_threshold=85.0,
        scale_down_threshold=25.0
    )
    
    logger.info(f"Scaling policies set for analysis and generation agents")
    
    # Get lifecycle status
    lifecycle_status = agent_lifecycle_manager.get_system_status()
    logger.info(f"Lifecycle status: {json.dumps(lifecycle_status, indent=2)}")

async def demo_load_balancer():
    """Demonstrate intelligent load balancer"""
    logger.info("=== Demo: Intelligent Load Balancer ===")
    
    # Configure and start load balancer
    config = LoadBalancerConfig(
        default_strategy=RoutingStrategy.INTELLIGENT,
        enable_predictive_routing=True,
        enable_adaptive_weights=True,
        max_queue_size=100
    )
    
    await start_load_balancer(config)
    
    # Create test requests
    test_requests = [
        AgentRequest(
            id=f"lb-test-{i}",
            type="analysis",
            payload={"type": "statistical", "data": list(range(10))},
            capabilities_required=["analysis"],
            priority=RequestPriority.NORMAL if i % 2 == 0 else RequestPriority.HIGH
        )
        for i in range(5)
    ]
    
    # Route requests
    for request in test_requests:
        decision = await route_request(request)
        logger.info(f"Request {request.id} routing decision: {decision.decision.value}")
        
        if decision.selected_agent:
            logger.info(f"  Selected agent: {decision.selected_agent.agent_id}")
            logger.info(f"  Agent score: {decision.selected_agent.total_score:.3f}")
            logger.info(f"  Selection reason: {decision.selected_agent.selection_reason}")
    
    # Get load balancer statistics
    lb_stats = intelligent_load_balancer.get_load_balancer_stats()
    logger.info(f"Load balancer stats: {json.dumps(lb_stats, indent=2, default=str)}")

async def demo_integration():
    """Demonstrate integration of all components"""
    logger.info("=== Demo: Full System Integration ===")
    
    # This would show how all components work together
    # in a real-world scenario
    
    logger.info("Creating integrated workflow with monitoring and load balancing...")
    
    # Create a request that goes through the full pipeline
    integrated_request = AgentRequest(
        id="integrated-test-001",
        type="complex_analysis",
        payload={
            "type": "predictive",
            "data": list(range(100)),
            "parameters": {
                "confidence_level": 0.95,
                "prediction_horizon": 30
            }
        },
        capabilities_required=["analysis", "prediction"],
        priority=RequestPriority.HIGH
    )
    
    # Route through load balancer
    routing_decision = await route_request(integrated_request)
    logger.info(f"Integrated request routed: {routing_decision.decision.value}")
    
    if routing_decision.selected_agent:
        # Simulate processing and record results
        await intelligent_load_balancer.record_request_result(
            integrated_request.id,
            routing_decision.selected_agent.agent_id,
            response_time=0.75,
            success=True
        )
        
        logger.info("Integrated request processed successfully!")
    
    # Show final system status
    monitoring_overview = agent_monitor.get_system_overview()
    lifecycle_status = agent_lifecycle_manager.get_system_status()
    lb_stats = intelligent_load_balancer.get_load_balancer_stats()
    
    logger.info("=== Final System Status ===")
    logger.info(f"Monitoring: {monitoring_overview['total_agents']} agents, {monitoring_overview['healthy_agents']} healthy")
    logger.info(f"Lifecycle: {lifecycle_status['running_agents']} running agents")
    logger.info(f"Load Balancer: {lb_stats['total_requests']} total requests, {lb_stats['success_rate']:.2%} success rate")

async def main():
    """Main demo function"""
    logger.info("Starting ScrollIntel Architecture Improvements Demo")
    logger.info("=" * 60)
    
    try:
        # Run all demos
        analysis_agent, generation_agent = await demo_enhanced_agents()
        await demo_schema_validation()
        await demo_monitoring_system()
        await demo_workflow_orchestration()
        await demo_error_handling()
        await demo_lifecycle_management()
        await demo_load_balancer()
        await demo_integration()
        
        logger.info("=" * 60)
        logger.info("Demo completed successfully!")
        
        # Cleanup
        await analysis_agent.shutdown()
        await generation_agent.shutdown()
        await agent_lifecycle_manager.stop()
        await intelligent_load_balancer.stop()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())