"""
Demo Script for ScrollIntel Modular Component Architecture

This script demonstrates the modular component system including:
- Component registration and discovery
- Dependency management
- Component orchestration
- Analysis and generation components working together
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, Any

# Import modular components
from scrollintel.core.modular_components import (
    component_registry, ComponentType, ComponentStatus
)
from scrollintel.core.component_orchestrator import (
    component_orchestrator, ComponentTask, OrchestrationPlan, 
    OrchestrationStrategy, create_analysis_generation_plan
)
from scrollintel.components.analysis_components import (
    register_analysis_components, AnalysisRequest
)
from scrollintel.components.generation_components import (
    register_generation_components, GenerationRequest
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_component_registration():
    """Demonstrate component registration and discovery"""
    logger.info("=== Demo: Component Registration and Discovery ===")
    
    # Register analysis components
    await register_analysis_components()
    
    # Register generation components
    await register_generation_components()
    
    # Show registered components
    logger.info("Registered Components:")
    for component_id, instance in component_registry._components.items():
        logger.info(f"  - {component_id}: {instance.metadata.name} ({instance.metadata.component_type.value})")
    
    # Show dependency graph
    dependency_order = component_registry.get_dependency_order()
    logger.info(f"Dependency order: {dependency_order}")
    
    # Show system status
    status = component_registry.get_system_status()
    logger.info(f"System status: {json.dumps(status, indent=2)}")

async def demo_component_initialization():
    """Demonstrate component initialization"""
    logger.info("=== Demo: Component Initialization ===")
    
    # Initialize all components
    success = await component_registry.initialize_all()
    logger.info(f"Component initialization: {'SUCCESS' if success else 'FAILED'}")
    
    # Check individual component status
    for component_id, instance in component_registry._components.items():
        logger.info(f"Component {component_id}: {instance.status.value}")

async def demo_individual_components():
    """Demonstrate individual component functionality"""
    logger.info("=== Demo: Individual Component Operations ===")
    
    # Test data validator
    validator = component_registry.get_component("data_validator")
    if validator and validator.status == ComponentStatus.READY:
        logger.info("Testing Data Validator...")
        
        # Test numerical data validation
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        validation_result = await validator.component.validate_data(test_data, "numerical")
        logger.info(f"Numerical validation: {validation_result}")
        
        # Test invalid data
        invalid_data = ["a", "b", "c"]
        validation_result = await validator.component.validate_data(invalid_data, "numerical")
        logger.info(f"Invalid data validation: {validation_result}")
    
    # Test statistical analyzer
    analyzer = component_registry.get_component("statistical_analyzer")
    if analyzer and analyzer.status == ComponentStatus.READY:
        logger.info("Testing Statistical Analyzer...")
        
        analysis_request = AnalysisRequest(
            request_id="test-analysis-001",
            analysis_type="descriptive",
            data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            parameters={"include_quartiles": True},
            metadata={}
        )
        
        analysis_result = await analyzer.component.analyze(analysis_request)
        logger.info(f"Analysis result: {analysis_result.results}")
        logger.info(f"Analysis confidence: {analysis_result.confidence}")
    
    # Test text generator
    generator = component_registry.get_component("text_generator")
    if generator and generator.status == ComponentStatus.READY:
        logger.info("Testing Text Generator...")
        
        generation_request = GenerationRequest(
            request_id="test-generation-001",
            generation_type="summary",
            prompt="Statistical analysis of numerical data",
            parameters={"format": "markdown"},
            constraints={"max_length": 300},
            metadata={}
        )
        
        generation_result = await generator.component.generate(generation_request)
        logger.info(f"Generated content:\n{generation_result.content}")
        logger.info(f"Quality metrics: {generation_result.quality_metrics}")

async def demo_component_orchestration():
    """Demonstrate component orchestration"""
    logger.info("=== Demo: Component Orchestration ===")
    
    # Create test data
    test_data = [10, 15, 12, 18, 20, 14, 16, 19, 13, 17, 21, 11, 22, 9, 25]
    
    # Create orchestration plan
    plan = create_analysis_generation_plan(
        analysis_data=test_data,
        generation_prompt="Analysis results for sales data"
    )
    
    logger.info(f"Executing orchestration plan: {plan.name}")
    logger.info(f"Strategy: {plan.strategy.value}")
    logger.info(f"Tasks: {[task.task_id for task in plan.tasks]}")
    
    # Execute the plan
    result = await component_orchestrator.execute_plan(plan)
    
    logger.info(f"Orchestration result: {result['status']}")
    logger.info(f"Execution duration: {result['duration']:.2f} seconds")
    
    if result["status"] == "success":
        logger.info("Task Results:")
        for task_id, task_result in result["results"].items():
            logger.info(f"  {task_id}: {task_result.status}")
            if task_result.status == "success" and task_result.output_data:
                if hasattr(task_result.output_data, 'content'):
                    # Generation result
                    logger.info(f"    Generated content preview: {task_result.output_data.content[:100]}...")
                elif hasattr(task_result.output_data, 'results'):
                    # Analysis result
                    logger.info(f"    Analysis results: {task_result.output_data.results}")
                else:
                    logger.info(f"    Output: {task_result.output_data}")

async def demo_parallel_orchestration():
    """Demonstrate parallel component orchestration"""
    logger.info("=== Demo: Parallel Component Orchestration ===")
    
    # Create tasks that can run in parallel
    tasks = [
        ComponentTask(
            task_id="analyze_sales_data",
            component_id="statistical_analyzer",
            operation="analyze",
            input_data=AnalysisRequest(
                request_id="sales-analysis",
                analysis_type="descriptive",
                data=[100, 120, 110, 130, 140, 125, 135, 145, 115, 150],
                parameters={},
                metadata={}
            )
        ),
        ComponentTask(
            task_id="analyze_customer_data",
            component_id="statistical_analyzer",
            operation="analyze",
            input_data=AnalysisRequest(
                request_id="customer-analysis",
                analysis_type="trend_analysis",
                data=[50, 55, 52, 58, 60, 54, 56, 59, 53, 61],
                parameters={},
                metadata={}
            )
        ),
        ComponentTask(
            task_id="generate_summary",
            component_id="text_generator",
            operation="generate",
            input_data=GenerationRequest(
                request_id="summary-generation",
                generation_type="summary",
                prompt="Business performance analysis",
                parameters={"format": "markdown"},
                constraints={"max_length": 200},
                metadata={}
            )
        )
    ]
    
    parallel_plan = OrchestrationPlan(
        plan_id="parallel-demo",
        name="Parallel Analysis Demo",
        strategy=OrchestrationStrategy.PARALLEL,
        tasks=tasks
    )
    
    logger.info("Executing parallel orchestration plan...")
    result = await component_orchestrator.execute_plan(parallel_plan)
    
    logger.info(f"Parallel orchestration result: {result['status']}")
    logger.info(f"Execution duration: {result['duration']:.2f} seconds")
    
    if result["status"] == "success":
        logger.info("Parallel task results:")
        for task_id, task_result in result["results"].items():
            logger.info(f"  {task_id}: {task_result.status} ({task_result.processing_time:.2f}s)")

async def demo_pipeline_orchestration():
    """Demonstrate pipeline orchestration"""
    logger.info("=== Demo: Pipeline Component Orchestration ===")
    
    # Create pipeline tasks where output flows to next task
    pipeline_tasks = [
        ComponentTask(
            task_id="validate_input",
            component_id="data_validator",
            operation="validate_data",
            input_data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            parameters={"data_type": "numerical"}
        ),
        ComponentTask(
            task_id="analyze_validated_data",
            component_id="statistical_analyzer",
            operation="analyze",
            input_data=None,  # Will be set from previous task
            parameters={"analysis_type": "descriptive"}
        ),
        ComponentTask(
            task_id="generate_analysis_report",
            component_id="text_generator",
            operation="generate",
            input_data=None,  # Will be set from previous task
            parameters={"generation_type": "report", "format": "markdown"}
        )
    ]
    
    pipeline_plan = OrchestrationPlan(
        plan_id="pipeline-demo",
        name="Pipeline Processing Demo",
        strategy=OrchestrationStrategy.PIPELINE,
        tasks=pipeline_tasks
    )
    
    logger.info("Executing pipeline orchestration plan...")
    result = await component_orchestrator.execute_plan(pipeline_plan)
    
    logger.info(f"Pipeline orchestration result: {result['status']}")
    logger.info(f"Execution duration: {result['duration']:.2f} seconds")
    
    if result["status"] == "success":
        logger.info("Pipeline results:")
        for task_id, task_result in result["results"].items():
            logger.info(f"  {task_id}: {task_result.status}")

async def demo_error_handling():
    """Demonstrate error handling in component orchestration"""
    logger.info("=== Demo: Error Handling ===")
    
    # Create a task that will fail
    failing_tasks = [
        ComponentTask(
            task_id="invalid_analysis",
            component_id="statistical_analyzer",
            operation="analyze",
            input_data=AnalysisRequest(
                request_id="failing-analysis",
                analysis_type="invalid_type",  # This will cause an error
                data=[1, 2, 3],
                parameters={},
                metadata={}
            )
        ),
        ComponentTask(
            task_id="backup_generation",
            component_id="text_generator",
            operation="generate",
            input_data=GenerationRequest(
                request_id="backup-generation",
                generation_type="summary",
                prompt="Fallback content generation",
                parameters={},
                constraints={},
                metadata={}
            )
        )
    ]
    
    error_plan = OrchestrationPlan(
        plan_id="error-demo",
        name="Error Handling Demo",
        strategy=OrchestrationStrategy.SEQUENTIAL,
        tasks=failing_tasks,
        error_handling="continue"  # Continue on errors
    )
    
    logger.info("Executing plan with error handling...")
    result = await component_orchestrator.execute_plan(error_plan)
    
    logger.info(f"Error handling result: {result['status']}")
    
    if "results" in result:
        for task_id, task_result in result["results"].items():
            logger.info(f"  {task_id}: {task_result.status}")
            if task_result.error:
                logger.info(f"    Error: {task_result.error}")

async def demo_metrics_and_monitoring():
    """Demonstrate metrics and monitoring"""
    logger.info("=== Demo: Metrics and Monitoring ===")
    
    # Get orchestration metrics
    metrics = component_orchestrator.get_orchestration_metrics()
    logger.info(f"Orchestration metrics: {json.dumps(metrics, indent=2)}")
    
    # Get execution history
    history = component_orchestrator.get_execution_history(5)
    logger.info(f"Recent executions: {len(history)}")
    
    for execution in history:
        logger.info(f"  {execution['plan'].name}: {execution['status']} ({execution.get('duration', 0):.2f}s)")
    
    # Get component registry status
    registry_status = component_registry.get_system_status()
    logger.info(f"Component registry status: {json.dumps(registry_status, indent=2)}")

async def demo_component_interfaces():
    """Demonstrate component interface standardization"""
    logger.info("=== Demo: Component Interface Standardization ===")
    
    # Show component interfaces and capabilities
    for component_id, instance in component_registry._components.items():
        metadata = instance.metadata
        logger.info(f"Component: {component_id}")
        logger.info(f"  Type: {metadata.component_type.value}")
        logger.info(f"  Version: {metadata.version}")
        logger.info(f"  Interface Version: {metadata.interface_version}")
        logger.info(f"  Provides: {metadata.provides}")
        logger.info(f"  Requires: {metadata.requires}")
        logger.info(f"  Dependencies: {metadata.dependencies}")
        logger.info(f"  Status: {instance.status.value}")
        logger.info("")

async def main():
    """Main demo function"""
    logger.info("Starting ScrollIntel Modular Component Architecture Demo")
    logger.info("=" * 70)
    
    try:
        # Run all demos
        await demo_component_registration()
        await demo_component_initialization()
        await demo_individual_components()
        await demo_component_orchestration()
        await demo_parallel_orchestration()
        await demo_pipeline_orchestration()
        await demo_error_handling()
        await demo_metrics_and_monitoring()
        await demo_component_interfaces()
        
        logger.info("=" * 70)
        logger.info("Modular Component Architecture Demo completed successfully!")
        
        # Cleanup
        await component_registry.shutdown_all()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())