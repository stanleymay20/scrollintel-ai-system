"""
Demo: Real-Time Orchestration Engine

This demo showcases the Real-Time Orchestration Engine capabilities including:
- Intelligent task distribution based on agent capabilities and load
- Real-time workload balancing across multiple agents
- Multi-agent collaboration protocols (sequential, parallel, consensus, competitive)
- Real-time monitoring and performance metrics

Requirements: 1.1, 1.2, 1.3
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any
import json

from scrollintel.core.realtime_orchestration_engine import (
    RealTimeOrchestrationEngine,
    TaskPriority,
    CollaborationMode,
    OrchestrationTask
)
from scrollintel.core.agent_registry import AgentRegistry, AgentRegistrationRequest
from scrollintel.core.message_bus import MessageBus, get_message_bus
from scrollintel.core.interfaces import AgentResponse, ResponseStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_demo_agents(agent_registry: AgentRegistry) -> Dict[str, str]:
    """Set up demo agents with different capabilities"""
    
    agents = {}
    
    # Data Analyst Agent
    data_analyst_request = AgentRegistrationRequest(
        name="Advanced Data Analyst",
        type="data_analysis",
        version="2.1.0",
        capabilities=[
            {"name": "data_analysis", "performance_score": 92.0},
            {"name": "statistical_analysis", "performance_score": 88.0},
            {"name": "reporting", "performance_score": 85.0}
        ],
        endpoint_url="http://localhost:8001/data-analyst",
        health_check_url="http://localhost:8001/health",
        resource_requirements={"cpu": 2, "memory": 4},
        configuration={"max_tasks": 8, "timeout": 300}
    )
    
    agent_id = await agent_registry.register_agent(data_analyst_request)
    if agent_id:
        agents["data_analyst"] = agent_id
        logger.info(f"Registered Data Analyst Agent: {agent_id}")
    
    # ML Engineer Agent
    ml_engineer_request = AgentRegistrationRequest(
        name="ML Engineering Specialist",
        type="machine_learning",
        version="3.0.1",
        capabilities=[
            {"name": "machine_learning", "performance_score": 95.0},
            {"name": "deep_learning", "performance_score": 90.0},
            {"name": "data_analysis", "performance_score": 82.0},
            {"name": "model_optimization", "performance_score": 88.0}
        ],
        endpoint_url="http://localhost:8002/ml-engineer",
        health_check_url="http://localhost:8002/health",
        resource_requirements={"cpu": 4, "memory": 8, "gpu": 1},
        configuration={"max_tasks": 6, "timeout": 600}
    )
    
    agent_id = await agent_registry.register_agent(ml_engineer_request)
    if agent_id:
        agents["ml_engineer"] = agent_id
        logger.info(f"Registered ML Engineer Agent: {agent_id}")
    
    # Visualization Specialist Agent
    viz_specialist_request = AgentRegistrationRequest(
        name="Visualization Specialist",
        type="visualization",
        version="1.8.2",
        capabilities=[
            {"name": "visualization", "performance_score": 94.0},
            {"name": "dashboard_creation", "performance_score": 91.0},
            {"name": "reporting", "performance_score": 87.0},
            {"name": "data_analysis", "performance_score": 75.0}
        ],
        endpoint_url="http://localhost:8003/viz-specialist",
        health_check_url="http://localhost:8003/health",
        resource_requirements={"cpu": 2, "memory": 6},
        configuration={"max_tasks": 10, "timeout": 240}
    )
    
    agent_id = await agent_registry.register_agent(viz_specialist_request)
    if agent_id:
        agents["viz_specialist"] = agent_id
        logger.info(f"Registered Visualization Specialist Agent: {agent_id}")
    
    # QA Specialist Agent
    qa_specialist_request = AgentRegistrationRequest(
        name="Quality Assurance Specialist",
        type="quality_assurance",
        version="2.3.0",
        capabilities=[
            {"name": "quality_assurance", "performance_score": 96.0},
            {"name": "testing", "performance_score": 93.0},
            {"name": "validation", "performance_score": 89.0},
            {"name": "reporting", "performance_score": 84.0}
        ],
        endpoint_url="http://localhost:8004/qa-specialist",
        health_check_url="http://localhost:8004/health",
        resource_requirements={"cpu": 1, "memory": 2},
        configuration={"max_tasks": 12, "timeout": 180}
    )
    
    agent_id = await agent_registry.register_agent(qa_specialist_request)
    if agent_id:
        agents["qa_specialist"] = agent_id
        logger.info(f"Registered QA Specialist Agent: {agent_id}")
    
    return agents


async def demo_single_agent_tasks(engine: RealTimeOrchestrationEngine):
    """Demonstrate single-agent task orchestration"""
    logger.info("\n=== DEMO: Single-Agent Task Orchestration ===")
    
    tasks = []
    
    # Submit various single-agent tasks
    task_configs = [
        {
            "name": "Customer Data Analysis",
            "description": "Analyze customer behavior patterns from transaction data",
            "required_capabilities": ["data_analysis", "statistical_analysis"],
            "payload": {
                "dataset": "customer_transactions_2024.csv",
                "analysis_type": "behavioral_patterns",
                "time_period": "last_6_months"
            },
            "priority": TaskPriority.HIGH
        },
        {
            "name": "ML Model Training",
            "description": "Train predictive model for customer churn",
            "required_capabilities": ["machine_learning", "model_optimization"],
            "payload": {
                "algorithm": "gradient_boosting",
                "features": ["transaction_frequency", "avg_amount", "tenure"],
                "target": "churn_probability"
            },
            "priority": TaskPriority.CRITICAL
        },
        {
            "name": "Sales Dashboard Creation",
            "description": "Create interactive sales performance dashboard",
            "required_capabilities": ["visualization", "dashboard_creation"],
            "payload": {
                "data_source": "sales_metrics_db",
                "dashboard_type": "executive_summary",
                "refresh_interval": "hourly"
            },
            "priority": TaskPriority.NORMAL
        },
        {
            "name": "Data Quality Validation",
            "description": "Validate data quality for ML pipeline",
            "required_capabilities": ["quality_assurance", "validation"],
            "payload": {
                "pipeline": "customer_analytics_pipeline",
                "validation_rules": ["completeness", "accuracy", "consistency"],
                "threshold": 0.95
            },
            "priority": TaskPriority.HIGH
        }
    ]
    
    # Submit all tasks
    for config in task_configs:
        task_id = await engine.submit_task(**config)
        tasks.append(task_id)
        logger.info(f"Submitted task: {config['name']} (ID: {task_id})")
    
    # Monitor task progress
    logger.info("\nMonitoring task progress...")
    await asyncio.sleep(2.0)  # Allow processing time
    
    for task_id in tasks:
        status = await engine.get_task_status(task_id)
        if status:
            logger.info(f"Task {status['name']}: {status['status']} "
                       f"(Agent: {status['assigned_agent_id']})")
    
    return tasks


async def demo_collaborative_tasks(engine: RealTimeOrchestrationEngine, agents: Dict[str, str]):
    """Demonstrate multi-agent collaborative task orchestration"""
    logger.info("\n=== DEMO: Multi-Agent Collaborative Tasks ===")
    
    collaborative_tasks = []
    
    # Sequential Collaboration: Data Pipeline
    logger.info("\n--- Sequential Collaboration: End-to-End Data Pipeline ---")
    
    sequential_task_id = await engine.submit_task(
        name="End-to-End Analytics Pipeline",
        description="Complete analytics pipeline from data processing to visualization",
        required_capabilities=["data_analysis", "machine_learning", "visualization", "quality_assurance"],
        payload={
            "raw_data": "customer_data_raw.csv",
            "pipeline_stages": ["clean", "analyze", "model", "visualize", "validate"],
            "output_format": "executive_dashboard"
        },
        collaboration_mode=CollaborationMode.SEQUENTIAL,
        collaboration_agents=[
            agents["data_analyst"],
            agents["ml_engineer"], 
            agents["viz_specialist"],
            agents["qa_specialist"]
        ],
        priority=TaskPriority.HIGH,
        timeout_seconds=900.0
    )
    
    collaborative_tasks.append(sequential_task_id)
    logger.info(f"Submitted sequential pipeline task: {sequential_task_id}")
    
    # Parallel Collaboration: Market Research
    logger.info("\n--- Parallel Collaboration: Comprehensive Market Analysis ---")
    
    parallel_task_id = await engine.submit_task(
        name="Comprehensive Market Analysis",
        description="Parallel analysis of market data from multiple perspectives",
        required_capabilities=["data_analysis", "machine_learning", "visualization"],
        payload={
            "market_data": "market_trends_2024.json",
            "analysis_dimensions": ["customer_segments", "product_performance", "competitive_landscape"],
            "output_requirements": ["statistical_summary", "predictive_models", "visual_reports"]
        },
        collaboration_mode=CollaborationMode.PARALLEL,
        collaboration_agents=[
            agents["data_analyst"],
            agents["ml_engineer"],
            agents["viz_specialist"]
        ],
        priority=TaskPriority.NORMAL,
        timeout_seconds=600.0
    )
    
    collaborative_tasks.append(parallel_task_id)
    logger.info(f"Submitted parallel analysis task: {parallel_task_id}")
    
    # Consensus Collaboration: Risk Assessment
    logger.info("\n--- Consensus Collaboration: Risk Assessment ---")
    
    consensus_task_id = await engine.submit_task(
        name="Multi-Perspective Risk Assessment",
        description="Consensus-based risk evaluation from multiple analytical perspectives",
        required_capabilities=["data_analysis", "machine_learning", "quality_assurance"],
        payload={
            "risk_factors": ["market_volatility", "customer_churn", "operational_risks"],
            "assessment_criteria": ["probability", "impact", "mitigation_options"],
            "consensus_threshold": 0.8
        },
        collaboration_mode=CollaborationMode.CONSENSUS,
        collaboration_agents=[
            agents["data_analyst"],
            agents["ml_engineer"],
            agents["qa_specialist"]
        ],
        requires_consensus=True,
        consensus_threshold=0.8,
        priority=TaskPriority.CRITICAL,
        timeout_seconds=720.0
    )
    
    collaborative_tasks.append(consensus_task_id)
    logger.info(f"Submitted consensus risk assessment task: {consensus_task_id}")
    
    # Competitive Collaboration: Algorithm Selection
    logger.info("\n--- Competitive Collaboration: Best Algorithm Selection ---")
    
    competitive_task_id = await engine.submit_task(
        name="Optimal Algorithm Competition",
        description="Competitive evaluation to select best performing algorithm",
        required_capabilities=["machine_learning", "data_analysis"],
        payload={
            "problem_type": "classification",
            "dataset": "customer_features.csv",
            "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score"],
            "algorithms_to_test": ["random_forest", "gradient_boosting", "neural_network"]
        },
        collaboration_mode=CollaborationMode.COMPETITIVE,
        collaboration_agents=[
            agents["ml_engineer"],
            agents["data_analyst"]
        ],
        priority=TaskPriority.HIGH,
        timeout_seconds=480.0
    )
    
    collaborative_tasks.append(competitive_task_id)
    logger.info(f"Submitted competitive algorithm task: {competitive_task_id}")
    
    # Monitor collaborative task progress
    logger.info("\nMonitoring collaborative task progress...")
    await asyncio.sleep(3.0)  # Allow more processing time for complex tasks
    
    for task_id in collaborative_tasks:
        status = await engine.get_task_status(task_id)
        if status:
            logger.info(f"Collaborative Task {status['name']}: {status['status']} "
                       f"(Mode: {status['collaboration_mode']}, "
                       f"Agents: {len(status['collaboration_agents'])})")
    
    return collaborative_tasks


async def demo_load_balancing(engine: RealTimeOrchestrationEngine):
    """Demonstrate real-time load balancing"""
    logger.info("\n=== DEMO: Real-Time Load Balancing ===")
    
    # Submit many tasks to trigger load balancing
    logger.info("Submitting high volume of tasks to demonstrate load balancing...")
    
    bulk_tasks = []
    for i in range(15):
        task_id = await engine.submit_task(
            name=f"Bulk Analysis Task {i+1}",
            description=f"Bulk data analysis task number {i+1}",
            required_capabilities=["data_analysis"],
            payload={"batch_id": i+1, "data_size": "medium"},
            priority=TaskPriority.NORMAL if i % 3 != 0 else TaskPriority.HIGH
        )
        bulk_tasks.append(task_id)
    
    logger.info(f"Submitted {len(bulk_tasks)} bulk tasks")
    
    # Monitor load balancing statistics
    await asyncio.sleep(1.0)
    
    load_stats = engine.load_balancer.get_load_balance_stats()
    logger.info(f"Load Balancing Stats: {json.dumps(load_stats, indent=2)}")
    
    # Show distribution statistics
    dist_stats = engine.task_distributor.distribution_stats
    logger.info(f"Task Distribution Stats: {json.dumps(dist_stats, indent=2)}")
    
    return bulk_tasks


async def demo_performance_monitoring(engine: RealTimeOrchestrationEngine):
    """Demonstrate real-time performance monitoring"""
    logger.info("\n=== DEMO: Real-Time Performance Monitoring ===")
    
    # Get comprehensive engine statistics
    stats = engine.get_engine_stats()
    
    logger.info("=== Engine Performance Statistics ===")
    logger.info(f"Engine Status: {stats['engine_status']}")
    logger.info(f"Task Statistics: {stats['task_statistics']}")
    logger.info(f"Component Status: {stats['component_status']}")
    
    # Show active tasks
    active_tasks = engine.get_active_tasks()
    logger.info(f"\nActive Tasks: {len(active_tasks)}")
    for task in active_tasks[:5]:  # Show first 5
        logger.info(f"  - {task['name']} ({task['status']}) - Priority: {task['priority']}")
    
    # Show coordination statistics
    coord_stats = engine.coordinator.get_coordination_stats()
    logger.info(f"\nCoordination Statistics: {json.dumps(coord_stats, indent=2)}")
    
    # Show load balancing metrics
    load_stats = engine.load_balancer.get_load_balance_stats()
    if "error" not in load_stats:
        logger.info(f"\nLoad Balancing Metrics:")
        logger.info(f"  - Average Load: {load_stats.get('average_load', 0):.1f}%")
        logger.info(f"  - Max Load: {load_stats.get('max_load', 0):.1f}%")
        logger.info(f"  - Min Load: {load_stats.get('min_load', 0):.1f}%")
        logger.info(f"  - Rebalancing Events: {load_stats.get('rebalancing_stats', {}).get('rebalancing_events', 0)}")


async def demo_error_handling_and_recovery(engine: RealTimeOrchestrationEngine):
    """Demonstrate error handling and recovery mechanisms"""
    logger.info("\n=== DEMO: Error Handling and Recovery ===")
    
    # Submit a task that will likely fail (nonexistent capability)
    logger.info("Testing error handling with invalid capability requirement...")
    
    error_task_id = await engine.submit_task(
        name="Error Test Task",
        description="Task designed to test error handling",
        required_capabilities=["nonexistent_capability"],
        payload={"test": "error_handling"},
        priority=TaskPriority.NORMAL,
        timeout_seconds=60.0
    )
    
    logger.info(f"Submitted error test task: {error_task_id}")
    
    # Wait and check status
    await asyncio.sleep(2.0)
    
    error_task_status = await engine.get_task_status(error_task_id)
    if error_task_status:
        logger.info(f"Error Task Status: {error_task_status['status']}")
        if error_task_status.get('error_message'):
            logger.info(f"Error Message: {error_task_status['error_message']}")
    
    # Test task cancellation
    logger.info("\nTesting task cancellation...")
    
    cancel_task_id = await engine.submit_task(
        name="Cancellation Test Task",
        description="Task to test cancellation mechanism",
        required_capabilities=["data_analysis"],
        payload={"test": "cancellation"},
        priority=TaskPriority.LOW,
        timeout_seconds=300.0
    )
    
    logger.info(f"Submitted cancellation test task: {cancel_task_id}")
    
    # Cancel immediately
    success = await engine.cancel_task(cancel_task_id)
    logger.info(f"Task cancellation {'successful' if success else 'failed'}")
    
    cancel_task_status = await engine.get_task_status(cancel_task_id)
    if cancel_task_status:
        logger.info(f"Cancelled Task Status: {cancel_task_status['status']}")


async def main():
    """Main demo function"""
    logger.info("=== Real-Time Orchestration Engine Demo ===")
    logger.info("Demonstrating enterprise-grade AI agent orchestration capabilities")
    
    # Initialize components
    message_bus = get_message_bus()
    agent_registry = AgentRegistry(message_bus)
    engine = RealTimeOrchestrationEngine(agent_registry, message_bus)
    
    try:
        # Start the orchestration engine
        logger.info("\nStarting Real-Time Orchestration Engine...")
        await engine.start()
        
        # Set up demo agents
        logger.info("\nSetting up demo agents...")
        agents = await setup_demo_agents(agent_registry)
        logger.info(f"Successfully registered {len(agents)} agents")
        
        # Run demonstrations
        await demo_single_agent_tasks(engine)
        await demo_collaborative_tasks(engine, agents)
        await demo_load_balancing(engine)
        await demo_performance_monitoring(engine)
        await demo_error_handling_and_recovery(engine)
        
        # Final statistics
        logger.info("\n=== FINAL DEMO STATISTICS ===")
        final_stats = engine.get_engine_stats()
        logger.info(f"Total Tasks Processed: {final_stats['task_statistics']['tasks_processed']}")
        logger.info(f"Tasks Completed: {final_stats['task_statistics']['tasks_completed']}")
        logger.info(f"Tasks Failed: {final_stats['task_statistics']['tasks_failed']}")
        logger.info(f"Average Processing Time: {final_stats['task_statistics']['average_processing_time']:.2f}s")
        logger.info(f"Peak Concurrent Tasks: {final_stats['task_statistics']['concurrent_tasks_peak']}")
        
        # Show registry statistics
        registry_stats = await agent_registry.get_registry_stats()
        logger.info(f"\nAgent Registry Stats:")
        logger.info(f"  - Total Agents: {registry_stats['agent_counts']['total']}")
        logger.info(f"  - Active Agents: {registry_stats['agent_counts']['active']}")
        logger.info(f"  - Average Response Time: {registry_stats['performance_metrics']['average_response_time']:.2f}s")
        
        logger.info("\n=== Demo completed successfully! ===")
        logger.info("The Real-Time Orchestration Engine has demonstrated:")
        logger.info("✓ Intelligent task distribution based on agent capabilities and load")
        logger.info("✓ Real-time workload balancing across multiple agents")
        logger.info("✓ Multi-agent collaboration protocols (sequential, parallel, consensus, competitive)")
        logger.info("✓ Comprehensive performance monitoring and metrics")
        logger.info("✓ Robust error handling and recovery mechanisms")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {str(e)}")
        raise
    
    finally:
        # Clean shutdown
        logger.info("\nShutting down orchestration engine...")
        await engine.stop()
        await message_bus.stop()
        logger.info("Demo cleanup completed")


if __name__ == "__main__":
    asyncio.run(main())