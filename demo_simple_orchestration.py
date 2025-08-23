"""
Simple Demo: Real-Time Orchestration Engine

This demo shows the basic functionality of the Real-Time Orchestration Engine
for coordinating multiple agents with task distribution and real-time processing.
"""

import asyncio
import logging
from scrollintel.core.realtime_orchestration_engine import (
    RealTimeOrchestrationEngine, 
    TaskPriority
)
from scrollintel.core.agent_registry import AgentRegistry
from scrollintel.core.message_bus import get_message_bus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main demo function"""
    logger.info("=== Real-Time Orchestration Engine Demo ===")
    
    # Initialize components
    message_bus = get_message_bus()
    agent_registry = AgentRegistry(message_bus)
    engine = RealTimeOrchestrationEngine(agent_registry, message_bus)
    
    try:
        # Start the orchestration engine
        logger.info("Starting orchestration engine...")
        await engine.start()
        
        # Submit some test tasks
        logger.info("Submitting test tasks...")
        
        task1_id = await engine.submit_task(
            name="Data Analysis Task",
            description="Analyze customer data for insights",
            required_capabilities=["data_analysis"],
            payload={"dataset": "customers.csv"},
            priority=TaskPriority.HIGH
        )
        logger.info(f"Submitted task 1: {task1_id}")
        
        task2_id = await engine.submit_task(
            name="Report Generation",
            description="Generate monthly sales report",
            required_capabilities=["reporting"],
            payload={"month": "January", "year": 2024},
            priority=TaskPriority.NORMAL
        )
        logger.info(f"Submitted task 2: {task2_id}")
        
        task3_id = await engine.submit_task(
            name="ML Model Training",
            description="Train predictive model",
            required_capabilities=["machine_learning"],
            payload={"algorithm": "random_forest"},
            priority=TaskPriority.CRITICAL
        )
        logger.info(f"Submitted task 3: {task3_id}")
        
        # Wait for tasks to process
        logger.info("Waiting for tasks to process...")
        await asyncio.sleep(2.0)
        
        # Check task statuses
        logger.info("Checking task statuses...")
        for task_id in [task1_id, task2_id, task3_id]:
            status = await engine.get_task_status(task_id)
            if status:
                logger.info(f"Task {status['name']}: {status['status']}")
        
        # Get engine statistics
        stats = engine.get_engine_stats()
        logger.info("Engine Statistics:")
        logger.info(f"  - Tasks Processed: {stats['task_statistics']['tasks_processed']}")
        logger.info(f"  - Tasks Completed: {stats['task_statistics']['tasks_completed']}")
        logger.info(f"  - Tasks Failed: {stats['task_statistics']['tasks_failed']}")
        logger.info(f"  - Active Tasks: {stats['task_statistics']['active_tasks']}")
        logger.info(f"  - Engine Running: {stats['engine_status']['running']}")
        
        # Show active tasks
        active_tasks = engine.get_active_tasks()
        logger.info(f"Active Tasks: {len(active_tasks)}")
        for task in active_tasks:
            logger.info(f"  - {task['name']} ({task['status']})")
        
        logger.info("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        raise
    
    finally:
        # Clean shutdown
        logger.info("Shutting down...")
        await engine.stop()
        await message_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())