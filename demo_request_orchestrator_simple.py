"""
Simple demo script for Request Orchestrator functionality.

Demonstrates basic task submission and execution without complex workflows.
"""

import asyncio
import time
from scrollintel.core.request_orchestrator import (
    RequestOrchestrator,
    TaskDefinition,
    TaskPriority,
    ResourceType,
    ResourceRequirement
)


async def simple_task(name: str, duration: float = 1.0) -> dict:
    """Simple task that sleeps for a specified duration."""
    print(f"🔄 Starting task: {name}")
    await asyncio.sleep(duration)
    result = {"name": name, "duration": duration, "status": "completed"}
    print(f"✅ Completed task: {name}")
    return result


async def demo_basic_functionality():
    """Demonstrate basic orchestrator functionality."""
    print("🚀 Request Orchestrator Simple Demo")
    print("=" * 50)
    
    orchestrator = RequestOrchestrator(max_concurrent_tasks=3)
    
    async with orchestrator.managed_execution():
        # Submit a few simple tasks
        tasks = []
        
        for i in range(3):
            task = TaskDefinition(
                id=f"task_{i}",
                name=f"Simple Task {i}",
                handler=simple_task,
                args=(f"Task {i}", 1.0 + i * 0.5),
                priority=TaskPriority.NORMAL,
                resource_requirements=[
                    ResourceRequirement(ResourceType.CPU, 1.0)
                ]
            )
            
            task_id = await orchestrator.submit_task(task)
            tasks.append(task_id)
            print(f"📝 Submitted: {task.name} (ID: {task_id})")
        
        # Wait for all tasks to complete
        print("\n📊 Waiting for tasks to complete...")
        completed = set()
        
        while len(completed) < len(tasks):
            await asyncio.sleep(0.5)
            
            for task_id in tasks:
                if task_id not in completed:
                    status = await orchestrator.get_task_status(task_id)
                    if status and status["status"] == "completed":
                        completed.add(task_id)
                        print(f"✅ Task {task_id} completed in {status.get('duration', 'N/A')}")
        
        # Show final system status
        system_status = await orchestrator.get_system_status()
        print(f"\n📈 System Status:")
        print(f"   Running tasks: {system_status['running_tasks']}")
        print(f"   Resource utilization: {system_status['resource_utilization']}")
        
        print("\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(demo_basic_functionality())