"""
Demo script for Request Orchestrator functionality.

Demonstrates task submission, workflow management, resource allocation,
and progress tracking capabilities.
"""

import asyncio
import time
import random
from datetime import datetime
from typing import List

from scrollintel.core.request_orchestrator import (
    RequestOrchestrator,
    TaskDefinition,
    WorkflowDefinition,
    TaskPriority,
    ResourceType,
    ResourceRequirement,
    create_image_generation_workflow,
    create_video_generation_workflow
)


# Demo task handlers

async def simulate_image_generation(prompt: str, model: str = "dalle3", resolution: tuple = (1024, 1024)) -> dict:
    """Simulate image generation task."""
    print(f"🎨 Generating image with {model}: '{prompt}' at {resolution[0]}x{resolution[1]}")
    
    # Simulate processing time based on resolution
    base_time = 2.0
    resolution_factor = (resolution[0] * resolution[1]) / (1024 * 1024)
    processing_time = base_time * resolution_factor
    
    await asyncio.sleep(processing_time)
    
    result = {
        "type": "image",
        "prompt": prompt,
        "model": model,
        "resolution": resolution,
        "url": f"https://example.com/images/{hash(prompt)}.jpg",
        "processing_time": processing_time
    }
    
    print(f"✅ Image generated successfully: {result['url']}")
    return result


async def simulate_video_generation(prompt: str, duration: float = 5.0, fps: int = 30) -> dict:
    """Simulate video generation task."""
    print(f"🎬 Generating video: '{prompt}' ({duration}s at {fps}fps)")
    
    # Simulate processing time based on duration and fps
    processing_time = duration * 0.5 + (fps / 30) * 2.0
    
    await asyncio.sleep(processing_time)
    
    result = {
        "type": "video",
        "prompt": prompt,
        "duration": duration,
        "fps": fps,
        "url": f"https://example.com/videos/{hash(prompt)}.mp4",
        "processing_time": processing_time
    }
    
    print(f"✅ Video generated successfully: {result['url']}")
    return result


async def simulate_image_enhancement(image_url: str, enhancement_type: str = "upscale") -> dict:
    """Simulate image enhancement task."""
    print(f"✨ Enhancing image: {image_url} ({enhancement_type})")
    
    # Simulate processing time
    processing_time = random.uniform(1.0, 3.0)
    await asyncio.sleep(processing_time)
    
    result = {
        "type": "enhancement",
        "input_url": image_url,
        "enhancement_type": enhancement_type,
        "output_url": f"https://example.com/enhanced/{hash(image_url)}.jpg",
        "processing_time": processing_time
    }
    
    print(f"✅ Image enhanced successfully: {result['output_url']}")
    return result


async def simulate_style_transfer(image_url: str, style: str = "artistic") -> dict:
    """Simulate style transfer task."""
    print(f"🎭 Applying style transfer: {image_url} -> {style}")
    
    # Simulate processing time
    processing_time = random.uniform(2.0, 4.0)
    await asyncio.sleep(processing_time)
    
    result = {
        "type": "style_transfer",
        "input_url": image_url,
        "style": style,
        "output_url": f"https://example.com/styled/{hash(image_url + style)}.jpg",
        "processing_time": processing_time
    }
    
    print(f"✅ Style transfer completed: {result['output_url']}")
    return result


def simulate_slow_task(duration: float = 5.0) -> dict:
    """Simulate a slow synchronous task."""
    print(f"⏳ Running slow task for {duration} seconds...")
    time.sleep(duration)
    print(f"✅ Slow task completed after {duration} seconds")
    return {"type": "slow_task", "duration": duration}


async def demo_basic_task_submission():
    """Demonstrate basic task submission and execution."""
    print("\n" + "="*60)
    print("🚀 DEMO: Basic Task Submission")
    print("="*60)
    
    orchestrator = RequestOrchestrator(max_concurrent_tasks=3)
    
    async with orchestrator.managed_execution():
        # Submit individual tasks
        tasks = []
        
        # High priority image generation
        task1 = TaskDefinition(
            id="img_gen_1",
            name="Generate Cat Image",
            handler=simulate_image_generation,
            args=("A cute cat sitting in a garden",),
            kwargs={"model": "dalle3", "resolution": (1024, 1024)},
            priority=TaskPriority.HIGH,
            resource_requirements=[
                ResourceRequirement(ResourceType.GPU, 1.0),
                ResourceRequirement(ResourceType.MEMORY, 2.0)
            ]
        )
        
        # Normal priority video generation
        task2 = TaskDefinition(
            id="vid_gen_1",
            name="Generate Flying Car Video",
            handler=simulate_video_generation,
            args=("A flying car in a futuristic city",),
            kwargs={"duration": 3.0, "fps": 24},
            priority=TaskPriority.NORMAL,
            resource_requirements=[
                ResourceRequirement(ResourceType.GPU, 2.0),
                ResourceRequirement(ResourceType.MEMORY, 4.0)
            ]
        )
        
        # Low priority slow task
        task3 = TaskDefinition(
            id="slow_task_1",
            name="Slow Processing Task",
            handler=simulate_slow_task,
            args=(2.0,),
            priority=TaskPriority.LOW,
            resource_requirements=[
                ResourceRequirement(ResourceType.CPU, 1.0)
            ]
        )
        
        # Submit tasks
        for task in [task1, task2, task3]:
            task_id = await orchestrator.submit_task(task)
            tasks.append(task_id)
            print(f"📝 Submitted task: {task.name} (ID: {task_id})")
        
        # Monitor progress
        print("\n📊 Monitoring task progress...")
        completed_tasks = set()
        
        while len(completed_tasks) < len(tasks):
            await asyncio.sleep(0.5)
            
            for task_id in tasks:
                if task_id not in completed_tasks:
                    status = await orchestrator.get_task_status(task_id)
                    if status and status["status"] in ["completed", "failed"]:
                        completed_tasks.add(task_id)
                        print(f"✅ Task {task_id} {status['status']}: {status.get('duration', 'N/A')}")
        
        # Show final system status
        system_status = await orchestrator.get_system_status()
        print(f"\n📈 Final system status:")
        print(f"   Running tasks: {system_status['running_tasks']}")
        print(f"   Queue status: {system_status['queue_status']}")


async def demo_workflow_execution():
    """Demonstrate workflow execution with dependencies."""
    print("\n" + "="*60)
    print("🔄 DEMO: Workflow Execution with Dependencies")
    print("="*60)
    
    orchestrator = RequestOrchestrator(max_concurrent_tasks=4)
    
    async with orchestrator.managed_execution():
        # Create a complex workflow: Generate -> Enhance -> Style Transfer
        
        # Task 1: Generate base image
        generate_task = TaskDefinition(
            id="generate_base",
            name="Generate Base Image",
            handler=simulate_image_generation,
            args=("A majestic mountain landscape",),
            kwargs={"model": "stable_diffusion", "resolution": (1024, 1024)},
            priority=TaskPriority.HIGH,
            resource_requirements=[
                ResourceRequirement(ResourceType.GPU, 1.0),
                ResourceRequirement(ResourceType.MEMORY, 2.0)
            ]
        )
        
        # Task 2: Enhance the generated image (depends on Task 1)
        enhance_task = TaskDefinition(
            id="enhance_image",
            name="Enhance Image Quality",
            handler=simulate_image_enhancement,
            args=("https://example.com/base_image.jpg",),
            kwargs={"enhancement_type": "upscale_4x"},
            dependencies=["generate_base"],
            priority=TaskPriority.NORMAL,
            resource_requirements=[
                ResourceRequirement(ResourceType.GPU, 0.5),
                ResourceRequirement(ResourceType.MEMORY, 1.0)
            ]
        )
        
        # Task 3: Apply style transfer (depends on Task 2)
        style_task = TaskDefinition(
            id="apply_style",
            name="Apply Artistic Style",
            handler=simulate_style_transfer,
            args=("https://example.com/enhanced_image.jpg",),
            kwargs={"style": "impressionist"},
            dependencies=["enhance_image"],
            priority=TaskPriority.NORMAL,
            resource_requirements=[
                ResourceRequirement(ResourceType.GPU, 1.0),
                ResourceRequirement(ResourceType.MEMORY, 1.5)
            ]
        )
        
        # Create workflow
        workflow = WorkflowDefinition(
            id="image_processing_pipeline",
            name="Image Processing Pipeline",
            tasks=[generate_task, enhance_task, style_task],
            description="Generate, enhance, and stylize an image"
        )
        
        # Submit workflow
        workflow_id = await orchestrator.submit_workflow(workflow)
        print(f"📋 Submitted workflow: {workflow.name} (ID: {workflow_id})")
        
        # Monitor workflow progress
        print("\n📊 Monitoring workflow progress...")
        
        while True:
            await asyncio.sleep(1.0)
            
            workflow_status = await orchestrator.get_workflow_status(workflow_id)
            progress = workflow_status.get("progress", 0)
            status = workflow_status.get("status", "unknown")
            
            print(f"   Progress: {progress:.1f}% - Status: {status}")
            
            # Show individual task statuses
            tasks_info = workflow_status.get("tasks", {})
            for task_id, task_info in tasks_info.items():
                task_status = task_info.get("status", "unknown")
                task_progress = task_info.get("progress", 0)
                print(f"     {task_id}: {task_status} ({task_progress:.1f}%)")
            
            if status in ["completed", "failed"]:
                break
        
        print(f"\n✅ Workflow completed with status: {status}")


async def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n" + "="*60)
    print("📦 DEMO: Batch Processing")
    print("="*60)
    
    orchestrator = RequestOrchestrator(max_concurrent_tasks=5)
    
    async with orchestrator.managed_execution():
        # Create batch image generation workflow
        prompts = [
            "A serene lake at sunset",
            "A bustling city street at night",
            "A cozy cabin in the woods",
            "A futuristic space station",
            "A tropical beach with palm trees"
        ]
        
        models = ["dalle3", "stable_diffusion", "midjourney", "dalle3", "stable_diffusion"]
        
        workflow = await create_image_generation_workflow(
            prompts=prompts,
            model_preferences=models,
            priority=TaskPriority.HIGH
        )
        
        workflow_id = await orchestrator.submit_workflow(workflow)
        print(f"📋 Submitted batch workflow: {len(prompts)} images")
        
        # Monitor batch progress
        start_time = time.time()
        
        while True:
            await asyncio.sleep(1.0)
            
            workflow_status = await orchestrator.get_workflow_status(workflow_id)
            progress = workflow_status.get("progress", 0)
            status = workflow_status.get("status", "unknown")
            completed = workflow_status.get("completed_tasks", 0)
            total = workflow_status.get("total_tasks", len(prompts))
            
            elapsed = time.time() - start_time
            print(f"   Batch Progress: {completed}/{total} images ({progress:.1f}%) - {elapsed:.1f}s elapsed")
            
            if status in ["completed", "failed"]:
                break
        
        print(f"\n✅ Batch processing completed: {completed}/{total} images in {elapsed:.1f}s")


async def demo_resource_management():
    """Demonstrate resource management and constraints."""
    print("\n" + "="*60)
    print("💾 DEMO: Resource Management")
    print("="*60)
    
    orchestrator = RequestOrchestrator(max_concurrent_tasks=10)
    
    async with orchestrator.managed_execution():
        # Submit resource-intensive tasks that exceed available resources
        tasks = []
        
        for i in range(8):  # More tasks than available GPUs
            task = TaskDefinition(
                id=f"gpu_intensive_{i}",
                name=f"GPU Intensive Task {i}",
                handler=simulate_image_generation,
                args=(f"High resolution image {i}",),
                kwargs={"resolution": (2048, 2048)},
                priority=TaskPriority.NORMAL,
                resource_requirements=[
                ResourceRequirement(ResourceType.GPU, 2.0),  # Each needs 2 GPUs
                ResourceRequirement(ResourceType.MEMORY, 8.0)
                ]
            )
            
            task_id = await orchestrator.submit_task(task)
            tasks.append(task_id)
            print(f"📝 Submitted GPU-intensive task {i}")
        
        # Monitor resource utilization
        print("\n📊 Monitoring resource utilization...")
        
        for _ in range(10):  # Monitor for 10 seconds
            await asyncio.sleep(1.0)
            
            system_status = await orchestrator.get_system_status()
            running = system_status["running_tasks"]
            utilization = system_status["resource_utilization"]
            queue_status = system_status["queue_status"]
            
            print(f"   Running: {running}/10 tasks")
            print(f"   GPU: {utilization.get('GPU', '0%')}, Memory: {utilization.get('MEMORY', '0%')}")
            print(f"   Queue: {sum(queue_status.values())} tasks waiting")
            print()


async def demo_error_handling():
    """Demonstrate error handling and retry logic."""
    print("\n" + "="*60)
    print("⚠️  DEMO: Error Handling and Retry Logic")
    print("="*60)
    
    async def failing_task(fail_count: int = 2):
        """Task that fails a specified number of times before succeeding."""
        if not hasattr(failing_task, 'attempts'):
            failing_task.attempts = {}
        
        task_id = id(asyncio.current_task())
        attempts = failing_task.attempts.get(task_id, 0)
        failing_task.attempts[task_id] = attempts + 1
        
        print(f"💥 Failing task attempt {attempts + 1}")
        
        if attempts < fail_count:
            raise RuntimeError(f"Task failed on attempt {attempts + 1}")
        
        print(f"✅ Task succeeded on attempt {attempts + 1}")
        return {"attempts": attempts + 1, "result": "success"}
    
    async def timeout_task():
        """Task that times out."""
        print("⏰ Starting task that will timeout...")
        await asyncio.sleep(10)  # Longer than timeout
        return "This should not complete"
    
    orchestrator = RequestOrchestrator(max_concurrent_tasks=3)
    
    async with orchestrator.managed_execution():
        # Task with retry logic
        retry_task = TaskDefinition(
            id="retry_task",
            name="Task with Retries",
            handler=failing_task,
            args=(2,),  # Fail 2 times before succeeding
            max_retries=3,
            priority=TaskPriority.NORMAL
        )
        
        # Task with timeout
        timeout_task_def = TaskDefinition(
            id="timeout_task",
            name="Task with Timeout",
            handler=timeout_task,
            timeout=3.0,  # 3 second timeout
            priority=TaskPriority.NORMAL
        )
        
        # Submit tasks
        retry_id = await orchestrator.submit_task(retry_task)
        timeout_id = await orchestrator.submit_task(timeout_task_def)
        
        print(f"📝 Submitted retry task: {retry_id}")
        print(f"📝 Submitted timeout task: {timeout_id}")
        
        # Monitor task completion
        completed = set()
        
        while len(completed) < 2:
            await asyncio.sleep(0.5)
            
            for task_id in [retry_id, timeout_id]:
                if task_id not in completed:
                    status = await orchestrator.get_task_status(task_id)
                    if status and status["status"] in ["completed", "failed"]:
                        completed.add(task_id)
                        print(f"📊 Task {task_id}: {status['status']}")
                        if status["error"]:
                            print(f"   Error: {status['error']}")


async def demo_priority_scheduling():
    """Demonstrate priority-based task scheduling."""
    print("\n" + "="*60)
    print("🎯 DEMO: Priority-Based Scheduling")
    print("="*60)
    
    orchestrator = RequestOrchestrator(max_concurrent_tasks=2)  # Limited concurrency
    
    async with orchestrator.managed_execution():
        # Submit tasks with different priorities
        tasks = [
            ("urgent_task", "Urgent Task", TaskPriority.URGENT, 1.0),
            ("low_task_1", "Low Priority Task 1", TaskPriority.LOW, 2.0),
            ("high_task", "High Priority Task", TaskPriority.HIGH, 1.5),
            ("normal_task", "Normal Priority Task", TaskPriority.NORMAL, 1.0),
            ("low_task_2", "Low Priority Task 2", TaskPriority.LOW, 2.0),
        ]
        
        submitted_tasks = []
        
        for task_id, name, priority, duration in tasks:
            task = TaskDefinition(
                id=task_id,
                name=name,
                handler=simulate_slow_task,
                args=(duration,),
                priority=priority
            )
            
            await orchestrator.submit_task(task)
            submitted_tasks.append(task_id)
            print(f"📝 Submitted {priority.name} priority task: {name}")
        
        # Monitor execution order
        print("\n📊 Monitoring execution order (should prioritize URGENT > HIGH > NORMAL > LOW)...")
        
        completed_order = []
        
        while len(completed_order) < len(submitted_tasks):
            await asyncio.sleep(0.2)
            
            for task_id in submitted_tasks:
                if task_id not in completed_order:
                    status = await orchestrator.get_task_status(task_id)
                    if status and status["status"] == "completed":
                        completed_order.append(task_id)
                        task_info = next(t for t in tasks if t[0] == task_id)
                        print(f"✅ Completed: {task_info[1]} ({task_info[2].name} priority)")
        
        print(f"\n📋 Execution order: {' -> '.join(completed_order)}")


async def main():
    """Run all demos."""
    print("🎭 Request Orchestrator Demo Suite")
    print("=" * 60)
    
    demos = [
        demo_basic_task_submission,
        demo_workflow_execution,
        demo_batch_processing,
        demo_resource_management,
        demo_error_handling,
        demo_priority_scheduling
    ]
    
    for i, demo in enumerate(demos, 1):
        try:
            await demo()
            print(f"\n✅ Demo {i}/{len(demos)} completed successfully!")
        except Exception as e:
            print(f"\n❌ Demo {i}/{len(demos)} failed: {e}")
        
        if i < len(demos):
            print("\n" + "⏳ Waiting 2 seconds before next demo...")
            await asyncio.sleep(2)
    
    print("\n" + "="*60)
    print("🎉 All demos completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())