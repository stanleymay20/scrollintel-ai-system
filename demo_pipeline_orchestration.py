"""
Demo: Pipeline Orchestration Engine
Demonstrates the complete pipeline orchestration system with scheduling, dependencies, and resource management.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.core.pipeline_orchestrator import (
    PipelineOrchestrator, ScheduleConfig, ScheduleType, ResourceType
)
from scrollintel.models.pipeline_models import Pipeline, PipelineNode, NodeType


def print_banner(title: str):
    """Print a formatted banner"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_status(message: str, status: str = "INFO"):
    """Print a status message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{status}] {message}")


def demo_basic_orchestration():
    """Demonstrate basic pipeline orchestration"""
    print_banner("Basic Pipeline Orchestration Demo")
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(max_concurrent_executions=3)
    orchestrator.start()
    
    try:
        # Create sample pipeline IDs
        pipeline_ids = [str(uuid.uuid4()) for _ in range(3)]
        
        print_status("Starting pipeline orchestration demo...")
        print_status(f"Created {len(pipeline_ids)} sample pipelines")
        
        # Execute pipelines with different priorities
        execution_ids = []
        priorities = [5, 10, 1]  # Medium, High, Low
        
        for i, (pipeline_id, priority) in enumerate(zip(pipeline_ids, priorities)):
            execution_id = orchestrator.execute_pipeline_now(
                pipeline_id=pipeline_id,
                priority=priority,
                resource_requirements={
                    ResourceType.CPU: 1.0 + i * 0.5,
                    ResourceType.MEMORY: 2.0 + i * 1.0
                }
            )
            execution_ids.append(execution_id)
            print_status(f"Scheduled pipeline {i+1} with priority {priority} (ID: {execution_id[:8]}...)")
        
        # Monitor executions
        print_status("Monitoring executions...")
        for _ in range(10):
            time.sleep(1)
            
            # Check status of all executions
            running_count = 0
            completed_count = 0
            
            for execution_id in execution_ids:
                status = orchestrator.get_execution_status(execution_id)
                if status:
                    if status['status'] == 'running':
                        running_count += 1
                    elif status['status'] == 'completed':
                        completed_count += 1
            
            print_status(f"Running: {running_count}, Completed: {completed_count}")
            
            if completed_count == len(execution_ids):
                break
        
        # Show final metrics
        metrics = orchestrator.get_orchestrator_metrics()
        print_status("Final Metrics:")
        print(f"  Total Executions: {metrics['total_executions']}")
        print(f"  Successful: {metrics['successful_executions']}")
        print(f"  Failed: {metrics['failed_executions']}")
        
    finally:
        orchestrator.stop()
        print_status("Orchestrator stopped")


def demo_scheduling_and_dependencies():
    """Demonstrate pipeline scheduling with dependencies"""
    print_banner("Pipeline Scheduling and Dependencies Demo")
    
    orchestrator = PipelineOrchestrator(max_concurrent_executions=2)
    orchestrator.start()
    
    try:
        # Create pipeline chain: A -> B -> C
        pipeline_a = str(uuid.uuid4())
        pipeline_b = str(uuid.uuid4())
        pipeline_c = str(uuid.uuid4())
        
        print_status("Creating pipeline dependency chain: A -> B -> C")
        
        # Schedule pipeline A (no dependencies)
        schedule_config = ScheduleConfig(schedule_type=ScheduleType.ONCE)
        execution_a = orchestrator.schedule_pipeline(
            pipeline_id=pipeline_a,
            schedule_config=schedule_config,
            priority=5
        )
        print_status(f"Scheduled Pipeline A (ID: {execution_a[:8]}...)")
        
        # Schedule pipeline B (depends on A)
        execution_b = orchestrator.schedule_pipeline(
            pipeline_id=pipeline_b,
            schedule_config=schedule_config,
            priority=5,
            dependencies=[execution_a]
        )
        print_status(f"Scheduled Pipeline B (depends on A) (ID: {execution_b[:8]}...)")
        
        # Schedule pipeline C (depends on B)
        execution_c = orchestrator.schedule_pipeline(
            pipeline_id=pipeline_c,
            schedule_config=schedule_config,
            priority=5,
            dependencies=[execution_b]
        )
        print_status(f"Scheduled Pipeline C (depends on B) (ID: {execution_c[:8]}...)")
        
        # Monitor dependency execution
        executions = [
            (execution_a, "Pipeline A"),
            (execution_b, "Pipeline B"),
            (execution_c, "Pipeline C")
        ]
        
        print_status("Monitoring dependency execution...")
        for _ in range(15):
            time.sleep(1)
            
            all_completed = True
            for execution_id, name in executions:
                status = orchestrator.get_execution_status(execution_id)
                if status:
                    print_status(f"{name}: {status['status']}")
                    if status['status'] not in ['completed', 'failed']:
                        all_completed = False
                else:
                    all_completed = False
            
            print_status("---")
            
            if all_completed:
                break
        
        print_status("Dependency chain execution completed!")
        
    finally:
        orchestrator.stop()


def demo_resource_management():
    """Demonstrate resource allocation and management"""
    print_banner("Resource Management Demo")
    
    orchestrator = PipelineOrchestrator(max_concurrent_executions=5)
    
    # Set limited resources for demonstration
    orchestrator.resource_manager.total_cpu = 6.0
    orchestrator.resource_manager.total_memory = 12.0
    
    orchestrator.start()
    
    try:
        print_status("Resource limits set:")
        print(f"  CPU: {orchestrator.resource_manager.total_cpu} cores")
        print(f"  Memory: {orchestrator.resource_manager.total_memory} GB")
        
        # Schedule multiple resource-intensive pipelines
        pipeline_ids = [str(uuid.uuid4()) for _ in range(5)]
        execution_ids = []
        
        resource_requirements = [
            {ResourceType.CPU: 2.0, ResourceType.MEMORY: 4.0},
            {ResourceType.CPU: 3.0, ResourceType.MEMORY: 6.0},
            {ResourceType.CPU: 1.0, ResourceType.MEMORY: 2.0},
            {ResourceType.CPU: 2.5, ResourceType.MEMORY: 5.0},
            {ResourceType.CPU: 1.5, ResourceType.MEMORY: 3.0}
        ]
        
        for i, (pipeline_id, resources) in enumerate(zip(pipeline_ids, resource_requirements)):
            execution_id = orchestrator.execute_pipeline_now(
                pipeline_id=pipeline_id,
                priority=5,
                resource_requirements=resources
            )
            execution_ids.append(execution_id)
            print_status(f"Scheduled Pipeline {i+1} requiring {resources[ResourceType.CPU]} CPU, {resources[ResourceType.MEMORY]} GB RAM")
        
        # Monitor resource utilization
        print_status("Monitoring resource utilization...")
        for _ in range(12):
            time.sleep(1)
            
            utilization = orchestrator.get_resource_utilization()
            metrics = orchestrator.get_orchestrator_metrics()
            
            print_status(f"CPU: {utilization['cpu_utilization']:.1f}%, "
                        f"Memory: {utilization['memory_utilization']:.1f}%, "
                        f"Running: {metrics['running_executions']}, "
                        f"Queue: {metrics['queue_size']}")
        
        print_status("Resource management demo completed!")
        
    finally:
        orchestrator.stop()


def demo_retry_mechanism():
    """Demonstrate retry mechanism with exponential backoff"""
    print_banner("Retry Mechanism Demo")
    
    orchestrator = PipelineOrchestrator(max_concurrent_executions=2)
    orchestrator.start()
    
    try:
        # Mock failing pipeline execution
        original_execute = orchestrator._execute_pipeline
        failure_count = 0
        
        def mock_failing_execute(context):
            nonlocal failure_count
            failure_count += 1
            
            if failure_count <= 2:  # Fail first 2 attempts
                print_status(f"Execution attempt {failure_count} failed (simulated)", "ERROR")
                raise Exception(f"Simulated failure #{failure_count}")
            else:
                print_status(f"Execution attempt {failure_count} succeeded", "SUCCESS")
                return {"success": True, "execution_time": 2}
        
        orchestrator._execute_pipeline = mock_failing_execute
        
        pipeline_id = str(uuid.uuid4())
        print_status("Scheduling pipeline with simulated failures...")
        
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=10
        )
        
        print_status(f"Pipeline scheduled (ID: {execution_id[:8]}...)")
        print_status("Monitoring retry attempts...")
        
        # Monitor execution with retries
        for _ in range(20):
            time.sleep(1)
            
            status = orchestrator.get_execution_status(execution_id)
            if status:
                print_status(f"Status: {status['status']}, Retry count: {status['retry_count']}")
                
                if status['status'] in ['completed', 'failed']:
                    break
        
        # Show retry metrics
        metrics = orchestrator.get_orchestrator_metrics()
        print_status(f"Total retry attempts: {metrics['retry_executions']}")
        
        # Restore original execution function
        orchestrator._execute_pipeline = original_execute
        
    finally:
        orchestrator.stop()


def demo_execution_control():
    """Demonstrate execution control (pause, resume, cancel)"""
    print_banner("Execution Control Demo")
    
    orchestrator = PipelineOrchestrator(max_concurrent_executions=2)
    orchestrator.start()
    
    try:
        # Mock long-running execution
        original_execute = orchestrator._execute_pipeline
        
        def mock_long_execute(context):
            print_status("Starting long-running execution...")
            for i in range(10):
                time.sleep(1)
                if context.status.value == 'paused':
                    print_status("Execution paused, waiting...")
                    while context.status.value == 'paused':
                        time.sleep(0.5)
                    print_status("Execution resumed!")
                
                if context.status.value == 'cancelled':
                    print_status("Execution cancelled!")
                    return {"success": False, "cancelled": True}
            
            print_status("Long execution completed!")
            return {"success": True, "execution_time": 10}
        
        orchestrator._execute_pipeline = mock_long_execute
        
        pipeline_id = str(uuid.uuid4())
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=10
        )
        
        print_status(f"Started long-running pipeline (ID: {execution_id[:8]}...)")
        
        # Wait for execution to start
        time.sleep(3)
        
        # Pause execution
        print_status("Pausing execution...")
        success = orchestrator.pause_execution(execution_id)
        if success:
            print_status("Execution paused successfully")
        
        time.sleep(3)
        
        # Resume execution
        print_status("Resuming execution...")
        success = orchestrator.resume_execution(execution_id)
        if success:
            print_status("Execution resumed successfully")
        
        # Wait for completion
        time.sleep(8)
        
        status = orchestrator.get_execution_status(execution_id)
        if status:
            print_status(f"Final status: {status['status']}")
        
        # Restore original execution function
        orchestrator._execute_pipeline = original_execute
        
    finally:
        orchestrator.stop()


def demo_metrics_and_monitoring():
    """Demonstrate metrics collection and monitoring"""
    print_banner("Metrics and Monitoring Demo")
    
    orchestrator = PipelineOrchestrator(max_concurrent_executions=3)
    orchestrator.start()
    
    try:
        # Execute multiple pipelines to generate metrics
        pipeline_ids = [str(uuid.uuid4()) for _ in range(5)]
        
        print_status("Executing multiple pipelines to generate metrics...")
        
        for i, pipeline_id in enumerate(pipeline_ids):
            orchestrator.execute_pipeline_now(
                pipeline_id=pipeline_id,
                priority=5 + (i % 3),  # Varying priorities
                resource_requirements={
                    ResourceType.CPU: 1.0 + (i * 0.3),
                    ResourceType.MEMORY: 2.0 + (i * 0.5)
                }
            )
        
        # Monitor and display metrics
        print_status("Collecting metrics...")
        for _ in range(10):
            time.sleep(1)
            
            metrics = orchestrator.get_orchestrator_metrics()
            utilization = orchestrator.get_resource_utilization()
            
            print_status("Current Metrics:")
            print(f"  Queue Size: {metrics['queue_size']}")
            print(f"  Running Executions: {metrics['running_executions']}")
            print(f"  Completed Executions: {metrics['completed_executions']}")
            print(f"  Total Executions: {metrics['total_executions']}")
            print(f"  Successful: {metrics['successful_executions']}")
            print(f"  Failed: {metrics['failed_executions']}")
            print(f"  Retries: {metrics['retry_executions']}")
            print(f"  CPU Utilization: {utilization['cpu_utilization']:.1f}%")
            print(f"  Memory Utilization: {utilization['memory_utilization']:.1f}%")
            print("---")
        
        print_status("Metrics collection completed!")
        
    finally:
        orchestrator.stop()


def main():
    """Run all orchestration demos"""
    print_banner("Pipeline Orchestration Engine Demo Suite")
    print_status("Starting comprehensive pipeline orchestration demonstration...")
    
    try:
        # Run all demos
        demo_basic_orchestration()
        time.sleep(2)
        
        demo_scheduling_and_dependencies()
        time.sleep(2)
        
        demo_resource_management()
        time.sleep(2)
        
        demo_retry_mechanism()
        time.sleep(2)
        
        demo_execution_control()
        time.sleep(2)
        
        demo_metrics_and_monitoring()
        
        print_banner("Demo Suite Completed Successfully!")
        print_status("All pipeline orchestration features demonstrated successfully!")
        
    except KeyboardInterrupt:
        print_status("Demo interrupted by user", "WARNING")
    except Exception as e:
        print_status(f"Demo failed with error: {e}", "ERROR")
        raise


if __name__ == "__main__":
    main()