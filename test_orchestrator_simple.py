"""
Simple test for Pipeline Orchestrator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scrollintel.core.pipeline_orchestrator import PipelineOrchestrator, ScheduleConfig, ScheduleType, ResourceType
import time
import uuid

def test_orchestrator():
    print('Testing Pipeline Orchestrator...')

    # Create orchestrator
    orchestrator = PipelineOrchestrator(max_concurrent_executions=2)
    orchestrator.start()

    try:
        # Test basic execution
        pipeline_id = str(uuid.uuid4())
        execution_id = orchestrator.execute_pipeline_now(
            pipeline_id=pipeline_id,
            priority=5,
            resource_requirements={
                ResourceType.CPU: 1.0,
                ResourceType.MEMORY: 2.0
            }
        )
        
        print(f'Scheduled execution: {execution_id[:8]}...')
        
        # Check status
        status = orchestrator.get_execution_status(execution_id)
        print(f'Initial status: {status["status"] if status else "Not found"}')
        
        # Wait and check again
        time.sleep(3)
        status = orchestrator.get_execution_status(execution_id)
        print(f'After 3s: {status["status"] if status else "Not found"}')
        
        # Get metrics
        metrics = orchestrator.get_orchestrator_metrics()
        print(f'Metrics: {metrics["total_executions"]} total, {metrics["successful_executions"]} successful')
        
        print('✓ Pipeline Orchestrator test completed successfully!')
        
    finally:
        orchestrator.stop()

if __name__ == "__main__":
    test_orchestrator()