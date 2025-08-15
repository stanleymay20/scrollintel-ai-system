"""
Integration tests for Request Orchestrator API routes.

Tests the complete API functionality including task submission,
workflow management, and status tracking.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi import FastAPI
from unittest.mock import AsyncMock, patch
import json

from scrollintel.api.routes.request_orchestrator_routes import router, orchestrator
from scrollintel.core.request_orchestrator import TaskStatus, TaskPriority


@pytest.fixture
def app():
    """Create FastAPI app for testing."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def mock_orchestrator():
    """Create mock orchestrator for testing."""
    mock = AsyncMock()
    
    # Mock task submission
    mock.submit_task.return_value = "test_task_id"
    mock.submit_workflow.return_value = "test_workflow_id"
    
    # Mock status responses
    mock.get_task_status.return_value = {
        "id": "test_task_id",
        "name": "Test Task",
        "status": "completed",
        "progress": 100.0,
        "created_at": "2024-01-01T00:00:00",
        "started_at": "2024-01-01T00:00:01",
        "completed_at": "2024-01-01T00:00:05",
        "duration": "0:00:04",
        "error": None,
        "allocated_resources": {"GPU": 1.0}
    }
    
    mock.get_workflow_status.return_value = {
        "progress": 100.0,
        "status": "completed",
        "total_tasks": 2,
        "completed_tasks": 2,
        "failed_tasks": 0,
        "running_tasks": 0,
        "tasks": {
            "task1": {"status": "completed", "progress": 100.0, "error": None},
            "task2": {"status": "completed", "progress": 100.0, "error": None}
        }
    }
    
    mock.get_system_status.return_value = {
        "running_tasks": 1,
        "max_concurrent_tasks": 10,
        "queue_status": {"LOW": 0, "NORMAL": 2, "HIGH": 1, "URGENT": 0},
        "resource_utilization": {"GPU": "25.0%", "CPU": "15.0%", "MEMORY": "30.0%"},
        "shutdown": False
    }
    
    mock.cancel_task.return_value = True
    
    return mock


class TestTaskSubmission:
    """Test task submission endpoints."""
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_submit_single_task(self, mock_orch, client):
        """Test submitting a single task."""
        mock_orch.submit_task.return_value = asyncio.Future()
        mock_orch.submit_task.return_value.set_result("test_task_id")
        
        task_data = {
            "name": "Test Image Generation",
            "handler_name": "image_generation",
            "args": ["A beautiful sunset"],
            "kwargs": {"model": "dalle3"},
            "priority": "HIGH",
            "resource_requirements": [
                {"resource_type": "gpu", "amount": 1.0},
                {"resource_type": "memory", "amount": 4.0}
            ],
            "timeout": 300.0,
            "max_retries": 3
        }
        
        response = client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test_task_id"
        assert data["status"] == "submitted"
    
    def test_submit_task_invalid_handler(self, client):
        """Test submitting task with invalid handler."""
        task_data = {
            "name": "Test Task",
            "handler_name": "nonexistent_handler",
            "priority": "NORMAL"
        }
        
        response = client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        assert response.status_code == 400
        assert "Unknown handler" in response.json()["detail"]
    
    def test_submit_task_invalid_priority(self, client):
        """Test submitting task with invalid priority."""
        task_data = {
            "name": "Test Task",
            "handler_name": "image_generation",
            "priority": "INVALID_PRIORITY"
        }
        
        response = client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        assert response.status_code == 400
        assert "Invalid priority" in response.json()["detail"]
    
    def test_submit_task_invalid_resource_type(self, client):
        """Test submitting task with invalid resource type."""
        task_data = {
            "name": "Test Task",
            "handler_name": "image_generation",
            "resource_requirements": [
                {"resource_type": "invalid_resource", "amount": 1.0}
            ]
        }
        
        response = client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        assert response.status_code == 400
        assert "Invalid resource type" in response.json()["detail"]


class TestWorkflowSubmission:
    """Test workflow submission endpoints."""
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_submit_workflow(self, mock_orch, client):
        """Test submitting a workflow."""
        mock_orch.submit_workflow.return_value = asyncio.Future()
        mock_orch.submit_workflow.return_value.set_result("test_workflow_id")
        
        workflow_data = {
            "name": "Image Processing Pipeline",
            "description": "Generate and enhance images",
            "tasks": [
                {
                    "name": "Generate Image",
                    "handler_name": "image_generation",
                    "args": ["A cat"],
                    "priority": "HIGH",
                    "resource_requirements": [{"resource_type": "gpu", "amount": 1.0}]
                },
                {
                    "name": "Enhance Image",
                    "handler_name": "image_enhancement",
                    "args": ["generated_image.jpg"],
                    "dependencies": ["Generate Image"],
                    "priority": "NORMAL"
                }
            ]
        }
        
        response = client.post("/api/v1/orchestrator/workflows", json=workflow_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "test_workflow_id"
        assert data["status"] == "submitted"
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.create_image_generation_workflow')
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_submit_batch_image_generation(self, mock_orch, mock_create_workflow, client):
        """Test submitting batch image generation workflow."""
        # Mock workflow creation
        mock_workflow = AsyncMock()
        mock_workflow.id = "batch_workflow_id"
        mock_create_workflow.return_value = asyncio.Future()
        mock_create_workflow.return_value.set_result(mock_workflow)
        
        # Mock orchestrator
        mock_orch.submit_workflow.return_value = asyncio.Future()
        mock_orch.submit_workflow.return_value.set_result("batch_workflow_id")
        
        request_data = {
            "prompts": ["A cat", "A dog", "A bird"],
            "model_preferences": ["dalle3", "stable_diffusion", "midjourney"],
            "priority": "HIGH"
        }
        
        response = client.post("/api/v1/orchestrator/workflows/batch-images", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "batch_workflow_id"
        assert data["image_count"] == 3
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.create_video_generation_workflow')
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_submit_video_generation(self, mock_orch, mock_create_workflow, client):
        """Test submitting video generation workflow."""
        # Mock workflow creation
        mock_workflow = AsyncMock()
        mock_workflow.id = "video_workflow_id"
        mock_create_workflow.return_value = asyncio.Future()
        mock_create_workflow.return_value.set_result(mock_workflow)
        
        # Mock orchestrator
        mock_orch.submit_workflow.return_value = asyncio.Future()
        mock_orch.submit_workflow.return_value.set_result("video_workflow_id")
        
        request_data = {
            "prompt": "A flying car in a futuristic city",
            "duration": 10.0,
            "resolution": [1920, 1080],
            "priority": "URGENT"
        }
        
        response = client.post("/api/v1/orchestrator/workflows/video-generation", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["workflow_id"] == "video_workflow_id"


class TestStatusEndpoints:
    """Test status and monitoring endpoints."""
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_get_task_status(self, mock_orch, client):
        """Test getting task status."""
        mock_orch.get_task_status.return_value = asyncio.Future()
        mock_orch.get_task_status.return_value.set_result({
            "id": "test_task_id",
            "name": "Test Task",
            "status": "running",
            "progress": 75.0,
            "created_at": "2024-01-01T00:00:00",
            "started_at": "2024-01-01T00:00:01",
            "completed_at": None,
            "duration": None,
            "error": None,
            "allocated_resources": {"GPU": 1.0, "MEMORY": 4.0}
        })
        
        response = client.get("/api/v1/orchestrator/tasks/test_task_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test_task_id"
        assert data["status"] == "running"
        assert data["progress"] == 75.0
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_get_task_status_not_found(self, mock_orch, client):
        """Test getting status for non-existent task."""
        mock_orch.get_task_status.return_value = asyncio.Future()
        mock_orch.get_task_status.return_value.set_result(None)
        
        response = client.get("/api/v1/orchestrator/tasks/nonexistent")
        
        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_get_workflow_status(self, mock_orch, client):
        """Test getting workflow status."""
        mock_orch.get_workflow_status.return_value = asyncio.Future()
        mock_orch.get_workflow_status.return_value.set_result({
            "progress": 50.0,
            "status": "running",
            "total_tasks": 4,
            "completed_tasks": 2,
            "failed_tasks": 0,
            "running_tasks": 2,
            "tasks": {
                "task1": {"status": "completed", "progress": 100.0, "error": None},
                "task2": {"status": "completed", "progress": 100.0, "error": None},
                "task3": {"status": "running", "progress": 50.0, "error": None},
                "task4": {"status": "pending", "progress": 0.0, "error": None}
            }
        })
        
        response = client.get("/api/v1/orchestrator/workflows/test_workflow_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["progress"] == 50.0
        assert data["status"] == "running"
        assert data["total_tasks"] == 4
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_get_system_status(self, mock_orch, client):
        """Test getting system status."""
        mock_orch.get_system_status.return_value = asyncio.Future()
        mock_orch.get_system_status.return_value.set_result({
            "running_tasks": 3,
            "max_concurrent_tasks": 10,
            "queue_status": {"LOW": 1, "NORMAL": 5, "HIGH": 2, "URGENT": 0},
            "resource_utilization": {
                "GPU": "60.0%",
                "CPU": "45.0%",
                "MEMORY": "70.0%",
                "STORAGE": "20.0%",
                "NETWORK": "10.0%"
            },
            "shutdown": False
        })
        
        response = client.get("/api/v1/orchestrator/system/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["running_tasks"] == 3
        assert data["max_concurrent_tasks"] == 10
        assert "queue_status" in data
        assert "resource_utilization" in data


class TestTaskManagement:
    """Test task management endpoints."""
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_cancel_task(self, mock_orch, client):
        """Test cancelling a task."""
        mock_orch.cancel_task.return_value = asyncio.Future()
        mock_orch.cancel_task.return_value.set_result(True)
        
        response = client.delete("/api/v1/orchestrator/tasks/test_task_id")
        
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test_task_id"
        assert data["status"] == "cancelled"
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_cancel_task_not_found(self, mock_orch, client):
        """Test cancelling non-existent task."""
        mock_orch.cancel_task.return_value = asyncio.Future()
        mock_orch.cancel_task.return_value.set_result(False)
        
        response = client.delete("/api/v1/orchestrator/tasks/nonexistent")
        
        assert response.status_code == 404
        assert "Task not found" in response.json()["detail"]


class TestUtilityEndpoints:
    """Test utility endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/api/v1/orchestrator/system/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_list_handlers(self, client):
        """Test listing available handlers."""
        response = client.get("/api/v1/orchestrator/handlers")
        
        assert response.status_code == 200
        data = response.json()
        assert "handlers" in data
        assert isinstance(data["handlers"], list)
        assert "image_generation" in data["handlers"]
        assert "video_generation" in data["handlers"]


class TestValidation:
    """Test input validation."""
    
    def test_task_validation_missing_fields(self, client):
        """Test task validation with missing required fields."""
        task_data = {
            "name": "Test Task"
            # Missing handler_name
        }
        
        response = client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_task_validation_invalid_timeout(self, client):
        """Test task validation with invalid timeout."""
        task_data = {
            "name": "Test Task",
            "handler_name": "image_generation",
            "timeout": -5.0  # Negative timeout
        }
        
        response = client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        assert response.status_code == 422
    
    def test_resource_validation_invalid_amount(self, client):
        """Test resource validation with invalid amount."""
        task_data = {
            "name": "Test Task",
            "handler_name": "image_generation",
            "resource_requirements": [
                {"resource_type": "gpu", "amount": -1.0}  # Negative amount
            ]
        }
        
        response = client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        assert response.status_code == 422
    
    def test_batch_image_validation_empty_prompts(self, client):
        """Test batch image validation with empty prompts."""
        request_data = {
            "prompts": []  # Empty list
        }
        
        response = client.post("/api/v1/orchestrator/workflows/batch-images", json=request_data)
        
        assert response.status_code == 422
    
    def test_video_validation_invalid_duration(self, client):
        """Test video validation with invalid duration."""
        request_data = {
            "prompt": "Test video",
            "duration": 0.0  # Zero duration
        }
        
        response = client.post("/api/v1/orchestrator/workflows/video-generation", json=request_data)
        
        assert response.status_code == 422


class TestErrorHandling:
    """Test error handling in API routes."""
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_internal_error_handling(self, mock_orch, client):
        """Test handling of internal errors."""
        mock_orch.submit_task.side_effect = Exception("Internal error")
        
        task_data = {
            "name": "Test Task",
            "handler_name": "image_generation"
        }
        
        response = client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        assert response.status_code == 500
        assert "Internal error" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_async_error_handling(self, mock_orch, client):
        """Test handling of async errors."""
        future = asyncio.Future()
        future.set_exception(RuntimeError("Async error"))
        mock_orch.get_task_status.return_value = future
        
        response = client.get("/api/v1/orchestrator/tasks/test_task")
        
        assert response.status_code == 500


@pytest.mark.asyncio
async def test_orchestrator_lifecycle_events():
    """Test orchestrator startup and shutdown events."""
    from scrollintel.api.routes.request_orchestrator_routes import startup_orchestrator, shutdown_orchestrator
    
    # Test startup
    with patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator') as mock_orch:
        mock_orch.start.return_value = asyncio.Future()
        mock_orch.start.return_value.set_result(None)
        
        await startup_orchestrator()
        mock_orch.start.assert_called_once()
    
    # Test shutdown
    with patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator') as mock_orch:
        mock_orch.stop.return_value = asyncio.Future()
        mock_orch.stop.return_value.set_result(None)
        
        await shutdown_orchestrator()
        mock_orch.stop.assert_called_once()


class TestConcurrentRequests:
    """Test handling of concurrent requests."""
    
    @patch('scrollintel.api.routes.request_orchestrator_routes.orchestrator')
    def test_concurrent_task_submissions(self, mock_orch, client):
        """Test submitting multiple tasks concurrently."""
        import threading
        
        # Mock orchestrator to return different task IDs
        task_ids = [f"task_{i}" for i in range(5)]
        mock_orch.submit_task.side_effect = [
            asyncio.Future() for _ in task_ids
        ]
        for i, future in enumerate(mock_orch.submit_task.side_effect):
            future.set_result(task_ids[i])
        
        def submit_task(task_id):
            task_data = {
                "name": f"Task {task_id}",
                "handler_name": "image_generation",
                "args": [f"Prompt {task_id}"]
            }
            return client.post("/api/v1/orchestrator/tasks", json=task_data)
        
        # Submit tasks concurrently
        threads = []
        results = []
        
        for i in range(5):
            thread = threading.Thread(target=lambda i=i: results.append(submit_task(i)))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Check all submissions succeeded
        assert len(results) == 5
        for response in results:
            assert response.status_code == 200