"""
Tests for workflow automation system.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Mock the database imports to avoid import errors
with patch.dict('sys.modules', {
    'scrollintel.core.database': Mock(),
    'scrollintel.models.workflow_models': Mock()
}):
    from scrollintel.engines.workflow_engine import (
        ZapierIntegration, PowerAutomateIntegration,
        AirflowIntegration, CustomIntegration, RetryManager
    )

# Import the enums directly
import sys
from enum import Enum

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingMode(str, Enum):
    BATCH = "batch"
    REAL_TIME = "real_time"

class IntegrationType(str, Enum):
    ZAPIER = "zapier"
    POWER_AUTOMATE = "power_automate"
    AIRFLOW = "airflow"
    CUSTOM = "custom"

class TriggerType(str, Enum):
    WEBHOOK = "webhook"
    MANUAL = "manual"

class TestWorkflowEngine:
    """Test workflow engine functionality."""
    
    @pytest.fixture
    def workflow_engine(self):
        return WorkflowEngine()
    
    @pytest.fixture
    def sample_workflow_data(self):
        return {
            "name": "Test Workflow",
            "description": "A test workflow",
            "integration_type": IntegrationType.CUSTOM,
            "trigger_config": {
                "type": TriggerType.MANUAL,
                "config": {}
            },
            "steps": [
                {
                    "name": "Step 1",
                    "type": "http_request",
                    "config": {
                        "url": "https://api.example.com/test",
                        "method": "POST"
                    }
                }
            ],
            "processing_mode": ProcessingMode.REAL_TIME
        }
    
    @patch('scrollintel.engines.workflow_engine.get_db_session')
    async def test_create_workflow(self, mock_db, workflow_engine, sample_workflow_data):
        """Test workflow creation."""
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        workflow_id = await workflow_engine.create_workflow(sample_workflow_data, "user123")
        
        assert workflow_id is not None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @patch('scrollintel.engines.workflow_engine.get_db_session')
    async def test_execute_workflow(self, mock_db, workflow_engine):
        """Test workflow execution."""
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Mock workflow
        mock_workflow = Mock()
        mock_workflow.id = "workflow123"
        mock_workflow.integration_type = IntegrationType.CUSTOM
        mock_workflow.processing_mode = ProcessingMode.REAL_TIME
        mock_workflow.steps = [
            {
                "name": "Test Step",
                "type": "http_request",
                "config": {"url": "https://api.example.com/test"}
            }
        ]
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_workflow
        
        with patch.object(workflow_engine, '_execute_realtime_workflow') as mock_execute:
            execution_id = await workflow_engine.execute_workflow("workflow123", {"test": "data"})
            
            assert execution_id is not None
            mock_execute.assert_called_once()
    
    @patch('scrollintel.engines.workflow_engine.get_db_session')
    async def test_execute_realtime_workflow(self, mock_db, workflow_engine):
        """Test real-time workflow execution."""
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        mock_workflow = Mock()
        mock_workflow.steps = [
            {
                "name": "Test Step",
                "type": "http_request",
                "config": {"url": "https://api.example.com/test"}
            }
        ]
        
        with patch.object(workflow_engine.integrations[IntegrationType.CUSTOM], 'execute_step') as mock_step:
            mock_step.return_value = {"result": "success"}
            
            await workflow_engine._execute_realtime_workflow("exec123", mock_workflow, {"input": "data"})
            
            mock_step.assert_called_once()

class TestZapierIntegration:
    """Test Zapier integration."""
    
    @pytest.fixture
    def zapier_integration(self):
        return ZapierIntegration()
    
    @patch('httpx.AsyncClient')
    async def test_execute_step(self, mock_client, zapier_integration):
        """Test Zapier step execution."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        step_config = {
            "config": {
                "webhook_url": "https://hooks.zapier.com/test"
            }
        }
        
        result = await zapier_integration.execute_step(step_config, {"test": "data"})
        
        assert result == {"status": "success"}
    
    async def test_execute_step_missing_url(self, zapier_integration):
        """Test Zapier step execution with missing URL."""
        step_config = {"config": {}}
        
        with pytest.raises(ValueError, match="Zapier webhook URL not configured"):
            await zapier_integration.execute_step(step_config, {"test": "data"})

class TestPowerAutomateIntegration:
    """Test Power Automate integration."""
    
    @pytest.fixture
    def power_automate_integration(self):
        return PowerAutomateIntegration()
    
    @patch('httpx.AsyncClient')
    async def test_execute_step(self, mock_client, power_automate_integration):
        """Test Power Automate step execution."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "completed"}
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        step_config = {
            "config": {
                "flow_url": "https://prod-123.westus.logic.azure.com/workflows/test"
            }
        }
        
        result = await power_automate_integration.execute_step(step_config, {"test": "data"})
        
        assert result == {"status": "completed"}
    
    async def test_execute_step_missing_url(self, power_automate_integration):
        """Test Power Automate step execution with missing URL."""
        step_config = {"config": {}}
        
        with pytest.raises(ValueError, match="Power Automate flow URL not configured"):
            await power_automate_integration.execute_step(step_config, {"test": "data"})

class TestAirflowIntegration:
    """Test Airflow integration."""
    
    @pytest.fixture
    def airflow_integration(self):
        return AirflowIntegration()
    
    @patch('httpx.AsyncClient')
    async def test_execute_step(self, mock_client, airflow_integration):
        """Test Airflow step execution."""
        # Mock DAG trigger response
        mock_trigger_response = Mock()
        mock_trigger_response.json.return_value = {"dag_run_id": "run123"}
        mock_trigger_response.raise_for_status.return_value = None
        
        # Mock DAG status response
        mock_status_response = Mock()
        mock_status_response.json.return_value = {"state": "success"}
        mock_status_response.raise_for_status.return_value = None
        
        mock_client_instance = Mock()
        mock_client_instance.post.return_value = mock_trigger_response
        mock_client_instance.get.return_value = mock_status_response
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        
        step_config = {
            "config": {
                "base_url": "http://airflow.example.com",
                "dag_id": "test_dag",
                "username": "admin",
                "password": "password"
            }
        }
        
        result = await airflow_integration.execute_step(step_config, {"test": "data"})
        
        assert result["status"] == "completed"
    
    async def test_execute_step_missing_config(self, airflow_integration):
        """Test Airflow step execution with missing configuration."""
        step_config = {"config": {}}
        
        with pytest.raises(ValueError, match="Airflow configuration incomplete"):
            await airflow_integration.execute_step(step_config, {"test": "data"})

class TestCustomIntegration:
    """Test custom integration."""
    
    @pytest.fixture
    def custom_integration(self):
        return CustomIntegration()
    
    @patch('httpx.AsyncClient')
    async def test_execute_http_request_step(self, mock_client, custom_integration):
        """Test HTTP request step execution."""
        mock_response = Mock()
        mock_response.json.return_value = {"result": "success"}
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__.return_value.request.return_value = mock_response
        
        step_config = {
            "type": "http_request",
            "config": {
                "url": "https://api.example.com/test",
                "method": "POST"
            }
        }
        
        result = await custom_integration.execute_step(step_config, {"test": "data"})
        
        assert result == {"result": "success"}
    
    async def test_execute_data_transformation_step(self, custom_integration):
        """Test data transformation step execution."""
        step_config = {
            "type": "data_transformation",
            "config": {
                "rules": [
                    {
                        "type": "map_field",
                        "source": "input_field",
                        "target": "output_field"
                    }
                ]
            }
        }
        
        input_data = {"input_field": "test_value"}
        result = await custom_integration.execute_step(step_config, input_data)
        
        assert result["output_field"] == "test_value"
    
    async def test_execute_condition_step(self, custom_integration):
        """Test condition step execution."""
        step_config = {
            "type": "condition",
            "config": {
                "condition": "data.get('value', 0) > 10"
            }
        }
        
        # Test condition met
        input_data = {"value": 15}
        result = await custom_integration.execute_step(step_config, input_data)
        assert result["condition_met"] is True
        
        # Test condition not met
        input_data = {"value": 5}
        result = await custom_integration.execute_step(step_config, input_data)
        assert result["condition_met"] is False
    
    async def test_execute_unknown_step_type(self, custom_integration):
        """Test execution with unknown step type."""
        step_config = {
            "type": "unknown_type",
            "config": {}
        }
        
        with pytest.raises(ValueError, match="Unknown custom step type"):
            await custom_integration.execute_step(step_config, {"test": "data"})

class TestWebhookManager:
    """Test webhook manager."""
    
    @pytest.fixture
    def webhook_manager(self):
        return WebhookManager()
    
    @patch('scrollintel.engines.workflow_engine.get_db_session')
    async def test_create_webhook(self, mock_db, webhook_manager):
        """Test webhook creation."""
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        webhook_config = {
            "url": "https://api.example.com/webhook",
            "method": "POST",
            "headers": {"Content-Type": "application/json"}
        }
        
        webhook_id = await webhook_manager.create_webhook("workflow123", webhook_config)
        
        assert webhook_id is not None
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
    
    @patch('scrollintel.engines.workflow_engine.get_db_session')
    @patch('scrollintel.engines.workflow_engine.WorkflowEngine')
    async def test_handle_webhook_callback(self, mock_engine_class, mock_db, webhook_manager):
        """Test webhook callback handling."""
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Mock webhook config
        mock_webhook = Mock()
        mock_webhook.workflow_id = "workflow123"
        mock_webhook.is_active = True
        mock_session.query.return_value.filter.return_value.first.return_value = mock_webhook
        
        # Mock workflow engine
        mock_engine = Mock()
        mock_engine.execute_workflow.return_value = "execution123"
        mock_engine_class.return_value = mock_engine
        
        result = await webhook_manager.handle_webhook_callback("webhook123", {"test": "data"})
        
        assert result["execution_id"] == "execution123"
        assert result["status"] == "triggered"

class TestRetryManager:
    """Test retry manager."""
    
    @pytest.fixture
    def retry_manager(self):
        return RetryManager()
    
    async def test_execute_with_retry_success(self, retry_manager):
        """Test successful execution without retries."""
        async def mock_func():
            return "success"
        
        result = await retry_manager.execute_with_retry(mock_func)
        assert result == "success"
    
    async def test_execute_with_retry_failure_then_success(self, retry_manager):
        """Test execution that fails then succeeds."""
        call_count = 0
        
        async def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await retry_manager.execute_with_retry(mock_func)
        assert result == "success"
        assert call_count == 2
    
    async def test_execute_with_retry_max_retries_exceeded(self, retry_manager):
        """Test execution that exceeds max retries."""
        async def mock_func():
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            await retry_manager.execute_with_retry(mock_func)

class TestWorkflowIntegration:
    """Integration tests for workflow system."""
    
    @pytest.fixture
    def workflow_engine(self):
        return WorkflowEngine()
    
    @patch('scrollintel.engines.workflow_engine.get_db_session')
    async def test_end_to_end_workflow_execution(self, mock_db, workflow_engine):
        """Test complete workflow execution flow."""
        mock_session = Mock()
        mock_db.return_value.__enter__.return_value = mock_session
        
        # Create workflow
        workflow_data = {
            "name": "E2E Test Workflow",
            "integration_type": IntegrationType.CUSTOM,
            "trigger_config": {
                "type": TriggerType.MANUAL,
                "config": {}
            },
            "steps": [
                {
                    "name": "Transform Data",
                    "type": "data_transformation",
                    "config": {
                        "rules": [
                            {
                                "type": "map_field",
                                "source": "input",
                                "target": "output"
                            }
                        ]
                    }
                }
            ]
        }
        
        workflow_id = await workflow_engine.create_workflow(workflow_data, "user123")
        
        # Mock workflow for execution
        mock_workflow = Mock()
        mock_workflow.id = workflow_id
        mock_workflow.integration_type = IntegrationType.CUSTOM
        mock_workflow.processing_mode = ProcessingMode.REAL_TIME
        mock_workflow.steps = workflow_data["steps"]
        
        mock_session.query.return_value.filter.return_value.first.return_value = mock_workflow
        
        # Execute workflow
        execution_id = await workflow_engine.execute_workflow(workflow_id, {"input": "test_data"})
        
        assert workflow_id is not None
        assert execution_id is not None

if __name__ == "__main__":
    pytest.main([__file__])