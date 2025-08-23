"""
Simple tests for workflow automation system components.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

pytestmark = pytest.mark.asyncio

class TestZapierIntegration:
    """Test Zapier integration."""
    
    def setup_method(self):
        # Import here to avoid database dependency issues
        from scrollintel.engines.workflow_engine import ZapierIntegration
        self.zapier_integration = ZapierIntegration()
    
    @patch('httpx.AsyncClient')
    async def test_execute_step_success(self, mock_client):
        """Test successful Zapier step execution."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "success", "data": "processed"}
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        step_config = {
            "config": {
                "webhook_url": "https://hooks.zapier.com/test"
            }
        }
        
        result = await self.zapier_integration.execute_step(step_config, {"test": "data"})
        
        assert result == {"status": "success", "data": "processed"}
    
    async def test_execute_step_missing_url(self):
        """Test Zapier step execution with missing URL."""
        step_config = {"config": {}}
        
        with pytest.raises(ValueError, match="Zapier webhook URL not configured"):
            await self.zapier_integration.execute_step(step_config, {"test": "data"})

class TestPowerAutomateIntegration:
    """Test Power Automate integration."""
    
    def setup_method(self):
        from scrollintel.engines.workflow_engine import PowerAutomateIntegration
        self.power_automate_integration = PowerAutomateIntegration()
    
    @patch('httpx.AsyncClient')
    async def test_execute_step_success(self, mock_client):
        """Test successful Power Automate step execution."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "completed", "result": "flow_executed"}
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_response
        
        step_config = {
            "config": {
                "flow_url": "https://prod-123.westus.logic.azure.com/workflows/test"
            }
        }
        
        result = await self.power_automate_integration.execute_step(step_config, {"test": "data"})
        
        assert result == {"status": "completed", "result": "flow_executed"}
    
    async def test_execute_step_missing_url(self):
        """Test Power Automate step execution with missing URL."""
        step_config = {"config": {}}
        
        with pytest.raises(ValueError, match="Power Automate flow URL not configured"):
            await self.power_automate_integration.execute_step(step_config, {"test": "data"})

class TestCustomIntegration:
    """Test custom integration."""
    
    def setup_method(self):
        from scrollintel.engines.workflow_engine import CustomIntegration
        self.custom_integration = CustomIntegration()
    
    @patch('httpx.AsyncClient')
    async def test_execute_http_request_step(self, mock_client):
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
        
        result = await self.custom_integration.execute_step(step_config, {"test": "data"})
        
        assert result == {"result": "success"}
    
    async def test_execute_data_transformation_step(self):
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
        result = await self.custom_integration.execute_step(step_config, input_data)
        
        assert result["output_field"] == "test_value"
    
    async def test_execute_condition_step_true(self):
        """Test condition step execution - condition met."""
        step_config = {
            "type": "condition",
            "config": {
                "condition": "data.get('value', 0) > 10"
            }
        }
        
        input_data = {"value": 15}
        result = await self.custom_integration.execute_step(step_config, input_data)
        assert result["condition_met"] is True
    
    async def test_execute_condition_step_false(self):
        """Test condition step execution - condition not met."""
        step_config = {
            "type": "condition",
            "config": {
                "condition": "data.get('value', 0) > 10"
            }
        }
        
        input_data = {"value": 5}
        result = await self.custom_integration.execute_step(step_config, input_data)
        assert result["condition_met"] is False
    
    async def test_execute_unknown_step_type(self):
        """Test execution with unknown step type."""
        step_config = {
            "type": "unknown_type",
            "config": {}
        }
        
        with pytest.raises(ValueError, match="Unknown custom step type"):
            await self.custom_integration.execute_step(step_config, {"test": "data"})

class TestRetryManager:
    """Test retry manager."""
    
    def setup_method(self):
        from scrollintel.engines.workflow_engine import RetryManager
        self.retry_manager = RetryManager()
    
    async def test_execute_with_retry_success(self):
        """Test successful execution without retries."""
        async def mock_func():
            return "success"
        
        result = await self.retry_manager.execute_with_retry(mock_func)
        assert result == "success"
    
    async def test_execute_with_retry_failure_then_success(self):
        """Test execution that fails then succeeds."""
        call_count = 0
        
        async def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success"
        
        result = await self.retry_manager.execute_with_retry(mock_func)
        assert result == "success"
        assert call_count == 2
    
    async def test_execute_with_retry_max_retries_exceeded(self):
        """Test execution that exceeds max retries."""
        async def mock_func():
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            await self.retry_manager.execute_with_retry(mock_func)

class TestWorkflowModels:
    """Test workflow model validation."""
    
    def test_workflow_status_enum(self):
        """Test workflow status enum values."""
        from scrollintel.models.workflow_models import WorkflowStatus
        
        assert WorkflowStatus.DRAFT == "draft"
        assert WorkflowStatus.ACTIVE == "active"
        assert WorkflowStatus.COMPLETED == "completed"
        assert WorkflowStatus.FAILED == "failed"
    
    def test_integration_type_enum(self):
        """Test integration type enum values."""
        from scrollintel.models.workflow_models import IntegrationType
        
        assert IntegrationType.ZAPIER == "zapier"
        assert IntegrationType.POWER_AUTOMATE == "power_automate"
        assert IntegrationType.AIRFLOW == "airflow"
        assert IntegrationType.CUSTOM == "custom"
    
    def test_trigger_type_enum(self):
        """Test trigger type enum values."""
        from scrollintel.models.workflow_models import TriggerType
        
        assert TriggerType.WEBHOOK == "webhook"
        assert TriggerType.SCHEDULE == "schedule"
        assert TriggerType.EVENT == "event"
        assert TriggerType.MANUAL == "manual"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])