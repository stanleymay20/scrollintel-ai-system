"""
Tests for ML Platform Integration Engine
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from ai_data_readiness.engines.ml_platform_integrator import (
    MLPlatformIntegrator,
    MLPlatformConfig,
    MLflowConnector,
    KubeflowConnector,
    GenericMLPlatformConnector,
    ModelDeploymentInfo,
    DataQualityCorrelation
)


class TestMLPlatformConfig:
    """Test ML Platform Configuration"""
    
    def test_config_creation(self):
        """Test creating ML platform configuration"""
        config = MLPlatformConfig(
            platform_type="mlflow",
            endpoint_url="http://localhost:5000",
            credentials={"api_key": "test_key"},
            metadata={"region": "us-west-2"}
        )
        
        assert config.platform_type == "mlflow"
        assert config.endpoint_url == "http://localhost:5000"
        assert config.credentials["api_key"] == "test_key"
        assert config.metadata["region"] == "us-west-2"


class TestGenericMLPlatformConnector:
    """Test Generic ML Platform Connector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MLPlatformConfig(
            platform_type="generic",
            endpoint_url="http://test-ml-platform.com",
            credentials={"api_key": "test_key"}
        )
        self.connector = GenericMLPlatformConnector(self.config)
    
    @patch('requests.Session.get')
    def test_connect_success(self, mock_get):
        """Test successful connection"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = self.connector.connect()
        assert result is True
        mock_get.assert_called_once_with("http://test-ml-platform.com/health")
    
    @patch('requests.Session.get')
    def test_connect_failure(self, mock_get):
        """Test connection failure"""
        mock_get.side_effect = Exception("Connection failed")
        
        result = self.connector.connect()
        assert result is False
    
    @patch('requests.Session.get')
    def test_get_models(self, mock_get):
        """Test retrieving models"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "model1", "name": "Test Model 1"},
            {"id": "model2", "name": "Test Model 2"}
        ]
        mock_get.return_value = mock_response
        
        models = self.connector.get_models()
        assert len(models) == 2
        assert models[0]["name"] == "Test Model 1"
    
    @patch('requests.Session.get')
    def test_get_model_info(self, mock_get):
        """Test getting model information"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "model1",
            "name": "Test Model 1",
            "version": "1.0",
            "status": "active"
        }
        mock_get.return_value = mock_response
        
        model_info = self.connector.get_model_info("model1")
        assert model_info["name"] == "Test Model 1"
        assert model_info["version"] == "1.0"
    
    @patch('requests.Session.post')
    def test_deploy_model(self, mock_post):
        """Test model deployment"""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "status": "deployed",
            "endpoint_url": "http://test-ml-platform.com/models/model1/predict"
        }
        mock_post.return_value = mock_response
        
        model_info = {"id": "model1", "name": "Test Model 1", "version": "1.0"}
        deployment_info = self.connector.deploy_model(model_info)
        
        assert isinstance(deployment_info, ModelDeploymentInfo)
        assert deployment_info.model_id == "model1"
        assert deployment_info.deployment_status == "deployed"
        assert deployment_info.endpoint_url == "http://test-ml-platform.com/models/model1/predict"
    
    @patch('requests.Session.get')
    def test_get_model_performance(self, mock_get):
        """Test getting model performance metrics"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.92,
            "f1_score": 0.925
        }
        mock_get.return_value = mock_response
        
        performance = self.connector.get_model_performance("model1")
        assert performance["accuracy"] == 0.95
        assert performance["precision"] == 0.93


@patch('ai_data_readiness.engines.ml_platform_integrator.MLFLOW_AVAILABLE', True)
class TestMLflowConnector:
    """Test MLflow Connector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MLPlatformConfig(
            platform_type="mlflow",
            endpoint_url="http://localhost:5000",
            credentials={}
        )
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.tracking.MlflowClient')
    def test_connect_success(self, mock_client_class, mock_set_uri):
        """Test successful MLflow connection"""
        mock_client = Mock()
        mock_client.search_experiments.return_value = [Mock(), Mock()]
        mock_client_class.return_value = mock_client
        
        connector = MLflowConnector(self.config)
        result = connector.connect()
        
        assert result is True
        mock_set_uri.assert_called_once_with("http://localhost:5000")
        mock_client.search_experiments.assert_called_once()
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.tracking.MlflowClient')
    def test_get_models(self, mock_client_class, mock_set_uri):
        """Test retrieving MLflow models"""
        mock_model = Mock()
        mock_model.name = "test_model"
        mock_model.description = "Test model description"
        mock_model.creation_timestamp = 1234567890
        mock_model.last_updated_timestamp = 1234567891
        mock_model.latest_versions = [Mock(version="1", current_stage="Production", status="READY", run_id="run123")]
        
        mock_client = Mock()
        mock_client.search_registered_models.return_value = [mock_model]
        mock_client_class.return_value = mock_client
        
        connector = MLflowConnector(self.config)
        connector.client = mock_client
        
        models = connector.get_models()
        assert len(models) == 1
        assert models[0]["name"] == "test_model"
        assert len(models[0]["latest_versions"]) == 1


@patch('ai_data_readiness.engines.ml_platform_integrator.KUBEFLOW_AVAILABLE', True)
class TestKubeflowConnector:
    """Test Kubeflow Connector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = MLPlatformConfig(
            platform_type="kubeflow",
            endpoint_url="http://kubeflow.example.com",
            credentials={}
        )
    
    @patch('kfp.Client')
    def test_connect_success(self, mock_client_class):
        """Test successful Kubeflow connection"""
        mock_pipelines = Mock()
        mock_pipelines.total_size = 5
        
        mock_client = Mock()
        mock_client.list_pipelines.return_value = mock_pipelines
        mock_client_class.return_value = mock_client
        
        connector = KubeflowConnector(self.config)
        result = connector.connect()
        
        assert result is True
        mock_client_class.assert_called_once_with(host="http://kubeflow.example.com")
        mock_client.list_pipelines.assert_called_once()


class TestMLPlatformIntegrator:
    """Test ML Platform Integrator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.integrator = MLPlatformIntegrator()
    
    def test_register_platform_generic(self):
        """Test registering a generic platform"""
        config = MLPlatformConfig(
            platform_type="generic",
            endpoint_url="http://test-platform.com",
            credentials={"api_key": "test_key"}
        )
        
        with patch.object(GenericMLPlatformConnector, 'connect', return_value=True):
            result = self.integrator.register_platform("test_platform", config)
            assert result is True
            assert "test_platform" in self.integrator.connectors
    
    def test_register_platform_connection_failure(self):
        """Test registering platform with connection failure"""
        config = MLPlatformConfig(
            platform_type="generic",
            endpoint_url="http://invalid-platform.com",
            credentials={}
        )
        
        with patch.object(GenericMLPlatformConnector, 'connect', return_value=False):
            result = self.integrator.register_platform("invalid_platform", config)
            assert result is False
            assert "invalid_platform" not in self.integrator.connectors
    
    def test_get_all_models(self):
        """Test getting models from all platforms"""
        # Register mock platforms
        mock_connector1 = Mock()
        mock_connector1.get_models.return_value = [{"id": "model1", "name": "Model 1"}]
        
        mock_connector2 = Mock()
        mock_connector2.get_models.return_value = [{"id": "model2", "name": "Model 2"}]
        
        self.integrator.connectors = {
            "platform1": mock_connector1,
            "platform2": mock_connector2
        }
        
        all_models = self.integrator.get_all_models()
        
        assert len(all_models) == 2
        assert "platform1" in all_models
        assert "platform2" in all_models
        assert len(all_models["platform1"]) == 1
        assert len(all_models["platform2"]) == 1
    
    def test_deploy_model_success(self):
        """Test successful model deployment"""
        mock_connector = Mock()
        mock_deployment_info = ModelDeploymentInfo(
            model_id="model1",
            model_name="Test Model",
            version="1.0",
            deployment_status="deployed",
            created_at=datetime.now()
        )
        mock_connector.deploy_model.return_value = mock_deployment_info
        
        self.integrator.connectors = {"test_platform": mock_connector}
        
        model_info = {"id": "model1", "name": "Test Model", "version": "1.0"}
        result = self.integrator.deploy_model("test_platform", model_info)
        
        assert result is not None
        assert result.model_id == "model1"
        assert result.deployment_status == "deployed"
    
    def test_deploy_model_platform_not_found(self):
        """Test deploying model to non-existent platform"""
        model_info = {"id": "model1", "name": "Test Model"}
        result = self.integrator.deploy_model("nonexistent_platform", model_info)
        
        assert result is None
    
    def test_correlate_data_quality_with_performance(self):
        """Test data quality and performance correlation"""
        # Mock connector with models and performance data
        mock_connector = Mock()
        mock_connector.get_models.return_value = [
            {"id": "model1", "name": "Model 1"},
            {"id": "model2", "name": "Model 2"}
        ]
        mock_connector.get_model_performance.side_effect = [
            {"accuracy": 0.95, "precision": 0.93},
            {"accuracy": 0.88, "f1_score": 0.85}
        ]
        
        self.integrator.connectors = {"test_platform": mock_connector}
        
        correlations = self.integrator.correlate_data_quality_with_performance(
            dataset_id="dataset1",
            quality_score=0.92,
            quality_dimensions={"completeness": 0.95, "accuracy": 0.90}
        )
        
        assert len(correlations) == 2
        assert all(isinstance(corr, DataQualityCorrelation) for corr in correlations)
        assert correlations[0].dataset_id == "dataset1"
        assert correlations[0].model_id == "model1"
    
    def test_calculate_performance_score(self):
        """Test performance score calculation"""
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.92,
            "f1_score": 0.925,
            "irrelevant_metric": 0.5
        }
        
        score = self.integrator._calculate_performance_score(metrics)
        
        # Should average accuracy, f1_score, precision, recall
        expected_score = (0.95 + 0.925 + 0.93 + 0.92) / 4
        assert abs(score - expected_score) < 0.001
    
    def test_calculate_correlation(self):
        """Test correlation coefficient calculation"""
        quality_score = 0.90
        performance_score = 0.92
        quality_dimensions = {"accuracy": 0.88, "completeness": 0.92}
        performance_metrics = {"accuracy": 0.90, "precision": 0.94}
        
        correlation = self.integrator._calculate_correlation(
            quality_score,
            performance_score,
            quality_dimensions,
            performance_metrics
        )
        
        assert 0.0 <= correlation <= 1.0
    
    def test_get_platform_status(self):
        """Test getting platform status"""
        mock_connector = Mock()
        mock_connector.connect.return_value = True
        mock_connector.get_models.return_value = [{"id": "model1"}, {"id": "model2"}]
        mock_connector.config.platform_type = "generic"
        mock_connector.config.endpoint_url = "http://test.com"
        
        self.integrator.connectors = {"test_platform": mock_connector}
        
        status = self.integrator.get_platform_status()
        
        assert "test_platform" in status
        assert status["test_platform"]["connected"] is True
        assert status["test_platform"]["model_count"] == 2
        assert status["test_platform"]["platform_type"] == "generic"


class TestModelDeploymentInfo:
    """Test ModelDeploymentInfo dataclass"""
    
    def test_creation(self):
        """Test creating ModelDeploymentInfo"""
        deployment_info = ModelDeploymentInfo(
            model_id="model1",
            model_name="Test Model",
            version="1.0",
            deployment_status="deployed",
            endpoint_url="http://api.example.com/predict",
            performance_metrics={"accuracy": 0.95},
            created_at=datetime.now()
        )
        
        assert deployment_info.model_id == "model1"
        assert deployment_info.model_name == "Test Model"
        assert deployment_info.version == "1.0"
        assert deployment_info.deployment_status == "deployed"
        assert deployment_info.performance_metrics["accuracy"] == 0.95


class TestDataQualityCorrelation:
    """Test DataQualityCorrelation dataclass"""
    
    def test_creation(self):
        """Test creating DataQualityCorrelation"""
        correlation = DataQualityCorrelation(
            dataset_id="dataset1",
            model_id="model1",
            quality_score=0.92,
            performance_score=0.95,
            correlation_coefficient=0.87,
            quality_dimensions={"completeness": 0.95, "accuracy": 0.90},
            performance_metrics={"accuracy": 0.95, "precision": 0.93},
            timestamp=datetime.now()
        )
        
        assert correlation.dataset_id == "dataset1"
        assert correlation.model_id == "model1"
        assert correlation.quality_score == 0.92
        assert correlation.performance_score == 0.95
        assert correlation.correlation_coefficient == 0.87


if __name__ == '__main__':
    pytest.main([__file__])