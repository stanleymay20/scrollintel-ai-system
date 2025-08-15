"""
Tests for ScrollModelFactory API routes.
Tests the FastAPI endpoints for custom model creation, validation, and deployment.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from scrollintel.api.routes.scroll_model_factory_routes import router
from scrollintel.models.database import MLModel, Dataset, User
from scrollintel.models.schemas import MLModelCreate
from scrollintel.engines.scroll_model_factory import ScrollModelFactory


# Mock dependencies
@pytest.fixture
def mock_db():
    """Mock database session."""
    return Mock(spec=Session)


@pytest.fixture
def mock_user():
    """Mock authenticated user."""
    user = Mock(spec=User)
    user.id = "test-user-id"
    user.email = "test@example.com"
    user.role = "analyst"
    user.permissions = ["model:create", "model:validate", "model:deploy"]
    return user


@pytest.fixture
def mock_engine():
    """Mock ScrollModelFactory engine."""
    engine = Mock(spec=ScrollModelFactory)
    engine.process = AsyncMock()
    engine.get_status = Mock()
    engine.health_check = AsyncMock()
    return engine


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_dataset():
    """Mock dataset."""
    dataset = Mock(spec=Dataset)
    dataset.id = "test-dataset-id"
    dataset.name = "Test Dataset"
    dataset.file_path = "test_data.csv"
    return dataset


class TestScrollModelFactoryRoutes:
    """Test ScrollModelFactory API routes."""
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    def test_get_templates(self, mock_get_engine, mock_get_user, client, mock_user, mock_engine):
        """Test GET /templates endpoint."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        
        # Mock engine response
        mock_engine.process.return_value = {
            "templates": {
                "binary_classification": {
                    "name": "Binary Classification",
                    "description": "Classify data into two categories"
                }
            },
            "count": 1
        }
        
        response = client.get("/api/model-factory/templates")
        
        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert "count" in data
        assert data["count"] == 1
        
        # Verify engine was called correctly
        mock_engine.process.assert_called_once_with(
            input_data=None,
            parameters={"action": "get_templates"}
        )
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    def test_get_algorithms(self, mock_get_engine, mock_get_user, client, mock_user, mock_engine):
        """Test GET /algorithms endpoint."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        
        # Mock engine response
        mock_engine.process.return_value = {
            "algorithms": {
                "random_forest": {
                    "name": "Random Forest",
                    "supports_classification": True,
                    "supports_regression": True
                }
            },
            "count": 1
        }
        
        response = client.get("/api/model-factory/algorithms")
        
        assert response.status_code == 200
        data = response.json()
        assert "algorithms" in data
        assert "count" in data
        assert data["count"] == 1
        
        # Verify engine was called correctly
        mock_engine.process.assert_called_once_with(
            input_data=None,
            parameters={"action": "get_algorithms"}
        )
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_db')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.require_permission')
    @patch('pandas.read_csv')
    def test_create_custom_model(self, mock_read_csv, mock_require_permission, 
                                mock_get_engine, mock_get_user, mock_get_db,
                                client, mock_user, mock_engine, mock_db, sample_dataset):
        """Test POST /models endpoint."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        mock_get_db.return_value = mock_db
        mock_require_permission.return_value = None
        
        # Mock database query
        mock_db.query.return_value.filter.return_value.first.return_value = sample_dataset
        mock_db.add = Mock()
        mock_db.commit = Mock()
        mock_db.refresh = Mock()
        
        # Mock pandas read_csv
        sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0]
        })
        mock_read_csv.return_value = sample_data
        
        # Mock engine response
        mock_engine.process.return_value = {
            "model_id": "test-model-id",
            "model_name": "Test Model",
            "algorithm": "random_forest",
            "template": "binary_classification",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"],
            "model_path": "/path/to/model.pkl",
            "is_classification": True,
            "parameters": {"n_estimators": 100},
            "metrics": {"accuracy": 0.85},
            "validation_strategy": "train_test_split",
            "training_duration": 10.5,
            "created_at": "2024-01-01T00:00:00"
        }
        
        request_data = {
            "model_name": "Test Model",
            "dataset_id": "test-dataset-id",
            "algorithm": "random_forest",
            "template": "binary_classification",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"],
            "custom_params": {},
            "validation_strategy": "train_test_split",
            "hyperparameter_tuning": False
        }
        
        response = client.post("/api/model-factory/models", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "Test Model"
        assert data["algorithm"] == "random_forest"
        
        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        
        # Verify engine was called correctly
        mock_engine.process.assert_called_once()
        call_args = mock_engine.process.call_args
        assert call_args[1]["parameters"]["action"] == "create_model"
        assert call_args[1]["parameters"]["model_name"] == "Test Model"
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.require_permission')
    def test_validate_model(self, mock_require_permission, mock_get_engine, 
                           mock_get_user, client, mock_user, mock_engine):
        """Test POST /models/{model_id}/validate endpoint."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        mock_require_permission.return_value = None
        
        # Mock engine response
        mock_engine.process.return_value = {
            "model_id": "test-model-id",
            "validation_status": "success",
            "predictions": [0, 1, 0],
            "validation_timestamp": "2024-01-01T00:00:00"
        }
        
        request_data = {
            "model_id": "test-model-id",
            "validation_data": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        }
        
        response = client.post("/api/model-factory/models/test-model-id/validate", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test-model-id"
        assert data["validation_status"] == "success"
        assert data["predictions"] == [0, 1, 0]
        
        # Verify engine was called correctly
        mock_engine.process.assert_called_once()
        call_args = mock_engine.process.call_args
        assert call_args[1]["parameters"]["action"] == "validate_model"
        assert call_args[1]["parameters"]["model_id"] == "test-model-id"
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_db')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.require_permission')
    def test_deploy_model(self, mock_require_permission, mock_get_engine, 
                         mock_get_user, mock_get_db, client, mock_user, mock_engine, mock_db):
        """Test POST /models/{model_id}/deploy endpoint."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        mock_get_db.return_value = mock_db
        mock_require_permission.return_value = None
        
        # Mock database model
        mock_model = Mock(spec=MLModel)
        mock_model.id = "test-model-id"
        mock_model.api_endpoint = None
        mock_model.is_deployed = False
        mock_db.query.return_value.filter.return_value.first.return_value = mock_model
        mock_db.commit = Mock()
        
        # Mock engine response
        mock_engine.process.return_value = {
            "model_id": "test-model-id",
            "endpoint_name": "test_endpoint",
            "api_endpoint": "/api/models/test-model-id/predict",
            "model_path": "/path/to/model.pkl",
            "deployment_timestamp": "2024-01-01T00:00:00",
            "status": "deployed"
        }
        
        request_data = {
            "model_id": "test-model-id",
            "endpoint_name": "test_endpoint"
        }
        
        response = client.post("/api/model-factory/models/test-model-id/deploy", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test-model-id"
        assert data["status"] == "deployed"
        
        # Verify database update
        assert mock_model.api_endpoint == "/api/models/test-model-id/predict"
        assert mock_model.is_deployed == True
        mock_db.commit.assert_called_once()
        
        # Verify engine was called correctly
        mock_engine.process.assert_called_once()
        call_args = mock_engine.process.call_args
        assert call_args[1]["parameters"]["action"] == "deploy_model"
        assert call_args[1]["parameters"]["model_id"] == "test-model-id"
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_db')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('joblib.load')
    def test_predict_with_model(self, mock_joblib_load, mock_get_user, mock_get_db,
                               client, mock_user, mock_db):
        """Test GET /models/{model_id}/predict endpoint."""
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = mock_db
        
        # Mock database model
        mock_model_db = Mock(spec=MLModel)
        mock_model_db.id = "test-model-id"
        mock_model_db.model_path = "/path/to/model.pkl"
        mock_model_db.is_deployed = True
        mock_model_db.is_active = True
        mock_db.query.return_value.filter.return_value.first.return_value = mock_model_db
        
        # Mock joblib model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_joblib_load.return_value = mock_model
        
        # Mock Path.exists
        with patch('pathlib.Path.exists', return_value=True):
            response = client.get("/api/model-factory/models/test-model-id/predict?features=1.0,2.0,3.0")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "test-model-id"
        assert data["prediction"] == [1]
        assert data["prediction_proba"] == [[0.3, 0.7]]
        assert "timestamp" in data
        
        # Verify model was called correctly
        mock_model.predict.assert_called_once()
        mock_model.predict_proba.assert_called_once()
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    def test_get_engine_status(self, mock_get_engine, mock_get_user, client, mock_user, mock_engine):
        """Test GET /status endpoint."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        
        # Mock engine status
        mock_engine.get_status.return_value = {
            "engine_id": "scroll_model_factory",
            "name": "ScrollModelFactory Engine",
            "status": "ready",
            "healthy": True
        }
        
        response = client.get("/api/model-factory/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["engine_id"] == "scroll_model_factory"
        assert data["healthy"] == True
        
        mock_engine.get_status.assert_called_once()
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    def test_health_check(self, mock_get_engine, client, mock_engine):
        """Test GET /health endpoint."""
        mock_get_engine.return_value = mock_engine
        mock_engine.health_check.return_value = True
        
        response = client.get("/api/model-factory/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        
        mock_engine.health_check.assert_called_once()
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_db')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.require_permission')
    def test_create_model_dataset_not_found(self, mock_require_permission, mock_get_engine,
                                           mock_get_user, mock_get_db, client, mock_user, mock_engine, mock_db):
        """Test create model with non-existent dataset."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        mock_get_db.return_value = mock_db
        mock_require_permission.return_value = None
        
        # Mock database query to return None (dataset not found)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        request_data = {
            "model_name": "Test Model",
            "dataset_id": "nonexistent-dataset-id",
            "algorithm": "random_forest",
            "target_column": "target"
        }
        
        response = client.post("/api/model-factory/models", json=request_data)
        
        assert response.status_code == 404
        assert "Dataset not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_db')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.require_permission')
    def test_deploy_model_not_found(self, mock_require_permission, mock_get_engine,
                                   mock_get_user, mock_get_db, client, mock_user, mock_engine, mock_db):
        """Test deploy model with non-existent model."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        mock_get_db.return_value = mock_db
        mock_require_permission.return_value = None
        
        # Mock database query to return None (model not found)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        request_data = {
            "model_id": "nonexistent-model-id",
            "endpoint_name": "test_endpoint"
        }
        
        response = client.post("/api/model-factory/models/nonexistent-model-id/deploy", json=request_data)
        
        assert response.status_code == 404
        assert "Model not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_db')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    def test_predict_model_not_deployed(self, mock_get_user, mock_get_db, client, mock_user, mock_db):
        """Test prediction with non-deployed model."""
        mock_get_user.return_value = mock_user
        mock_get_db.return_value = mock_db
        
        # Mock database query to return None (deployed model not found)
        mock_db.query.return_value.filter.return_value.first.return_value = None
        
        response = client.get("/api/model-factory/models/test-model-id/predict?features=1.0,2.0,3.0")
        
        assert response.status_code == 404
        assert "Deployed model not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_current_user')
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    def test_engine_error_handling(self, mock_get_engine, mock_get_user, client, mock_user, mock_engine):
        """Test handling of engine errors."""
        mock_get_user.return_value = mock_user
        mock_get_engine.return_value = mock_engine
        
        # Mock engine to raise an exception
        mock_engine.process.side_effect = Exception("Engine error")
        
        response = client.get("/api/model-factory/templates")
        
        assert response.status_code == 500
        assert "Engine error" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.scroll_model_factory_routes.get_model_factory_engine')
    def test_health_check_unhealthy(self, mock_get_engine, client, mock_engine):
        """Test health check when engine is unhealthy."""
        mock_get_engine.return_value = mock_engine
        mock_engine.health_check.side_effect = Exception("Health check failed")
        
        response = client.get("/api/model-factory/health")
        
        assert response.status_code == 200  # Health endpoint should always return 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "error" in data
        assert "Health check failed" in data["error"]