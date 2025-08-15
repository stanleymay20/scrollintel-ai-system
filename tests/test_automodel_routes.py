"""
Unit tests for AutoModel API routes.
Tests FastAPI endpoints for model training, prediction, and management.
"""

import pytest
import asyncio
import tempfile
import os
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.api.routes.automodel_routes import router, automodel_engine
from scrollintel.models.database import User
from scrollintel.core.interfaces import UserRole


class TestAutoModelRoutes:
    """Test cases for AutoModel API routes."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Create mock user for authentication."""
        user = Mock(spec=User)
        user.id = "test-user-id"
        user.email = "test@example.com"
        user.role = UserRole.ADMIN
        return user
    
    @pytest.fixture
    def sample_data_file(self):
        """Create temporary CSV file with sample data."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
        df['target'] = y
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_train_model_endpoint(self, mock_get_user, client, mock_user, sample_data_file):
        """Test model training endpoint."""
        mock_get_user.return_value = mock_user
        
        # Mock engine execution
        mock_result = {
            "model_name": "test_model",
            "model_type": "classification",
            "algorithms_tested": ["random_forest"],
            "best_model": {
                "algorithm": "random_forest",
                "score": 0.85,
                "metrics": {"accuracy": 0.85}
            },
            "training_duration_seconds": 10.5,
            "model_path": "/path/to/model.pkl",
            "results": {"random_forest": {"metrics": {"accuracy": 0.85}}}
        }
        
        with patch.object(automodel_engine, 'execute', return_value=mock_result):
            response = client.post("/automodel/train", json={
                "dataset_path": sample_data_file,
                "target_column": "target",
                "model_name": "test_model",
                "algorithms": ["random_forest"]
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test_model"
        assert data["model_type"] == "classification"
        assert data["training_duration_seconds"] == 10.5
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_predict_endpoint(self, mock_get_user, client, mock_user):
        """Test prediction endpoint."""
        mock_get_user.return_value = mock_user
        
        mock_result = {
            "model_name": "test_model",
            "predictions": [1, 0, 1],
            "model_info": {
                "algorithm": "random_forest",
                "model_type": "classification",
                "metrics": {"accuracy": 0.85}
            }
        }
        
        with patch.object(automodel_engine, 'execute', return_value=mock_result):
            response = client.post("/automodel/predict", json={
                "model_name": "test_model",
                "data": [
                    {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0},
                    {"feature1": -1.0, "feature2": -2.0, "feature3": -3.0},
                    {"feature1": 0.5, "feature2": 1.5, "feature3": 2.5}
                ]
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test_model"
        assert len(data["predictions"]) == 3
        assert data["predictions"] == [1, 0, 1]
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_list_models_endpoint(self, mock_get_user, client, mock_user):
        """Test list models endpoint."""
        mock_get_user.return_value = mock_user
        
        # Mock trained models
        mock_models = {
            "model1": {
                "algorithm": "random_forest",
                "model_type": "classification",
                "metrics": {"accuracy": 0.85}
            },
            "model2": {
                "algorithm": "xgboost",
                "model_type": "regression",
                "metrics": {"r2_score": 0.92}
            }
        }
        
        with patch.object(automodel_engine, 'trained_models', mock_models):
            response = client.get("/automodel/models")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_models"] == 2
        assert "model1" in data["models"]
        assert "model2" in data["models"]
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_get_model_info_endpoint(self, mock_get_user, client, mock_user):
        """Test get model info endpoint."""
        mock_get_user.return_value = mock_user
        
        mock_model_info = {
            "algorithm": "random_forest",
            "model_type": "classification",
            "metrics": {"accuracy": 0.85},
            "feature_columns": ["feature1", "feature2", "feature3"],
            "target_column": "target"
        }
        
        with patch.object(automodel_engine, 'trained_models', {"test_model": mock_model_info}):
            response = client.get("/automodel/models/test_model")
        
        assert response.status_code == 200
        data = response.json()
        assert data["algorithm"] == "random_forest"
        assert data["model_type"] == "classification"
        assert data["metrics"]["accuracy"] == 0.85
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_get_model_info_not_found(self, mock_get_user, client, mock_user):
        """Test get model info for nonexistent model."""
        mock_get_user.return_value = mock_user
        
        with patch.object(automodel_engine, 'trained_models', {}):
            response = client.get("/automodel/models/nonexistent_model")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_compare_models_endpoint(self, mock_get_user, client, mock_user):
        """Test model comparison endpoint."""
        mock_get_user.return_value = mock_user
        
        mock_result = {
            "comparison": {
                "model1": {
                    "algorithm": "random_forest",
                    "model_type": "classification",
                    "metrics": {"accuracy": 0.85}
                },
                "model2": {
                    "algorithm": "xgboost",
                    "model_type": "classification",
                    "metrics": {"accuracy": 0.87}
                }
            },
            "total_models": 2
        }
        
        with patch.object(automodel_engine, 'execute', return_value=mock_result):
            response = client.post("/automodel/compare", json=["model1", "model2"])
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_models"] == 2
        assert "model1" in data["comparison"]
        assert "model2" in data["comparison"]
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_export_model_endpoint(self, mock_get_user, client, mock_user):
        """Test model export endpoint."""
        mock_get_user.return_value = mock_user
        
        mock_result = {
            "model_name": "test_model",
            "export_path": "/path/to/export",
            "export_format": "joblib",
            "files_created": ["test_model_random_forest.pkl", "metadata.json", "deploy.py"]
        }
        
        with patch.object(automodel_engine, 'execute', return_value=mock_result):
            response = client.post("/automodel/export", json={
                "model_name": "test_model",
                "format": "joblib"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test_model"
        assert data["export_format"] == "joblib"
        assert len(data["files_created"]) == 3
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_delete_model_endpoint(self, mock_get_user, client, mock_user):
        """Test model deletion endpoint."""
        mock_get_user.return_value = mock_user
        
        mock_model_info = {
            "algorithm": "random_forest",
            "model_path": "/path/to/model.pkl"
        }
        
        with patch.object(automodel_engine, 'trained_models', {"test_model": mock_model_info}):
            with patch('os.path.exists', return_value=True):
                with patch('os.remove') as mock_remove:
                    response = client.delete("/automodel/models/test_model")
        
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_delete_model_not_found(self, mock_get_user, client, mock_user):
        """Test deletion of nonexistent model."""
        mock_get_user.return_value = mock_user
        
        with patch.object(automodel_engine, 'trained_models', {}):
            response = client.delete("/automodel/models/nonexistent_model")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_get_engine_status_endpoint(self, mock_get_user, client, mock_user):
        """Test engine status endpoint."""
        mock_get_user.return_value = mock_user
        
        mock_status = {
            "healthy": True,
            "models_trained": 5,
            "supported_algorithms": ["random_forest", "xgboost", "neural_network"]
        }
        
        mock_metrics = {
            "engine_id": "automodel_engine",
            "usage_count": 10,
            "error_count": 0
        }
        
        with patch.object(automodel_engine, 'get_status', return_value=mock_status):
            with patch.object(automodel_engine, 'get_metrics', return_value=mock_metrics):
                response = client.get("/automodel/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"]["healthy"] is True
        assert data["metrics"]["usage_count"] == 10
        assert "engine_info" in data
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_retrain_model_endpoint(self, mock_get_user, client, mock_user):
        """Test model retraining endpoint."""
        mock_get_user.return_value = mock_user
        
        mock_model_info = {
            "algorithm": "random_forest",
            "target_column": "target",
            "feature_columns": ["feature1", "feature2", "feature3"]
        }
        
        with patch.object(automodel_engine, 'trained_models', {"test_model": mock_model_info}):
            response = client.post("/automodel/retrain/test_model")
        
        assert response.status_code == 200
        assert "Retraining initiated" in response.json()["message"]
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_retrain_model_not_found(self, mock_get_user, client, mock_user):
        """Test retraining nonexistent model."""
        mock_get_user.return_value = mock_user
        
        with patch.object(automodel_engine, 'trained_models', {}):
            response = client.post("/automodel/retrain/nonexistent_model")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_train_model_validation_error(self, mock_get_user, client, mock_user):
        """Test training with invalid request data."""
        mock_get_user.return_value = mock_user
        
        # Missing required fields
        response = client.post("/automodel/train", json={
            "model_name": "test_model"
            # Missing dataset_path and target_column
        })
        
        assert response.status_code == 422  # Validation error
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_predict_validation_error(self, mock_get_user, client, mock_user):
        """Test prediction with invalid request data."""
        mock_get_user.return_value = mock_user
        
        # Missing required fields
        response = client.post("/automodel/predict", json={
            "model_name": "test_model"
            # Missing data field
        })
        
        assert response.status_code == 422  # Validation error
    
    @patch('scrollintel.api.routes.automodel_routes.get_current_user')
    async def test_engine_error_handling(self, mock_get_user, client, mock_user, sample_data_file):
        """Test error handling when engine execution fails."""
        mock_get_user.return_value = mock_user
        
        # Mock engine to raise an exception
        with patch.object(automodel_engine, 'execute', side_effect=Exception("Engine error")):
            response = client.post("/automodel/train", json={
                "dataset_path": sample_data_file,
                "target_column": "target",
                "model_name": "test_model"
            })
        
        assert response.status_code == 500
        assert "Engine error" in response.json()["detail"]
    
    async def test_authentication_required(self, client, sample_data_file):
        """Test that authentication is required for all endpoints."""
        # Test without authentication (should fail)
        response = client.post("/automodel/train", json={
            "dataset_path": sample_data_file,
            "target_column": "target",
            "model_name": "test_model"
        })
        
        # This will fail because get_current_user dependency is not satisfied
        assert response.status_code in [401, 422]  # Unauthorized or dependency error


@pytest.mark.asyncio
class TestAutoModelRoutesIntegration:
    """Integration tests for AutoModel routes with real engine."""
    
    async def test_full_workflow_integration(self):
        """Test complete workflow through API endpoints."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        
        # Create test app
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Mock authentication
        mock_user = Mock(spec=User)
        mock_user.id = "test-user-id"
        mock_user.email = "test@example.com"
        mock_user.role = UserRole.ADMIN
        
        # Create sample data
        np.random.seed(42)
        n_samples = 200
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(X, columns=['feature1', 'feature2'])
        df['target'] = y
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            with patch('scrollintel.api.routes.automodel_routes.get_current_user', return_value=mock_user):
                # Initialize engine
                await automodel_engine.start()
                
                # Train model
                train_response = client.post("/automodel/train", json={
                    "dataset_path": temp_file,
                    "target_column": "target",
                    "model_name": "integration_test_model",
                    "algorithms": ["random_forest"]
                })
                
                assert train_response.status_code == 200
                train_data = train_response.json()
                assert train_data["model_name"] == "integration_test_model"
                
                # List models
                list_response = client.get("/automodel/models")
                assert list_response.status_code == 200
                list_data = list_response.json()
                assert "integration_test_model" in list_data["models"]
                
                # Get model info
                info_response = client.get("/automodel/models/integration_test_model")
                assert info_response.status_code == 200
                info_data = info_response.json()
                assert info_data["algorithm"] == "random_forest"
                
                # Make prediction
                predict_response = client.post("/automodel/predict", json={
                    "model_name": "integration_test_model",
                    "data": [
                        {"feature1": 1.0, "feature2": 1.0},
                        {"feature1": -1.0, "feature2": -1.0}
                    ]
                })
                
                assert predict_response.status_code == 200
                predict_data = predict_response.json()
                assert len(predict_data["predictions"]) == 2
                
                # Export model
                export_response = client.post("/automodel/export", json={
                    "model_name": "integration_test_model",
                    "format": "joblib"
                })
                
                assert export_response.status_code == 200
                export_data = export_response.json()
                assert "export_path" in export_data
                
                # Get status
                status_response = client.get("/automodel/status")
                assert status_response.status_code == 200
                status_data = status_response.json()
                assert status_data["status"]["healthy"] is True
                
        finally:
            os.unlink(temp_file)
            await automodel_engine.stop()


if __name__ == "__main__":
    pytest.main([__file__])