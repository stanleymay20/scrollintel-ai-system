"""
Integration tests for ScrollModelFactory engine.
Tests the complete workflow of custom model creation, validation, and deployment.
"""

import pytest
import pytest_asyncio
import pandas as pd
import numpy as np
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from scrollintel.engines.scroll_model_factory import (
    ScrollModelFactory, 
    ModelTemplate, 
    ModelAlgorithm, 
    ValidationStrategy
)
from scrollintel.models.database import MLModel, Dataset
from scrollintel.models.schemas import MLModelCreate


class TestScrollModelFactoryIntegration:
    """Integration tests for ScrollModelFactory engine."""
    
    @pytest_asyncio.fixture
    async def engine(self):
        """Create and initialize ScrollModelFactory engine."""
        engine = ScrollModelFactory()
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create classification dataset
        X = np.random.randn(n_samples, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def regression_data(self):
        """Create sample regression dataset."""
        np.random.seed(42)
        n_samples = 100
        
        X = np.random.randn(n_samples, 3)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 0.5 + np.random.randn(n_samples) * 0.1
        
        df = pd.DataFrame(X, columns=['x1', 'x2', 'x3'])
        df['y'] = y
        
        return df
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.engine_id == "scroll_model_factory"
        assert engine.name == "ScrollModelFactory Engine"
        assert engine.status.value == "ready"
        
        # Check templates and algorithms are loaded
        assert len(engine.model_templates) > 0
        assert len(engine.algorithm_configs) > 0
        
        # Check directories are created
        assert engine.models_dir.exists()
        assert engine.templates_dir.exists()
    
    @pytest.mark.asyncio
    async def test_get_templates(self, engine):
        """Test getting model templates."""
        result = await engine.process(
            input_data=None,
            parameters={"action": "get_templates"}
        )
        
        assert "templates" in result
        assert "count" in result
        assert result["count"] > 0
        
        # Check specific templates exist
        templates = result["templates"]
        assert ModelTemplate.BINARY_CLASSIFICATION in templates
        assert ModelTemplate.REGRESSION in templates
        
        # Check template structure
        binary_template = templates[ModelTemplate.BINARY_CLASSIFICATION]
        assert "name" in binary_template
        assert "description" in binary_template
        assert "recommended_algorithms" in binary_template
        assert "evaluation_metrics" in binary_template
    
    @pytest.mark.asyncio
    async def test_get_algorithms(self, engine):
        """Test getting available algorithms."""
        result = await engine.process(
            input_data=None,
            parameters={"action": "get_algorithms"}
        )
        
        assert "algorithms" in result
        assert "count" in result
        assert result["count"] > 0
        
        # Check specific algorithms exist
        algorithms = result["algorithms"]
        assert ModelAlgorithm.RANDOM_FOREST.value in algorithms
        assert ModelAlgorithm.LOGISTIC_REGRESSION.value in algorithms
        
        # Check algorithm structure
        rf_algorithm = algorithms[ModelAlgorithm.RANDOM_FOREST.value]
        assert "name" in rf_algorithm
        assert "supports_classification" in rf_algorithm
        assert "supports_regression" in rf_algorithm
        assert "default_params" in rf_algorithm
        assert "tunable_params" in rf_algorithm
    
    @pytest.mark.asyncio
    async def test_create_classification_model(self, engine, sample_data):
        """Test creating a classification model."""
        parameters = {
            "action": "create_model",
            "model_name": "test_classification_model",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "template": ModelTemplate.BINARY_CLASSIFICATION.value,
            "target_column": "target",
            "feature_columns": ["feature1", "feature2", "feature3", "feature4"],
            "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT.value,
            "hyperparameter_tuning": False,
            "custom_params": {"n_estimators": 50}
        }
        
        result = await engine.process(input_data=sample_data, parameters=parameters)
        
        # Check result structure
        assert "model_id" in result
        assert "model_name" in result
        assert result["model_name"] == "test_classification_model"
        assert result["algorithm"] == ModelAlgorithm.RANDOM_FOREST.value
        assert result["is_classification"] == True
        assert "metrics" in result
        assert "model_path" in result
        assert "training_duration" in result
        
        # Check metrics
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "cv_mean" in metrics
        assert "cv_std" in metrics
        
        # Check model file exists
        model_path = Path(result["model_path"])
        assert model_path.exists()
        
        # Verify model can be loaded
        import joblib
        model = joblib.load(model_path)
        assert hasattr(model, 'predict')
    
    @pytest.mark.asyncio
    async def test_create_regression_model(self, engine, regression_data):
        """Test creating a regression model."""
        parameters = {
            "action": "create_model",
            "model_name": "test_regression_model",
            "algorithm": ModelAlgorithm.LINEAR_REGRESSION.value,
            "template": ModelTemplate.REGRESSION.value,
            "target_column": "y",
            "feature_columns": ["x1", "x2", "x3"],
            "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT.value,
            "hyperparameter_tuning": False
        }
        
        result = await engine.process(input_data=regression_data, parameters=parameters)
        
        # Check result structure
        assert result["is_classification"] == False
        assert result["algorithm"] == ModelAlgorithm.LINEAR_REGRESSION.value
        
        # Check regression metrics
        metrics = result["metrics"]
        assert "r2" in metrics
        assert "mse" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        
        # Check model file exists and can be loaded
        model_path = Path(result["model_path"])
        assert model_path.exists()
        
        import joblib
        model = joblib.load(model_path)
        assert hasattr(model, 'predict')
    
    @pytest.mark.asyncio
    async def test_create_model_with_hyperparameter_tuning(self, engine, sample_data):
        """Test creating a model with hyperparameter tuning."""
        parameters = {
            "action": "create_model",
            "model_name": "test_tuned_model",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "target_column": "target",
            "validation_strategy": ValidationStrategy.CROSS_VALIDATION.value,
            "hyperparameter_tuning": True
        }
        
        result = await engine.process(input_data=sample_data, parameters=parameters)
        
        # Check that hyperparameter tuning was performed
        assert "model_id" in result
        assert result["hyperparameter_tuning"] == True
        
        # Training should take longer with hyperparameter tuning
        assert result["training_duration"] > 0
        
        # Model should still be created successfully
        model_path = Path(result["model_path"])
        assert model_path.exists()
    
    @pytest.mark.asyncio
    async def test_validate_model(self, engine, sample_data):
        """Test model validation."""
        # First create a model
        create_params = {
            "action": "create_model",
            "model_name": "test_validation_model",
            "algorithm": ModelAlgorithm.LOGISTIC_REGRESSION.value,
            "target_column": "target",
            "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT.value
        }
        
        create_result = await engine.process(input_data=sample_data, parameters=create_params)
        model_id = create_result["model_id"]
        
        # Test validation without data (just model loading)
        validate_params = {
            "action": "validate_model",
            "model_id": model_id
        }
        
        validate_result = await engine.process(input_data=None, parameters=validate_params)
        
        assert validate_result["model_id"] == model_id
        assert validate_result["validation_status"] == "model_loaded"
        assert "validation_timestamp" in validate_result
        
        # Test validation with data
        validation_data = [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]
        validate_params["validation_data"] = validation_data
        
        validate_result = await engine.process(input_data=None, parameters=validate_params)
        
        assert validate_result["validation_status"] == "success"
        assert "predictions" in validate_result
        assert len(validate_result["predictions"]) == 2
    
    @pytest.mark.asyncio
    async def test_deploy_model(self, engine, sample_data):
        """Test model deployment."""
        # First create a model
        create_params = {
            "action": "create_model",
            "model_name": "test_deployment_model",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "target_column": "target",
            "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT.value
        }
        
        create_result = await engine.process(input_data=sample_data, parameters=create_params)
        model_id = create_result["model_id"]
        
        # Test deployment
        deploy_params = {
            "action": "deploy_model",
            "model_id": model_id,
            "endpoint_name": "test_endpoint"
        }
        
        deploy_result = await engine.process(input_data=None, parameters=deploy_params)
        
        assert deploy_result["model_id"] == model_id
        assert deploy_result["endpoint_name"] == "test_endpoint"
        assert deploy_result["api_endpoint"] == f"/api/models/{model_id}/predict"
        assert deploy_result["status"] == "deployed"
        assert "deployment_timestamp" in deploy_result
        
        # Check deployment config file exists
        deployment_file = engine.models_dir / f"{model_id}_deployment.json"
        assert deployment_file.exists()
        
        with open(deployment_file) as f:
            config = json.load(f)
            assert config["model_id"] == model_id
            assert config["endpoint_name"] == "test_endpoint"
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_algorithm(self, engine, sample_data):
        """Test error handling for invalid algorithm."""
        parameters = {
            "action": "create_model",
            "model_name": "test_invalid_algorithm",
            "algorithm": "invalid_algorithm",
            "target_column": "target"
        }
        
        with pytest.raises(ValueError, match="invalid_algorithm"):
            await engine.process(input_data=sample_data, parameters=parameters)
    
    @pytest.mark.asyncio
    async def test_error_handling_missing_target_column(self, engine, sample_data):
        """Test error handling for missing target column."""
        parameters = {
            "action": "create_model",
            "model_name": "test_missing_target",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "target_column": "nonexistent_column"
        }
        
        with pytest.raises(ValueError, match="Target column 'nonexistent_column' not found"):
            await engine.process(input_data=sample_data, parameters=parameters)
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_model_id(self, engine):
        """Test error handling for invalid model ID in validation."""
        parameters = {
            "action": "validate_model",
            "model_id": "nonexistent_model_id"
        }
        
        with pytest.raises(ValueError, match="Model nonexistent_model_id not found"):
            await engine.process(input_data=None, parameters=parameters)
    
    @pytest.mark.asyncio
    async def test_classification_vs_regression_detection(self, engine, sample_data, regression_data):
        """Test automatic detection of classification vs regression problems."""
        # Test classification detection
        class_params = {
            "action": "create_model",
            "model_name": "test_classification_detection",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "target_column": "target"
        }
        
        class_result = await engine.process(input_data=sample_data, parameters=class_params)
        assert class_result["is_classification"] == True
        
        # Test regression detection
        reg_params = {
            "action": "create_model",
            "model_name": "test_regression_detection",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "target_column": "y"
        }
        
        reg_result = await engine.process(input_data=regression_data, parameters=reg_params)
        assert reg_result["is_classification"] == False
    
    @pytest.mark.asyncio
    async def test_template_configuration_application(self, engine, sample_data):
        """Test that template configurations are properly applied."""
        parameters = {
            "action": "create_model",
            "model_name": "test_template_config",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "template": ModelTemplate.BINARY_CLASSIFICATION.value,
            "target_column": "target"
        }
        
        result = await engine.process(input_data=sample_data, parameters=parameters)
        
        # Check that template-specific metrics are included
        metrics = result["metrics"]
        template_config = engine.model_templates[ModelTemplate.BINARY_CLASSIFICATION]
        expected_metrics = template_config["evaluation_metrics"]
        
        for metric in expected_metrics:
            if metric == "roc_auc":
                continue  # ROC AUC might not be calculated in all cases
            assert metric in metrics or metric.replace("_", "") in metrics
    
    @pytest.mark.asyncio
    async def test_engine_status_and_health(self, engine):
        """Test engine status and health check."""
        # Test status
        status = engine.get_status()
        assert status["engine_id"] == "scroll_model_factory"
        assert status["status"] == "ready"
        assert status["healthy"] == True
        assert "available_templates" in status
        assert "available_algorithms" in status
        
        # Test health check
        is_healthy = await engine.health_check()
        assert is_healthy == True
    
    @pytest.mark.asyncio
    async def test_concurrent_model_creation(self, engine, sample_data):
        """Test concurrent model creation."""
        async def create_model(model_name):
            parameters = {
                "action": "create_model",
                "model_name": model_name,
                "algorithm": ModelAlgorithm.LOGISTIC_REGRESSION.value,
                "target_column": "target"
            }
            return await engine.process(input_data=sample_data, parameters=parameters)
        
        # Create multiple models concurrently
        tasks = [
            create_model(f"concurrent_model_{i}")
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Check all models were created successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["model_name"] == f"concurrent_model_{i}"
            assert "model_id" in result
            
            # Check model files exist
            model_path = Path(result["model_path"])
            assert model_path.exists()
    
    @pytest.mark.asyncio
    async def test_custom_parameters_application(self, engine, sample_data):
        """Test that custom parameters are properly applied."""
        custom_params = {
            "n_estimators": 25,
            "max_depth": 5,
            "min_samples_split": 10
        }
        
        parameters = {
            "action": "create_model",
            "model_name": "test_custom_params",
            "algorithm": ModelAlgorithm.RANDOM_FOREST.value,
            "target_column": "target",
            "custom_params": custom_params
        }
        
        result = await engine.process(input_data=sample_data, parameters=parameters)
        
        # Check that custom parameters were applied
        applied_params = result["parameters"]
        for param, value in custom_params.items():
            assert applied_params.get(param) == value
    
    @pytest.mark.asyncio
    async def test_feature_column_selection(self, engine, sample_data):
        """Test feature column selection."""
        # Test with specific feature columns
        parameters = {
            "action": "create_model",
            "model_name": "test_feature_selection",
            "algorithm": ModelAlgorithm.LOGISTIC_REGRESSION.value,
            "target_column": "target",
            "feature_columns": ["feature1", "feature2"]  # Only use 2 features
        }
        
        result = await engine.process(input_data=sample_data, parameters=parameters)
        
        assert result["feature_columns"] == ["feature1", "feature2"]
        
        # Model should still work with reduced features
        model_path = Path(result["model_path"])
        assert model_path.exists()
        
        import joblib
        model = joblib.load(model_path)
        
        # Test prediction with correct number of features
        test_features = np.array([[1.0, 2.0]])  # 2 features
        prediction = model.predict(test_features)
        assert len(prediction) == 1