"""
Unit tests for ScrollModelFactory engine.
Tests individual components and methods of the ScrollModelFactory.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from scrollintel.engines.scroll_model_factory import (
    ScrollModelFactory,
    ModelTemplate,
    ModelAlgorithm,
    ValidationStrategy
)


class TestScrollModelFactory:
    """Unit tests for ScrollModelFactory engine."""
    
    @pytest.fixture
    def engine(self):
        """Create ScrollModelFactory engine instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = ScrollModelFactory()
            engine.models_dir = Path(temp_dir) / "models"
            engine.templates_dir = Path(temp_dir) / "templates"
            engine.models_dir.mkdir(parents=True, exist_ok=True)
            engine.templates_dir.mkdir(parents=True, exist_ok=True)
            yield engine
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification dataset."""
        np.random.seed(42)
        data = {
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50),
            'target': np.random.choice([0, 1], 50)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression dataset."""
        np.random.seed(42)
        data = {
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'x3': np.random.randn(50),
            'y': np.random.randn(50) * 10 + 50  # Continuous target
        }
        return pd.DataFrame(data)
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.engine_id == "scroll_model_factory"
        assert engine.name == "ScrollModelFactory Engine"
        assert len(engine.model_templates) > 0
        assert len(engine.algorithm_configs) > 0
        
        # Check that all expected templates are present
        expected_templates = [
            ModelTemplate.BINARY_CLASSIFICATION,
            ModelTemplate.MULTICLASS_CLASSIFICATION,
            ModelTemplate.REGRESSION,
            ModelTemplate.TIME_SERIES_FORECASTING,
            ModelTemplate.ANOMALY_DETECTION
        ]
        
        for template in expected_templates:
            assert template in engine.model_templates
    
    def test_template_structure(self, engine):
        """Test model template structure."""
        template = engine.model_templates[ModelTemplate.BINARY_CLASSIFICATION]
        
        required_keys = [
            "name", "description", "recommended_algorithms",
            "default_parameters", "preprocessing", "evaluation_metrics"
        ]
        
        for key in required_keys:
            assert key in template
        
        assert isinstance(template["recommended_algorithms"], list)
        assert isinstance(template["default_parameters"], dict)
        assert isinstance(template["preprocessing"], list)
        assert isinstance(template["evaluation_metrics"], list)
    
    def test_algorithm_configuration(self, engine):
        """Test algorithm configuration structure."""
        algorithm = engine.algorithm_configs[ModelAlgorithm.RANDOM_FOREST]
        
        required_keys = ["classifier", "regressor", "default_params", "param_grid"]
        
        for key in required_keys:
            assert key in algorithm
        
        assert isinstance(algorithm["default_params"], dict)
        assert isinstance(algorithm["param_grid"], dict)
        
        # Random Forest should support both classification and regression
        assert algorithm["classifier"] is not None
        assert algorithm["regressor"] is not None
    
    def test_is_classification_problem(self, engine, sample_classification_data, sample_regression_data):
        """Test classification vs regression problem detection."""
        # Test classification detection
        y_classification = sample_classification_data['target']
        assert engine._is_classification_problem(y_classification) == True
        
        # Test regression detection
        y_regression = sample_regression_data['y']
        assert engine._is_classification_problem(y_regression) == False
        
        # Test categorical string data
        y_categorical = pd.Series(['cat', 'dog', 'cat', 'bird', 'dog'])
        assert engine._is_classification_problem(y_categorical) == True
        
        # Test numeric data with many unique values (regression)
        y_continuous = pd.Series(np.random.randn(100) * 100)
        assert engine._is_classification_problem(y_continuous) == False
    
    def test_create_preprocessing_pipeline(self, engine):
        """Test preprocessing pipeline creation."""
        # Test with binary classification template
        steps = engine._create_preprocessing_pipeline(
            ModelTemplate.BINARY_CLASSIFICATION.value, True
        )
        
        assert len(steps) > 0
        assert any("scaler" in step[0] for step in steps)
        
        # Test with no template (default preprocessing)
        steps_default = engine._create_preprocessing_pipeline(None, True)
        assert len(steps_default) > 0
    
    def test_calculate_metrics_classification(self, engine):
        """Test metrics calculation for classification."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = engine._calculate_metrics(y_true, y_pred, is_classification=True)
        
        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
            assert 0 <= metrics[metric] <= 1
    
    def test_calculate_metrics_regression(self, engine):
        """Test metrics calculation for regression."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        
        metrics = engine._calculate_metrics(y_true, y_pred, is_classification=False)
        
        expected_metrics = ["r2", "mse", "mae", "rmse"]
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    @pytest.mark.asyncio
    async def test_save_templates(self, engine):
        """Test saving templates to disk."""
        await engine._save_templates()
        
        # Check templates file exists
        templates_file = engine.templates_dir / "model_templates.json"
        assert templates_file.exists()
        
        # Check algorithms file exists
        algorithms_file = engine.templates_dir / "algorithm_configs.json"
        assert algorithms_file.exists()
        
        # Verify templates file content
        with open(templates_file) as f:
            templates_data = json.load(f)
            assert ModelTemplate.BINARY_CLASSIFICATION.value in templates_data
            
            template = templates_data[ModelTemplate.BINARY_CLASSIFICATION.value]
            assert "name" in template
            assert "description" in template
        
        # Verify algorithms file content
        with open(algorithms_file) as f:
            algorithms_data = json.load(f)
            assert ModelAlgorithm.RANDOM_FOREST.value in algorithms_data
            
            algorithm = algorithms_data[ModelAlgorithm.RANDOM_FOREST.value]
            assert "default_params" in algorithm
            assert "supports_classification" in algorithm
            assert "supports_regression" in algorithm
    
    @pytest.mark.asyncio
    async def test_get_templates_action(self, engine):
        """Test get_templates action."""
        result = await engine.process(
            input_data=None,
            parameters={"action": "get_templates"}
        )
        
        assert "templates" in result
        assert "count" in result
        assert result["count"] == len(engine.model_templates)
        assert isinstance(result["templates"], dict)
    
    @pytest.mark.asyncio
    async def test_get_algorithms_action(self, engine):
        """Test get_algorithms action."""
        result = await engine.process(
            input_data=None,
            parameters={"action": "get_algorithms"}
        )
        
        assert "algorithms" in result
        assert "count" in result
        assert result["count"] == len(engine.algorithm_configs)
        
        algorithms = result["algorithms"]
        for alg_key, alg_info in algorithms.items():
            assert "name" in alg_info
            assert "supports_classification" in alg_info
            assert "supports_regression" in alg_info
            assert "default_params" in alg_info
            assert "tunable_params" in alg_info
    
    @pytest.mark.asyncio
    async def test_invalid_action(self, engine):
        """Test handling of invalid action."""
        with pytest.raises(ValueError, match="Unknown action"):
            await engine.process(
                input_data=None,
                parameters={"action": "invalid_action"}
            )
    
    @pytest.mark.asyncio
    async def test_missing_parameters(self, engine):
        """Test handling of missing parameters."""
        with pytest.raises(ValueError, match="Parameters required"):
            await engine.process(input_data=None, parameters=None)
    
    @pytest.mark.asyncio
    async def test_validate_model_missing_id(self, engine):
        """Test model validation with missing model ID."""
        with pytest.raises(ValueError, match="Model ID required"):
            await engine.process(
                input_data=None,
                parameters={"action": "validate_model"}
            )
    
    @pytest.mark.asyncio
    async def test_deploy_model_missing_id(self, engine):
        """Test model deployment with missing model ID."""
        with pytest.raises(ValueError, match="Model ID required"):
            await engine.process(
                input_data=None,
                parameters={"action": "deploy_model"}
            )
    
    def test_get_status(self, engine):
        """Test engine status retrieval."""
        status = engine.get_status()
        
        required_keys = [
            "engine_id", "name", "status", "capabilities",
            "models_directory", "templates_directory",
            "available_templates", "available_algorithms", "healthy"
        ]
        
        for key in required_keys:
            assert key in status
        
        assert status["engine_id"] == "scroll_model_factory"
        assert status["healthy"] == True
        assert status["available_templates"] == len(engine.model_templates)
        assert status["available_algorithms"] == len(engine.algorithm_configs)
    
    @pytest.mark.asyncio
    async def test_health_check(self, engine):
        """Test engine health check."""
        is_healthy = await engine.health_check()
        assert is_healthy == True
    
    def test_validation_strategy_enum(self):
        """Test ValidationStrategy enum values."""
        expected_strategies = [
            "train_test_split",
            "cross_validation", 
            "time_series_split",
            "stratified_split"
        ]
        
        for strategy in expected_strategies:
            assert hasattr(ValidationStrategy, strategy.upper())
            assert ValidationStrategy(strategy).value == strategy
    
    def test_model_template_enum(self):
        """Test ModelTemplate enum values."""
        expected_templates = [
            "binary_classification",
            "multiclass_classification",
            "regression",
            "time_series_forecasting",
            "anomaly_detection",
            "clustering",
            "recommendation",
            "text_classification",
            "image_classification"
        ]
        
        for template in expected_templates:
            assert hasattr(ModelTemplate, template.upper())
            assert ModelTemplate(template).value == template
    
    def test_model_algorithm_enum(self):
        """Test ModelAlgorithm enum values."""
        expected_algorithms = [
            "random_forest",
            "logistic_regression",
            "linear_regression",
            "svm",
            "knn",
            "decision_tree",
            "naive_bayes",
            "xgboost",
            "neural_network"
        ]
        
        for algorithm in expected_algorithms:
            assert hasattr(ModelAlgorithm, algorithm.upper())
            assert ModelAlgorithm(algorithm).value == algorithm
    
    @patch('scrollintel.engines.scroll_model_factory.cross_val_score')
    @patch('scrollintel.engines.scroll_model_factory.joblib.dump')
    @patch('scrollintel.engines.scroll_model_factory.Pipeline')
    @patch('scrollintel.engines.scroll_model_factory.train_test_split')
    @pytest.mark.asyncio
    async def test_create_custom_model_mocked(self, mock_split, mock_pipeline, mock_dump, mock_cv_score, engine, sample_classification_data):
        """Test custom model creation with mocked dependencies."""
        # Mock train_test_split with consistent lengths
        X = sample_classification_data[['feature1', 'feature2', 'feature3']]
        y = sample_classification_data['target']
        X_train, X_test = X[:30], X[30:]
        y_train, y_test = y[:30], y[30:]
        mock_split.return_value = (X_train, X_test, y_train, y_test)
        
        # Mock pipeline
        mock_model = Mock()
        mock_model.fit.return_value = None
        # Make sure prediction length matches test set length
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])  # 20 predictions for 20 test samples
        mock_pipeline.return_value = mock_model
        
        # Mock cross_val_score
        mock_cv_score.return_value = np.array([0.8, 0.85, 0.82, 0.88, 0.83])
        
        # Mock joblib.dump
        mock_dump.return_value = None
        
        parameters = {
            "action": "create_model",
            "model_name": "test_mocked_model",
            "algorithm": ModelAlgorithm.LOGISTIC_REGRESSION.value,
            "target_column": "target",
            "feature_columns": ["feature1", "feature2", "feature3"],
            "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT.value,
            "hyperparameter_tuning": False
        }
        
        result = await engine.process(input_data=sample_classification_data, parameters=parameters)
        
        # Verify mocks were called
        mock_split.assert_called_once()
        mock_pipeline.assert_called_once()
        mock_dump.assert_called_once()
        mock_cv_score.assert_called_once()
        
        # Verify result structure
        assert "model_id" in result
        assert result["model_name"] == "test_mocked_model"
        assert result["algorithm"] == ModelAlgorithm.LOGISTIC_REGRESSION.value
    
    def test_algorithm_support_matrix(self, engine):
        """Test algorithm support for classification and regression."""
        # Algorithms that should support classification
        classification_algorithms = [
            ModelAlgorithm.RANDOM_FOREST,
            ModelAlgorithm.LOGISTIC_REGRESSION,
            ModelAlgorithm.SVM,
            ModelAlgorithm.KNN,
            ModelAlgorithm.DECISION_TREE,
            ModelAlgorithm.NAIVE_BAYES
        ]
        
        for algorithm in classification_algorithms:
            config = engine.algorithm_configs[algorithm]
            assert config["classifier"] is not None, f"{algorithm} should support classification"
        
        # Algorithms that should support regression
        regression_algorithms = [
            ModelAlgorithm.RANDOM_FOREST,
            ModelAlgorithm.LINEAR_REGRESSION,
            ModelAlgorithm.SVM,
            ModelAlgorithm.KNN,
            ModelAlgorithm.DECISION_TREE
        ]
        
        for algorithm in regression_algorithms:
            config = engine.algorithm_configs[algorithm]
            assert config["regressor"] is not None, f"{algorithm} should support regression"
        
        # Algorithms that should NOT support regression
        classification_only = [ModelAlgorithm.LOGISTIC_REGRESSION, ModelAlgorithm.NAIVE_BAYES]
        
        for algorithm in classification_only:
            config = engine.algorithm_configs[algorithm]
            assert config["regressor"] is None, f"{algorithm} should not support regression"
    
    def test_template_algorithm_recommendations(self, engine):
        """Test that template algorithm recommendations are valid."""
        for template_key, template_config in engine.model_templates.items():
            recommended_algorithms = template_config["recommended_algorithms"]
            
            for algorithm in recommended_algorithms:
                # Check that recommended algorithm exists in algorithm configs
                assert ModelAlgorithm(algorithm) in engine.algorithm_configs
                
                # Check that algorithm supports the template's problem type
                alg_config = engine.algorithm_configs[ModelAlgorithm(algorithm)]
                
                if "classification" in template_key.value:
                    assert alg_config["classifier"] is not None, \
                        f"Algorithm {algorithm} recommended for {template_key} but doesn't support classification"
                elif template_key == ModelTemplate.REGRESSION:
                    assert alg_config["regressor"] is not None, \
                        f"Algorithm {algorithm} recommended for {template_key} but doesn't support regression"