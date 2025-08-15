"""
Unit tests for AutoModel engine.
Tests model training, evaluation, export functionality, and API endpoints.
"""

import pytest
import pytest_asyncio
import asyncio
import pandas as pd
import numpy as np
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.engines.automodel_engine import (
    AutoModelEngine, ModelType, AlgorithmType
)


class TestAutoModelEngine:
    """Test cases for AutoModel engine functionality."""
    
    @pytest_asyncio.fixture
    async def engine(self):
        """Create AutoModel engine instance for testing."""
        engine = AutoModelEngine()
        await engine.start()
        yield engine
        await engine.stop()
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification dataset."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X = np.random.randn(n_samples, 4)
        
        # Generate target (binary classification)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3', 'feature4'])
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression dataset."""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X = np.random.randn(n_samples, 3)
        
        # Generate target (continuous)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
        
        df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def temp_csv_file(self, sample_classification_data):
        """Create temporary CSV file with sample data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_classification_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.engine_id == "automodel_engine"
        assert engine.name == "AutoModel Engine"
        assert engine.status.value == "ready"
        assert engine.models_dir.exists()
    
    @pytest.mark.asyncio
    async def test_engine_status(self, engine):
        """Test engine status reporting."""
        status = engine.get_status()
        
        assert status["healthy"] is True
        assert "models_trained" in status
        assert "supported_algorithms" in status
        assert "models_directory" in status
    
    @pytest.mark.asyncio
    async def test_load_dataset_csv(self, engine, temp_csv_file):
        """Test loading CSV dataset."""
        df = await engine._load_dataset(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'target' in df.columns
    
    @pytest.mark.asyncio
    async def test_prepare_data_classification(self, engine, sample_classification_data):
        """Test data preparation for classification."""
        X, y, model_type = await engine._prepare_data(
            sample_classification_data, 
            'target'
        )
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert model_type == ModelType.CLASSIFICATION
        assert len(X) == len(y)
        assert 'target' not in X.columns
    
    @pytest.mark.asyncio
    async def test_prepare_data_regression(self, engine, sample_regression_data):
        """Test data preparation for regression."""
        X, y, model_type = await engine._prepare_data(
            sample_regression_data, 
            'target'
        )
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert model_type == ModelType.REGRESSION
        assert len(X) == len(y)
        assert 'target' not in X.columns
    
    @pytest.mark.asyncio
    async def test_train_random_forest_classification(self, engine, sample_classification_data):
        """Test Random Forest classification training."""
        X, y, model_type = await engine._prepare_data(
            sample_classification_data, 
            'target'
        )
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        result = await engine._train_single_model(
            AlgorithmType.RANDOM_FOREST,
            X_train, X_test, y_train, y_test,
            model_type
        )
        
        assert "model" in result
        assert "best_params" in result
        assert "metrics" in result
        assert "cv_scores" in result
        
        # Check metrics
        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert metrics["accuracy"] > 0.5  # Should be better than random
    
    @pytest.mark.asyncio
    async def test_train_xgboost_regression(self, engine, sample_regression_data):
        """Test XGBoost regression training."""
        X, y, model_type = await engine._prepare_data(
            sample_regression_data, 
            'target'
        )
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        result = await engine._train_single_model(
            AlgorithmType.XGBOOST,
            X_train, X_test, y_train, y_test,
            model_type
        )
        
        assert "model" in result
        assert "best_params" in result
        assert "metrics" in result
        assert "cv_scores" in result
        
        # Check metrics
        metrics = result["metrics"]
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics
        assert metrics["r2_score"] > 0.5  # Should explain some variance
    
    @pytest.mark.asyncio
    async def test_train_neural_network(self, engine, sample_classification_data):
        """Test neural network training."""
        X, y, model_type = await engine._prepare_data(
            sample_classification_data, 
            'target'
        )
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        result = await engine._train_neural_network(
            X_train, X_test, y_train, y_test,
            model_type
        )
        
        assert "model" in result
        assert "metrics" in result
        assert "training_history" in result
        
        # Check that model can make predictions
        predictions = result["model"].predict(X_test)
        assert len(predictions) == len(y_test)
    
    @pytest.mark.asyncio
    async def test_calculate_classification_metrics(self, engine):
        """Test classification metrics calculation."""
        y_true = pd.Series([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = await engine._calculate_metrics(
            y_true, y_pred, ModelType.CLASSIFICATION
        )
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["primary_score"] == metrics["accuracy"]
    
    @pytest.mark.asyncio
    async def test_calculate_regression_metrics(self, engine):
        """Test regression metrics calculation."""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        
        metrics = await engine._calculate_metrics(
            y_true, y_pred, ModelType.REGRESSION
        )
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2_score" in metrics
        assert "mean_residual" in metrics
        assert "std_residual" in metrics
        assert metrics["primary_score"] == metrics["r2_score"]
    
    @pytest.mark.asyncio
    async def test_full_training_workflow(self, engine, temp_csv_file):
        """Test complete model training workflow."""
        training_data = {
            "action": "train",
            "dataset_path": temp_csv_file,
            "target_column": "target",
            "model_name": "test_model",
            "algorithms": [AlgorithmType.RANDOM_FOREST]
        }
        
        result = await engine.execute(training_data)
        
        assert "model_name" in result
        assert "model_type" in result
        assert "algorithms_tested" in result
        assert "results" in result
        assert "best_model" in result
        assert "training_duration_seconds" in result
        
        # Check that model was stored
        assert "test_model" in engine.trained_models
        
        model_info = engine.trained_models["test_model"]
        assert "algorithm" in model_info
        assert "model_type" in model_info
        assert "metrics" in model_info
        assert "model_path" in model_info
    
    @pytest.mark.asyncio
    async def test_model_prediction(self, engine, temp_csv_file):
        """Test model prediction functionality."""
        # First train a model
        training_data = {
            "action": "train",
            "dataset_path": temp_csv_file,
            "target_column": "target",
            "model_name": "prediction_test_model",
            "algorithms": [AlgorithmType.RANDOM_FOREST]
        }
        
        await engine.execute(training_data)
        
        # Now test prediction
        prediction_data = {
            "action": "predict",
            "model_name": "prediction_test_model",
            "data": [
                {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "feature4": 4.0}
            ]
        }
        
        result = await engine.execute(prediction_data)
        
        assert "model_name" in result
        assert "predictions" in result
        assert "model_info" in result
        assert len(result["predictions"]) == 1
    
    @pytest.mark.asyncio
    async def test_model_comparison(self, engine, temp_csv_file):
        """Test model comparison functionality."""
        # Train multiple models
        for i, algorithm in enumerate([AlgorithmType.RANDOM_FOREST, AlgorithmType.XGBOOST]):
            training_data = {
                "action": "train",
                "dataset_path": temp_csv_file,
                "target_column": "target",
                "model_name": f"comparison_model_{i}",
                "algorithms": [algorithm]
            }
            await engine.execute(training_data)
        
        # Compare models
        comparison_data = {
            "action": "compare",
            "model_names": ["comparison_model_0", "comparison_model_1"]
        }
        
        result = await engine.execute(comparison_data)
        
        assert "comparison" in result
        assert "total_models" in result
        assert len(result["comparison"]) == 2
    
    @pytest.mark.asyncio
    async def test_model_export(self, engine, temp_csv_file):
        """Test model export functionality."""
        # First train a model
        training_data = {
            "action": "train",
            "dataset_path": temp_csv_file,
            "target_column": "target",
            "model_name": "export_test_model",
            "algorithms": [AlgorithmType.RANDOM_FOREST]
        }
        
        await engine.execute(training_data)
        
        # Export the model
        export_data = {
            "action": "export",
            "model_name": "export_test_model",
            "format": "joblib"
        }
        
        result = await engine.execute(export_data)
        
        assert "model_name" in result
        assert "export_path" in result
        assert "export_format" in result
        assert "files_created" in result
        
        # Check that export directory exists
        export_path = Path(result["export_path"])
        assert export_path.exists()
        assert (export_path / "metadata.json").exists()
        assert (export_path / "deploy.py").exists()
    
    @pytest.mark.asyncio
    async def test_save_and_load_model(self, engine, sample_classification_data):
        """Test model saving and loading."""
        X, y, model_type = await engine._prepare_data(
            sample_classification_data, 
            'target'
        )
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Save model
        model_path = await engine._save_model(
            model, "test_save_model", AlgorithmType.RANDOM_FOREST
        )
        
        assert os.path.exists(model_path)
        
        # Load and test
        import joblib
        loaded_model = joblib.load(model_path)
        predictions = loaded_model.predict(X_test)
        
        assert len(predictions) == len(y_test)
    
    async def test_error_handling_invalid_dataset(self, engine):
        """Test error handling for invalid dataset."""
        training_data = {
            "action": "train",
            "dataset_path": "nonexistent_file.csv",
            "target_column": "target",
            "model_name": "error_test_model"
        }
        
        with pytest.raises(Exception):
            await engine.execute(training_data)
    
    async def test_error_handling_invalid_target_column(self, engine, temp_csv_file):
        """Test error handling for invalid target column."""
        training_data = {
            "action": "train",
            "dataset_path": temp_csv_file,
            "target_column": "nonexistent_column",
            "model_name": "error_test_model"
        }
        
        with pytest.raises(Exception):
            await engine.execute(training_data)
    
    async def test_error_handling_prediction_nonexistent_model(self, engine):
        """Test error handling for prediction with nonexistent model."""
        prediction_data = {
            "action": "predict",
            "model_name": "nonexistent_model",
            "data": [{"feature1": 1.0}]
        }
        
        with pytest.raises(ValueError):
            await engine.execute(prediction_data)
    
    async def test_unsupported_file_format(self, engine):
        """Test error handling for unsupported file formats."""
        with pytest.raises(ValueError):
            await engine._load_dataset("test.unsupported")
    
    async def test_engine_metrics(self, engine):
        """Test engine metrics reporting."""
        metrics = engine.get_metrics()
        
        assert "engine_id" in metrics
        assert "name" in metrics
        assert "status" in metrics
        assert "capabilities" in metrics
        assert "usage_count" in metrics
        assert "error_count" in metrics
        assert "error_rate" in metrics
        assert "created_at" in metrics


@pytest.mark.asyncio
class TestAutoModelEngineIntegration:
    """Integration tests for AutoModel engine with real ML workflows."""
    
    async def test_end_to_end_classification_workflow(self):
        """Test complete classification workflow from training to prediction."""
        engine = AutoModelEngine()
        await engine.initialize()
        
        try:
            # Create sample data
            np.random.seed(42)
            n_samples = 500
            X = np.random.randn(n_samples, 3)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            
            df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
            df['target'] = y
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_file = f.name
            
            try:
                # Train model
                training_data = {
                    "action": "train",
                    "dataset_path": temp_file,
                    "target_column": "target",
                    "model_name": "integration_test_model",
                    "algorithms": [AlgorithmType.RANDOM_FOREST, AlgorithmType.XGBOOST]
                }
                
                training_result = await engine.execute(training_data)
                
                assert training_result["model_name"] == "integration_test_model"
                assert training_result["model_type"] == ModelType.CLASSIFICATION
                assert len(training_result["algorithms_tested"]) == 2
                
                # Make predictions
                prediction_data = {
                    "action": "predict",
                    "model_name": "integration_test_model",
                    "data": [
                        {"feature1": 1.0, "feature2": 1.0, "feature3": 0.0},
                        {"feature1": -1.0, "feature2": -1.0, "feature3": 0.0}
                    ]
                }
                
                prediction_result = await engine.execute(prediction_data)
                
                assert len(prediction_result["predictions"]) == 2
                assert all(pred in [0, 1] for pred in prediction_result["predictions"])
                
                # Export model
                export_data = {
                    "action": "export",
                    "model_name": "integration_test_model",
                    "format": "joblib"
                }
                
                export_result = await engine.execute(export_data)
                
                assert "export_path" in export_result
                assert Path(export_result["export_path"]).exists()
                
            finally:
                os.unlink(temp_file)
                
        finally:
            await engine.cleanup()
    
    async def test_multiple_model_comparison(self):
        """Test training and comparing multiple models."""
        engine = AutoModelEngine()
        await engine.initialize()
        
        try:
            # Create sample regression data
            np.random.seed(42)
            n_samples = 300
            X = np.random.randn(n_samples, 2)
            y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(n_samples) * 0.1
            
            df = pd.DataFrame(X, columns=['feature1', 'feature2'])
            df['target'] = y
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_file = f.name
            
            try:
                # Train multiple models
                models_to_train = [
                    ("rf_model", [AlgorithmType.RANDOM_FOREST]),
                    ("xgb_model", [AlgorithmType.XGBOOST])
                ]
                
                for model_name, algorithms in models_to_train:
                    training_data = {
                        "action": "train",
                        "dataset_path": temp_file,
                        "target_column": "target",
                        "model_name": model_name,
                        "algorithms": algorithms
                    }
                    
                    await engine.execute(training_data)
                
                # Compare models
                comparison_data = {
                    "action": "compare",
                    "model_names": ["rf_model", "xgb_model"]
                }
                
                comparison_result = await engine.execute(comparison_data)
                
                assert comparison_result["total_models"] == 2
                assert "rf_model" in comparison_result["comparison"]
                assert "xgb_model" in comparison_result["comparison"]
                
                # Each model should have metrics
                for model_name in ["rf_model", "xgb_model"]:
                    model_info = comparison_result["comparison"][model_name]
                    assert "algorithm" in model_info
                    assert "model_type" in model_info
                    assert "metrics" in model_info
                    assert "r2_score" in model_info["metrics"]
                
            finally:
                os.unlink(temp_file)
                
        finally:
            await engine.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])