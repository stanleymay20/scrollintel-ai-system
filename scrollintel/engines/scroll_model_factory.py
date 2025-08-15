"""
ScrollModelFactory Engine for custom model creation and deployment.
Implements requirement 7.1: UI-driven model creation interface with parameter configuration.
"""

import os
import json
import joblib
import pickle
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from uuid import uuid4
from enum import Enum

# ML libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    train_test_split, StratifiedKFold, KFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

# Optional ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    xgb = None

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    keras = None
    layers = None

import warnings
warnings.filterwarnings('ignore')

from .base_engine import BaseEngine, EngineCapability, EngineStatus
from ..models.database import MLModel, Dataset
from ..models.schemas import MLModelCreate, MLModelResponse

logger = logging.getLogger(__name__)


class ModelTemplate(str, Enum):
    """Pre-defined model templates for common use cases."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    TEXT_CLASSIFICATION = "text_classification"
    IMAGE_CLASSIFICATION = "image_classification"


class ModelAlgorithm(str, Enum):
    """Supported model algorithms."""
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    SVM = "svm"
    KNN = "knn"
    DECISION_TREE = "decision_tree"
    NAIVE_BAYES = "naive_bayes"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"


class ValidationStrategy(str, Enum):
    """Model validation strategies."""
    TRAIN_TEST_SPLIT = "train_test_split"
    CROSS_VALIDATION = "cross_validation"
    TIME_SERIES_SPLIT = "time_series_split"
    STRATIFIED_SPLIT = "stratified_split"


class ScrollModelFactory(BaseEngine):
    """
    ScrollModelFactory engine for custom model creation and deployment.
    
    Capabilities:
    - UI-driven model creation with parameter configuration
    - Custom model training pipeline with user-defined parameters
    - Model template system for common use cases
    - Model validation and testing framework
    - Model deployment automation with API endpoint generation
    """
    
    def __init__(self):
        super().__init__(
            engine_id="scroll_model_factory",
            name="ScrollModelFactory Engine",
            capabilities=[
                EngineCapability.ML_TRAINING,
                EngineCapability.DATA_ANALYSIS
            ]
        )
        self.models_dir = Path("models/custom")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir = Path("models/templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model templates and algorithms
        self._initialize_templates()
        self._initialize_algorithms()
    
    def _initialize_templates(self):
        """Initialize model templates for common use cases."""
        self.model_templates = {
            ModelTemplate.BINARY_CLASSIFICATION: {
                "name": "Binary Classification",
                "description": "Classify data into two categories",
                "recommended_algorithms": [
                    ModelAlgorithm.LOGISTIC_REGRESSION,
                    ModelAlgorithm.RANDOM_FOREST,
                    ModelAlgorithm.SVM
                ],
                "default_parameters": {
                    "test_size": 0.2,
                    "validation_strategy": ValidationStrategy.STRATIFIED_SPLIT,
                    "scoring_metric": "accuracy"
                },
                "preprocessing": ["standard_scaler"],
                "evaluation_metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            },
            ModelTemplate.MULTICLASS_CLASSIFICATION: {
                "name": "Multiclass Classification",
                "description": "Classify data into multiple categories",
                "recommended_algorithms": [
                    ModelAlgorithm.RANDOM_FOREST,
                    ModelAlgorithm.SVM,
                    ModelAlgorithm.DECISION_TREE
                ],
                "default_parameters": {
                    "test_size": 0.2,
                    "validation_strategy": ValidationStrategy.STRATIFIED_SPLIT,
                    "scoring_metric": "accuracy"
                },
                "preprocessing": ["standard_scaler"],
                "evaluation_metrics": ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
            },
            ModelTemplate.REGRESSION: {
                "name": "Regression",
                "description": "Predict continuous numerical values",
                "recommended_algorithms": [
                    ModelAlgorithm.LINEAR_REGRESSION,
                    ModelAlgorithm.RANDOM_FOREST,
                    ModelAlgorithm.DECISION_TREE
                ],
                "default_parameters": {
                    "test_size": 0.2,
                    "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT,
                    "scoring_metric": "r2"
                },
                "preprocessing": ["standard_scaler"],
                "evaluation_metrics": ["r2", "mse", "mae", "rmse"]
            },
            ModelTemplate.TIME_SERIES_FORECASTING: {
                "name": "Time Series Forecasting",
                "description": "Predict future values based on historical time series data",
                "recommended_algorithms": [
                    ModelAlgorithm.LINEAR_REGRESSION,
                    ModelAlgorithm.RANDOM_FOREST,
                    ModelAlgorithm.DECISION_TREE
                ],
                "default_parameters": {
                    "test_size": 0.2,
                    "validation_strategy": ValidationStrategy.TIME_SERIES_SPLIT,
                    "scoring_metric": "mse"
                },
                "preprocessing": ["time_features", "lag_features"],
                "evaluation_metrics": ["mse", "mae", "mape"]
            },
            ModelTemplate.ANOMALY_DETECTION: {
                "name": "Anomaly Detection",
                "description": "Identify unusual patterns or outliers in data",
                "recommended_algorithms": [
                    ModelAlgorithm.RANDOM_FOREST,
                    ModelAlgorithm.SVM
                ],
                "default_parameters": {
                    "test_size": 0.2,
                    "validation_strategy": ValidationStrategy.TRAIN_TEST_SPLIT,
                    "scoring_metric": "f1"
                },
                "preprocessing": ["standard_scaler", "outlier_detection"],
                "evaluation_metrics": ["precision", "recall", "f1_score"]
            }
        }
    
    def _initialize_algorithms(self):
        """Initialize algorithm configurations."""
        self.algorithm_configs = {
            ModelAlgorithm.RANDOM_FOREST: {
                "classifier": RandomForestClassifier,
                "regressor": RandomForestRegressor,
                "default_params": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "min_samples_split": 2,
                    "min_samples_leaf": 1,
                    "random_state": 42
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            ModelAlgorithm.LOGISTIC_REGRESSION: {
                "classifier": LogisticRegression,
                "regressor": None,
                "default_params": {
                    "random_state": 42,
                    "max_iter": 1000
                },
                "param_grid": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                }
            },
            ModelAlgorithm.LINEAR_REGRESSION: {
                "classifier": None,
                "regressor": LinearRegression,
                "default_params": {},
                "param_grid": {
                    "fit_intercept": [True, False],
                    "normalize": [True, False]
                }
            },
            ModelAlgorithm.SVM: {
                "classifier": SVC,
                "regressor": SVR,
                "default_params": {
                    "random_state": 42
                },
                "param_grid": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf", "poly"],
                    "gamma": ["scale", "auto"]
                }
            },
            ModelAlgorithm.KNN: {
                "classifier": KNeighborsClassifier,
                "regressor": KNeighborsRegressor,
                "default_params": {
                    "n_neighbors": 5
                },
                "param_grid": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                }
            },
            ModelAlgorithm.DECISION_TREE: {
                "classifier": DecisionTreeClassifier,
                "regressor": DecisionTreeRegressor,
                "default_params": {
                    "random_state": 42
                },
                "param_grid": {
                    "max_depth": [None, 5, 10, 15],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            ModelAlgorithm.NAIVE_BAYES: {
                "classifier": GaussianNB,
                "regressor": None,
                "default_params": {},
                "param_grid": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7]
                }
            }
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.algorithm_configs[ModelAlgorithm.XGBOOST] = {
                "classifier": xgb.XGBClassifier,
                "regressor": xgb.XGBRegressor,
                "default_params": {
                    "random_state": 42,
                    "eval_metric": "logloss"
                },
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0]
                }
            }
    
    async def initialize(self) -> None:
        """Initialize the ScrollModelFactory engine."""
        try:
            # Create necessary directories
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.templates_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model templates to disk
            await self._save_templates()
            
            logger.info("ScrollModelFactory engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ScrollModelFactory engine: {e}")
            raise
    
    async def _save_templates(self):
        """Save model templates to disk for UI consumption."""
        templates_file = self.templates_dir / "model_templates.json"
        
        # Convert templates to JSON-serializable format
        templates_data = {}
        for template_key, template_config in self.model_templates.items():
            templates_data[template_key.value] = {
                **template_config,
                "recommended_algorithms": [alg.value for alg in template_config["recommended_algorithms"]],
                "default_parameters": {
                    **template_config["default_parameters"],
                    "validation_strategy": template_config["default_parameters"]["validation_strategy"].value
                }
            }
        
        with open(templates_file, 'w') as f:
            json.dump(templates_data, f, indent=2)
        
        # Save algorithm configurations
        algorithms_file = self.templates_dir / "algorithm_configs.json"
        algorithms_data = {}
        for alg_key, alg_config in self.algorithm_configs.items():
            algorithms_data[alg_key.value] = {
                "default_params": alg_config["default_params"],
                "param_grid": alg_config["param_grid"],
                "supports_classification": alg_config["classifier"] is not None,
                "supports_regression": alg_config["regressor"] is not None
            }
        
        with open(algorithms_file, 'w') as f:
            json.dump(algorithms_data, f, indent=2)
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """Process custom model creation request."""
        if not parameters:
            raise ValueError("Parameters required for model creation")
        
        action = parameters.get("action")
        
        if action == "get_templates":
            return await self._get_templates()
        elif action == "get_algorithms":
            return await self._get_algorithms()
        elif action == "create_model":
            return await self._create_custom_model(input_data, parameters)
        elif action == "validate_model":
            return await self._validate_model(parameters)
        elif action == "deploy_model":
            return await self._deploy_model(parameters)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    async def _get_templates(self) -> Dict[str, Any]:
        """Get available model templates."""
        return {
            "templates": self.model_templates,
            "count": len(self.model_templates)
        }
    
    async def _get_algorithms(self) -> Dict[str, Any]:
        """Get available algorithms."""
        algorithms_info = {}
        for alg_key, alg_config in self.algorithm_configs.items():
            algorithms_info[alg_key.value] = {
                "name": alg_key.value.replace("_", " ").title(),
                "supports_classification": alg_config["classifier"] is not None,
                "supports_regression": alg_config["regressor"] is not None,
                "default_params": alg_config["default_params"],
                "tunable_params": list(alg_config["param_grid"].keys())
            }
        
        return {
            "algorithms": algorithms_info,
            "count": len(algorithms_info)
        }
    
    async def _create_custom_model(self, data: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom model with user-defined parameters."""
        try:
            # Extract parameters
            model_name = parameters.get("model_name", f"custom_model_{uuid4().hex[:8]}")
            algorithm = ModelAlgorithm(parameters.get("algorithm"))
            template = parameters.get("template")
            target_column = parameters.get("target_column")
            feature_columns = parameters.get("feature_columns", [])
            custom_params = parameters.get("custom_params", {})
            validation_strategy = ValidationStrategy(parameters.get("validation_strategy", "train_test_split"))
            hyperparameter_tuning = parameters.get("hyperparameter_tuning", False)
            
            # Validate inputs
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            if not feature_columns:
                feature_columns = [col for col in data.columns if col != target_column]
            
            # Prepare data
            X = data[feature_columns]
            y = data[target_column]
            
            # Determine problem type
            is_classification = self._is_classification_problem(y)
            
            # Get algorithm class
            alg_config = self.algorithm_configs[algorithm]
            if is_classification and alg_config["classifier"] is None:
                raise ValueError(f"Algorithm {algorithm.value} does not support classification")
            if not is_classification and alg_config["regressor"] is None:
                raise ValueError(f"Algorithm {algorithm.value} does not support regression")
            
            model_class = alg_config["classifier"] if is_classification else alg_config["regressor"]
            
            # Merge parameters
            model_params = {**alg_config["default_params"], **custom_params}
            
            # Create preprocessing pipeline
            preprocessing_steps = self._create_preprocessing_pipeline(template, is_classification)
            
            # Create model pipeline
            pipeline_steps = preprocessing_steps + [("model", model_class(**model_params))]
            pipeline = Pipeline(pipeline_steps)
            
            # Train model
            training_results = await self._train_model(
                pipeline, X, y, validation_strategy, is_classification, hyperparameter_tuning, alg_config
            )
            
            # Save model
            model_id = str(uuid4())
            model_path = self.models_dir / f"{model_id}.pkl"
            joblib.dump(training_results["model"], model_path)
            
            # Prepare response
            result = {
                "model_id": model_id,
                "model_name": model_name,
                "algorithm": algorithm.value,
                "template": template,
                "target_column": target_column,
                "feature_columns": feature_columns,
                "model_path": str(model_path),
                "is_classification": is_classification,
                "parameters": model_params,
                "metrics": training_results["metrics"],
                "validation_strategy": validation_strategy.value,
                "training_duration": training_results["training_duration"],
                "created_at": datetime.utcnow().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating custom model: {e}")
            raise
    
    def _is_classification_problem(self, y: pd.Series) -> bool:
        """Determine if the problem is classification or regression."""
        # Check if target is categorical or has limited unique values
        if y.dtype == 'object' or y.dtype.name == 'category':
            return True
        
        # If numeric, check if it looks like classification (limited unique values)
        unique_values = y.nunique()
        total_values = len(y)
        
        # If less than 20 unique values and they represent less than 5% of total, likely classification
        if unique_values < 20 and unique_values / total_values < 0.05:
            return True
        
        return False
    
    def _create_preprocessing_pipeline(self, template: str, is_classification: bool) -> List[Tuple[str, Any]]:
        """Create preprocessing pipeline based on template."""
        steps = []
        
        if template:
            template_config = self.model_templates.get(ModelTemplate(template), {})
            preprocessing = template_config.get("preprocessing", [])
            
            for prep_step in preprocessing:
                if prep_step == "standard_scaler":
                    steps.append(("scaler", StandardScaler()))
                elif prep_step == "minmax_scaler":
                    steps.append(("scaler", MinMaxScaler()))
                # Add more preprocessing steps as needed
        else:
            # Default preprocessing
            steps.append(("scaler", StandardScaler()))
        
        return steps
    
    async def _train_model(
        self, 
        pipeline: Pipeline, 
        X: pd.DataFrame, 
        y: pd.Series, 
        validation_strategy: ValidationStrategy,
        is_classification: bool,
        hyperparameter_tuning: bool,
        alg_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train the model with specified validation strategy."""
        start_time = datetime.utcnow()
        
        # Split data based on validation strategy
        if validation_strategy == ValidationStrategy.TRAIN_TEST_SPLIT:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=y if is_classification else None
            )
        elif validation_strategy == ValidationStrategy.STRATIFIED_SPLIT:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # Default to train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Hyperparameter tuning if requested
        if hyperparameter_tuning:
            param_grid = {}
            for param, values in alg_config["param_grid"].items():
                param_grid[f"model__{param}"] = values
            
            cv = StratifiedKFold(n_splits=5) if is_classification else KFold(n_splits=5)
            scoring = "accuracy" if is_classification else "r2"
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, is_classification)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            best_model, X_train, y_train, 
            cv=5, scoring="accuracy" if is_classification else "r2"
        )
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()
        
        training_duration = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "model": best_model,
            "metrics": metrics,
            "best_params": best_params,
            "training_duration": training_duration
        }
    
    def _calculate_metrics(self, y_true, y_pred, is_classification: bool) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        metrics = {}
        
        if is_classification:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        else:
            metrics["r2"] = r2_score(y_true, y_pred)
            metrics["mse"] = mean_squared_error(y_true, y_pred)
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])
        
        return metrics
    
    async def _validate_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a trained model."""
        model_id = parameters.get("model_id")
        validation_data = parameters.get("validation_data")
        
        if not model_id:
            raise ValueError("Model ID required for validation")
        
        # Load model
        model_path = self.models_dir / f"{model_id}.pkl"
        if not model_path.exists():
            raise ValueError(f"Model {model_id} not found")
        
        model = joblib.load(model_path)
        
        # Validate on new data if provided
        if validation_data is not None:
            # Perform validation
            predictions = model.predict(validation_data)
            
            return {
                "model_id": model_id,
                "validation_status": "success",
                "predictions": predictions.tolist(),
                "validation_timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "model_id": model_id,
            "validation_status": "model_loaded",
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _deploy_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy a model with API endpoint generation."""
        model_id = parameters.get("model_id")
        endpoint_name = parameters.get("endpoint_name", f"model_{model_id}")
        
        if not model_id:
            raise ValueError("Model ID required for deployment")
        
        # Load model
        model_path = self.models_dir / f"{model_id}.pkl"
        if not model_path.exists():
            raise ValueError(f"Model {model_id} not found")
        
        # Generate API endpoint
        api_endpoint = f"/api/models/{model_id}/predict"
        
        # Create deployment configuration
        deployment_config = {
            "model_id": model_id,
            "endpoint_name": endpoint_name,
            "api_endpoint": api_endpoint,
            "model_path": str(model_path),
            "deployment_timestamp": datetime.utcnow().isoformat(),
            "status": "deployed"
        }
        
        # Save deployment config
        deployment_file = self.models_dir / f"{model_id}_deployment.json"
        with open(deployment_file, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        return deployment_config
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("ScrollModelFactory engine cleanup completed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "models_directory": str(self.models_dir),
            "templates_directory": str(self.templates_dir),
            "available_templates": len(self.model_templates),
            "available_algorithms": len(self.algorithm_configs),
            "healthy": True
        }