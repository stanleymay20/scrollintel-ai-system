"""
AutoModel Engine for automated ML model training and deployment.
Implements requirement 3: Automated ML model building and deployment.
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

# ML libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score, 
    train_test_split, StratifiedKFold, KFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder

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


class ModelType:
    """Supported model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class AlgorithmType:
    """Supported algorithm types."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    NEURAL_NETWORK = "neural_network"


class AutoModelEngine(BaseEngine):
    """
    AutoModel engine for automated ML model training and deployment.
    
    Capabilities:
    - Multiple algorithm support (Random Forest, XGBoost, Neural Networks)
    - Automated hyperparameter tuning
    - Model comparison with cross-validation
    - Model export functionality
    - Performance metrics calculation
    """
    
    def __init__(self):
        super().__init__(
            engine_id="automodel_engine",
            name="AutoModel Engine",
            capabilities=[
                EngineCapability.ML_TRAINING,
                EngineCapability.DATA_ANALYSIS,
                EngineCapability.FORECASTING
            ]
        )
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        self.trained_models = {}
        self.model_configs = {
            AlgorithmType.RANDOM_FOREST: {
                "classifier": RandomForestClassifier,
                "regressor": RandomForestRegressor,
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
        }
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            self.model_configs[AlgorithmType.XGBOOST] = {
                "classifier": xgb.XGBClassifier,
                "regressor": xgb.XGBRegressor,
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 6, 10],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0]
                }
            }
        else:
            logger.warning("XGBoost not available. Install with: pip install xgboost")
    
    async def initialize(self) -> None:
        """Initialize the AutoModel engine."""
        logger.info("Initializing AutoModel engine...")
        
        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)
        
        # Load any existing trained models
        await self._load_existing_models()
        
        # Set status to ready
        self.status = EngineStatus.READY
        
        logger.info("AutoModel engine initialized successfully")
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """
        Process ML training request.
        
        Args:
            input_data: Dictionary containing dataset and training parameters
            parameters: Additional processing parameters
            
        Returns:
            Training results and model information
        """
        try:
            action = input_data.get("action", "train")
            
            if action == "train":
                return await self._train_models(input_data, parameters)
            elif action == "predict":
                return await self._predict(input_data, parameters)
            elif action == "compare":
                return await self._compare_models(input_data, parameters)
            elif action == "export":
                return await self._export_model(input_data, parameters)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error in AutoModel processing: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up AutoModel engine...")
        self.trained_models.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "healthy": True,
            "models_trained": len(self.trained_models),
            "supported_algorithms": list(self.model_configs.keys()),
            "models_directory": str(self.models_dir)
        }
    
    async def _train_models(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train multiple ML models automatically.
        
        Args:
            input_data: Contains dataset, target column, and training config
            parameters: Additional training parameters
            
        Returns:
            Training results with model performance metrics
        """
        start_time = datetime.utcnow()
        
        # Extract training data
        dataset_path = input_data.get("dataset_path")
        target_column = input_data.get("target_column")
        feature_columns = input_data.get("feature_columns", [])
        model_name = input_data.get("model_name", f"automodel_{uuid4().hex[:8]}")
        algorithms = input_data.get("algorithms", [AlgorithmType.RANDOM_FOREST, AlgorithmType.XGBOOST])
        
        if not dataset_path or not target_column:
            raise ValueError("dataset_path and target_column are required")
        
        # Load and prepare data
        df = await self._load_dataset(dataset_path)
        X, y, model_type = await self._prepare_data(df, target_column, feature_columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if model_type == ModelType.CLASSIFICATION else None
        )
        
        # Train models with different algorithms
        results = {}
        best_model = None
        best_score = -np.inf if model_type == ModelType.REGRESSION else 0
        
        for algorithm in algorithms:
            try:
                # Check if algorithm is available
                if algorithm == AlgorithmType.XGBOOST and not HAS_XGBOOST:
                    results[algorithm] = {"error": "XGBoost not installed"}
                    continue
                elif algorithm == AlgorithmType.NEURAL_NETWORK and not HAS_TENSORFLOW:
                    results[algorithm] = {"error": "TensorFlow not installed"}
                    continue
                elif algorithm not in self.model_configs:
                    results[algorithm] = {"error": f"Algorithm {algorithm} not supported"}
                    continue
                
                logger.info(f"Training {algorithm} model...")
                model_result = await self._train_single_model(
                    algorithm, X_train, X_test, y_train, y_test, model_type
                )
                results[algorithm] = model_result
                
                # Track best model
                score = model_result["metrics"]["primary_score"]
                if (model_type == ModelType.REGRESSION and score > best_score) or \
                   (model_type == ModelType.CLASSIFICATION and score > best_score):
                    best_score = score
                    best_model = {
                        "algorithm": algorithm,
                        "model": model_result["model"],
                        "metrics": model_result["metrics"]
                    }
                    
            except Exception as e:
                logger.error(f"Error training {algorithm}: {e}")
                results[algorithm] = {"error": str(e)}
        
        # Add neural network if requested
        if AlgorithmType.NEURAL_NETWORK in algorithms:
            try:
                logger.info("Training neural network model...")
                nn_result = await self._train_neural_network(
                    X_train, X_test, y_train, y_test, model_type
                )
                results[AlgorithmType.NEURAL_NETWORK] = nn_result
                
                score = nn_result["metrics"]["primary_score"]
                if (model_type == ModelType.REGRESSION and score > best_score) or \
                   (model_type == ModelType.CLASSIFICATION and score > best_score):
                    best_score = score
                    best_model = {
                        "algorithm": AlgorithmType.NEURAL_NETWORK,
                        "model": nn_result["model"],
                        "metrics": nn_result["metrics"]
                    }
                    
            except Exception as e:
                logger.error(f"Error training neural network: {e}")
                results[AlgorithmType.NEURAL_NETWORK] = {"error": str(e)}
        
        # Save best model
        if best_model:
            model_path = await self._save_model(
                best_model["model"], 
                model_name, 
                best_model["algorithm"]
            )
            
            # Store model info
            self.trained_models[model_name] = {
                "algorithm": best_model["algorithm"],
                "model_type": model_type,
                "model_path": model_path,
                "metrics": best_model["metrics"],
                "feature_columns": feature_columns or list(X.columns),
                "target_column": target_column,
                "trained_at": start_time
            }
        
        training_duration = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "model_name": model_name,
            "model_type": model_type,
            "algorithms_tested": algorithms,
            "results": results,
            "best_model": {
                "algorithm": best_model["algorithm"] if best_model else None,
                "score": best_score,
                "metrics": best_model["metrics"] if best_model else None
            },
            "training_duration_seconds": training_duration,
            "model_path": model_path if best_model else None
        }
    
    async def _train_single_model(
        self, 
        algorithm: str, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series, 
        model_type: str
    ) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning."""
        
        config = self.model_configs[algorithm]
        model_class = config["classifier"] if model_type == ModelType.CLASSIFICATION else config["regressor"]
        param_grid = config["param_grid"]
        
        # Create base model
        base_model = model_class(random_state=42)
        
        # Hyperparameter tuning
        cv_folds = 5
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) \
            if model_type == ModelType.CLASSIFICATION else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Use RandomizedSearchCV for faster tuning
        search = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=20,
            cv=cv,
            scoring='accuracy' if model_type == ModelType.CLASSIFICATION else 'r2',
            n_jobs=-1,
            random_state=42
        )
        
        # Fit the model
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
        # Calculate metrics
        metrics = await self._calculate_metrics(y_test, y_pred, model_type)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            best_model, X_train, y_train, cv=cv,
            scoring='accuracy' if model_type == ModelType.CLASSIFICATION else 'r2'
        )
        
        return {
            "model": best_model,
            "best_params": search.best_params_,
            "metrics": metrics,
            "cv_scores": {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "scores": cv_scores.tolist()
            }
        }
    
    async def _train_neural_network(
        self, 
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series, 
        model_type: str
    ) -> Dict[str, Any]:
        """Train a neural network model."""
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check if TensorFlow is available
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow not available. Install with: pip install tensorflow")
        
        # Prepare target for neural network
        if model_type == ModelType.CLASSIFICATION:
            num_classes = len(np.unique(y_train))
            y_train_encoded = keras.utils.to_categorical(y_train, num_classes)
            y_test_encoded = keras.utils.to_categorical(y_test, num_classes)
        else:
            y_train_encoded = y_train.values
            y_test_encoded = y_test.values
        
        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
        ])
        
        if model_type == ModelType.CLASSIFICATION:
            model.add(layers.Dense(num_classes, activation='softmax'))
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(layers.Dense(1))
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train_encoded,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Make predictions
        if model_type == ModelType.CLASSIFICATION:
            y_pred_proba = model.predict(X_test_scaled)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test_scaled).flatten()
        
        # Calculate metrics
        metrics = await self._calculate_metrics(y_test, y_pred, model_type)
        
        # Create wrapper for consistency with sklearn models
        class NeuralNetworkWrapper:
            def __init__(self, model, scaler):
                self.model = model
                self.scaler = scaler
                self.model_type = model_type
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                if self.model_type == ModelType.CLASSIFICATION:
                    pred_proba = self.model.predict(X_scaled)
                    return np.argmax(pred_proba, axis=1)
                else:
                    return self.model.predict(X_scaled).flatten()
        
        wrapped_model = NeuralNetworkWrapper(model, scaler)
        
        return {
            "model": wrapped_model,
            "metrics": metrics,
            "training_history": {
                "loss": history.history['loss'],
                "val_loss": history.history['val_loss']
            }
        }
    
    async def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, model_type: str) -> Dict[str, Any]:
        """Calculate performance metrics based on model type."""
        
        if model_type == ModelType.CLASSIFICATION:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted'),
                "recall": recall_score(y_true, y_pred, average='weighted'),
                "f1_score": f1_score(y_true, y_pred, average='weighted'),
                "primary_score": accuracy_score(y_true, y_pred),
                "classification_report": classification_report(y_true, y_pred, output_dict=True),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }
        else:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            metrics = {
                "mse": mse,
                "rmse": np.sqrt(mse),
                "mae": mae,
                "r2_score": r2,
                "primary_score": r2,
                "mean_residual": np.mean(y_true - y_pred),
                "std_residual": np.std(y_true - y_pred)
            }
        
        return metrics
    
    async def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset from file path."""
        
        if dataset_path.endswith('.csv'):
            return pd.read_csv(dataset_path)
        elif dataset_path.endswith('.xlsx'):
            return pd.read_excel(dataset_path)
        elif dataset_path.endswith('.json'):
            return pd.read_json(dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {dataset_path}")
    
    async def _prepare_data(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        feature_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, str]:
        """Prepare data for training."""
        
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        y = df[target_column]
        
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
        
        # Encode categorical variables
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
        
        # Determine model type
        if y.dtype == 'object' or len(y.unique()) < 20:
            model_type = ModelType.CLASSIFICATION
            if y.dtype == 'object':
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y), index=y.index)
        else:
            model_type = ModelType.REGRESSION
        
        return X, y, model_type
    
    async def _save_model(self, model: Any, model_name: str, algorithm: str) -> str:
        """Save trained model to disk."""
        
        model_path = self.models_dir / f"{model_name}_{algorithm}.pkl"
        
        if algorithm == AlgorithmType.NEURAL_NETWORK:
            # Save neural network separately
            model.model.save(str(model_path).replace('.pkl', '_nn.h5'))
            # Save scaler and wrapper info
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'scaler': model.scaler,
                    'model_type': model.model_type,
                    'nn_path': str(model_path).replace('.pkl', '_nn.h5')
                }, f)
        else:
            # Save sklearn models
            joblib.dump(model, model_path)
        
        return str(model_path)
    
    async def _load_existing_models(self) -> None:
        """Load existing trained models from disk."""
        
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem.split('_')[0]
                if model_name not in self.trained_models:
                    # Load model metadata if available
                    metadata_file = model_file.with_suffix('.json')
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            self.trained_models[model_name] = metadata
                            
            except Exception as e:
                logger.warning(f"Could not load model {model_file}: {e}")
    
    async def _predict(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using a trained model."""
        
        model_name = input_data.get("model_name")
        data = input_data.get("data")
        
        if not model_name or model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.trained_models[model_name]
        model_path = model_info["model_path"]
        
        # Load model
        if model_info["algorithm"] == AlgorithmType.NEURAL_NETWORK:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            nn_model = keras.models.load_model(model_data['nn_path'])
            scaler = model_data['scaler']
            
            class NeuralNetworkWrapper:
                def __init__(self, model, scaler, model_type):
                    self.model = model
                    self.scaler = scaler
                    self.model_type = model_type
                
                def predict(self, X):
                    X_scaled = self.scaler.transform(X)
                    if self.model_type == ModelType.CLASSIFICATION:
                        pred_proba = self.model.predict(X_scaled)
                        return np.argmax(pred_proba, axis=1)
                    else:
                        return self.model.predict(X_scaled).flatten()
            
            model = NeuralNetworkWrapper(nn_model, scaler, model_data['model_type'])
        else:
            model = joblib.load(model_path)
        
        # Prepare data
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        return {
            "model_name": model_name,
            "predictions": predictions.tolist(),
            "model_info": {
                "algorithm": model_info["algorithm"],
                "model_type": model_info["model_type"],
                "metrics": model_info["metrics"]
            }
        }
    
    async def _compare_models(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance of multiple trained models."""
        
        model_names = input_data.get("model_names", list(self.trained_models.keys()))
        
        comparison = {}
        for model_name in model_names:
            if model_name in self.trained_models:
                model_info = self.trained_models[model_name]
                comparison[model_name] = {
                    "algorithm": model_info["algorithm"],
                    "model_type": model_info["model_type"],
                    "metrics": model_info["metrics"],
                    "trained_at": model_info["trained_at"].isoformat()
                }
        
        return {
            "comparison": comparison,
            "total_models": len(comparison)
        }
    
    async def _export_model(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Export model for deployment."""
        
        model_name = input_data.get("model_name")
        export_format = input_data.get("format", "joblib")
        
        if not model_name or model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found")
        
        model_info = self.trained_models[model_name]
        model_path = model_info["model_path"]
        
        # Create export package
        export_dir = self.models_dir / f"{model_name}_export"
        export_dir.mkdir(exist_ok=True)
        
        # Copy model file
        import shutil
        if model_info["algorithm"] == AlgorithmType.NEURAL_NETWORK:
            # Copy both pickle and h5 files
            shutil.copy2(model_path, export_dir)
            nn_path = model_path.replace('.pkl', '_nn.h5')
            if os.path.exists(nn_path):
                shutil.copy2(nn_path, export_dir)
        else:
            shutil.copy2(model_path, export_dir)
        
        # Create metadata file
        metadata = {
            "model_name": model_name,
            "algorithm": model_info["algorithm"],
            "model_type": model_info["model_type"],
            "feature_columns": model_info["feature_columns"],
            "target_column": model_info["target_column"],
            "metrics": model_info["metrics"],
            "trained_at": model_info["trained_at"].isoformat(),
            "export_format": export_format
        }
        
        with open(export_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create deployment script
        deployment_script = f"""
# Model Deployment Script for {model_name}
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('{model_name}_{model_info["algorithm"]}.pkl')

# Example prediction function
def predict(data):
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    return model.predict(df)

# Example usage
# result = predict({{"feature1": 1.0, "feature2": 2.0}})
"""
        
        with open(export_dir / "deploy.py", 'w') as f:
            f.write(deployment_script)
        
        return {
            "model_name": model_name,
            "export_path": str(export_dir),
            "export_format": export_format,
            "files_created": [
                f"{model_name}_{model_info['algorithm']}.pkl",
                "metadata.json",
                "deploy.py"
            ]
        }