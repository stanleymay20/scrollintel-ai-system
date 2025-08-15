"""
ML Engineer Agent - Automated model building and deployment
"""
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import joblib
import os

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from .base import Agent, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)


class MLEngineerAgent(Agent):
    """ML Engineer Agent for automated model building and deployment"""
    
    def __init__(self):
        super().__init__(
            name="ML Engineer Agent",
            description="Builds, trains, and deploys machine learning models automatically"
        )
        self.models_dir = "models"
        self.ensure_models_directory()
        
        # Model configurations
        self.classification_models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42, probability=True)
        }
        
        self.regression_models = {
            'random_forest': RandomForestRegressor(random_state=42),
            'linear_regression': LinearRegression(),
            'svm': SVR()
        }
        
        # Hyperparameter grids
        self.param_grids = {
            'random_forest_clf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest_reg': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'svm_clf': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'svm_reg': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def ensure_models_directory(self):
        """Ensure models directory exists"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def get_capabilities(self) -> List[str]:
        """Return ML Engineer agent capabilities"""
        return [
            "Automated model selection based on data characteristics",
            "Hyperparameter tuning with GridSearch and RandomSearch",
            "Cross-validation with multiple strategies",
            "Model performance evaluation and comparison",
            "Model deployment pipeline with FastAPI endpoints",
            "Feature preprocessing and engineering",
            "Model persistence and versioning",
            "Performance monitoring and drift detection"
        ]
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process ML engineering requests"""
        start_time = time.time()
        
        try:
            query = request.query.lower()
            context = request.context
            
            # Handle different ML tasks
            if "build" in query or "train" in query:
                result = await self._build_and_train_model(context)
            elif "deploy" in query:
                result = await self._deploy_model(context)
            elif "evaluate" in query or "performance" in query:
                result = await self._evaluate_model(context)
            elif "hyperparameter" in query or "tune" in query:
                result = await self._tune_hyperparameters(context)
            elif "predict" in query:
                result = await self._make_predictions(context)
            elif "compare" in query:
                result = await self._compare_models(context)
            else:
                result = self._provide_ml_guidance(request.query, context)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                result=result,
                metadata={
                    "task_type": self._classify_ml_task(query),
                    "timestamp": datetime.utcnow().isoformat()
                },
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"ML Engineer Agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _build_and_train_model(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build and train ML model with automated pipeline"""
        try:
            # Extract data and parameters
            data = context.get("data")
            target_column = context.get("target_column")
            problem_type = context.get("problem_type", "auto")
            test_size = context.get("test_size", 0.2)
            
            if data is None:
                return self._get_model_building_guide()
            
            # Convert data to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            if target_column not in df.columns:
                return {
                    "error": f"Target column '{target_column}' not found in data",
                    "available_columns": list(df.columns)
                }
            
            # Automatic problem type detection
            if problem_type == "auto":
                problem_type = self._detect_problem_type(df[target_column])
            
            # Prepare data
            X, y = self._prepare_data(df, target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, 
                stratify=y if problem_type == "classification" else None
            )
            
            # Build and evaluate models
            results = await self._train_multiple_models(
                X_train, X_test, y_train, y_test, problem_type
            )
            
            # Select best model
            best_model_info = self._select_best_model(results, problem_type)
            
            # Save best model
            model_id = self._save_model(best_model_info["model"], best_model_info["preprocessor"])
            
            return {
                "success": True,
                "model_id": model_id,
                "problem_type": problem_type,
                "best_model": best_model_info["name"],
                "performance": best_model_info["metrics"],
                "all_results": results,
                "data_info": {
                    "total_samples": len(df),
                    "features": len(X.columns),
                    "train_samples": len(X_train),
                    "test_samples": len(X_test)
                },
                "next_steps": [
                    "Review model performance metrics",
                    "Consider hyperparameter tuning for better results",
                    "Deploy model for predictions",
                    "Monitor model performance over time"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in model building: {e}")
            return {"error": str(e), "guide": self._get_model_building_guide()}
    
    def _detect_problem_type(self, target_series: pd.Series) -> str:
        """Automatically detect if problem is classification or regression"""
        if target_series.dtype == 'object' or target_series.nunique() < 10:
            return "classification"
        else:
            return "regression"
    
    def _prepare_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle categorical variables in features
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle categorical target for classification
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
        
        return X, y
    
    async def _train_multiple_models(self, X_train, X_test, y_train, y_test, problem_type: str) -> Dict[str, Any]:
        """Train multiple models and compare performance"""
        results = {}
        
        # Select models based on problem type
        if problem_type == "classification":
            models = self.classification_models
        else:
            models = self.regression_models
        
        # Create preprocessor
        preprocessor = self._create_preprocessor(X_train)
        
        for name, model in models.items():
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, problem_type)
                
                # Cross-validation
                cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
                
                results[name] = {
                    "model": pipeline,
                    "preprocessor": preprocessor,
                    "metrics": metrics,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "predictions": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred
                }
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {"error": str(e)}
        
        return results
    
    def _create_preprocessor(self, X: pd.DataFrame):
        """Create preprocessing pipeline"""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
    
    def _calculate_metrics(self, y_true, y_pred, problem_type: str) -> Dict[str, float]:
        """Calculate appropriate metrics based on problem type"""
        if problem_type == "classification":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        else:
            return {
                "mae": mean_absolute_error(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "r2_score": r2_score(y_true, y_pred)
            }
    
    def _select_best_model(self, results: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
        """Select the best performing model"""
        best_model = None
        best_score = float('-inf') if problem_type == "classification" else float('inf')
        best_name = ""
        
        for name, result in results.items():
            if "error" in result:
                continue
                
            if problem_type == "classification":
                score = result["metrics"]["f1_score"]
                if score > best_score:
                    best_score = score
                    best_model = result
                    best_name = name
            else:
                score = result["metrics"]["r2_score"]
                if score > best_score:
                    best_score = score
                    best_model = result
                    best_name = name
        
        if best_model:
            best_model["name"] = best_name
        
        return best_model
    
    def _save_model(self, model, preprocessor) -> str:
        """Save trained model and return model ID"""
        model_id = f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        
        model_data = {
            "model": model,
            "preprocessor": preprocessor,
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved: {model_path}")
        
        return model_id
    
    async def _deploy_model(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy ML model with FastAPI endpoint"""
        model_id = context.get("model_id")
        
        if not model_id:
            return {
                "error": "Model ID required for deployment",
                "available_models": self._list_available_models()
            }
        
        # Generate FastAPI deployment code
        deployment_code = self._generate_deployment_code(model_id)
        
        return {
            "deployment_status": "ready",
            "model_id": model_id,
            "endpoint_code": deployment_code,
            "deployment_options": {
                "local": {
                    "command": f"uvicorn model_api:app --host 0.0.0.0 --port 8000",
                    "url": "http://localhost:8000"
                },
                "docker": {
                    "dockerfile": self._generate_dockerfile(),
                    "build_command": "docker build -t ml-model .",
                    "run_command": "docker run -p 8000:8000 ml-model"
                }
            },
            "api_endpoints": {
                "predict": "/predict",
                "health": "/health",
                "model_info": "/model/info"
            },
            "monitoring": {
                "metrics": ["prediction_latency", "throughput", "error_rate"],
                "logging": "structured_json",
                "health_checks": "automated"
            }
        }
    
    async def _evaluate_model(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        model_id = context.get("model_id")
        test_data = context.get("test_data")
        
        if not model_id:
            return {
                "error": "Model ID required for evaluation",
                "available_models": self._list_available_models()
            }
        
        try:
            # Load model
            model_data = self._load_model(model_id)
            if not model_data:
                return {"error": f"Model {model_id} not found"}
            
            model = model_data["model"]
            
            if test_data is not None:
                # Evaluate on provided test data
                if isinstance(test_data, dict):
                    test_df = pd.DataFrame(test_data)
                else:
                    test_df = test_data
                
                # Make predictions
                predictions = model.predict(test_df)
                
                return {
                    "model_id": model_id,
                    "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                    "evaluation_completed": True,
                    "model_info": {
                        "created_at": model_data.get("created_at"),
                        "version": model_data.get("version")
                    }
                }
            else:
                # Return model evaluation guide
                return {
                    "model_id": model_id,
                    "evaluation_guide": {
                        "classification_metrics": [
                            "Accuracy - Overall correctness",
                            "Precision - True positives / (True positives + False positives)",
                            "Recall - True positives / (True positives + False negatives)",
                            "F1-score - Harmonic mean of precision and recall",
                            "ROC-AUC - Area under ROC curve"
                        ],
                        "regression_metrics": [
                            "MAE - Mean Absolute Error",
                            "MSE - Mean Squared Error", 
                            "RMSE - Root Mean Squared Error",
                            "R² - Coefficient of determination"
                        ],
                        "validation_strategies": [
                            "Cross-validation for robust performance estimation",
                            "Hold-out validation for final model assessment",
                            "Time series validation for temporal data"
                        ]
                    },
                    "next_steps": [
                        "Provide test data for evaluation",
                        "Use model for predictions",
                        "Monitor model performance over time"
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {"error": str(e)}
    
    async def _tune_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Automated hyperparameter tuning"""
        try:
            data = context.get("data")
            target_column = context.get("target_column")
            model_type = context.get("model_type", "random_forest")
            problem_type = context.get("problem_type", "auto")
            search_method = context.get("search_method", "grid")  # grid or random
            cv_folds = context.get("cv_folds", 5)
            
            if not data or not target_column:
                return self._get_hyperparameter_tuning_guide()
            
            # Prepare data
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
                
            X, y = self._prepare_data(df, target_column)
            
            if problem_type == "auto":
                problem_type = self._detect_problem_type(y)
            
            # Select model and parameter grid
            if problem_type == "classification":
                if model_type == "random_forest":
                    model = RandomForestClassifier(random_state=42)
                    param_grid = self.param_grids['random_forest_clf']
                elif model_type == "logistic_regression":
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    param_grid = self.param_grids['logistic_regression']
                elif model_type == "svm":
                    model = SVC(random_state=42)
                    param_grid = self.param_grids['svm_clf']
                else:
                    return {"error": f"Unsupported classification model: {model_type}"}
            else:
                if model_type == "random_forest":
                    model = RandomForestRegressor(random_state=42)
                    param_grid = self.param_grids['random_forest_reg']
                elif model_type == "svm":
                    model = SVR()
                    param_grid = self.param_grids['svm_reg']
                else:
                    return {"error": f"Unsupported regression model: {model_type}"}
            
            # Create preprocessor and pipeline
            preprocessor = self._create_preprocessor(X)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Add model prefix to parameter names
            param_grid_pipeline = {f'model__{k}': v for k, v in param_grid.items()}
            
            # Perform hyperparameter search
            if search_method == "grid":
                search = GridSearchCV(
                    pipeline, param_grid_pipeline, cv=cv_folds, 
                    scoring='f1_weighted' if problem_type == "classification" else 'r2',
                    n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    pipeline, param_grid_pipeline, cv=cv_folds,
                    scoring='f1_weighted' if problem_type == "classification" else 'r2',
                    n_iter=20, random_state=42, n_jobs=-1
                )
            
            # Fit the search
            search.fit(X, y)
            
            # Save the best model
            model_id = self._save_model(search.best_estimator_, preprocessor)
            
            return {
                "success": True,
                "model_id": model_id,
                "model_type": model_type,
                "problem_type": problem_type,
                "search_method": search_method,
                "best_params": search.best_params_,
                "best_score": search.best_score_,
                "cv_results": {
                    "mean_test_scores": search.cv_results_['mean_test_score'].tolist(),
                    "std_test_scores": search.cv_results_['std_test_score'].tolist()
                },
                "tuning_summary": {
                    "total_combinations_tested": len(search.cv_results_['mean_test_score']),
                    "best_score_improvement": f"{search.best_score_:.4f}",
                    "cv_folds": cv_folds
                },
                "next_steps": [
                    "Evaluate tuned model on test set",
                    "Deploy model for predictions",
                    "Monitor model performance"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {e}")
            return {"error": str(e), "guide": self._get_hyperparameter_tuning_guide()}
    
    async def _make_predictions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using trained model"""
        model_id = context.get("model_id")
        input_data = context.get("input_data")
        
        if not model_id or not input_data:
            return {
                "error": "Both model_id and input_data required for predictions",
                "available_models": self._list_available_models()
            }
        
        try:
            # Load model
            model_data = self._load_model(model_id)
            if not model_data:
                return {"error": f"Model {model_id} not found"}
            
            model = model_data["model"]
            
            # Prepare input data
            if isinstance(input_data, dict):
                # Handle single prediction case
                if all(isinstance(v, (int, float, str)) for v in input_data.values()):
                    input_df = pd.DataFrame([input_data])
                else:
                    input_df = pd.DataFrame(input_data)
            elif isinstance(input_data, list):
                input_df = pd.DataFrame(input_data)
            else:
                input_df = input_data
            
            # Make predictions
            predictions = model.predict(input_df)
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(input_df)
                except:
                    pass
            
            result = {
                "model_id": model_id,
                "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                "input_samples": len(input_df)
            }
            
            if probabilities is not None:
                result["probabilities"] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {"error": str(e)}
    
    async def _compare_models(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compare multiple models"""
        model_ids = context.get("model_ids", [])
        
        if not model_ids:
            return {
                "error": "List of model_ids required for comparison",
                "available_models": self._list_available_models()
            }
        
        comparison_results = {}
        
        for model_id in model_ids:
            try:
                model_data = self._load_model(model_id)
                if model_data:
                    comparison_results[model_id] = {
                        "created_at": model_data.get("created_at"),
                        "version": model_data.get("version"),
                        "status": "loaded"
                    }
                else:
                    comparison_results[model_id] = {"status": "not_found"}
            except Exception as e:
                comparison_results[model_id] = {"status": "error", "error": str(e)}
        
        return {
            "comparison_results": comparison_results,
            "recommendation": "Use test data to evaluate and compare model performance",
            "next_steps": [
                "Evaluate each model on the same test dataset",
                "Compare performance metrics",
                "Select best performing model for deployment"
            ]
        }
    
    def _load_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved model"""
        model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
        
        if os.path.exists(model_path):
            try:
                return joblib.load(model_path)
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {e}")
                return None
        return None
    
    def _list_available_models(self) -> List[str]:
        """List all available saved models"""
        if not os.path.exists(self.models_dir):
            return []
        
        models = []
        for file in os.listdir(self.models_dir):
            if file.endswith('.joblib'):
                models.append(file.replace('.joblib', ''))
        
        return models
    
    def _generate_deployment_code(self, model_id: str) -> str:
        """Generate FastAPI deployment code"""
        return f'''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List, Dict, Any
import os

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model
MODEL_PATH = "models/{model_id}.joblib"
model_data = joblib.load(MODEL_PATH)
model = model_data["model"]

class PredictionRequest(BaseModel):
    data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_id: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(df)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_id="{model_id}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    return {{"status": "healthy", "model_id": "{model_id}"}}

@app.get("/model/info")
async def model_info():
    return {{
        "model_id": "{model_id}",
        "created_at": model_data.get("created_at"),
        "version": model_data.get("version")
    }}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile for model deployment"""
        return '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "model_api:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    def _get_model_building_guide(self) -> Dict[str, Any]:
        """Get model building guide"""
        return {
            "required_parameters": {
                "data": "Training dataset (DataFrame, dict, or list)",
                "target_column": "Name of the target variable column",
                "problem_type": "auto, classification, or regression",
                "test_size": "Fraction of data for testing (default: 0.2)"
            },
            "example_usage": {
                "data": [
                    {"feature1": 1.0, "feature2": 2.0, "target": 0},
                    {"feature1": 1.5, "feature2": 2.5, "target": 1}
                ],
                "target_column": "target",
                "problem_type": "classification"
            },
            "supported_models": {
                "classification": ["Random Forest", "Logistic Regression", "SVM"],
                "regression": ["Random Forest", "Linear Regression", "SVM"]
            }
        }
    
    def _get_hyperparameter_tuning_guide(self) -> Dict[str, Any]:
        """Get hyperparameter tuning guide"""
        return {
            "required_parameters": {
                "data": "Training dataset",
                "target_column": "Target variable column name",
                "model_type": "random_forest, logistic_regression, or svm",
                "search_method": "grid or random (default: grid)",
                "cv_folds": "Number of cross-validation folds (default: 5)"
            },
            "supported_models": {
                "classification": ["random_forest", "logistic_regression", "svm"],
                "regression": ["random_forest", "svm"]
            },
            "search_methods": {
                "grid": "Exhaustive search over all parameter combinations",
                "random": "Random sampling of parameter space (faster)"
            }
        }
    
    def _provide_ml_guidance(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide general ML engineering guidance"""
        return {
            "ml_workflow": [
                "1. Problem definition and data collection",
                "2. Exploratory data analysis",
                "3. Data preprocessing and feature engineering",
                "4. Model selection and training",
                "5. Model evaluation and validation",
                "6. Hyperparameter tuning",
                "7. Deployment and monitoring"
            ],
            "available_commands": {
                "build_model": "Build and train ML models automatically",
                "tune_hyperparameters": "Optimize model parameters",
                "evaluate_model": "Assess model performance",
                "make_predictions": "Use trained models for predictions",
                "deploy_model": "Generate deployment code",
                "compare_models": "Compare multiple trained models"
            },
            "best_practices": [
                "Start with simple models and baseline",
                "Always validate on unseen data",
                "Use cross-validation for robust evaluation",
                "Monitor model performance in production",
                "Version control for models and data",
                "Document model assumptions and limitations"
            ],
            "common_pitfalls": [
                "Data leakage from future information",
                "Overfitting to training data",
                "Insufficient validation",
                "Ignoring data drift over time",
                "Poor feature engineering"
            ],
            "next_steps": [
                "Provide training data to build your first model",
                "Specify target column and problem type",
                "Review model performance and tune if needed",
                "Deploy model for production use"
            ]
        }
    
    def _classify_ml_task(self, query: str) -> str:
        """Classify the type of ML task"""
        if "build" in query or "train" in query:
            return "model_building"
        elif "deploy" in query:
            return "model_deployment"
        elif "evaluate" in query or "performance" in query:
            return "model_evaluation"
        elif "hyperparameter" in query or "tune" in query:
            return "hyperparameter_tuning"
        elif "predict" in query:
            return "prediction"
        elif "compare" in query:
            return "model_comparison"
        else:
            return "general_ml_guidance"