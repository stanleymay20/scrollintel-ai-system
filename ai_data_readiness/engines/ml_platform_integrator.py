"""
ML Platform Integration Engine for AI Data Readiness Platform

This module provides connectors and integration capabilities for popular ML platforms
including MLflow, Kubeflow, and other ML lifecycle management tools.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import json
import requests
from abc import ABC, abstractmethod

# MLflow integration
try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Kubeflow integration
try:
    from kfp import Client as KubeflowClient
    KUBEFLOW_AVAILABLE = True
except ImportError:
    KUBEFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MLPlatformConfig:
    """Configuration for ML platform connections"""
    platform_type: str
    endpoint_url: str
    credentials: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class ModelDeploymentInfo:
    """Information about model deployment"""
    model_id: str
    model_name: str
    version: str
    deployment_status: str
    endpoint_url: Optional[str] = None
    performance_metrics: Dict[str, float] = None
    data_quality_correlation: Dict[str, float] = None
    created_at: datetime = None
    updated_at: datetime = None


@dataclass
class DataQualityCorrelation:
    """Correlation between data quality and model performance"""
    dataset_id: str
    model_id: str
    quality_score: float
    performance_score: float
    correlation_coefficient: float
    quality_dimensions: Dict[str, float]
    performance_metrics: Dict[str, float]
    timestamp: datetime


class MLPlatformConnector(ABC):
    """Abstract base class for ML platform connectors"""
    
    def __init__(self, config: MLPlatformConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the ML platform"""
        pass
    
    @abstractmethod
    def get_models(self) -> List[Dict[str, Any]]:
        """Retrieve list of models from the platform"""
        pass
    
    @abstractmethod
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        pass
    
    @abstractmethod
    def deploy_model(self, model_info: Dict[str, Any]) -> ModelDeploymentInfo:
        """Deploy a model to the platform"""
        pass
    
    @abstractmethod
    def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """Get performance metrics for a deployed model"""
        pass


class MLflowConnector(MLPlatformConnector):
    """Connector for MLflow platform"""
    
    def __init__(self, config: MLPlatformConfig):
        super().__init__(config)
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed. Install with: pip install mlflow")
        self.client = None
    
    def connect(self) -> bool:
        """Establish connection to MLflow tracking server"""
        try:
            mlflow.set_tracking_uri(self.config.endpoint_url)
            self.client = mlflow.tracking.MlflowClient()
            
            # Test connection by listing experiments
            experiments = self.client.search_experiments()
            self.logger.info(f"Connected to MLflow. Found {len(experiments)} experiments.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MLflow: {str(e)}")
            return False
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Retrieve list of registered models from MLflow"""
        try:
            models = []
            registered_models = self.client.search_registered_models()
            
            for model in registered_models:
                model_info = {
                    'id': model.name,
                    'name': model.name,
                    'description': model.description,
                    'creation_timestamp': model.creation_timestamp,
                    'last_updated_timestamp': model.last_updated_timestamp,
                    'latest_versions': []
                }
                
                # Get latest versions
                for version in model.latest_versions:
                    version_info = {
                        'version': version.version,
                        'stage': version.current_stage,
                        'status': version.status,
                        'run_id': version.run_id
                    }
                    model_info['latest_versions'].append(version_info)
                
                models.append(model_info)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve models from MLflow: {str(e)}")
            return []
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific MLflow model"""
        try:
            model = self.client.get_registered_model(model_id)
            
            model_info = {
                'id': model.name,
                'name': model.name,
                'description': model.description,
                'creation_timestamp': model.creation_timestamp,
                'last_updated_timestamp': model.last_updated_timestamp,
                'versions': []
            }
            
            # Get all versions
            for version in model.latest_versions:
                run = self.client.get_run(version.run_id)
                version_info = {
                    'version': version.version,
                    'stage': version.current_stage,
                    'status': version.status,
                    'run_id': version.run_id,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                }
                model_info['versions'].append(version_info)
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info from MLflow: {str(e)}")
            return {}
    
    def deploy_model(self, model_info: Dict[str, Any]) -> ModelDeploymentInfo:
        """Deploy a model using MLflow"""
        try:
            model_name = model_info['name']
            version = model_info.get('version', 'latest')
            
            # For MLflow, deployment typically involves transitioning to Production stage
            if version != 'latest':
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Production"
                )
            
            deployment_info = ModelDeploymentInfo(
                model_id=model_name,
                model_name=model_name,
                version=version,
                deployment_status="deployed",
                created_at=datetime.now()
            )
            
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model in MLflow: {str(e)}")
            raise
    
    def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """Get performance metrics for an MLflow model"""
        try:
            model_info = self.get_model_info(model_id)
            performance_metrics = {}
            
            # Aggregate metrics from all versions
            for version in model_info.get('versions', []):
                if version['stage'] == 'Production':
                    metrics = version.get('metrics', {})
                    performance_metrics.update(metrics)
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get model performance from MLflow: {str(e)}")
            return {}


class KubeflowConnector(MLPlatformConnector):
    """Connector for Kubeflow platform"""
    
    def __init__(self, config: MLPlatformConfig):
        super().__init__(config)
        if not KUBEFLOW_AVAILABLE:
            raise ImportError("Kubeflow Pipelines SDK is not installed. Install with: pip install kfp")
        self.client = None
    
    def connect(self) -> bool:
        """Establish connection to Kubeflow Pipelines"""
        try:
            self.client = KubeflowClient(host=self.config.endpoint_url)
            
            # Test connection by listing pipelines
            pipelines = self.client.list_pipelines()
            self.logger.info(f"Connected to Kubeflow. Found {pipelines.total_size} pipelines.")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Kubeflow: {str(e)}")
            return False
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Retrieve list of models from Kubeflow (via pipelines)"""
        try:
            models = []
            pipelines = self.client.list_pipelines()
            
            for pipeline in pipelines.pipelines:
                model_info = {
                    'id': pipeline.id,
                    'name': pipeline.name,
                    'description': pipeline.description,
                    'created_at': pipeline.created_at,
                    'default_version': pipeline.default_version
                }
                models.append(model_info)
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve models from Kubeflow: {str(e)}")
            return []
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a Kubeflow pipeline/model"""
        try:
            pipeline = self.client.get_pipeline(model_id)
            
            model_info = {
                'id': pipeline.id,
                'name': pipeline.name,
                'description': pipeline.description,
                'created_at': pipeline.created_at,
                'default_version': pipeline.default_version
            }
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to get model info from Kubeflow: {str(e)}")
            return {}
    
    def deploy_model(self, model_info: Dict[str, Any]) -> ModelDeploymentInfo:
        """Deploy a model using Kubeflow Pipelines"""
        try:
            pipeline_id = model_info['id']
            
            # Create a run of the pipeline
            run = self.client.run_pipeline(
                experiment_id=None,
                job_name=f"deployment_{model_info['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                pipeline_id=pipeline_id
            )
            
            deployment_info = ModelDeploymentInfo(
                model_id=pipeline_id,
                model_name=model_info['name'],
                version=model_info.get('version', '1.0'),
                deployment_status="running",
                created_at=datetime.now()
            )
            
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model in Kubeflow: {str(e)}")
            raise
    
    def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """Get performance metrics for a Kubeflow model"""
        try:
            # Get recent runs for the pipeline
            runs = self.client.list_runs(pipeline_id=model_id)
            performance_metrics = {}
            
            # Extract metrics from the most recent successful run
            for run in runs.runs:
                if run.status == 'Succeeded':
                    # This would need to be customized based on how metrics are stored
                    # in your Kubeflow setup
                    performance_metrics = {
                        'accuracy': 0.95,  # Placeholder
                        'precision': 0.93,  # Placeholder
                        'recall': 0.92     # Placeholder
                    }
                    break
            
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get model performance from Kubeflow: {str(e)}")
            return {}


class GenericMLPlatformConnector(MLPlatformConnector):
    """Generic connector for ML platforms with REST APIs"""
    
    def __init__(self, config: MLPlatformConfig):
        super().__init__(config)
        self.session = requests.Session()
        
        # Set up authentication if provided
        if 'api_key' in config.credentials:
            self.session.headers.update({
                'Authorization': f"Bearer {config.credentials['api_key']}"
            })
    
    def connect(self) -> bool:
        """Test connection to the ML platform"""
        try:
            response = self.session.get(f"{self.config.endpoint_url}/health")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Failed to connect to generic ML platform: {str(e)}")
            return False
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Retrieve models via REST API"""
        try:
            response = self.session.get(f"{self.config.endpoint_url}/models")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            self.logger.error(f"Failed to retrieve models: {str(e)}")
            return []
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get model information via REST API"""
        try:
            response = self.session.get(f"{self.config.endpoint_url}/models/{model_id}")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get model info: {str(e)}")
            return {}
    
    def deploy_model(self, model_info: Dict[str, Any]) -> ModelDeploymentInfo:
        """Deploy model via REST API"""
        try:
            response = self.session.post(
                f"{self.config.endpoint_url}/models/{model_info['id']}/deploy",
                json=model_info
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                return ModelDeploymentInfo(
                    model_id=model_info['id'],
                    model_name=model_info['name'],
                    version=model_info.get('version', '1.0'),
                    deployment_status=result.get('status', 'deployed'),
                    endpoint_url=result.get('endpoint_url'),
                    created_at=datetime.now()
                )
            else:
                raise Exception(f"Deployment failed with status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to deploy model: {str(e)}")
            raise
    
    def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """Get model performance via REST API"""
        try:
            response = self.session.get(f"{self.config.endpoint_url}/models/{model_id}/metrics")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            self.logger.error(f"Failed to get model performance: {str(e)}")
            return {}


class MLPlatformIntegrator:
    """Main integration engine for ML platforms"""
    
    def __init__(self):
        self.connectors: Dict[str, MLPlatformConnector] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_platform(self, platform_name: str, config: MLPlatformConfig) -> bool:
        """Register a new ML platform"""
        try:
            if config.platform_type.lower() == 'mlflow':
                connector = MLflowConnector(config)
            elif config.platform_type.lower() == 'kubeflow':
                connector = KubeflowConnector(config)
            else:
                connector = GenericMLPlatformConnector(config)
            
            if connector.connect():
                self.connectors[platform_name] = connector
                self.logger.info(f"Successfully registered platform: {platform_name}")
                return True
            else:
                self.logger.error(f"Failed to connect to platform: {platform_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to register platform {platform_name}: {str(e)}")
            return False
    
    def get_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get models from all registered platforms"""
        all_models = {}
        
        for platform_name, connector in self.connectors.items():
            try:
                models = connector.get_models()
                all_models[platform_name] = models
                self.logger.info(f"Retrieved {len(models)} models from {platform_name}")
            except Exception as e:
                self.logger.error(f"Failed to get models from {platform_name}: {str(e)}")
                all_models[platform_name] = []
        
        return all_models
    
    def deploy_model(self, platform_name: str, model_info: Dict[str, Any]) -> Optional[ModelDeploymentInfo]:
        """Deploy a model to a specific platform"""
        if platform_name not in self.connectors:
            self.logger.error(f"Platform {platform_name} not registered")
            return None
        
        try:
            connector = self.connectors[platform_name]
            deployment_info = connector.deploy_model(model_info)
            self.logger.info(f"Successfully deployed model {model_info['name']} to {platform_name}")
            return deployment_info
            
        except Exception as e:
            self.logger.error(f"Failed to deploy model to {platform_name}: {str(e)}")
            return None
    
    def correlate_data_quality_with_performance(
        self, 
        dataset_id: str, 
        quality_score: float,
        quality_dimensions: Dict[str, float]
    ) -> List[DataQualityCorrelation]:
        """Correlate data quality metrics with model performance across platforms"""
        correlations = []
        
        for platform_name, connector in self.connectors.items():
            try:
                models = connector.get_models()
                
                for model in models:
                    model_id = model['id']
                    performance_metrics = connector.get_model_performance(model_id)
                    
                    if performance_metrics:
                        # Calculate overall performance score
                        performance_score = self._calculate_performance_score(performance_metrics)
                        
                        # Calculate correlation coefficient
                        correlation_coeff = self._calculate_correlation(
                            quality_score, 
                            performance_score,
                            quality_dimensions,
                            performance_metrics
                        )
                        
                        correlation = DataQualityCorrelation(
                            dataset_id=dataset_id,
                            model_id=model_id,
                            quality_score=quality_score,
                            performance_score=performance_score,
                            correlation_coefficient=correlation_coeff,
                            quality_dimensions=quality_dimensions,
                            performance_metrics=performance_metrics,
                            timestamp=datetime.now()
                        )
                        
                        correlations.append(correlation)
                        
            except Exception as e:
                self.logger.error(f"Failed to correlate data quality for {platform_name}: {str(e)}")
        
        return correlations
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall performance score from metrics"""
        # Common performance metrics
        score_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
        
        scores = []
        for metric in score_metrics:
            if metric in metrics:
                scores.append(metrics[metric])
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_correlation(
        self, 
        quality_score: float, 
        performance_score: float,
        quality_dimensions: Dict[str, float],
        performance_metrics: Dict[str, float]
    ) -> float:
        """Calculate correlation coefficient between quality and performance"""
        # Simple correlation calculation
        # In practice, this would use more sophisticated statistical methods
        
        if quality_score == 0 or performance_score == 0:
            return 0.0
        
        # Basic correlation based on score similarity
        correlation = 1.0 - abs(quality_score - performance_score)
        
        # Adjust based on dimension-specific correlations
        dimension_correlations = []
        for dim, dim_score in quality_dimensions.items():
            if dim in ['accuracy', 'completeness'] and 'accuracy' in performance_metrics:
                dim_corr = 1.0 - abs(dim_score - performance_metrics['accuracy'])
                dimension_correlations.append(dim_corr)
        
        if dimension_correlations:
            avg_dim_corr = sum(dimension_correlations) / len(dimension_correlations)
            correlation = (correlation + avg_dim_corr) / 2
        
        return max(0.0, min(1.0, correlation))
    
    def get_platform_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered platforms"""
        status = {}
        
        for platform_name, connector in self.connectors.items():
            try:
                is_connected = connector.connect()
                models = connector.get_models()
                
                status[platform_name] = {
                    'connected': is_connected,
                    'platform_type': connector.config.platform_type,
                    'endpoint_url': connector.config.endpoint_url,
                    'model_count': len(models),
                    'last_checked': datetime.now().isoformat()
                }
                
            except Exception as e:
                status[platform_name] = {
                    'connected': False,
                    'error': str(e),
                    'last_checked': datetime.now().isoformat()
                }
        
        return status