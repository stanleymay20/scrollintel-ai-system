"""
Predictive Failure Prevention Engine for ScrollIntel
Advanced system that predicts and prevents failures before they impact users
"""

import asyncio
import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import json
import pickle
import os
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import aiohttp
import sqlite3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from .monitoring import metrics_collector, PerformanceMetrics
from .failure_prevention import failure_prevention, FailureEvent, FailureType

logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    """Confidence levels for failure predictions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(Enum):
    """Types of anomalies detected"""
    RESOURCE_SPIKE = "resource_spike"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_INCREASE = "error_rate_increase"
    DEPENDENCY_FAILURE = "dependency_failure"
    UNUSUAL_PATTERN = "unusual_pattern"
    CAPACITY_THRESHOLD = "capacity_threshold"


class ScalingAction(Enum):
    """Types of scaling actions"""
    SCALE_UP_CPU = "scale_up_cpu"
    SCALE_UP_MEMORY = "scale_up_memory"
    SCALE_UP_STORAGE = "scale_up_storage"
    SCALE_DOWN = "scale_down"
    OPTIMIZE_QUERIES = "optimize_queries"
    CLEAR_CACHE = "clear_cache"
    RESTART_SERVICE = "restart_service"


@dataclass
class SystemHealthMetrics:
    """Comprehensive system health metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    disk_io: Dict[str, float]
    active_connections: int
    response_time_avg: float
    response_time_p95: float
    error_rate: float
    request_rate: float
    queue_depth: int
    cache_hit_rate: float
    database_connections: int
    database_query_time: float
    external_api_latency: Dict[str, float]
    user_sessions: int
    agent_processing_time: float


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_type: AnomalyType
    confidence: PredictionConfidence
    timestamp: datetime
    affected_metrics: List[str]
    anomaly_score: float
    description: str
    recommended_actions: List[str]
    predicted_impact: str
    time_to_failure: Optional[timedelta]


@dataclass
class FailurePrediction:
    """Failure prediction result"""
    failure_type: FailureType
    confidence: PredictionConfidence
    predicted_time: datetime
    affected_components: List[str]
    root_cause_analysis: Dict[str, Any]
    prevention_actions: List[str]
    impact_assessment: str
    probability: float


@dataclass
class DependencyHealth:
    """Health status of external dependencies"""
    service_name: str
    endpoint: str
    status: str  # healthy, degraded, failed
    response_time: float
    error_rate: float
    last_check: datetime
    consecutive_failures: int
    availability: float  # percentage over time window


@dataclass
class ResourceOptimization:
    """Resource optimization recommendation"""
    resource_type: str
    current_usage: float
    predicted_usage: float
    recommended_action: ScalingAction
    urgency: PredictionConfidence
    estimated_impact: str
    cost_benefit: Dict[str, float]


class HealthMonitor:
    """Advanced system health monitoring with anomaly detection"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.anomalies: List[AnomalyDetection] = []
        self.baseline_metrics = {}
        self._lock = threading.Lock()
        
    async def collect_comprehensive_metrics(self) -> SystemHealthMetrics:
        """Collect comprehensive system health metrics"""
        try:
            # Basic system metrics
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_io = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
            
            # Disk I/O
            disk_io_counters = psutil.disk_io_counters()
            disk_io = {
                'read_bytes': disk_io_counters.read_bytes if disk_io_counters else 0,
                'write_bytes': disk_io_counters.write_bytes if disk_io_counters else 0,
                'read_count': disk_io_counters.read_count if disk_io_counters else 0,
                'write_count': disk_io_counters.write_count if disk_io_counters else 0
            }
            
            # Application-specific metrics (mock values for now)
            metrics = SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=(disk.used / disk.total) * 100,
                network_io=network_io,
                disk_io=disk_io,
                active_connections=len(psutil.net_connections()),
                response_time_avg=0.5,  # Would come from request metrics
                response_time_p95=1.2,  # Would come from request metrics
                error_rate=0.1,  # Would come from error metrics
                request_rate=100.0,  # Would come from request metrics
                queue_depth=5,  # Would come from queue metrics
                cache_hit_rate=0.85,  # Would come from cache metrics
                database_connections=10,  # Would come from DB metrics
                database_query_time=0.05,  # Would come from DB metrics
                external_api_latency={'openai': 1.5, 'anthropic': 2.0},  # Mock
                user_sessions=25,  # Would come from session metrics
                agent_processing_time=3.0  # Would come from agent metrics
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting comprehensive metrics: {e}")
            return None
    
    def add_metrics(self, metrics: SystemHealthMetrics):
        """Add metrics to history and update baseline"""
        with self._lock:
            self.metrics_history.append(metrics)
            self._update_baseline(metrics)
            
            # Train anomaly detector if we have enough data
            if len(self.metrics_history) >= 50 and not self.is_trained:
                self._train_anomaly_detector()
    
    def _update_baseline(self, metrics: SystemHealthMetrics):
        """Update baseline metrics for comparison"""
        if not self.baseline_metrics:
            self.baseline_metrics = asdict(metrics)
        else:
            # Exponential moving average
            alpha = 0.1
            for key, value in asdict(metrics).items():
                if isinstance(value, (int, float)):
                    if key in self.baseline_metrics:
                        self.baseline_metrics[key] = (
                            alpha * value + (1 - alpha) * self.baseline_metrics[key]
                        )
                    else:
                        self.baseline_metrics[key] = value
    
    def _train_anomaly_detector(self):
        """Train the anomaly detection model"""
        try:
            # Convert metrics to feature matrix
            features = self._extract_features()
            if len(features) < 10:
                return
                
            # Normalize features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train anomaly detector
            self.anomaly_detector.fit(features_scaled)
            self.is_trained = True
            
            logger.info("Anomaly detector trained successfully")
            
        except Exception as e:
            logger.error(f"Error training anomaly detector: {e}")
    
    def _extract_features(self) -> np.ndarray:
        """Extract numerical features from metrics history"""
        features = []
        
        for metrics in self.metrics_history:
            feature_vector = [
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.disk_usage,
                metrics.active_connections,
                metrics.response_time_avg,
                metrics.response_time_p95,
                metrics.error_rate,
                metrics.request_rate,
                metrics.queue_depth,
                metrics.cache_hit_rate,
                metrics.database_connections,
                metrics.database_query_time,
                metrics.user_sessions,
                metrics.agent_processing_time
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    async def detect_anomalies(self, metrics: SystemHealthMetrics) -> List[AnomalyDetection]:
        """Detect anomalies in current metrics"""
        anomalies = []
        
        if not self.is_trained or not self.baseline_metrics:
            return anomalies
        
        try:
            # Extract features for current metrics
            current_features = np.array([[
                metrics.cpu_usage,
                metrics.memory_usage,
                metrics.disk_usage,
                metrics.active_connections,
                metrics.response_time_avg,
                metrics.response_time_p95,
                metrics.error_rate,
                metrics.request_rate,
                metrics.queue_depth,
                metrics.cache_hit_rate,
                metrics.database_connections,
                metrics.database_query_time,
                metrics.user_sessions,
                metrics.agent_processing_time
            ]])
            
            # Scale features
            current_features_scaled = self.scaler.transform(current_features)
            
            # Detect anomaly
            anomaly_score = self.anomaly_detector.decision_function(current_features_scaled)[0]
            is_anomaly = self.anomaly_detector.predict(current_features_scaled)[0] == -1
            
            if is_anomaly:
                # Determine anomaly type and details
                anomaly = await self._analyze_anomaly(metrics, anomaly_score)
                if anomaly:
                    anomalies.append(anomaly)
            
            # Check for specific threshold-based anomalies
            threshold_anomalies = await self._check_threshold_anomalies(metrics)
            anomalies.extend(threshold_anomalies)
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
        
        return anomalies
    
    async def _analyze_anomaly(self, metrics: SystemHealthMetrics, 
                             anomaly_score: float) -> Optional[AnomalyDetection]:
        """Analyze detected anomaly to determine type and severity"""
        try:
            affected_metrics = []
            anomaly_type = AnomalyType.UNUSUAL_PATTERN
            
            # Compare with baseline to identify affected metrics
            baseline = self.baseline_metrics
            
            if metrics.cpu_usage > baseline.get('cpu_usage', 0) * 1.5:
                affected_metrics.append('cpu_usage')
                anomaly_type = AnomalyType.RESOURCE_SPIKE
            
            if metrics.memory_usage > baseline.get('memory_usage', 0) * 1.3:
                affected_metrics.append('memory_usage')
                anomaly_type = AnomalyType.RESOURCE_SPIKE
            
            if metrics.response_time_avg > baseline.get('response_time_avg', 0) * 2:
                affected_metrics.append('response_time_avg')
                anomaly_type = AnomalyType.PERFORMANCE_DEGRADATION
            
            if metrics.error_rate > baseline.get('error_rate', 0) * 3:
                affected_metrics.append('error_rate')
                anomaly_type = AnomalyType.ERROR_RATE_INCREASE
            
            # Determine confidence based on anomaly score
            if anomaly_score < -0.5:
                confidence = PredictionConfidence.CRITICAL
            elif anomaly_score < -0.3:
                confidence = PredictionConfidence.HIGH
            elif anomaly_score < -0.1:
                confidence = PredictionConfidence.MEDIUM
            else:
                confidence = PredictionConfidence.LOW
            
            # Generate recommendations
            recommendations = self._generate_anomaly_recommendations(
                anomaly_type, affected_metrics, metrics
            )
            
            return AnomalyDetection(
                anomaly_type=anomaly_type,
                confidence=confidence,
                timestamp=metrics.timestamp,
                affected_metrics=affected_metrics,
                anomaly_score=anomaly_score,
                description=f"Anomaly detected in {', '.join(affected_metrics)}",
                recommended_actions=recommendations,
                predicted_impact="Potential service degradation",
                time_to_failure=timedelta(minutes=15) if confidence == PredictionConfidence.CRITICAL else None
            )
            
        except Exception as e:
            logger.error(f"Error analyzing anomaly: {e}")
            return None
    
    async def _check_threshold_anomalies(self, metrics: SystemHealthMetrics) -> List[AnomalyDetection]:
        """Check for threshold-based anomalies"""
        anomalies = []
        
        # CPU threshold
        if metrics.cpu_usage > 85:
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.CAPACITY_THRESHOLD,
                confidence=PredictionConfidence.HIGH,
                timestamp=metrics.timestamp,
                affected_metrics=['cpu_usage'],
                anomaly_score=-0.8,
                description=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                recommended_actions=['Scale up CPU resources', 'Optimize CPU-intensive processes'],
                predicted_impact="Performance degradation imminent",
                time_to_failure=timedelta(minutes=5)
            ))
        
        # Memory threshold
        if metrics.memory_usage > 90:
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.CAPACITY_THRESHOLD,
                confidence=PredictionConfidence.CRITICAL,
                timestamp=metrics.timestamp,
                affected_metrics=['memory_usage'],
                anomaly_score=-0.9,
                description=f"Critical memory usage: {metrics.memory_usage:.1f}%",
                recommended_actions=['Scale up memory', 'Clear caches', 'Restart memory-intensive services'],
                predicted_impact="System failure imminent",
                time_to_failure=timedelta(minutes=2)
            ))
        
        # Disk threshold
        if metrics.disk_usage > 95:
            anomalies.append(AnomalyDetection(
                anomaly_type=AnomalyType.CAPACITY_THRESHOLD,
                confidence=PredictionConfidence.CRITICAL,
                timestamp=metrics.timestamp,
                affected_metrics=['disk_usage'],
                anomaly_score=-0.95,
                description=f"Critical disk usage: {metrics.disk_usage:.1f}%",
                recommended_actions=['Clean up disk space', 'Archive old data', 'Scale up storage'],
                predicted_impact="System failure imminent",
                time_to_failure=timedelta(minutes=1)
            ))
        
        return anomalies
    
    def _generate_anomaly_recommendations(self, anomaly_type: AnomalyType, 
                                        affected_metrics: List[str], 
                                        metrics: SystemHealthMetrics) -> List[str]:
        """Generate recommendations based on anomaly type"""
        recommendations = []
        
        if anomaly_type == AnomalyType.RESOURCE_SPIKE:
            if 'cpu_usage' in affected_metrics:
                recommendations.extend(['Scale up CPU resources', 'Optimize CPU-intensive processes'])
            if 'memory_usage' in affected_metrics:
                recommendations.extend(['Scale up memory', 'Clear caches'])
        
        elif anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION:
            recommendations.extend([
                'Check database query performance',
                'Review recent deployments',
                'Scale up resources',
                'Enable performance profiling'
            ])
        
        elif anomaly_type == AnomalyType.ERROR_RATE_INCREASE:
            recommendations.extend([
                'Check application logs',
                'Review recent changes',
                'Enable circuit breakers',
                'Scale up error handling capacity'
            ])
        
        return recommendations


class FailurePredictor:
    """Machine learning-based failure prediction system"""
    
    def __init__(self):
        self.models = {}
        self.feature_history = deque(maxlen=1000)
        self.failure_history = []
        self.prediction_accuracy = {}
        self._lock = threading.Lock()
        
        # Load existing models if available
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            models_dir = Path("data/models")
            if models_dir.exists():
                for model_file in models_dir.glob("*.pkl"):
                    failure_type = model_file.stem
                    with open(model_file, 'rb') as f:
                        self.models[failure_type] = pickle.load(f)
                logger.info(f"Loaded {len(self.models)} prediction models")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            models_dir = Path("data/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            for failure_type, model in self.models.items():
                model_file = models_dir / f"{failure_type}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"Saved {len(self.models)} prediction models")
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def add_training_data(self, metrics: SystemHealthMetrics, 
                              failure_event: Optional[FailureEvent] = None):
        """Add training data for model improvement"""
        with self._lock:
            # Extract features
            features = self._extract_prediction_features(metrics)
            self.feature_history.append((features, metrics.timestamp))
            
            # If there was a failure, record it for training
            if failure_event:
                self.failure_history.append({
                    'failure_type': failure_event.failure_type.value,
                    'timestamp': failure_event.timestamp,
                    'features': features
                })
                
                # Retrain models periodically
                if len(self.failure_history) % 10 == 0:
                    await self._retrain_models()
    
    def _extract_prediction_features(self, metrics: SystemHealthMetrics) -> List[float]:
        """Extract features for failure prediction"""
        return [
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.disk_usage,
            metrics.response_time_avg,
            metrics.response_time_p95,
            metrics.error_rate,
            metrics.request_rate,
            metrics.database_query_time,
            metrics.agent_processing_time,
            # Derived features
            metrics.cpu_usage / 100.0 * metrics.memory_usage / 100.0,  # Resource pressure
            metrics.error_rate * metrics.request_rate,  # Error volume
            metrics.response_time_p95 / max(metrics.response_time_avg, 0.001),  # Response time variance
        ]
    
    async def predict_failures(self, metrics: SystemHealthMetrics) -> List[FailurePrediction]:
        """Predict potential failures based on current metrics"""
        predictions = []
        
        try:
            features = self._extract_prediction_features(metrics)
            
            # Use pattern-based prediction if no trained models
            if not self.models:
                predictions.extend(await self._pattern_based_prediction(metrics, features))
            else:
                # Use trained models for prediction
                for failure_type, model in self.models.items():
                    try:
                        probability = model.predict_proba([features])[0][1]  # Probability of failure
                        
                        if probability > 0.3:  # Threshold for prediction
                            confidence = self._probability_to_confidence(probability)
                            predicted_time = datetime.utcnow() + self._estimate_time_to_failure(
                                probability, failure_type
                            )
                            
                            predictions.append(FailurePrediction(
                                failure_type=FailureType(failure_type),
                                confidence=confidence,
                                predicted_time=predicted_time,
                                affected_components=self._identify_affected_components(failure_type, features),
                                root_cause_analysis=self._analyze_root_cause(failure_type, features),
                                prevention_actions=self._generate_prevention_actions(failure_type),
                                impact_assessment=self._assess_impact(failure_type),
                                probability=probability
                            ))
                    except Exception as e:
                        logger.error(f"Error predicting {failure_type}: {e}")
            
        except Exception as e:
            logger.error(f"Error in failure prediction: {e}")
        
        return predictions
    
    async def _pattern_based_prediction(self, metrics: SystemHealthMetrics, 
                                      features: List[float]) -> List[FailurePrediction]:
        """Pattern-based failure prediction when no trained models available"""
        predictions = []
        
        # CPU overload prediction
        if metrics.cpu_usage > 80:
            probability = min((metrics.cpu_usage - 80) / 20, 1.0)
            predictions.append(FailurePrediction(
                failure_type=FailureType.CPU_OVERLOAD,
                confidence=self._probability_to_confidence(probability),
                predicted_time=datetime.utcnow() + timedelta(minutes=10),
                affected_components=['application_server', 'background_jobs'],
                root_cause_analysis={'cpu_usage': metrics.cpu_usage, 'threshold': 80},
                prevention_actions=['Scale up CPU', 'Optimize processes'],
                impact_assessment='Performance degradation, potential timeouts',
                probability=probability
            ))
        
        # Memory exhaustion prediction
        if metrics.memory_usage > 85:
            probability = min((metrics.memory_usage - 85) / 15, 1.0)
            predictions.append(FailurePrediction(
                failure_type=FailureType.MEMORY_ERROR,
                confidence=self._probability_to_confidence(probability),
                predicted_time=datetime.utcnow() + timedelta(minutes=5),
                affected_components=['application_server', 'cache'],
                root_cause_analysis={'memory_usage': metrics.memory_usage, 'threshold': 85},
                prevention_actions=['Scale up memory', 'Clear caches', 'Restart services'],
                impact_assessment='Application crashes, data loss risk',
                probability=probability
            ))
        
        # Database performance prediction
        if metrics.database_query_time > 1.0:
            probability = min(metrics.database_query_time / 5.0, 1.0)
            predictions.append(FailurePrediction(
                failure_type=FailureType.DATABASE_ERROR,
                confidence=self._probability_to_confidence(probability),
                predicted_time=datetime.utcnow() + timedelta(minutes=15),
                affected_components=['database', 'api_endpoints'],
                root_cause_analysis={'query_time': metrics.database_query_time, 'threshold': 1.0},
                prevention_actions=['Optimize queries', 'Scale database', 'Add indexes'],
                impact_assessment='Slow responses, potential timeouts',
                probability=probability
            ))
        
        return predictions
    
    def _probability_to_confidence(self, probability: float) -> PredictionConfidence:
        """Convert probability to confidence level"""
        if probability >= 0.8:
            return PredictionConfidence.CRITICAL
        elif probability >= 0.6:
            return PredictionConfidence.HIGH
        elif probability >= 0.4:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def _estimate_time_to_failure(self, probability: float, failure_type: str) -> timedelta:
        """Estimate time until failure based on probability and type"""
        base_times = {
            'memory_error': timedelta(minutes=5),
            'cpu_overload': timedelta(minutes=10),
            'database_error': timedelta(minutes=15),
            'disk_full': timedelta(minutes=30),
            'network_error': timedelta(minutes=2)
        }
        
        base_time = base_times.get(failure_type, timedelta(minutes=10))
        # Higher probability means shorter time to failure
        multiplier = max(0.1, 1.0 - probability)
        return base_time * multiplier
    
    def _identify_affected_components(self, failure_type: str, features: List[float]) -> List[str]:
        """Identify components that would be affected by the failure"""
        component_map = {
            'memory_error': ['application_server', 'cache', 'background_jobs'],
            'cpu_overload': ['application_server', 'api_endpoints', 'agents'],
            'database_error': ['database', 'api_endpoints', 'data_processing'],
            'disk_full': ['logging', 'file_storage', 'database'],
            'network_error': ['external_apis', 'user_connections', 'data_sync']
        }
        
        return component_map.get(failure_type, ['unknown'])
    
    def _analyze_root_cause(self, failure_type: str, features: List[float]) -> Dict[str, Any]:
        """Analyze potential root causes of the predicted failure"""
        return {
            'failure_type': failure_type,
            'contributing_factors': {
                'cpu_usage': features[0],
                'memory_usage': features[1],
                'disk_usage': features[2],
                'response_time': features[3],
                'error_rate': features[5]
            },
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_prevention_actions(self, failure_type: str) -> List[str]:
        """Generate prevention actions for the predicted failure"""
        action_map = {
            'memory_error': [
                'Scale up memory resources',
                'Clear application caches',
                'Restart memory-intensive services',
                'Enable memory monitoring alerts'
            ],
            'cpu_overload': [
                'Scale up CPU resources',
                'Optimize CPU-intensive processes',
                'Distribute load across instances',
                'Enable CPU throttling'
            ],
            'database_error': [
                'Optimize database queries',
                'Scale database resources',
                'Add database indexes',
                'Enable query caching'
            ],
            'disk_full': [
                'Clean up temporary files',
                'Archive old data',
                'Scale up storage',
                'Enable log rotation'
            ],
            'network_error': [
                'Check network connectivity',
                'Enable circuit breakers',
                'Add retry mechanisms',
                'Scale network resources'
            ]
        }
        
        return action_map.get(failure_type, ['Monitor system closely'])
    
    def _assess_impact(self, failure_type: str) -> str:
        """Assess the potential impact of the predicted failure"""
        impact_map = {
            'memory_error': 'Application crashes, potential data loss, user session interruption',
            'cpu_overload': 'Slow response times, request timeouts, degraded user experience',
            'database_error': 'Data access failures, slow queries, potential data corruption',
            'disk_full': 'Unable to write logs or data, application crashes, data loss',
            'network_error': 'External API failures, user connectivity issues, data sync problems'
        }
        
        return impact_map.get(failure_type, 'Unknown impact')
    
    async def _retrain_models(self):
        """Retrain prediction models with new data"""
        try:
            # This would implement actual model training
            # For now, just log that retraining would happen
            logger.info(f"Retraining models with {len(self.failure_history)} failure events")
            
            # Save models after retraining
            self._save_models()
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")


class DependencyMonitor:
    """Monitor health of external dependencies with automatic failover"""
    
    def __init__(self):
        self.dependencies: Dict[str, DependencyHealth] = {}
        self.failover_strategies: Dict[str, Callable] = {}
        self.monitoring_active = False
        self._lock = threading.Lock()
        
        # Initialize default dependencies
        self._setup_default_dependencies()
    
    def _setup_default_dependencies(self):
        """Setup default external dependencies to monitor"""
        default_deps = [
            {'name': 'openai_api', 'endpoint': 'https://api.openai.com/v1/models'},
            {'name': 'anthropic_api', 'endpoint': 'https://api.anthropic.com/v1/messages'},
            {'name': 'database', 'endpoint': 'localhost:5432'},
            {'name': 'redis_cache', 'endpoint': 'localhost:6379'},
            {'name': 'elasticsearch', 'endpoint': 'localhost:9200'}
        ]
        
        for dep in default_deps:
            self.dependencies[dep['name']] = DependencyHealth(
                service_name=dep['name'],
                endpoint=dep['endpoint'],
                status='unknown',
                response_time=0.0,
                error_rate=0.0,
                last_check=datetime.utcnow(),
                consecutive_failures=0,
                availability=100.0
            )
    
    def register_dependency(self, name: str, endpoint: str, 
                          failover_strategy: Optional[Callable] = None):
        """Register a new dependency to monitor"""
        with self._lock:
            self.dependencies[name] = DependencyHealth(
                service_name=name,
                endpoint=endpoint,
                status='unknown',
                response_time=0.0,
                error_rate=0.0,
                last_check=datetime.utcnow(),
                consecutive_failures=0,
                availability=100.0
            )
            
            if failover_strategy:
                self.failover_strategies[name] = failover_strategy
    
    async def check_dependency_health(self, name: str) -> DependencyHealth:
        """Check health of a specific dependency"""
        if name not in self.dependencies:
            raise ValueError(f"Unknown dependency: {name}")
        
        dependency = self.dependencies[name]
        start_time = time.time()
        
        try:
            # Perform health check based on dependency type
            if 'api' in name.lower():
                success = await self._check_api_health(dependency.endpoint)
            elif 'database' in name.lower():
                success = await self._check_database_health(dependency.endpoint)
            elif 'redis' in name.lower():
                success = await self._check_redis_health(dependency.endpoint)
            else:
                success = await self._check_generic_health(dependency.endpoint)
            
            response_time = time.time() - start_time
            
            with self._lock:
                dependency.response_time = response_time
                dependency.last_check = datetime.utcnow()
                
                if success:
                    dependency.status = 'healthy'
                    dependency.consecutive_failures = 0
                    dependency.error_rate = max(0, dependency.error_rate - 0.1)
                else:
                    dependency.consecutive_failures += 1
                    dependency.error_rate = min(1.0, dependency.error_rate + 0.1)
                    
                    if dependency.consecutive_failures >= 3:
                        dependency.status = 'failed'
                        await self._trigger_failover(name)
                    else:
                        dependency.status = 'degraded'
                
                # Update availability (rolling average)
                if success:
                    dependency.availability = min(100.0, dependency.availability + 1.0)
                else:
                    dependency.availability = max(0.0, dependency.availability - 5.0)
        
        except Exception as e:
            logger.error(f"Error checking dependency {name}: {e}")
            with self._lock:
                dependency.status = 'failed'
                dependency.consecutive_failures += 1
                dependency.error_rate = min(1.0, dependency.error_rate + 0.2)
        
        return dependency
    
    async def _check_api_health(self, endpoint: str) -> bool:
        """Check health of an API endpoint"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(endpoint) as response:
                    return response.status < 500
        except Exception:
            return False
    
    async def _check_database_health(self, endpoint: str) -> bool:
        """Check database health"""
        try:
            # This would check actual database connection
            # For now, assume it's healthy
            await asyncio.sleep(0.1)  # Simulate check
            return True
        except Exception:
            return False
    
    async def _check_redis_health(self, endpoint: str) -> bool:
        """Check Redis health"""
        try:
            # This would check actual Redis connection
            # For now, assume it's healthy
            await asyncio.sleep(0.05)  # Simulate check
            return True
        except Exception:
            return False
    
    async def _check_generic_health(self, endpoint: str) -> bool:
        """Generic health check"""
        try:
            # Basic connectivity check
            await asyncio.sleep(0.1)  # Simulate check
            return True
        except Exception:
            return False
    
    async def _trigger_failover(self, dependency_name: str):
        """Trigger failover for a failed dependency"""
        logger.warning(f"Triggering failover for dependency: {dependency_name}")
        
        failover_strategy = self.failover_strategies.get(dependency_name)
        if failover_strategy:
            try:
                await failover_strategy(dependency_name)
                logger.info(f"Failover completed for {dependency_name}")
            except Exception as e:
                logger.error(f"Failover failed for {dependency_name}: {e}")
        else:
            logger.warning(f"No failover strategy defined for {dependency_name}")
    
    async def monitor_all_dependencies(self):
        """Monitor all registered dependencies"""
        while self.monitoring_active:
            try:
                tasks = []
                for name in self.dependencies.keys():
                    tasks.append(self.check_dependency_health(name))
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Check for critical failures
                failed_deps = [
                    name for name, dep in self.dependencies.items()
                    if dep.status == 'failed'
                ]
                
                if failed_deps:
                    logger.warning(f"Failed dependencies: {failed_deps}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in dependency monitoring: {e}")
                await asyncio.sleep(60)
    
    def start_monitoring(self):
        """Start dependency monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            asyncio.create_task(self.monitor_all_dependencies())
    
    def stop_monitoring(self):
        """Stop dependency monitoring"""
        self.monitoring_active = False
    
    def get_dependency_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all dependencies"""
        with self._lock:
            return {
                name: {
                    'status': dep.status,
                    'response_time': dep.response_time,
                    'error_rate': dep.error_rate,
                    'availability': dep.availability,
                    'consecutive_failures': dep.consecutive_failures,
                    'last_check': dep.last_check.isoformat()
                }
                for name, dep in self.dependencies.items()
            }


class ResourceScaler:
    """Proactive resource scaling and optimization"""
    
    def __init__(self):
        self.scaling_history = []
        self.optimization_strategies = {}
        self.scaling_thresholds = {
            'cpu_scale_up': 75.0,
            'cpu_scale_down': 30.0,
            'memory_scale_up': 80.0,
            'memory_scale_down': 40.0,
            'disk_scale_up': 85.0
        }
        
    async def analyze_resource_needs(self, metrics: SystemHealthMetrics, 
                                   predictions: List[FailurePrediction]) -> List[ResourceOptimization]:
        """Analyze current and predicted resource needs"""
        optimizations = []
        
        # Current resource analysis
        current_optimizations = await self._analyze_current_resources(metrics)
        optimizations.extend(current_optimizations)
        
        # Predictive resource analysis
        predictive_optimizations = await self._analyze_predicted_resources(predictions)
        optimizations.extend(predictive_optimizations)
        
        return optimizations
    
    async def _analyze_current_resources(self, metrics: SystemHealthMetrics) -> List[ResourceOptimization]:
        """Analyze current resource usage and recommend optimizations"""
        optimizations = []
        
        # CPU analysis
        if metrics.cpu_usage > self.scaling_thresholds['cpu_scale_up']:
            optimizations.append(ResourceOptimization(
                resource_type='cpu',
                current_usage=metrics.cpu_usage,
                predicted_usage=metrics.cpu_usage * 1.2,  # Assume 20% growth
                recommended_action=ScalingAction.SCALE_UP_CPU,
                urgency=PredictionConfidence.HIGH if metrics.cpu_usage > 85 else PredictionConfidence.MEDIUM,
                estimated_impact='Improved response times, reduced timeouts',
                cost_benefit={'cost_increase': 0.3, 'performance_gain': 0.8}
            ))
        elif metrics.cpu_usage < self.scaling_thresholds['cpu_scale_down']:
            optimizations.append(ResourceOptimization(
                resource_type='cpu',
                current_usage=metrics.cpu_usage,
                predicted_usage=metrics.cpu_usage,
                recommended_action=ScalingAction.SCALE_DOWN,
                urgency=PredictionConfidence.LOW,
                estimated_impact='Cost savings with minimal performance impact',
                cost_benefit={'cost_decrease': 0.2, 'performance_impact': 0.1}
            ))
        
        # Memory analysis
        if metrics.memory_usage > self.scaling_thresholds['memory_scale_up']:
            optimizations.append(ResourceOptimization(
                resource_type='memory',
                current_usage=metrics.memory_usage,
                predicted_usage=metrics.memory_usage * 1.15,
                recommended_action=ScalingAction.SCALE_UP_MEMORY,
                urgency=PredictionConfidence.CRITICAL if metrics.memory_usage > 90 else PredictionConfidence.HIGH,
                estimated_impact='Prevent memory exhaustion, improve stability',
                cost_benefit={'cost_increase': 0.25, 'stability_gain': 0.9}
            ))
        
        # Disk analysis
        if metrics.disk_usage > self.scaling_thresholds['disk_scale_up']:
            optimizations.append(ResourceOptimization(
                resource_type='storage',
                current_usage=metrics.disk_usage,
                predicted_usage=metrics.disk_usage * 1.1,
                recommended_action=ScalingAction.SCALE_UP_STORAGE,
                urgency=PredictionConfidence.CRITICAL if metrics.disk_usage > 95 else PredictionConfidence.HIGH,
                estimated_impact='Prevent disk full errors, maintain operations',
                cost_benefit={'cost_increase': 0.2, 'reliability_gain': 0.95}
            ))
        
        # Performance-based optimizations
        if metrics.database_query_time > 1.0:
            optimizations.append(ResourceOptimization(
                resource_type='database',
                current_usage=metrics.database_query_time,
                predicted_usage=metrics.database_query_time * 1.3,
                recommended_action=ScalingAction.OPTIMIZE_QUERIES,
                urgency=PredictionConfidence.MEDIUM,
                estimated_impact='Faster database responses, improved user experience',
                cost_benefit={'cost_increase': 0.1, 'performance_gain': 0.6}
            ))
        
        if metrics.cache_hit_rate < 0.7:
            optimizations.append(ResourceOptimization(
                resource_type='cache',
                current_usage=metrics.cache_hit_rate,
                predicted_usage=metrics.cache_hit_rate * 0.9,  # May degrade further
                recommended_action=ScalingAction.CLEAR_CACHE,
                urgency=PredictionConfidence.MEDIUM,
                estimated_impact='Improved cache efficiency, faster responses',
                cost_benefit={'cost_increase': 0.05, 'performance_gain': 0.4}
            ))
        
        return optimizations
    
    async def _analyze_predicted_resources(self, predictions: List[FailurePrediction]) -> List[ResourceOptimization]:
        """Analyze predicted failures and recommend preemptive scaling"""
        optimizations = []
        
        for prediction in predictions:
            if prediction.failure_type == FailureType.CPU_OVERLOAD:
                optimizations.append(ResourceOptimization(
                    resource_type='cpu',
                    current_usage=0,  # Will be filled from current metrics
                    predicted_usage=100.0,  # Predicted overload
                    recommended_action=ScalingAction.SCALE_UP_CPU,
                    urgency=prediction.confidence,
                    estimated_impact='Prevent predicted CPU overload',
                    cost_benefit={'cost_increase': 0.3, 'failure_prevention': 1.0}
                ))
            
            elif prediction.failure_type == FailureType.MEMORY_ERROR:
                optimizations.append(ResourceOptimization(
                    resource_type='memory',
                    current_usage=0,  # Will be filled from current metrics
                    predicted_usage=100.0,  # Predicted exhaustion
                    recommended_action=ScalingAction.SCALE_UP_MEMORY,
                    urgency=prediction.confidence,
                    estimated_impact='Prevent predicted memory exhaustion',
                    cost_benefit={'cost_increase': 0.25, 'failure_prevention': 1.0}
                ))
            
            elif prediction.failure_type == FailureType.DISK_FULL:
                optimizations.append(ResourceOptimization(
                    resource_type='storage',
                    current_usage=0,  # Will be filled from current metrics
                    predicted_usage=100.0,  # Predicted full disk
                    recommended_action=ScalingAction.SCALE_UP_STORAGE,
                    urgency=prediction.confidence,
                    estimated_impact='Prevent predicted disk full error',
                    cost_benefit={'cost_increase': 0.2, 'failure_prevention': 1.0}
                ))
        
        return optimizations
    
    async def execute_scaling_action(self, optimization: ResourceOptimization) -> bool:
        """Execute a scaling action"""
        try:
            logger.info(f"Executing scaling action: {optimization.recommended_action.value}")
            
            # Record scaling action
            self.scaling_history.append({
                'timestamp': datetime.utcnow(),
                'action': optimization.recommended_action.value,
                'resource_type': optimization.resource_type,
                'urgency': optimization.urgency.value,
                'estimated_impact': optimization.estimated_impact
            })
            
            # Execute the actual scaling (mock implementation)
            if optimization.recommended_action == ScalingAction.SCALE_UP_CPU:
                await self._scale_up_cpu()
            elif optimization.recommended_action == ScalingAction.SCALE_UP_MEMORY:
                await self._scale_up_memory()
            elif optimization.recommended_action == ScalingAction.SCALE_UP_STORAGE:
                await self._scale_up_storage()
            elif optimization.recommended_action == ScalingAction.CLEAR_CACHE:
                await self._clear_cache()
            elif optimization.recommended_action == ScalingAction.OPTIMIZE_QUERIES:
                await self._optimize_queries()
            elif optimization.recommended_action == ScalingAction.RESTART_SERVICE:
                await self._restart_service()
            
            logger.info(f"Scaling action completed: {optimization.recommended_action.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
            return False
    
    async def _scale_up_cpu(self):
        """Scale up CPU resources"""
        # This would integrate with cloud provider APIs
        logger.info("Scaling up CPU resources (mock)")
        await asyncio.sleep(1)  # Simulate scaling time
    
    async def _scale_up_memory(self):
        """Scale up memory resources"""
        # This would integrate with cloud provider APIs
        logger.info("Scaling up memory resources (mock)")
        await asyncio.sleep(1)  # Simulate scaling time
    
    async def _scale_up_storage(self):
        """Scale up storage resources"""
        # This would integrate with cloud provider APIs
        logger.info("Scaling up storage resources (mock)")
        await asyncio.sleep(1)  # Simulate scaling time
    
    async def _clear_cache(self):
        """Clear application caches"""
        # This would clear actual caches
        logger.info("Clearing application caches (mock)")
        await asyncio.sleep(0.5)  # Simulate cache clear time
    
    async def _optimize_queries(self):
        """Optimize database queries"""
        # This would trigger query optimization
        logger.info("Optimizing database queries (mock)")
        await asyncio.sleep(2)  # Simulate optimization time
    
    async def _restart_service(self):
        """Restart a service"""
        # This would restart specific services
        logger.info("Restarting service (mock)")
        await asyncio.sleep(3)  # Simulate restart time


class PredictiveFailurePreventionEngine:
    """Main engine that coordinates all predictive failure prevention components"""
    
    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.failure_predictor = FailurePredictor()
        self.dependency_monitor = DependencyMonitor()
        self.resource_scaler = ResourceScaler()
        
        self.running = False
        self.prevention_history = []
        self._lock = threading.Lock()
        
        # Initialize database for persistence
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for persistence"""
        try:
            db_path = Path("data/predictive_prevention.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    metrics_json TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    anomaly_type TEXT,
                    confidence TEXT,
                    description TEXT,
                    actions_taken TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    failure_type TEXT,
                    confidence TEXT,
                    predicted_time TEXT,
                    prevention_actions TEXT,
                    outcome TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Predictive prevention database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    async def start(self):
        """Start the predictive failure prevention engine"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting Predictive Failure Prevention Engine")
        
        # Start dependency monitoring
        self.dependency_monitor.start_monitoring()
        
        # Start main prevention loop
        asyncio.create_task(self._prevention_loop())
        
        logger.info("Predictive Failure Prevention Engine started")
    
    async def stop(self):
        """Stop the predictive failure prevention engine"""
        self.running = False
        self.dependency_monitor.stop_monitoring()
        logger.info("Predictive Failure Prevention Engine stopped")
    
    async def _prevention_loop(self):
        """Main prevention loop"""
        while self.running:
            try:
                # Collect comprehensive metrics
                metrics = await self.health_monitor.collect_comprehensive_metrics()
                if not metrics:
                    await asyncio.sleep(30)
                    continue
                
                # Add metrics to history and train models
                self.health_monitor.add_metrics(metrics)
                await self.failure_predictor.add_training_data(metrics)
                
                # Detect anomalies
                anomalies = await self.health_monitor.detect_anomalies(metrics)
                
                # Predict failures
                predictions = await self.failure_predictor.predict_failures(metrics)
                
                # Analyze resource needs
                optimizations = await self.resource_scaler.analyze_resource_needs(metrics, predictions)
                
                # Take preventive actions
                await self._take_preventive_actions(anomalies, predictions, optimizations)
                
                # Store results
                await self._store_results(metrics, anomalies, predictions)
                
                # Wait before next cycle
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in prevention loop: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _take_preventive_actions(self, anomalies: List[AnomalyDetection],
                                     predictions: List[FailurePrediction],
                                     optimizations: List[ResourceOptimization]):
        """Take preventive actions based on analysis results"""
        actions_taken = []
        
        # Handle critical anomalies
        for anomaly in anomalies:
            if anomaly.confidence in [PredictionConfidence.HIGH, PredictionConfidence.CRITICAL]:
                logger.warning(f"Critical anomaly detected: {anomaly.description}")
                
                # Execute recommended actions
                for action in anomaly.recommended_actions[:2]:  # Limit to 2 actions
                    try:
                        await self._execute_action(action)
                        actions_taken.append(f"Anomaly action: {action}")
                    except Exception as e:
                        logger.error(f"Error executing anomaly action {action}: {e}")
        
        # Handle high-confidence predictions
        for prediction in predictions:
            if prediction.confidence in [PredictionConfidence.HIGH, PredictionConfidence.CRITICAL]:
                logger.warning(f"High-confidence failure prediction: {prediction.failure_type.value}")
                
                # Execute prevention actions
                for action in prediction.prevention_actions[:2]:  # Limit to 2 actions
                    try:
                        await self._execute_action(action)
                        actions_taken.append(f"Prediction action: {action}")
                    except Exception as e:
                        logger.error(f"Error executing prediction action {action}: {e}")
        
        # Handle urgent resource optimizations
        for optimization in optimizations:
            if optimization.urgency in [PredictionConfidence.HIGH, PredictionConfidence.CRITICAL]:
                logger.info(f"Executing urgent resource optimization: {optimization.recommended_action.value}")
                
                success = await self.resource_scaler.execute_scaling_action(optimization)
                if success:
                    actions_taken.append(f"Resource action: {optimization.recommended_action.value}")
        
        # Record actions taken
        if actions_taken:
            with self._lock:
                self.prevention_history.append({
                    'timestamp': datetime.utcnow(),
                    'actions': actions_taken,
                    'anomaly_count': len(anomalies),
                    'prediction_count': len(predictions),
                    'optimization_count': len(optimizations)
                })
    
    async def _execute_action(self, action: str):
        """Execute a specific preventive action"""
        action_lower = action.lower()
        
        if 'scale up cpu' in action_lower:
            await self.resource_scaler._scale_up_cpu()
        elif 'scale up memory' in action_lower:
            await self.resource_scaler._scale_up_memory()
        elif 'clear cache' in action_lower:
            await self.resource_scaler._clear_cache()
        elif 'optimize' in action_lower and 'queries' in action_lower:
            await self.resource_scaler._optimize_queries()
        elif 'restart' in action_lower:
            await self.resource_scaler._restart_service()
        else:
            logger.info(f"Action noted but not executed: {action}")
    
    async def _store_results(self, metrics: SystemHealthMetrics,
                           anomalies: List[AnomalyDetection],
                           predictions: List[FailurePrediction]):
        """Store analysis results in database"""
        try:
            db_path = Path("data/predictive_prevention.db")
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Store metrics
            cursor.execute(
                "INSERT INTO metrics_history (timestamp, metrics_json) VALUES (?, ?)",
                (metrics.timestamp.isoformat(), json.dumps(asdict(metrics), default=str))
            )
            
            # Store anomalies
            for anomaly in anomalies:
                cursor.execute(
                    "INSERT INTO anomalies (timestamp, anomaly_type, confidence, description, actions_taken) VALUES (?, ?, ?, ?, ?)",
                    (
                        anomaly.timestamp.isoformat(),
                        anomaly.anomaly_type.value,
                        anomaly.confidence.value,
                        anomaly.description,
                        json.dumps(anomaly.recommended_actions)
                    )
                )
            
            # Store predictions
            for prediction in predictions:
                cursor.execute(
                    "INSERT INTO predictions (timestamp, failure_type, confidence, predicted_time, prevention_actions, outcome) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        datetime.utcnow().isoformat(),
                        prediction.failure_type.value,
                        prediction.confidence.value,
                        prediction.predicted_time.isoformat(),
                        json.dumps(prediction.prevention_actions),
                        'pending'
                    )
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing results: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'engine_running': self.running,
            'metrics_history_size': len(self.health_monitor.metrics_history),
            'anomaly_detector_trained': self.health_monitor.is_trained,
            'recent_anomalies': len([a for a in self.health_monitor.anomalies if a.timestamp > datetime.utcnow() - timedelta(hours=1)]),
            'dependency_status': self.dependency_monitor.get_dependency_status(),
            'recent_actions': len(self.prevention_history),
            'prediction_models': len(self.failure_predictor.models),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        # Get current metrics
        current_metrics = await self.health_monitor.collect_comprehensive_metrics()
        
        # Get recent anomalies
        recent_anomalies = [
            {
                'type': a.anomaly_type.value,
                'confidence': a.confidence.value,
                'description': a.description,
                'timestamp': a.timestamp.isoformat()
            }
            for a in self.health_monitor.anomalies[-10:]  # Last 10 anomalies
        ]
        
        # Get failure predictions
        predictions = await self.failure_predictor.predict_failures(current_metrics) if current_metrics else []
        prediction_summary = [
            {
                'failure_type': p.failure_type.value,
                'confidence': p.confidence.value,
                'predicted_time': p.predicted_time.isoformat(),
                'probability': p.probability
            }
            for p in predictions
        ]
        
        return {
            'current_metrics': asdict(current_metrics) if current_metrics else None,
            'system_status': self.get_system_status(),
            'recent_anomalies': recent_anomalies,
            'failure_predictions': prediction_summary,
            'dependency_health': self.dependency_monitor.get_dependency_status(),
            'prevention_history': self.prevention_history[-5:],  # Last 5 actions
            'timestamp': datetime.utcnow().isoformat()
        }


# Global instance
predictive_engine = PredictiveFailurePreventionEngine()


# Convenience functions
async def start_predictive_prevention():
    """Start the predictive failure prevention engine"""
    await predictive_engine.start()


async def stop_predictive_prevention():
    """Stop the predictive failure prevention engine"""
    await predictive_engine.stop()


def get_prevention_status():
    """Get current prevention system status"""
    return predictive_engine.get_system_status()


async def get_health_report():
    """Get comprehensive health report"""
    return await predictive_engine.get_health_report()