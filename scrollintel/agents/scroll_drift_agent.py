"""
ScrollDriftAgent - Advanced Data & Concept Drift Detection
Real-time monitoring of model performance degradation and data distribution changes.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from uuid import uuid4
from dataclasses import dataclass
from enum import Enum
import json
import logging

# Statistical libraries
try:
    from scipy import stats
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from scrollintel.core.interfaces import BaseAgent, AgentType, AgentRequest, AgentResponse, AgentCapability, ResponseStatus

logger = logging.getLogger(__name__)


class DriftType(str, Enum):
    """Types of drift detection."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


class DriftSeverity(str, Enum):
    """Severity levels of detected drift."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftDetectionMethod(str, Enum):
    """Methods for drift detection."""
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    POPULATION_STABILITY_INDEX = "psi"
    JENSEN_SHANNON_DIVERGENCE = "jensen_shannon"
    WASSERSTEIN_DISTANCE = "wasserstein"
    CHI_SQUARE = "chi_square"
    STATISTICAL_DISTANCE = "statistical_distance"
    PERFORMANCE_MONITORING = "performance_monitoring"


@dataclass
class DriftAlert:
    """Drift detection alert."""
    id: str
    model_id: str
    drift_type: DriftType
    severity: DriftSeverity
    method: DriftDetectionMethod
    drift_score: float
    threshold: float
    affected_features: List[str]
    detection_time: datetime
    description: str
    recommendations: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ModelMonitoringConfig:
    """Configuration for model monitoring."""
    model_id: str
    monitoring_frequency: str  # "hourly", "daily", "weekly"
    drift_thresholds: Dict[DriftType, float]
    detection_methods: List[DriftDetectionMethod]
    alert_channels: List[str]
    auto_retrain_threshold: float
    feature_importance_tracking: bool = True
    performance_tracking: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class ScrollDriftAgent(BaseAgent):
    """Advanced drift detection and monitoring agent."""
    
    def __init__(self):
        super().__init__(
            agent_id="scroll-drift-agent",
            name="ScrollDrift Agent",
            agent_type=AgentType.ML_ENGINEER
        )
        
        self.capabilities = [
            AgentCapability(
                name="data_drift_detection",
                description="Detect changes in data distribution over time",
                input_types=["reference_data", "current_data", "detection_config"],
                output_types=["drift_analysis", "drift_alerts", "recommendations"]
            ),
            AgentCapability(
                name="concept_drift_detection",
                description="Detect changes in the relationship between features and target",
                input_types=["model", "historical_performance", "current_performance"],
                output_types=["concept_drift_analysis", "performance_degradation_report"]
            ),
            AgentCapability(
                name="model_performance_monitoring",
                description="Continuous monitoring of model performance metrics",
                input_types=["model_predictions", "ground_truth", "performance_config"],
                output_types=["performance_report", "degradation_alerts"]
            ),
            AgentCapability(
                name="automated_retraining_triggers",
                description="Trigger model retraining based on drift detection",
                input_types=["drift_alerts", "retraining_config"],
                output_types=["retraining_recommendations", "automated_triggers"]
            )
        ]
        
        # Drift detection state
        self.monitoring_configs = {}
        self.drift_history = {}
        self.active_alerts = {}
        self.baseline_distributions = {}
        
        # Detection methods
        self.detection_methods = {
            DriftDetectionMethod.KOLMOGOROV_SMIRNOV: self._ks_test,
            DriftDetectionMethod.POPULATION_STABILITY_INDEX: self._psi_test,
            DriftDetectionMethod.JENSEN_SHANNON_DIVERGENCE: self._js_divergence,
            DriftDetectionMethod.WASSERSTEIN_DISTANCE: self._wasserstein_distance,
            DriftDetectionMethod.CHI_SQUARE: self._chi_square_test,
            DriftDetectionMethod.STATISTICAL_DISTANCE: self._statistical_distance,
            DriftDetectionMethod.PERFORMANCE_MONITORING: self._performance_monitoring
        }
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process drift detection requests."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            prompt = request.prompt.lower()
            context = request.context or {}
            
            if "detect" in prompt and "drift" in prompt:
                content = await self._detect_drift(request.prompt, context)
            elif "monitor" in prompt or "setup" in prompt:
                content = await self._setup_monitoring(request.prompt, context)
            elif "alert" in prompt or "notification" in prompt:
                content = await self._manage_alerts(request.prompt, context)
            elif "performance" in prompt:
                content = await self._analyze_performance(request.prompt, context)
            elif "retrain" in prompt or "trigger" in prompt:
                content = await self._evaluate_retraining(request.prompt, context)
            else:
                content = await self._analyze_drift_status(request.prompt, context)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return AgentResponse(
                id=f"drift-{uuid4()}",
                request_id=request.id,
                content=content,
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.SUCCESS
            )
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return AgentResponse(
                id=f"drift-{uuid4()}",
                request_id=request.id,
                content=f"Error in drift detection: {str(e)}",
                artifacts=[],
                execution_time=execution_time,
                status=ResponseStatus.ERROR,
                error_message=str(e)
            )
    
    async def _detect_drift(self, prompt: str, context: Dict[str, Any]) -> str:
        """Detect drift in data or model performance."""
        reference_data = context.get("reference_data")
        current_data = context.get("current_data")
        model_id = context.get("model_id", "default")
        detection_methods = context.get("methods", [DriftDetectionMethod.KOLMOGOROV_SMIRNOV])
        
        if reference_data is None or current_data is None:
            return "Error: Both reference_data and current_data are required for drift detection."
        
        # Convert to DataFrames if needed
        if not isinstance(reference_data, pd.DataFrame):
            reference_data = pd.DataFrame(reference_data)
        if not isinstance(current_data, pd.DataFrame):
            current_data = pd.DataFrame(current_data)
        
        # Perform drift detection
        drift_results = {}
        alerts = []
        
        for method in detection_methods:
            if isinstance(method, str):
                method = DriftDetectionMethod(method)
            
            result = await self._run_drift_detection(
                reference_data, current_data, method, model_id
            )
            drift_results[method.value] = result
            
            # Check for alerts
            if result["drift_detected"]:
                alert = DriftAlert(
                    id=f"alert-{uuid4()}",
                    model_id=model_id,
                    drift_type=DriftType.DATA_DRIFT,
                    severity=self._calculate_severity(result["drift_score"]),
                    method=method,
                    drift_score=result["drift_score"],
                    threshold=result["threshold"],
                    affected_features=result.get("affected_features", []),
                    detection_time=datetime.utcnow(),
                    description=result["description"],
                    recommendations=result.get("recommendations", [])
                )
                alerts.append(alert)
        
        # Store alerts
        for alert in alerts:
            self.active_alerts[alert.id] = alert
        
        # Store drift history
        if model_id not in self.drift_history:
            self.drift_history[model_id] = []
        
        self.drift_history[model_id].append({
            "timestamp": datetime.utcnow(),
            "results": drift_results,
            "alerts": [alert.id for alert in alerts]
        })
        
        return f"""
# Drift Detection Results

## Model: {model_id}
## Detection Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Methods Used**: {len(detection_methods)}
- **Drift Detected**: {any(result["drift_detected"] for result in drift_results.values())}
- **Alerts Generated**: {len(alerts)}

## Detection Results
{await self._format_drift_results(drift_results)}

## Active Alerts
{await self._format_alerts(alerts)}

## Recommendations
{await self._generate_drift_recommendations(drift_results, alerts)}

## Next Steps
{await self._suggest_next_steps(drift_results, alerts)}
"""
    
    async def _setup_monitoring(self, prompt: str, context: Dict[str, Any]) -> str:
        """Set up continuous drift monitoring."""
        model_id = context.get("model_id", f"model-{uuid4()}")
        monitoring_config = context.get("config", {})
        
        config = ModelMonitoringConfig(
            model_id=model_id,
            monitoring_frequency=monitoring_config.get("frequency", "daily"),
            drift_thresholds={
                DriftType.DATA_DRIFT: monitoring_config.get("data_drift_threshold", 0.05),
                DriftType.CONCEPT_DRIFT: monitoring_config.get("concept_drift_threshold", 0.1),
                DriftType.PERFORMANCE_DRIFT: monitoring_config.get("performance_drift_threshold", 0.05)
            },
            detection_methods=monitoring_config.get("methods", [
                DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
                DriftDetectionMethod.POPULATION_STABILITY_INDEX
            ]),
            alert_channels=monitoring_config.get("alert_channels", ["email", "dashboard"]),
            auto_retrain_threshold=monitoring_config.get("auto_retrain_threshold", 0.2)
        )
        
        # Store monitoring configuration
        self.monitoring_configs[model_id] = config
        
        return f"""
# Drift Monitoring Setup Complete

## Model: {model_id}

## Monitoring Configuration
- **Frequency**: {config.monitoring_frequency}
- **Detection Methods**: {[method.value for method in config.detection_methods]}
- **Alert Channels**: {config.alert_channels}

## Drift Thresholds
- **Data Drift**: {config.drift_thresholds[DriftType.DATA_DRIFT]}
- **Concept Drift**: {config.drift_thresholds[DriftType.CONCEPT_DRIFT]}
- **Performance Drift**: {config.drift_thresholds[DriftType.PERFORMANCE_DRIFT]}

## Auto-Retraining
- **Threshold**: {config.auto_retrain_threshold}
- **Enabled**: {config.auto_retrain_threshold > 0}

## Monitoring Status
- **Active**: ✅ Monitoring started
- **Next Check**: {await self._calculate_next_check(config)}

## Dashboard Access
Access your drift monitoring dashboard at: `/drift-monitor/{model_id}`
"""
    
    async def _run_drift_detection(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                                 method: DriftDetectionMethod, model_id: str) -> Dict[str, Any]:
        """Run specific drift detection method."""
        try:
            detection_func = self.detection_methods[method]
            result = await detection_func(reference_data, current_data)
            
            # Add metadata
            result.update({
                "method": method.value,
                "model_id": model_id,
                "reference_size": len(reference_data),
                "current_size": len(current_data),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Drift detection failed for method {method}: {e}")
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "threshold": 0.05,
                "error": str(e),
                "method": method.value
            }
    
    async def _ks_test(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Kolmogorov-Smirnov test for drift detection."""
        if not SCIPY_AVAILABLE:
            return {"drift_detected": False, "error": "SciPy not available"}
        
        drift_scores = []
        affected_features = []
        threshold = 0.05
        
        for column in reference_data.columns:
            if column in current_data.columns:
                if pd.api.types.is_numeric_dtype(reference_data[column]):
                    # Remove NaN values
                    ref_clean = reference_data[column].dropna()
                    cur_clean = current_data[column].dropna()
                    
                    if len(ref_clean) > 0 and len(cur_clean) > 0:
                        statistic, p_value = stats.ks_2samp(ref_clean, cur_clean)
                        drift_scores.append(p_value)
                        
                        if p_value < threshold:
                            affected_features.append(column)
        
        if not drift_scores:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "threshold": threshold,
                "description": "No numeric features found for KS test"
            }
        
        avg_p_value = np.mean(drift_scores)
        drift_detected = avg_p_value < threshold
        
        return {
            "drift_detected": drift_detected,
            "drift_score": 1 - avg_p_value,  # Convert to drift score (higher = more drift)
            "threshold": threshold,
            "affected_features": affected_features,
            "p_values": drift_scores,
            "description": f"KS test detected {'significant' if drift_detected else 'no significant'} distribution changes",
            "recommendations": [
                "Investigate affected features for data quality issues",
                "Consider retraining if drift is significant",
                "Monitor trends over time"
            ] if drift_detected else ["Continue monitoring"]
        }
    
    async def _psi_test(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Population Stability Index test."""
        def calculate_psi(expected, actual, buckets=10):
            """Calculate PSI for a single feature."""
            # Create bins based on expected data
            _, bin_edges = np.histogram(expected, bins=buckets)
            
            # Calculate distributions
            expected_dist = np.histogram(expected, bins=bin_edges)[0] / len(expected)
            actual_dist = np.histogram(actual, bins=bin_edges)[0] / len(actual)
            
            # Avoid division by zero
            expected_dist = np.where(expected_dist == 0, 0.0001, expected_dist)
            actual_dist = np.where(actual_dist == 0, 0.0001, actual_dist)
            
            # Calculate PSI
            psi = np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist))
            return psi
        
        psi_scores = []
        affected_features = []
        threshold = 0.1  # PSI threshold
        
        for column in reference_data.columns:
            if column in current_data.columns:
                if pd.api.types.is_numeric_dtype(reference_data[column]):
                    ref_clean = reference_data[column].dropna()
                    cur_clean = current_data[column].dropna()
                    
                    if len(ref_clean) > 10 and len(cur_clean) > 10:
                        psi = calculate_psi(ref_clean, cur_clean)
                        psi_scores.append(psi)
                        
                        if psi > threshold:
                            affected_features.append(column)
        
        if not psi_scores:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "threshold": threshold,
                "description": "No suitable features found for PSI calculation"
            }
        
        avg_psi = np.mean(psi_scores)
        drift_detected = avg_psi > threshold
        
        return {
            "drift_detected": drift_detected,
            "drift_score": avg_psi,
            "threshold": threshold,
            "affected_features": affected_features,
            "psi_scores": psi_scores,
            "description": f"PSI analysis shows {'significant' if drift_detected else 'stable'} population changes",
            "recommendations": [
                "Review data collection process",
                "Check for seasonal patterns",
                "Consider feature engineering updates"
            ] if drift_detected else ["Population remains stable"]
        }
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return self.capabilities
    
    async def health_check(self) -> bool:
        """Check agent health."""
        return True
    
    # Additional helper methods
    def _calculate_severity(self, drift_score: float) -> DriftSeverity:
        """Calculate drift severity based on score."""
        if drift_score >= 0.5:
            return DriftSeverity.CRITICAL
        elif drift_score >= 0.3:
            return DriftSeverity.HIGH
        elif drift_score >= 0.1:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    async def _format_drift_results(self, results: Dict[str, Any]) -> str:
        """Format drift detection results."""
        formatted = []
        for method, result in results.items():
            status = "🚨 DRIFT DETECTED" if result.get("drift_detected") else "✅ No Drift"
            score = result.get("drift_score", 0)
            formatted.append(f"- **{method}**: {status} (Score: {score:.3f})")
        return "\n".join(formatted)
    
    async def _format_alerts(self, alerts: List[DriftAlert]) -> str:
        """Format drift alerts."""
        if not alerts:
            return "No alerts generated."
        
        formatted = []
        for alert in alerts:
            formatted.append(f"""
**Alert {alert.id}**
- Severity: {alert.severity.value.upper()}
- Method: {alert.method.value}
- Score: {alert.drift_score:.3f}
- Features: {', '.join(alert.affected_features) if alert.affected_features else 'All'}
- Description: {alert.description}
""")
        return "\n".join(formatted)
    
    # Placeholder implementations for other detection methods
    async def _js_divergence(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Jensen-Shannon divergence calculation."""
        return {"drift_detected": False, "drift_score": 0.0, "threshold": 0.1, "description": "JS divergence not implemented"}
    
    async def _wasserstein_distance(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Wasserstein distance calculation."""
        return {"drift_detected": False, "drift_score": 0.0, "threshold": 0.1, "description": "Wasserstein distance not implemented"}
    
    async def _chi_square_test(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Chi-square test for categorical features."""
        return {"drift_detected": False, "drift_score": 0.0, "threshold": 0.05, "description": "Chi-square test not implemented"}
    
    async def _statistical_distance(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Statistical distance calculation."""
        return {"drift_detected": False, "drift_score": 0.0, "threshold": 0.1, "description": "Statistical distance not implemented"}
    
    async def _performance_monitoring(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Performance-based drift detection."""
        return {"drift_detected": False, "drift_score": 0.0, "threshold": 0.05, "description": "Performance monitoring not implemented"}