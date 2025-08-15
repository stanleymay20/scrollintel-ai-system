"""
Data models for predictive analytics engine.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import uuid


class ForecastModel(Enum):
    """Available forecasting models."""
    PROPHET = "prophet"
    ARIMA = "arima"
    LSTM = "lstm"
    LINEAR_REGRESSION = "linear_regression"
    ENSEMBLE = "ensemble"


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MetricCategory(Enum):
    """Business metric categories."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    CUSTOMER = "customer"
    TECHNICAL = "technical"


@dataclass
class BusinessMetric:
    """Business metric data structure."""
    id: str
    name: str
    category: MetricCategory
    value: float
    unit: str
    timestamp: datetime
    source: str
    context: Dict[str, Any]
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Forecast:
    """Forecast result with confidence intervals."""
    metric_id: str
    model_type: ForecastModel
    predictions: List[float]
    timestamps: List[datetime]
    confidence_lower: List[float]
    confidence_upper: List[float]
    confidence_level: float
    accuracy_score: Optional[float]
    created_at: datetime
    horizon_days: int
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow()


@dataclass
class ScenarioConfig:
    """Configuration for scenario modeling."""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    target_metrics: List[str]
    time_horizon: int
    created_by: str
    created_at: datetime
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()


@dataclass
class ScenarioResult:
    """Results from scenario modeling."""
    scenario_id: str
    baseline_forecast: Dict[str, Forecast]
    scenario_forecast: Dict[str, Forecast]
    impact_analysis: Dict[str, float]
    recommendations: List[str]
    confidence_score: float
    created_at: datetime
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow()


@dataclass
class RiskPrediction:
    """Risk prediction with early warning."""
    id: str
    metric_id: str
    risk_type: str
    risk_level: RiskLevel
    probability: float
    impact_score: float
    description: str
    early_warning_threshold: float
    mitigation_strategies: List[str]
    predicted_date: Optional[datetime]
    created_at: datetime
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow()


@dataclass
class PredictionAccuracy:
    """Tracking prediction accuracy over time."""
    model_type: ForecastModel
    metric_id: str
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    r2_score: float  # R-squared
    accuracy_trend: List[float]
    evaluation_date: datetime
    sample_size: int
    
    def __post_init__(self):
        if not self.evaluation_date:
            self.evaluation_date = datetime.utcnow()


@dataclass
class PredictionUpdate:
    """Update notification for predictions."""
    metric_id: str
    previous_forecast: Forecast
    updated_forecast: Forecast
    change_magnitude: float
    change_reason: str
    stakeholders_notified: List[str]
    update_timestamp: datetime
    
    def __post_init__(self):
        if not self.update_timestamp:
            self.update_timestamp = datetime.utcnow()


@dataclass
class BusinessContext:
    """Business context for predictions and risk analysis."""
    industry: str
    company_size: str
    market_conditions: Dict[str, Any]
    seasonal_factors: Dict[str, Any]
    external_factors: Dict[str, Any]
    historical_patterns: Dict[str, Any]