"""
Predictive Security Analytics with Trend Analysis and Risk Forecasting
Advanced ML-based security analytics for proactive threat detection
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    SECURITY_INCIDENT = "security_incident"
    VULNERABILITY_DISCOVERY = "vulnerability_discovery"
    THREAT_CAMPAIGN = "threat_campaign"
    COMPLIANCE_VIOLATION = "compliance_violation"
    SYSTEM_COMPROMISE = "system_compromise"

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityPrediction:
    prediction_id: str
    prediction_type: PredictionType
    predicted_probability: float
    confidence_interval: Tuple[float, float]
    risk_level: RiskLevel
    time_horizon: int  # days
    contributing_factors: List[str]
    recommended_actions: List[str]
    timestamp: datetime

@dataclass
class TrendAnalysis:
    trend_id: str
    metric_name: str
    trend_direction: str  # increasing, decreasing, stable
    trend_strength: float  # 0-1
    seasonal_pattern: bool
    anomaly_detected: bool
    forecast_values: List[float]
    forecast_dates: List[str]
    confidence_bands: List[Tuple[float, float]]

@dataclass
class RiskForecast:
    forecast_id: str
    risk_category: str
    current_risk_score: float
    forecasted_risk_scores: List[float]
    forecast_dates: List[str]
    risk_drivers: List[str]
    mitigation_recommendations: List[str]
    confidence_level: float

class SecurityPredictiveAnalytics:
    """Advanced predictive analytics for security operations"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or "sqlite:///security_analytics.db"
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.historical_data = pd.DataFrame()
        
    async def initialize(self):
        """Initialize predictive analytics system"""
        await self._setup_database()
        await self._load_historical_data()
        await self._train_models()
        
    async def _setup_database(self):
        """Setup analytics database"""
        logger.info("Setting up predictive analytics database")
        
    async def _load_historical_data(self):
        """Load historical security data for training"""
        # Generate synthetic historical data for demonstration
        dates = pd.date_range(start='2023-01-01', end=datetime.now(), freq='D')
        
        data = []
        for date in dates:
            # Simulate security metrics with trends and seasonality
            day_of_week = date.weekday()
            day_of_year = date.timetuple().tm_yday
            
            # Base metrics with weekly and seasonal patterns
            failed_logins = max(0, np.random.poisson(20) + 
                              5 * np.sin(2 * np.pi * day_of_week / 7) +
                              10 * np.sin(2 * np.pi * day_of_year / 365))
            
            network_anomalies = max(0, np.random.poisson(5) + 
                                  2 * np.sin(2 * np.pi * day_of_week / 7))
            
            vulnerability_scans = max(0, np.random.poisson(10) + 
                                    np.random.normal(0, 2))
            
            malware_detections = max(0, np.random.poisson(3) + 
                                   np.random.exponential(1))
            
            # Incident indicator (target variable)
            incident_probability = (
                0.1 + 
                0.3 * (failed_logins > 30) +
                0.4 * (network_anomalies > 8) +
                0.2 * (malware_detections > 5) +
                0.1 * np.random.random()
            )
            
            incident_occurred = np.random.random() < incident_probability
            
            data.append({
                'date': date,
                'day_of_week': day_of_week,
                'day_of_year': day_of_year,
                'failed_logins': failed_logins,
                'network_anomalies': network_anomalies,
                'vulnerability_scans': vulnerability_scans,
                'malware_detections': malware_detections,
                'incident_occurred': incident_occurred,
                'risk_score': min(100, incident_probability * 100 + np.random.normal(0, 5))
            })
            
        self.historical_data = pd.DataFrame(data)
        logger.info(f"Loaded {len(self.historical_data)} historical records")
        
    async def _train_models(self):
        """Train predictive models"""
        if self.historical_data.empty:
            logger.warning("No historical data available for training")
            return
            
        # Prepare features
        feature_columns = [
            'day_of_week', 'day_of_year', 'failed_logins', 
            'network_anomalies', 'vulnerability_scans', 'malware_detections'
        ]
        
        X = self.historical_data[feature_columns]
        y_incident = self.historical_data['incident_occurred']
        y_risk = self.historical_data['risk_score']
        
        # Train incident prediction model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_incident, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest for incident prediction
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test_scaled)
        logger.info("Incident Prediction Model Performance:")
        logger.info(classification_report(y_test, y_pred))
        
        # Store models
        self.models['incident_prediction'] = rf_model
        self.scalers['incident_prediction'] = scaler
        self.feature_columns = feature_columns
        
        # Train anomaly detection model
        isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        isolation_forest.fit(X_train_scaled)
        
        self.models['anomaly_detection'] = isolation_forest
        
        logger.info("Predictive models trained successfully")
        
    def predict_security_incidents(self, time_horizon: int = 7) -> List[SecurityPrediction]:
        """Predict security incidents for the next time_horizon days"""
        predictions = []
        
        if 'incident_prediction' not in self.models:
            logger.warning("Incident prediction model not available")
            return predictions
            
        model = self.models['incident_prediction']
        scaler = self.scalers['incident_prediction']
        
        # Generate predictions for each day in the time horizon
        for days_ahead in range(1, time_horizon + 1):
            future_date = datetime.now() + timedelta(days=days_ahead)
            
            # Create feature vector for future date
            features = self._create_feature_vector(future_date)
            features_scaled = scaler.transform([features])
            
            # Get prediction probability
            prob = model.predict_proba(features_scaled)[0][1]  # Probability of incident
            
            # Calculate confidence interval (simplified)
            confidence_interval = (max(0, prob - 0.1), min(1, prob + 0.1))
            
            # Determine risk level
            risk_level = self._probability_to_risk_level(prob)
            
            # Identify contributing factors
            contributing_factors = self._identify_contributing_factors(features, model)
            
            # Generate recommendations
            recommendations = self._generate_incident_recommendations(prob, contributing_factors)
            
            prediction = SecurityPrediction(
                prediction_id=f"incident_pred_{int(future_date.timestamp())}",
                prediction_type=PredictionType.SECURITY_INCIDENT,
                predicted_probability=prob,
                confidence_interval=confidence_interval,
                risk_level=risk_level,
                time_horizon=days_ahead,
                contributing_factors=contributing_factors,
                recommended_actions=recommendations,
                timestamp=datetime.now()
            )
            
            predictions.append(prediction)
            
        return predictions
        
    def _create_feature_vector(self, target_date: datetime) -> List[float]:
        """Create feature vector for a target date"""
        day_of_week = target_date.weekday()
        day_of_year = target_date.timetuple().tm_yday
        
        # Estimate future metrics based on historical patterns
        recent_data = self.historical_data.tail(7)  # Last 7 days
        
        failed_logins = recent_data['failed_logins'].mean() + np.random.normal(0, 2)
        network_anomalies = recent_data['network_anomalies'].mean() + np.random.normal(0, 1)
        vulnerability_scans = recent_data['vulnerability_scans'].mean() + np.random.normal(0, 1)
        malware_detections = recent_data['malware_detections'].mean() + np.random.normal(0, 0.5)
        
        return [
            day_of_week, day_of_year, failed_logins,
            network_anomalies, vulnerability_scans, malware_detections
        ]
        
    def _probability_to_risk_level(self, probability: float) -> RiskLevel:
        """Convert probability to risk level"""
        if probability >= 0.8:
            return RiskLevel.CRITICAL
        elif probability >= 0.6:
            return RiskLevel.HIGH
        elif probability >= 0.4:
            return RiskLevel.MEDIUM
        elif probability >= 0.2:
            return RiskLevel.LOW
        else:
            return RiskLevel.VERY_LOW
            
    def _identify_contributing_factors(self, features: List[float], model) -> List[str]:
        """Identify factors contributing to the prediction"""
        feature_names = self.feature_columns
        feature_importance = model.feature_importances_
        
        # Get top contributing features
        feature_contributions = list(zip(feature_names, features, feature_importance))
        feature_contributions.sort(key=lambda x: x[2], reverse=True)
        
        contributing_factors = []
        for name, value, importance in feature_contributions[:3]:
            if importance > 0.1:  # Only include significant factors
                if name == 'failed_logins' and value > 25:
                    contributing_factors.append("High number of failed login attempts")
                elif name == 'network_anomalies' and value > 7:
                    contributing_factors.append("Elevated network anomaly detection")
                elif name == 'malware_detections' and value > 4:
                    contributing_factors.append("Increased malware detection activity")
                elif name == 'vulnerability_scans' and value > 15:
                    contributing_factors.append("High vulnerability discovery rate")
                    
        return contributing_factors
        
    def _generate_incident_recommendations(self, probability: float, factors: List[str]) -> List[str]:
        """Generate recommendations based on incident probability"""
        recommendations = []
        
        if probability >= 0.7:
            recommendations.extend([
                "Activate enhanced monitoring protocols",
                "Brief security team on elevated threat level",
                "Review and test incident response procedures",
                "Consider implementing additional security controls"
            ])
        elif probability >= 0.5:
            recommendations.extend([
                "Increase security monitoring frequency",
                "Review recent security logs for anomalies",
                "Ensure backup systems are current"
            ])
        elif probability >= 0.3:
            recommendations.extend([
                "Maintain standard monitoring protocols",
                "Schedule routine security reviews"
            ])
            
        # Add factor-specific recommendations
        if "failed login attempts" in str(factors).lower():
            recommendations.append("Review authentication logs and consider MFA enforcement")
        if "network anomaly" in str(factors).lower():
            recommendations.append("Investigate network traffic patterns and update firewall rules")
        if "malware detection" in str(factors).lower():
            recommendations.append("Update antivirus signatures and scan critical systems")
            
        return recommendations
        
    def analyze_security_trends(self, metrics: List[str], days_back: int = 30) -> List[TrendAnalysis]:
        """Analyze trends in security metrics"""
        trend_analyses = []
        
        if self.historical_data.empty:
            return trend_analyses
            
        # Get recent data
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_data = self.historical_data[self.historical_data['date'] >= cutoff_date].copy()
        
        for metric in metrics:
            if metric not in recent_data.columns:
                continue
                
            values = recent_data[metric].values
            dates = recent_data['date'].dt.strftime('%Y-%m-%d').tolist()
            
            # Calculate trend
            trend_direction, trend_strength = self._calculate_trend(values)
            
            # Detect seasonality
            seasonal_pattern = self._detect_seasonality(values)
            
            # Detect anomalies
            anomaly_detected = self._detect_anomalies(values)
            
            # Generate forecast
            forecast_values, confidence_bands = self._generate_forecast(values, days_ahead=7)
            forecast_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                            for i in range(1, 8)]
            
            trend_analysis = TrendAnalysis(
                trend_id=f"trend_{metric}_{int(datetime.now().timestamp())}",
                metric_name=metric,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                seasonal_pattern=seasonal_pattern,
                anomaly_detected=anomaly_detected,
                forecast_values=forecast_values,
                forecast_dates=forecast_dates,
                confidence_bands=confidence_bands
            )
            
            trend_analyses.append(trend_analysis)
            
        return trend_analyses
        
    def _calculate_trend(self, values: np.ndarray) -> Tuple[str, float]:
        """Calculate trend direction and strength"""
        if len(values) < 2:
            return "stable", 0.0
            
        # Simple linear regression to determine trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        # Calculate correlation coefficient as trend strength
        correlation = np.corrcoef(x, values)[0, 1]
        trend_strength = abs(correlation)
        
        if slope > 0.1:
            trend_direction = "increasing"
        elif slope < -0.1:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
            
        return trend_direction, trend_strength
        
    def _detect_seasonality(self, values: np.ndarray) -> bool:
        """Detect seasonal patterns in data"""
        if len(values) < 14:  # Need at least 2 weeks of data
            return False
            
        # Simple autocorrelation check for weekly patterns
        if len(values) >= 7:
            weekly_correlation = np.corrcoef(values[:-7], values[7:])[0, 1]
            return abs(weekly_correlation) > 0.3
            
        return False
        
    def _detect_anomalies(self, values: np.ndarray) -> bool:
        """Detect anomalies in recent data"""
        if len(values) < 5:
            return False
            
        # Use IQR method for anomaly detection
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Check if recent values are anomalous
        recent_values = values[-3:]  # Last 3 values
        anomalies = (recent_values < lower_bound) | (recent_values > upper_bound)
        
        return np.any(anomalies)
        
    def _generate_forecast(self, values: np.ndarray, days_ahead: int = 7) -> Tuple[List[float], List[Tuple[float, float]]]:
        """Generate forecast with confidence bands"""
        if len(values) < 3:
            return [values[-1]] * days_ahead, [(values[-1], values[-1])] * days_ahead
            
        # Simple exponential smoothing
        alpha = 0.3
        forecast = []
        
        # Initialize with last value
        last_forecast = values[-1]
        
        # Calculate forecast error for confidence bands
        errors = []
        for i in range(1, len(values)):
            predicted = values[i-1]  # Simple persistence model
            actual = values[i]
            errors.append(abs(actual - predicted))
            
        avg_error = np.mean(errors) if errors else 0
        
        for _ in range(days_ahead):
            # Simple trend continuation with smoothing
            if len(values) >= 2:
                trend = values[-1] - values[-2]
                last_forecast = last_forecast + alpha * trend
            
            forecast.append(last_forecast)
            
        # Generate confidence bands
        confidence_bands = [
            (max(0, f - 1.96 * avg_error), f + 1.96 * avg_error) 
            for f in forecast
        ]
        
        return forecast, confidence_bands
        
    def generate_risk_forecast(self, risk_categories: List[str], time_horizon: int = 30) -> List[RiskForecast]:
        """Generate comprehensive risk forecasts"""
        risk_forecasts = []
        
        for category in risk_categories:
            # Calculate current risk score
            current_risk = self._calculate_current_risk_score(category)
            
            # Generate forecast
            forecasted_scores = self._forecast_risk_scores(category, time_horizon)
            forecast_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                            for i in range(1, time_horizon + 1)]
            
            # Identify risk drivers
            risk_drivers = self._identify_risk_drivers(category)
            
            # Generate mitigation recommendations
            mitigation_recommendations = self._generate_mitigation_recommendations(category, current_risk)
            
            risk_forecast = RiskForecast(
                forecast_id=f"risk_forecast_{category}_{int(datetime.now().timestamp())}",
                risk_category=category,
                current_risk_score=current_risk,
                forecasted_risk_scores=forecasted_scores,
                forecast_dates=forecast_dates,
                risk_drivers=risk_drivers,
                mitigation_recommendations=mitigation_recommendations,
                confidence_level=0.75  # Default confidence level
            )
            
            risk_forecasts.append(risk_forecast)
            
        return risk_forecasts
        
    def _calculate_current_risk_score(self, category: str) -> float:
        """Calculate current risk score for a category"""
        # Simulate risk calculation based on category
        base_scores = {
            "cyber_attacks": 65,
            "data_breaches": 45,
            "compliance_violations": 30,
            "insider_threats": 40,
            "system_failures": 35
        }
        
        base_score = base_scores.get(category, 50)
        # Add some randomness
        return max(0, min(100, base_score + np.random.normal(0, 5)))
        
    def _forecast_risk_scores(self, category: str, days: int) -> List[float]:
        """Forecast risk scores for a category"""
        current_score = self._calculate_current_risk_score(category)
        scores = []
        
        for day in range(days):
            # Simple random walk with slight mean reversion
            change = np.random.normal(0, 2) - 0.1 * (current_score - 50) / 50
            current_score = max(0, min(100, current_score + change))
            scores.append(round(current_score, 1))
            
        return scores
        
    def _identify_risk_drivers(self, category: str) -> List[str]:
        """Identify key risk drivers for a category"""
        risk_drivers = {
            "cyber_attacks": [
                "Increased threat actor activity",
                "New vulnerability discoveries",
                "Phishing campaign trends",
                "Geopolitical tensions"
            ],
            "data_breaches": [
                "Weak access controls",
                "Unencrypted data stores",
                "Third-party integrations",
                "Employee security awareness"
            ],
            "compliance_violations": [
                "Regulatory changes",
                "Audit findings",
                "Process gaps",
                "Training deficiencies"
            ],
            "insider_threats": [
                "Employee satisfaction levels",
                "Access privilege creep",
                "Monitoring gaps",
                "Background check processes"
            ],
            "system_failures": [
                "Infrastructure age",
                "Maintenance schedules",
                "Capacity utilization",
                "Redundancy levels"
            ]
        }
        
        return risk_drivers.get(category, ["Unknown risk factors"])
        
    def _generate_mitigation_recommendations(self, category: str, risk_score: float) -> List[str]:
        """Generate risk mitigation recommendations"""
        base_recommendations = {
            "cyber_attacks": [
                "Enhance threat monitoring capabilities",
                "Update security awareness training",
                "Review and update incident response procedures",
                "Implement additional network segmentation"
            ],
            "data_breaches": [
                "Strengthen data encryption practices",
                "Review access control policies",
                "Implement data loss prevention tools",
                "Conduct regular security assessments"
            ],
            "compliance_violations": [
                "Update compliance monitoring procedures",
                "Provide additional staff training",
                "Review regulatory requirements",
                "Implement automated compliance checks"
            ],
            "insider_threats": [
                "Enhance user behavior monitoring",
                "Review privileged access management",
                "Implement zero-trust principles",
                "Conduct regular access reviews"
            ],
            "system_failures": [
                "Improve system monitoring and alerting",
                "Review backup and recovery procedures",
                "Plan infrastructure upgrades",
                "Implement redundancy measures"
            ]
        }
        
        recommendations = base_recommendations.get(category, ["Review security posture"])
        
        # Add urgency-based recommendations
        if risk_score >= 70:
            recommendations.insert(0, "Immediate action required - risk level critical")
        elif risk_score >= 50:
            recommendations.insert(0, "Prioritize mitigation efforts - elevated risk")
            
        return recommendations
        
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        return {
            "model_status": {
                "incident_prediction": "trained" if "incident_prediction" in self.models else "not_trained",
                "anomaly_detection": "trained" if "anomaly_detection" in self.models else "not_trained"
            },
            "data_summary": {
                "historical_records": len(self.historical_data),
                "date_range": {
                    "start": self.historical_data['date'].min().isoformat() if not self.historical_data.empty else None,
                    "end": self.historical_data['date'].max().isoformat() if not self.historical_data.empty else None
                }
            },
            "recent_predictions": self._get_recent_predictions_summary(),
            "trend_analysis": self._get_trend_summary(),
            "risk_forecasts": self._get_risk_forecast_summary()
        }
        
    def _get_recent_predictions_summary(self) -> Dict[str, Any]:
        """Get summary of recent predictions"""
        return {
            "last_prediction_run": datetime.now().isoformat(),
            "prediction_accuracy": 0.85,  # Simulated accuracy
            "high_risk_predictions": 2,
            "medium_risk_predictions": 3
        }
        
    def _get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of trend analysis"""
        return {
            "metrics_analyzed": len(self.feature_columns),
            "trending_up": 2,
            "trending_down": 1,
            "stable": 3,
            "anomalies_detected": 1
        }
        
    def _get_risk_forecast_summary(self) -> Dict[str, Any]:
        """Get summary of risk forecasts"""
        return {
            "forecast_horizon": "30 days",
            "categories_analyzed": 5,
            "high_risk_categories": 1,
            "improving_categories": 2,
            "stable_categories": 2
        }