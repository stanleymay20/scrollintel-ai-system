"""
Predictive Security Analytics with 30-day risk forecasting capabilities
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib
import json

from .ml_siem_engine import SecurityEvent, ThreatLevel
from .behavioral_analytics_engine import BehaviorAnomaly

logger = logging.getLogger(__name__)

@dataclass
class RiskForecast:
    forecast_id: str
    entity_type: str  # 'user', 'ip', 'resource', 'organization'
    entity_id: str
    forecast_date: datetime
    predicted_risk_score: float
    confidence_interval: Tuple[float, float]
    risk_factors: Dict[str, float]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ThreatPrediction:
    prediction_id: str
    threat_type: str
    probability: float
    expected_timeframe: timedelta
    affected_entities: List[str]
    contributing_factors: Dict[str, float]
    mitigation_strategies: List[str]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SecurityTrend:
    trend_id: str
    metric_name: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1
    historical_data: List[Tuple[datetime, float]]
    forecast_data: List[Tuple[datetime, float]]
    statistical_significance: float

class PredictiveSecurityAnalytics:
    """
    Advanced predictive security analytics system with 30-day risk forecasting
    """
    
    def __init__(self):
        # ML models for prediction
        self.risk_forecasting_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        self.threat_prediction_model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        self.trend_analysis_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Historical data storage
        self.historical_events: List[SecurityEvent] = []
        self.historical_anomalies: List[BehaviorAnomaly] = []
        self.risk_history: Dict[str, List[Tuple[datetime, float]]] = {}
        
        # Forecasting results
        self.active_forecasts: Dict[str, RiskForecast] = {}
        self.threat_predictions: Dict[str, ThreatPrediction] = {}
        self.security_trends: Dict[str, SecurityTrend] = {}
        
        # Performance metrics
        self.metrics = {
            'forecasts_generated': 0,
            'predictions_made': 0,
            'forecast_accuracy': 0.0,
            'prediction_accuracy': 0.0,
            'trends_identified': 0
        }
        
        # Feature engineering parameters
        self.feature_windows = [1, 7, 14, 30]  # Days for feature calculation
        self.forecast_horizon = 30  # Days to forecast ahead
    
    async def initialize(self):
        """Initialize the predictive analytics system"""
        try:
            await self._load_models()
            await self._load_historical_data()
            if not self.is_trained:
                await self._train_initial_models()
            logger.info("Predictive Security Analytics initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Predictive Analytics: {e}")
            raise
    
    async def _load_models(self):
        """Load pre-trained models"""
        try:
            self.risk_forecasting_model = joblib.load('security/models/risk_forecasting.pkl')
            self.threat_prediction_model = joblib.load('security/models/threat_prediction.pkl')
            self.trend_analysis_model = joblib.load('security/models/trend_analysis.pkl')
            self.scaler = joblib.load('security/models/predictive_scaler.pkl')
            self.is_trained = True
            logger.info("Loaded pre-trained predictive models")
        except FileNotFoundError:
            logger.info("No pre-trained models found, will train new ones")
    
    async def _load_historical_data(self):
        """Load historical security data for analysis"""
        # In a real implementation, this would load from database
        # For demo, we'll generate synthetic historical data
        await self._generate_synthetic_historical_data()
    
    async def _generate_synthetic_historical_data(self):
        """Generate synthetic historical data for training"""
        # Generate 90 days of historical events
        base_date = datetime.now() - timedelta(days=90)
        
        for day in range(90):
            current_date = base_date + timedelta(days=day)
            
            # Generate daily events with trends
            daily_events = np.random.poisson(20)  # Average 20 events per day
            
            for _ in range(daily_events):
                # Create synthetic event
                event_time = current_date + timedelta(
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                )
                
                # Risk score with weekly and monthly patterns
                base_risk = 0.3
                weekly_pattern = 0.2 * np.sin(2 * np.pi * day / 7)  # Weekly cycle
                monthly_pattern = 0.1 * np.sin(2 * np.pi * day / 30)  # Monthly cycle
                trend = 0.002 * day  # Slight upward trend
                noise = np.random.normal(0, 0.1)
                
                # Add some high-risk events to ensure we have both classes
                if np.random.random() < 0.1:  # 10% chance of high-risk event
                    base_risk = 0.8
                
                risk_score = max(0, min(1, base_risk + weekly_pattern + monthly_pattern + trend + noise))
                
                # Store risk history
                entity_id = f"user_{np.random.randint(1, 10)}"  # Fewer users for more data per user
                if entity_id not in self.risk_history:
                    self.risk_history[entity_id] = []
                self.risk_history[entity_id].append((event_time, risk_score))
    
    async def _train_initial_models(self):
        """Train initial predictive models"""
        if not self.risk_history:
            await self._generate_synthetic_historical_data()
        
        # Prepare training data
        X_risk, y_risk = self._prepare_risk_forecasting_data()
        X_threat, y_threat = self._prepare_threat_prediction_data()
        X_trend, y_trend = self._prepare_trend_analysis_data()
        
        if len(X_risk) > 0:
            # Scale features
            X_risk_scaled = self.scaler.fit_transform(X_risk)
            
            # Train risk forecasting model
            self.risk_forecasting_model.fit(X_risk_scaled, y_risk)
            
            # Train threat prediction model with separate scaler if needed
            if len(X_threat) > 0:
                if X_threat.shape[1] != X_risk.shape[1]:
                    # Use separate scaler for threat prediction if dimensions differ
                    threat_scaler = StandardScaler()
                    X_threat_scaled = threat_scaler.fit_transform(X_threat)
                else:
                    X_threat_scaled = self.scaler.transform(X_threat)
                self.threat_prediction_model.fit(X_threat_scaled, y_threat)
            
            # Train trend analysis model with separate scaler if needed
            if len(X_trend) > 0:
                if X_trend.shape[1] != X_risk.shape[1]:
                    # Use separate scaler for trend analysis if dimensions differ
                    trend_scaler = StandardScaler()
                    X_trend_scaled = trend_scaler.fit_transform(X_trend)
                else:
                    X_trend_scaled = self.scaler.transform(X_trend)
                self.trend_analysis_model.fit(X_trend_scaled, y_trend)
            
            self.is_trained = True
            await self._save_models()
            logger.info("Trained predictive models successfully")
    
    def _prepare_risk_forecasting_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for risk forecasting model"""
        X, y = [], []
        
        for entity_id, history in self.risk_history.items():
            if len(history) < 30:  # Need at least 30 days of history
                continue
            
            # Sort by timestamp
            history.sort(key=lambda x: x[0])
            
            # Create sliding windows
            for i in range(30, len(history) - 7):  # Predict 7 days ahead
                # Features: risk scores for past 30 days
                features = []
                
                # Historical risk scores
                for j in range(i - 30, i):
                    features.append(history[j][1])
                
                # Time-based features
                current_date = history[i][0]
                features.extend([
                    current_date.weekday() / 7.0,
                    current_date.day / 31.0,
                    current_date.month / 12.0,
                    (current_date.timestamp() % (24 * 3600)) / (24 * 3600)  # Time of day
                ])
                
                # Statistical features
                recent_scores = [h[1] for h in history[i-7:i]]
                features.extend([
                    np.mean(recent_scores),
                    np.std(recent_scores),
                    np.max(recent_scores),
                    np.min(recent_scores)
                ])
                
                # Target: risk score 7 days later
                target = history[i + 7][1]
                
                X.append(features)
                y.append(target)
        
        return np.array(X), np.array(y)
    
    def _prepare_threat_prediction_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for threat prediction model"""
        X, y = [], []
        
        # Simplified threat prediction data
        for entity_id, history in self.risk_history.items():
            if len(history) < 14:
                continue
            
            history.sort(key=lambda x: x[0])
            
            for i in range(7, len(history) - 7):
                # Features: recent risk patterns
                recent_scores = [h[1] for h in history[i-7:i]]
                features = [
                    np.mean(recent_scores),
                    np.std(recent_scores),
                    np.max(recent_scores) - np.min(recent_scores),
                    len([s for s in recent_scores if s > 0.5]) / len(recent_scores)
                ]
                
                # Target: whether a high-risk event occurs in next 7 days
                future_scores = [h[1] for h in history[i:i+7]]
                # Make sure we have both classes in the data
                target = 1 if max(future_scores) > 0.6 else 0
                
                X.append(features)
                y.append(target)
        
        return np.array(X), np.array(y)
    
    def _prepare_trend_analysis_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for trend analysis"""
        X, y = [], []
        
        # Aggregate daily risk scores
        daily_risks = {}
        for entity_id, history in self.risk_history.items():
            for timestamp, risk_score in history:
                date_key = timestamp.date()
                if date_key not in daily_risks:
                    daily_risks[date_key] = []
                daily_risks[date_key].append(risk_score)
        
        # Calculate daily averages
        daily_averages = [(date, np.mean(scores)) for date, scores in daily_risks.items()]
        daily_averages.sort(key=lambda x: x[0])
        
        # Create trend prediction features
        for i in range(14, len(daily_averages) - 1):
            # Features: past 14 days of average risk
            features = [avg for _, avg in daily_averages[i-14:i]]
            
            # Target: next day's average risk
            target = daily_averages[i + 1][1]
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    async def generate_risk_forecast(self, entity_type: str, entity_id: str, 
                                   forecast_days: int = 30) -> RiskForecast:
        """Generate risk forecast for specific entity"""
        if not self.is_trained:
            await self.initialize()
        
        try:
            # Get historical data for entity
            if entity_id not in self.risk_history:
                # Create baseline forecast for new entity
                return await self._create_baseline_forecast(entity_type, entity_id, forecast_days)
            
            history = self.risk_history[entity_id]
            if len(history) < 30:
                return await self._create_baseline_forecast(entity_type, entity_id, forecast_days)
            
            # Prepare features for prediction
            features = self._extract_forecasting_features(history)
            features_scaled = self.scaler.transform([features])
            
            # Generate prediction
            predicted_risk = self.risk_forecasting_model.predict(features_scaled)[0]
            
            # Calculate confidence interval (simplified)
            confidence_interval = (
                max(0, predicted_risk - 0.1),
                min(1, predicted_risk + 0.1)
            )
            
            # Identify risk factors
            risk_factors = self._analyze_risk_factors(history, features)
            
            # Generate recommendations
            recommendations = self._generate_risk_recommendations(predicted_risk, risk_factors)
            
            forecast = RiskForecast(
                forecast_id=f"forecast_{entity_id}_{int(datetime.now().timestamp())}",
                entity_type=entity_type,
                entity_id=entity_id,
                forecast_date=datetime.now() + timedelta(days=forecast_days),
                predicted_risk_score=predicted_risk,
                confidence_interval=confidence_interval,
                risk_factors=risk_factors,
                recommendations=recommendations
            )
            
            self.active_forecasts[forecast.forecast_id] = forecast
            self.metrics['forecasts_generated'] += 1
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating risk forecast for {entity_id}: {e}")
            return await self._create_baseline_forecast(entity_type, entity_id, forecast_days)
    
    async def _create_baseline_forecast(self, entity_type: str, entity_id: str, 
                                      forecast_days: int) -> RiskForecast:
        """Create baseline forecast for entities without history"""
        baseline_risk = 0.3  # Default baseline risk
        
        return RiskForecast(
            forecast_id=f"baseline_{entity_id}_{int(datetime.now().timestamp())}",
            entity_type=entity_type,
            entity_id=entity_id,
            forecast_date=datetime.now() + timedelta(days=forecast_days),
            predicted_risk_score=baseline_risk,
            confidence_interval=(0.2, 0.4),
            risk_factors={'baseline': 1.0},
            recommendations=['Establish baseline behavior patterns', 'Monitor initial activity']
        )
    
    def _extract_forecasting_features(self, history: List[Tuple[datetime, float]]) -> List[float]:
        """Extract features for risk forecasting"""
        # Sort by timestamp
        history.sort(key=lambda x: x[0])
        
        # Get recent risk scores (last 30 days)
        recent_scores = [score for _, score in history[-30:]]
        
        # Pad if necessary
        while len(recent_scores) < 30:
            recent_scores.insert(0, 0.2)  # Default baseline
        
        features = recent_scores.copy()
        
        # Add time-based features
        latest_date = history[-1][0]
        features.extend([
            latest_date.weekday() / 7.0,
            latest_date.day / 31.0,
            latest_date.month / 12.0,
            (latest_date.timestamp() % (24 * 3600)) / (24 * 3600)
        ])
        
        # Add statistical features
        features.extend([
            np.mean(recent_scores),
            np.std(recent_scores),
            np.max(recent_scores),
            np.min(recent_scores)
        ])
        
        return features
    
    def _analyze_risk_factors(self, history: List[Tuple[datetime, float]], 
                            features: List[float]) -> Dict[str, float]:
        """Analyze contributing risk factors"""
        recent_scores = features[:30]  # First 30 features are historical scores
        
        risk_factors = {
            'trend': self._calculate_trend(recent_scores),
            'volatility': np.std(recent_scores),
            'recent_peak': max(recent_scores[-7:]) if len(recent_scores) >= 7 else 0,
            'baseline_deviation': abs(np.mean(recent_scores) - 0.2),
            'weekend_activity': self._calculate_weekend_factor(history)
        }
        
        return risk_factors
    
    def _calculate_trend(self, scores: List[float]) -> float:
        """Calculate trend direction and strength"""
        if len(scores) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(scores))
        coeffs = np.polyfit(x, scores, 1)
        return coeffs[0]  # Slope indicates trend
    
    def _calculate_weekend_factor(self, history: List[Tuple[datetime, float]]) -> float:
        """Calculate weekend activity risk factor"""
        weekend_scores = [score for timestamp, score in history 
                         if timestamp.weekday() >= 5]
        weekday_scores = [score for timestamp, score in history 
                         if timestamp.weekday() < 5]
        
        if not weekend_scores or not weekday_scores:
            return 0.0
        
        return np.mean(weekend_scores) - np.mean(weekday_scores)
    
    def _generate_risk_recommendations(self, predicted_risk: float, 
                                     risk_factors: Dict[str, float]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if predicted_risk > 0.7:
            recommendations.append("Implement enhanced monitoring")
            recommendations.append("Consider access restrictions")
        
        if risk_factors.get('trend', 0) > 0.01:
            recommendations.append("Investigate increasing risk trend")
        
        if risk_factors.get('volatility', 0) > 0.2:
            recommendations.append("Review inconsistent behavior patterns")
        
        if risk_factors.get('weekend_activity', 0) > 0.1:
            recommendations.append("Monitor weekend activity closely")
        
        if risk_factors.get('recent_peak', 0) > 0.8:
            recommendations.append("Investigate recent high-risk events")
        
        if not recommendations:
            recommendations.append("Continue standard monitoring")
        
        return recommendations
    
    async def predict_threat_likelihood(self, threat_type: str, 
                                      timeframe_days: int = 30) -> ThreatPrediction:
        """Predict likelihood of specific threat type"""
        if not self.is_trained:
            await self.initialize()
        
        try:
            # Analyze current security posture
            current_features = self._extract_threat_prediction_features()
            features_scaled = self.scaler.transform([current_features])
            
            # Predict threat probability
            probability = self.threat_prediction_model.predict_proba(features_scaled)[0][1]
            
            # Identify contributing factors
            contributing_factors = self._analyze_threat_factors(current_features, threat_type)
            
            # Identify affected entities
            affected_entities = self._identify_high_risk_entities()
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(threat_type, probability)
            
            prediction = ThreatPrediction(
                prediction_id=f"threat_{threat_type}_{int(datetime.now().timestamp())}",
                threat_type=threat_type,
                probability=probability,
                expected_timeframe=timedelta(days=timeframe_days),
                affected_entities=affected_entities,
                contributing_factors=contributing_factors,
                mitigation_strategies=mitigation_strategies
            )
            
            self.threat_predictions[prediction.prediction_id] = prediction
            self.metrics['predictions_made'] += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error predicting threat {threat_type}: {e}")
            # Return baseline prediction
            return ThreatPrediction(
                prediction_id=f"baseline_{threat_type}_{int(datetime.now().timestamp())}",
                threat_type=threat_type,
                probability=0.3,
                expected_timeframe=timedelta(days=timeframe_days),
                affected_entities=[],
                contributing_factors={'baseline': 1.0},
                mitigation_strategies=['Maintain standard security posture']
            )
    
    def _extract_threat_prediction_features(self) -> List[float]:
        """Extract features for threat prediction"""
        # Calculate aggregate risk metrics
        all_recent_scores = []
        for history in self.risk_history.values():
            if history:
                recent = [score for _, score in history[-7:]]  # Last 7 days
                all_recent_scores.extend(recent)
        
        if not all_recent_scores:
            return [0.2, 0.1, 0.0, 0.0]  # Default features
        
        features = [
            np.mean(all_recent_scores),
            np.std(all_recent_scores),
            np.max(all_recent_scores) - np.min(all_recent_scores),
            len([s for s in all_recent_scores if s > 0.5]) / len(all_recent_scores)
        ]
        
        return features
    
    def _analyze_threat_factors(self, features: List[float], threat_type: str) -> Dict[str, float]:
        """Analyze factors contributing to threat likelihood"""
        avg_risk, risk_volatility, risk_range, high_risk_ratio = features
        
        factors = {
            'average_risk_level': avg_risk,
            'risk_volatility': risk_volatility,
            'risk_range': risk_range,
            'high_risk_events_ratio': high_risk_ratio
        }
        
        # Add threat-specific factors
        if threat_type == 'data_breach':
            factors['data_access_anomalies'] = high_risk_ratio * 1.2
        elif threat_type == 'insider_threat':
            factors['behavioral_anomalies'] = risk_volatility * 1.5
        elif threat_type == 'malware':
            factors['external_connections'] = avg_risk * 0.8
        
        return factors
    
    def _identify_high_risk_entities(self) -> List[str]:
        """Identify entities at highest risk"""
        high_risk_entities = []
        
        for entity_id, history in self.risk_history.items():
            if history:
                recent_avg = np.mean([score for _, score in history[-7:]])
                if recent_avg > 0.6:
                    high_risk_entities.append(entity_id)
        
        return high_risk_entities[:10]  # Top 10 highest risk
    
    def _generate_mitigation_strategies(self, threat_type: str, probability: float) -> List[str]:
        """Generate threat-specific mitigation strategies"""
        base_strategies = ["Enhance monitoring", "Review access controls"]
        
        threat_strategies = {
            'data_breach': [
                "Implement data loss prevention",
                "Encrypt sensitive data",
                "Monitor data access patterns"
            ],
            'insider_threat': [
                "Implement user behavior analytics",
                "Enforce principle of least privilege",
                "Conduct security awareness training"
            ],
            'malware': [
                "Update antivirus signatures",
                "Implement email filtering",
                "Patch vulnerable systems"
            ],
            'phishing': [
                "Deploy email security gateway",
                "Conduct phishing simulations",
                "Implement multi-factor authentication"
            ]
        }
        
        strategies = base_strategies.copy()
        if threat_type in threat_strategies:
            strategies.extend(threat_strategies[threat_type])
        
        if probability > 0.7:
            strategies.append("Activate incident response team")
            strategies.append("Consider threat hunting activities")
        
        return strategies
    
    async def analyze_security_trends(self) -> List[SecurityTrend]:
        """Analyze security trends and patterns"""
        trends = []
        
        # Analyze overall risk trend
        overall_trend = await self._analyze_overall_risk_trend()
        if overall_trend:
            trends.append(overall_trend)
        
        # Analyze threat type trends
        threat_trends = await self._analyze_threat_type_trends()
        trends.extend(threat_trends)
        
        # Analyze temporal patterns
        temporal_trends = await self._analyze_temporal_patterns()
        trends.extend(temporal_trends)
        
        # Store trends
        for trend in trends:
            self.security_trends[trend.trend_id] = trend
        
        self.metrics['trends_identified'] = len(trends)
        
        return trends
    
    async def _analyze_overall_risk_trend(self) -> Optional[SecurityTrend]:
        """Analyze overall organizational risk trend"""
        # Aggregate daily risk scores
        daily_risks = {}
        for history in self.risk_history.values():
            for timestamp, risk_score in history:
                date_key = timestamp.date()
                if date_key not in daily_risks:
                    daily_risks[date_key] = []
                daily_risks[date_key].append(risk_score)
        
        if len(daily_risks) < 14:
            return None
        
        # Calculate daily averages
        daily_data = [(date, np.mean(scores)) for date, scores in daily_risks.items()]
        daily_data.sort(key=lambda x: x[0])
        
        # Calculate trend
        scores = [score for _, score in daily_data]
        trend_slope = self._calculate_trend(scores)
        
        # Determine trend direction
        if abs(trend_slope) < 0.001:
            direction = 'stable'
        elif trend_slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Generate forecast
        forecast_data = []
        last_score = scores[-1]
        for i in range(1, 31):  # 30-day forecast
            forecast_date = daily_data[-1][0] + timedelta(days=i)
            forecast_score = max(0, min(1, last_score + trend_slope * i))
            forecast_data.append((forecast_date, forecast_score))
        
        return SecurityTrend(
            trend_id=f"overall_risk_{int(datetime.now().timestamp())}",
            metric_name="Overall Risk Score",
            trend_direction=direction,
            trend_strength=abs(trend_slope) * 100,
            historical_data=[(datetime.combine(date, datetime.min.time()), score) 
                           for date, score in daily_data],
            forecast_data=[(datetime.combine(date, datetime.min.time()), score) 
                         for date, score in forecast_data],
            statistical_significance=0.95 if abs(trend_slope) > 0.005 else 0.5
        )
    
    async def _analyze_threat_type_trends(self) -> List[SecurityTrend]:
        """Analyze trends for different threat types"""
        # This would analyze trends in specific threat types
        # For demo, return empty list
        return []
    
    async def _analyze_temporal_patterns(self) -> List[SecurityTrend]:
        """Analyze temporal security patterns"""
        # This would analyze hourly, daily, weekly patterns
        # For demo, return empty list
        return []
    
    async def _save_models(self):
        """Save trained models"""
        import os
        os.makedirs('security/models', exist_ok=True)
        
        joblib.dump(self.risk_forecasting_model, 'security/models/risk_forecasting.pkl')
        joblib.dump(self.threat_prediction_model, 'security/models/threat_prediction.pkl')
        joblib.dump(self.trend_analysis_model, 'security/models/trend_analysis.pkl')
        joblib.dump(self.scaler, 'security/models/predictive_scaler.pkl')
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get predictive analytics performance metrics"""
        return {
            'forecasts_generated': self.metrics['forecasts_generated'],
            'predictions_made': self.metrics['predictions_made'],
            'forecast_accuracy': self.metrics.get('forecast_accuracy', 0.0),
            'prediction_accuracy': self.metrics.get('prediction_accuracy', 0.0),
            'trends_identified': self.metrics['trends_identified'],
            'active_forecasts': len(self.active_forecasts),
            'active_predictions': len(self.threat_predictions),
            'model_trained': self.is_trained,
            'forecast_horizon_days': self.forecast_horizon
        }