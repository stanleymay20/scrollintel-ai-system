"""
Predictive Analytics Engine for forecasting and risk prediction.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import warnings
warnings.filterwarnings('ignore')

# Import forecasting libraries
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Install with: pip install statsmodels")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Install with: pip install tensorflow")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from ..models.predictive_models import (
    BusinessMetric, Forecast, ScenarioConfig, ScenarioResult,
    RiskPrediction, PredictionAccuracy, PredictionUpdate, BusinessContext,
    ForecastModel, RiskLevel, MetricCategory
)


class PredictiveEngine:
    """
    Comprehensive predictive analytics engine with multiple forecasting models,
    scenario modeling, and risk prediction capabilities.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.accuracy_tracker = {}
        self.confidence_tracker = {}  # Track confidence intervals performance
        self.early_warning_system = {}  # Track early warning thresholds
        self.risk_thresholds = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.7,
            RiskLevel.CRITICAL: 0.9
        }
        
    def forecast_metrics(self, metric: BusinessMetric, horizon: int, 
                        historical_data: List[BusinessMetric],
                        model_type: ForecastModel = ForecastModel.ENSEMBLE) -> Forecast:
        """
        Generate forecasts for business metrics with confidence intervals.
        
        Args:
            metric: Current business metric
            horizon: Forecast horizon in days
            historical_data: Historical metric data
            model_type: Forecasting model to use
            
        Returns:
            Forecast with predictions and confidence intervals
        """
        # Prepare data
        df = self._prepare_time_series_data(historical_data)
        
        if len(df) < 10:
            raise ValueError("Insufficient historical data for forecasting")
        
        try:
            # Generate forecast based on model type
            if model_type == ForecastModel.ENSEMBLE:
                forecast = self._ensemble_forecast(df, horizon, metric.id)
            elif model_type == ForecastModel.PROPHET:
                forecast = self._prophet_forecast(df, horizon, metric.id)
            elif model_type == ForecastModel.ARIMA:
                forecast = self._arima_forecast(df, horizon, metric.id)
            elif model_type == ForecastModel.LSTM:
                forecast = self._lstm_forecast(df, horizon, metric.id)
            else:
                forecast = self._linear_forecast(df, horizon, metric.id)
            
            # Calculate accuracy if we have recent data
            accuracy_score = self._calculate_forecast_accuracy(
                metric.id, model_type, df
            )
            forecast.accuracy_score = accuracy_score
            
            self.logger.info(f"Generated {model_type.value} forecast for metric {metric.id}")
            return forecast
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {str(e)}")
            # Return simple linear forecast as fallback
            return self._linear_forecast(df, horizon, metric.id)
    
    def model_scenario(self, scenario: ScenarioConfig,
                      historical_data: Dict[str, List[BusinessMetric]]) -> ScenarioResult:
        """
        Perform scenario modeling and what-if analysis.
        
        Args:
            scenario: Scenario configuration
            historical_data: Historical data for target metrics
            
        Returns:
            Scenario analysis results
        """
        try:
            baseline_forecasts = {}
            scenario_forecasts = {}
            impact_analysis = {}
            
            # Generate baseline forecasts
            for metric_id in scenario.target_metrics:
                if metric_id in historical_data:
                    # Create dummy metric for forecasting
                    latest_metric = historical_data[metric_id][-1]
                    baseline_forecast = self.forecast_metrics(
                        latest_metric, scenario.time_horizon, 
                        historical_data[metric_id]
                    )
                    baseline_forecasts[metric_id] = baseline_forecast
            
            # Apply scenario parameters and generate modified forecasts
            for metric_id in scenario.target_metrics:
                if metric_id in baseline_forecasts:
                    scenario_forecast = self._apply_scenario_parameters(
                        baseline_forecasts[metric_id], scenario.parameters
                    )
                    scenario_forecasts[metric_id] = scenario_forecast
                    
                    # Calculate impact
                    baseline_avg = np.mean(baseline_forecasts[metric_id].predictions)
                    scenario_avg = np.mean(scenario_forecast.predictions)
                    impact_analysis[metric_id] = (scenario_avg - baseline_avg) / baseline_avg * 100
            
            # Generate recommendations
            recommendations = self._generate_scenario_recommendations(
                impact_analysis, scenario.parameters
            )
            
            # Calculate overall confidence
            confidence_scores = [f.confidence_level for f in baseline_forecasts.values()]
            confidence_score = np.mean(confidence_scores) if confidence_scores else 0.8
            
            result = ScenarioResult(
                scenario_id=scenario.id,
                baseline_forecast=baseline_forecasts,
                scenario_forecast=scenario_forecasts,
                impact_analysis=impact_analysis,
                recommendations=recommendations,
                confidence_score=confidence_score,
                created_at=datetime.utcnow()
            )
            
            self.logger.info(f"Completed scenario modeling for {scenario.name}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in scenario modeling: {str(e)}")
            raise
    
    def predict_risks(self, context: BusinessContext,
                     current_metrics: List[BusinessMetric],
                     historical_data: Dict[str, List[BusinessMetric]]) -> List[RiskPrediction]:
        """
        Predict business risks with early warning systems.
        
        Args:
            context: Business context for risk analysis
            current_metrics: Current metric values
            historical_data: Historical metric data
            
        Returns:
            List of risk predictions
        """
        try:
            risk_predictions = []
            
            for metric in current_metrics:
                # Get historical data for this metric
                metric_history = historical_data.get(metric.id, [])
                if len(metric_history) < 5:
                    continue
                
                # Detect anomalies and trends
                risks = self._detect_metric_risks(metric, metric_history, context)
                risk_predictions.extend(risks)
            
            # Add systemic risks based on overall patterns
            systemic_risks = self._detect_systemic_risks(current_metrics, context)
            risk_predictions.extend(systemic_risks)
            
            # Sort by risk level and probability
            risk_predictions.sort(
                key=lambda x: (x.risk_level.value, -x.probability), 
                reverse=True
            )
            
            self.logger.info(f"Identified {len(risk_predictions)} potential risks")
            return risk_predictions
            
        except Exception as e:
            self.logger.error(f"Error predicting risks: {str(e)}")
            return []
    
    def update_predictions(self, new_data: List[BusinessMetric]) -> List[PredictionUpdate]:
        """
        Update predictions based on new data and notify stakeholders.
        
        Args:
            new_data: New metric data
            
        Returns:
            List of prediction updates
        """
        try:
            updates = []
            
            for metric in new_data:
                # Check if we have existing predictions for this metric
                if metric.id in self.models:
                    # Generate new forecast
                    historical_data = self._get_historical_data(metric.id)
                    new_forecast = self.forecast_metrics(metric, 30, historical_data)
                    
                    # Compare with previous forecast
                    previous_forecast = self.models[metric.id].get('last_forecast')
                    if previous_forecast:
                        change_magnitude = self._calculate_forecast_change(
                            previous_forecast, new_forecast
                        )
                        
                        if abs(change_magnitude) > 0.1:  # 10% threshold
                            update = PredictionUpdate(
                                metric_id=metric.id,
                                previous_forecast=previous_forecast,
                                updated_forecast=new_forecast,
                                change_magnitude=change_magnitude,
                                change_reason=self._determine_change_reason(
                                    metric, change_magnitude
                                ),
                                stakeholders_notified=[],
                                update_timestamp=datetime.utcnow()
                            )
                            updates.append(update)
                    
                    # Store new forecast
                    self.models[metric.id]['last_forecast'] = new_forecast
            
            self.logger.info(f"Generated {len(updates)} prediction updates")
            return updates
            
        except Exception as e:
            self.logger.error(f"Error updating predictions: {str(e)}")
            return []
    
    def _prepare_time_series_data(self, historical_data: List[BusinessMetric]) -> pd.DataFrame:
        """Prepare time series data for forecasting."""
        if not historical_data:
            return pd.DataFrame(columns=['ds', 'y', 'metric_id'])
            
        data = []
        for metric in historical_data:
            data.append({
                'ds': metric.timestamp,
                'y': metric.value,
                'metric_id': metric.id
            })
        
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)
        return df
    
    def _prophet_forecast(self, df: pd.DataFrame, horizon: int, metric_id: str) -> Forecast:
        """Generate forecast using Prophet model."""
        if not PROPHET_AVAILABLE:
            return self._linear_forecast(df, horizon, metric_id)
        
        try:
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=True,
                interval_width=0.8
            )
            model.fit(df[['ds', 'y']])
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=horizon)
            forecast_df = model.predict(future)
            
            # Extract predictions for future periods
            future_predictions = forecast_df.tail(horizon)
            
            return Forecast(
                metric_id=metric_id,
                model_type=ForecastModel.PROPHET,
                predictions=future_predictions['yhat'].tolist(),
                timestamps=[d.to_pydatetime() for d in future_predictions['ds']],
                confidence_lower=future_predictions['yhat_lower'].tolist(),
                confidence_upper=future_predictions['yhat_upper'].tolist(),
                confidence_level=0.8,
                accuracy_score=None,
                created_at=datetime.utcnow(),
                horizon_days=horizon
            )
            
        except Exception as e:
            self.logger.warning(f"Prophet forecast failed: {str(e)}")
            return self._linear_forecast(df, horizon, metric_id)
    
    def _arima_forecast(self, df: pd.DataFrame, horizon: int, metric_id: str) -> Forecast:
        """Generate forecast using ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            return self._linear_forecast(df, horizon, metric_id)
        
        try:
            # Prepare data
            ts = df.set_index('ds')['y']
            
            # Fit ARIMA model (using auto-selection for simplicity)
            model = ARIMA(ts, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast_result = fitted_model.forecast(steps=horizon, alpha=0.2)
            conf_int = fitted_model.get_forecast(steps=horizon).conf_int(alpha=0.2)
            
            # Generate future timestamps
            last_date = df['ds'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
            
            return Forecast(
                metric_id=metric_id,
                model_type=ForecastModel.ARIMA,
                predictions=forecast_result.tolist(),
                timestamps=future_dates,
                confidence_lower=conf_int.iloc[:, 0].tolist(),
                confidence_upper=conf_int.iloc[:, 1].tolist(),
                confidence_level=0.8,
                accuracy_score=None,
                created_at=datetime.utcnow(),
                horizon_days=horizon
            )
            
        except Exception as e:
            self.logger.warning(f"ARIMA forecast failed: {str(e)}")
            return self._linear_forecast(df, horizon, metric_id)
    
    def _lstm_forecast(self, df: pd.DataFrame, horizon: int, metric_id: str) -> Forecast:
        """Generate forecast using LSTM neural network."""
        if not TENSORFLOW_AVAILABLE:
            return self._linear_forecast(df, horizon, metric_id)
        
        try:
            # Prepare data
            values = df['y'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(values)
            
            # Create sequences for LSTM
            sequence_length = min(10, len(scaled_values) // 2)
            X, y = [], []
            for i in range(sequence_length, len(scaled_values)):
                X.append(scaled_values[i-sequence_length:i, 0])
                y.append(scaled_values[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            # Generate predictions
            predictions = []
            current_sequence = scaled_values[-sequence_length:].reshape(1, sequence_length, 1)
            
            for _ in range(horizon):
                pred = model.predict(current_sequence, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions).flatten()
            
            # Generate confidence intervals (simplified)
            std_dev = np.std(df['y'].values)
            confidence_lower = predictions - 1.96 * std_dev
            confidence_upper = predictions + 1.96 * std_dev
            
            # Generate future timestamps
            last_date = df['ds'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
            
            return Forecast(
                metric_id=metric_id,
                model_type=ForecastModel.LSTM,
                predictions=predictions.tolist(),
                timestamps=future_dates,
                confidence_lower=confidence_lower.tolist(),
                confidence_upper=confidence_upper.tolist(),
                confidence_level=0.95,
                accuracy_score=None,
                created_at=datetime.utcnow(),
                horizon_days=horizon
            )
            
        except Exception as e:
            self.logger.warning(f"LSTM forecast failed: {str(e)}")
            return self._linear_forecast(df, horizon, metric_id)
    
    def _linear_forecast(self, df: pd.DataFrame, horizon: int, metric_id: str) -> Forecast:
        """Generate simple linear regression forecast."""
        try:
            # Prepare data
            df['days'] = (df['ds'] - df['ds'].min()).dt.days
            X = df[['days']].values
            y = df['y'].values
            
            # Fit linear model
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate predictions
            last_day = df['days'].max()
            future_days = np.array([[last_day + i + 1] for i in range(horizon)])
            predictions = model.predict(future_days)
            
            # Calculate confidence intervals
            residuals = y - model.predict(X)
            std_error = np.std(residuals)
            confidence_lower = predictions - 1.96 * std_error
            confidence_upper = predictions + 1.96 * std_error
            
            # Generate future timestamps
            last_date = df['ds'].max()
            future_dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
            
            return Forecast(
                metric_id=metric_id,
                model_type=ForecastModel.LINEAR_REGRESSION,
                predictions=predictions.tolist(),
                timestamps=future_dates,
                confidence_lower=confidence_lower.tolist(),
                confidence_upper=confidence_upper.tolist(),
                confidence_level=0.95,
                accuracy_score=None,
                created_at=datetime.utcnow(),
                horizon_days=horizon
            )
            
        except Exception as e:
            self.logger.error(f"Linear forecast failed: {str(e)}")
            raise
    
    def _ensemble_forecast(self, df: pd.DataFrame, horizon: int, metric_id: str) -> Forecast:
        """Generate ensemble forecast combining multiple models."""
        try:
            forecasts = []
            weights = []
            
            # Try different models and collect results
            models_to_try = [
                (ForecastModel.LINEAR_REGRESSION, 0.2),
                (ForecastModel.PROPHET, 0.4),
                (ForecastModel.ARIMA, 0.3),
                (ForecastModel.LSTM, 0.1)
            ]
            
            for model_type, weight in models_to_try:
                try:
                    if model_type == ForecastModel.PROPHET:
                        forecast = self._prophet_forecast(df, horizon, metric_id)
                    elif model_type == ForecastModel.ARIMA:
                        forecast = self._arima_forecast(df, horizon, metric_id)
                    elif model_type == ForecastModel.LSTM:
                        forecast = self._lstm_forecast(df, horizon, metric_id)
                    else:
                        forecast = self._linear_forecast(df, horizon, metric_id)
                    
                    forecasts.append(forecast)
                    weights.append(weight)
                except Exception as e:
                    self.logger.warning(f"Model {model_type.value} failed in ensemble: {str(e)}")
                    continue
            
            if not forecasts:
                return self._linear_forecast(df, horizon, metric_id)
            
            # Normalize weights
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Combine predictions
            ensemble_predictions = np.zeros(horizon)
            ensemble_lower = np.zeros(horizon)
            ensemble_upper = np.zeros(horizon)
            
            for forecast, weight in zip(forecasts, weights):
                ensemble_predictions += np.array(forecast.predictions) * weight
                ensemble_lower += np.array(forecast.confidence_lower) * weight
                ensemble_upper += np.array(forecast.confidence_upper) * weight
            
            return Forecast(
                metric_id=metric_id,
                model_type=ForecastModel.ENSEMBLE,
                predictions=ensemble_predictions.tolist(),
                timestamps=forecasts[0].timestamps,
                confidence_lower=ensemble_lower.tolist(),
                confidence_upper=ensemble_upper.tolist(),
                confidence_level=0.85,
                accuracy_score=None,
                created_at=datetime.utcnow(),
                horizon_days=horizon
            )
            
        except Exception as e:
            self.logger.error(f"Ensemble forecast failed: {str(e)}")
            return self._linear_forecast(df, horizon, metric_id)   
 
    def _apply_scenario_parameters(self, baseline_forecast: Forecast, 
                                 parameters: Dict[str, Any]) -> Forecast:
        """Apply scenario parameters to modify baseline forecast."""
        try:
            modified_predictions = baseline_forecast.predictions.copy()
            
            # Apply percentage changes
            if 'percentage_change' in parameters:
                multiplier = 1 + (parameters['percentage_change'] / 100)
                modified_predictions = [p * multiplier for p in modified_predictions]
            
            # Apply seasonal adjustments
            if 'seasonal_adjustment' in parameters:
                seasonal_factor = parameters['seasonal_adjustment']
                for i, pred in enumerate(modified_predictions):
                    # Apply seasonal pattern (simplified)
                    seasonal_multiplier = 1 + seasonal_factor * np.sin(2 * np.pi * i / 365)
                    modified_predictions[i] = pred * seasonal_multiplier
            
            # Apply trend adjustments
            if 'trend_adjustment' in parameters:
                trend_factor = parameters['trend_adjustment']
                for i, pred in enumerate(modified_predictions):
                    trend_multiplier = 1 + (trend_factor * i / len(modified_predictions))
                    modified_predictions[i] = pred * trend_multiplier
            
            # Adjust confidence intervals proportionally
            baseline_avg = np.mean(baseline_forecast.predictions)
            modified_avg = np.mean(modified_predictions)
            adjustment_ratio = modified_avg / baseline_avg if baseline_avg != 0 else 1
            
            modified_lower = [l * adjustment_ratio for l in baseline_forecast.confidence_lower]
            modified_upper = [u * adjustment_ratio for u in baseline_forecast.confidence_upper]
            
            return Forecast(
                metric_id=baseline_forecast.metric_id,
                model_type=baseline_forecast.model_type,
                predictions=modified_predictions,
                timestamps=baseline_forecast.timestamps,
                confidence_lower=modified_lower,
                confidence_upper=modified_upper,
                confidence_level=baseline_forecast.confidence_level * 0.9,  # Reduce confidence
                accuracy_score=baseline_forecast.accuracy_score,
                created_at=datetime.utcnow(),
                horizon_days=baseline_forecast.horizon_days
            )
            
        except Exception as e:
            self.logger.error(f"Error applying scenario parameters: {str(e)}")
            return baseline_forecast
    
    def _generate_scenario_recommendations(self, impact_analysis: Dict[str, float],
                                         parameters: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on scenario analysis."""
        recommendations = []
        
        try:
            # Analyze impacts
            positive_impacts = {k: v for k, v in impact_analysis.items() if v > 0}
            negative_impacts = {k: v for k, v in impact_analysis.items() if v < 0}
            
            if positive_impacts:
                best_metric = max(positive_impacts, key=positive_impacts.get)
                recommendations.append(
                    f"Scenario shows {positive_impacts[best_metric]:.1f}% improvement in {best_metric}. "
                    "Consider implementing this strategy."
                )
            
            if negative_impacts:
                worst_metric = min(negative_impacts, key=negative_impacts.get)
                recommendations.append(
                    f"Scenario shows {abs(negative_impacts[worst_metric]):.1f}% decline in {worst_metric}. "
                    "Implement mitigation strategies."
                )
            
            # Parameter-specific recommendations
            if 'percentage_change' in parameters:
                change = parameters['percentage_change']
                if change > 10:
                    recommendations.append("Large positive change detected. Monitor for sustainability.")
                elif change < -10:
                    recommendations.append("Significant negative impact. Consider alternative approaches.")
            
            if not recommendations:
                recommendations.append("Scenario analysis complete. Review detailed metrics for insights.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations. Review scenario results manually."]
    
    def _detect_metric_risks(self, metric: BusinessMetric, 
                           history: List[BusinessMetric],
                           context: BusinessContext) -> List[RiskPrediction]:
        """Detect risks for a specific metric."""
        risks = []
        
        try:
            # Calculate recent trend
            recent_values = [m.value for m in history[-10:]]
            if len(recent_values) < 3:
                return risks
            
            trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
            current_value = metric.value
            historical_mean = np.mean([m.value for m in history])
            historical_std = np.std([m.value for m in history])
            
            # Anomaly detection
            z_score = abs((current_value - historical_mean) / historical_std) if historical_std > 0 else 0
            
            if z_score > 2:  # Significant anomaly
                risk_level = RiskLevel.HIGH if z_score > 3 else RiskLevel.MEDIUM
                risks.append(RiskPrediction(
                    id="",
                    metric_id=metric.id,
                    risk_type="anomaly",
                    risk_level=risk_level,
                    probability=min(z_score / 4, 1.0),
                    impact_score=z_score * 0.2,
                    description=f"Metric {metric.name} shows unusual deviation from historical patterns",
                    early_warning_threshold=historical_mean + 2 * historical_std,
                    mitigation_strategies=[
                        "Investigate root cause of deviation",
                        "Implement monitoring alerts",
                        "Review data quality"
                    ],
                    predicted_date=None,
                    created_at=datetime.utcnow()
                ))
            
            # Trend-based risks
            if trend < 0 and metric.category in [MetricCategory.FINANCIAL, MetricCategory.PERFORMANCE]:
                risk_level = RiskLevel.HIGH if abs(trend) > historical_std else RiskLevel.MEDIUM
                risks.append(RiskPrediction(
                    id="",
                    metric_id=metric.id,
                    risk_type="declining_trend",
                    risk_level=risk_level,
                    probability=0.7,
                    impact_score=abs(trend) / historical_mean if historical_mean > 0 else 0.5,
                    description=f"Declining trend detected in {metric.name}",
                    early_warning_threshold=current_value * 0.9,
                    mitigation_strategies=[
                        "Analyze trend drivers",
                        "Implement corrective measures",
                        "Set up early warning system"
                    ],
                    predicted_date=datetime.utcnow() + timedelta(days=30),
                    created_at=datetime.utcnow()
                ))
            
            # Threshold-based risks
            if hasattr(context, 'risk_thresholds') and metric.name in context.risk_thresholds:
                threshold = context.risk_thresholds[metric.name]
                if current_value > threshold:
                    risks.append(RiskPrediction(
                        id="",
                        metric_id=metric.id,
                        risk_type="threshold_breach",
                        risk_level=RiskLevel.CRITICAL,
                        probability=1.0,
                        impact_score=0.8,
                        description=f"Metric {metric.name} exceeded critical threshold",
                        early_warning_threshold=threshold * 0.9,
                        mitigation_strategies=[
                            "Immediate intervention required",
                            "Escalate to management",
                            "Implement emergency protocols"
                        ],
                        predicted_date=datetime.utcnow(),
                        created_at=datetime.utcnow()
                    ))
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error detecting metric risks: {str(e)}")
            return []
    
    def _detect_systemic_risks(self, metrics: List[BusinessMetric],
                             context: BusinessContext) -> List[RiskPrediction]:
        """Detect systemic risks across multiple metrics."""
        risks = []
        
        try:
            # Group metrics by category
            category_metrics = {}
            for metric in metrics:
                if metric.category not in category_metrics:
                    category_metrics[metric.category] = []
                category_metrics[metric.category].append(metric)
            
            # Check for correlated declines
            declining_categories = []
            for category, cat_metrics in category_metrics.items():
                if len(cat_metrics) >= 2:
                    # Simple correlation check (would be more sophisticated in production)
                    values = [m.value for m in cat_metrics]
                    if all(v < np.mean(values) * 0.9 for v in values[-2:]):
                        declining_categories.append(category)
            
            if len(declining_categories) >= 2:
                risks.append(RiskPrediction(
                    id="",
                    metric_id="systemic",
                    risk_type="systemic_decline",
                    risk_level=RiskLevel.HIGH,
                    probability=0.8,
                    impact_score=0.9,
                    description="Multiple metric categories showing simultaneous decline",
                    early_warning_threshold=0,
                    mitigation_strategies=[
                        "Conduct comprehensive business review",
                        "Implement cross-functional response team",
                        "Review strategic initiatives"
                    ],
                    predicted_date=datetime.utcnow() + timedelta(days=14),
                    created_at=datetime.utcnow()
                ))
            
            # Market condition risks
            if hasattr(context, 'market_conditions'):
                market_sentiment = context.market_conditions.get('sentiment', 'neutral')
                if market_sentiment == 'negative':
                    risks.append(RiskPrediction(
                        id="",
                        metric_id="market",
                        risk_type="market_conditions",
                        risk_level=RiskLevel.MEDIUM,
                        probability=0.6,
                        impact_score=0.7,
                        description="Negative market conditions may impact business metrics",
                        early_warning_threshold=0,
                        mitigation_strategies=[
                            "Implement defensive strategies",
                            "Focus on core business areas",
                            "Prepare contingency plans"
                        ],
                        predicted_date=datetime.utcnow() + timedelta(days=60),
                        created_at=datetime.utcnow()
                    ))
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Error detecting systemic risks: {str(e)}")
            return []
    
    def _calculate_forecast_accuracy(self, metric_id: str, model_type: ForecastModel,
                                   df: pd.DataFrame) -> Optional[float]:
        """Calculate forecast accuracy using historical data."""
        try:
            if len(df) < 20:  # Need sufficient data for validation
                return None
            
            # Use last 20% of data for validation
            split_point = int(len(df) * 0.8)
            train_df = df[:split_point]
            test_df = df[split_point:]
            
            # Generate forecast for test period
            test_horizon = len(test_df)
            if model_type == ForecastModel.PROPHET:
                forecast = self._prophet_forecast(train_df, test_horizon, metric_id)
            elif model_type == ForecastModel.ARIMA:
                forecast = self._arima_forecast(train_df, test_horizon, metric_id)
            elif model_type == ForecastModel.LSTM:
                forecast = self._lstm_forecast(train_df, test_horizon, metric_id)
            else:
                forecast = self._linear_forecast(train_df, test_horizon, metric_id)
            
            # Calculate accuracy metrics
            actual_values = test_df['y'].values
            predicted_values = np.array(forecast.predictions[:len(actual_values)])
            
            mae = mean_absolute_error(actual_values, predicted_values)
            mse = mean_squared_error(actual_values, predicted_values)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
            
            # Store accuracy metrics
            accuracy = PredictionAccuracy(
                model_type=model_type,
                metric_id=metric_id,
                mae=mae,
                mape=mape,
                rmse=rmse,
                r2_score=r2_score(actual_values, predicted_values),
                accuracy_trend=[],
                evaluation_date=datetime.utcnow(),
                sample_size=len(actual_values)
            )
            
            self.accuracy_tracker[f"{metric_id}_{model_type.value}"] = accuracy
            
            # Return overall accuracy score (inverse of normalized MAPE)
            return max(0, 1 - (mape / 100))
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast accuracy: {str(e)}")
            return None
    
    def _calculate_forecast_change(self, previous: Forecast, current: Forecast) -> float:
        """Calculate magnitude of change between forecasts."""
        try:
            prev_avg = np.mean(previous.predictions)
            curr_avg = np.mean(current.predictions)
            
            if prev_avg == 0:
                return 0
            
            return (curr_avg - prev_avg) / prev_avg
            
        except Exception as e:
            self.logger.error(f"Error calculating forecast change: {str(e)}")
            return 0
    
    def _determine_change_reason(self, metric: BusinessMetric, change_magnitude: float) -> str:
        """Determine reason for forecast change."""
        try:
            if abs(change_magnitude) > 0.3:
                return "Significant data shift detected"
            elif abs(change_magnitude) > 0.2:
                return "Notable trend change observed"
            elif abs(change_magnitude) > 0.1:
                return "Moderate pattern adjustment"
            else:
                return "Minor forecast refinement"
                
        except Exception as e:
            self.logger.error(f"Error determining change reason: {str(e)}")
            return "Forecast updated with new data"
    
    def _get_historical_data(self, metric_id: str) -> List[BusinessMetric]:
        """Get historical data for a metric (placeholder - would connect to database)."""
        # This would typically query a database
        # For now, return empty list as placeholder
        return []
    
    def get_model_performance(self, metric_id: str) -> Dict[str, PredictionAccuracy]:
        """Get performance metrics for all models for a specific metric."""
        performance = {}
        for key, accuracy in self.accuracy_tracker.items():
            if key.startswith(metric_id):
                # Extract model type from key (e.g., "test_metric_linear_regression" -> "linear_regression")
                parts = key.split('_')
                if len(parts) >= 2:
                    model_type = '_'.join(parts[2:])  # Join all parts after metric name
                    performance[model_type] = accuracy
        return performance
    
    def get_risk_summary(self, risks: List[RiskPrediction]) -> Dict[str, Any]:
        """Generate summary of risk predictions."""
        try:
            if not risks:
                return {"total_risks": 0, "risk_levels": {}, "top_risks": []}
            
            risk_levels = {}
            for risk in risks:
                level = risk.risk_level.value
                risk_levels[level] = risk_levels.get(level, 0) + 1
            
            # Get top 5 risks by impact
            top_risks = sorted(risks, key=lambda x: x.impact_score, reverse=True)[:5]
            
            return {
                "total_risks": len(risks),
                "risk_levels": risk_levels,
                "top_risks": [asdict(risk) for risk in top_risks],
                "average_probability": np.mean([r.probability for r in risks]),
                "average_impact": np.mean([r.impact_score for r in risks])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk summary: {str(e)}")
            return {"total_risks": 0, "risk_levels": {}, "top_risks": []}
    
    def track_confidence_intervals(self, forecast: Forecast, actual_values: List[float]) -> Dict[str, float]:
        """Track confidence interval performance."""
        try:
            if len(actual_values) != len(forecast.predictions):
                return {"coverage": 0.0, "width": 0.0, "reliability": 0.0}
            
            # Calculate coverage (percentage of actual values within confidence intervals)
            within_intervals = 0
            total_width = 0
            
            for i, actual in enumerate(actual_values):
                if i < len(forecast.confidence_lower) and i < len(forecast.confidence_upper):
                    lower = forecast.confidence_lower[i]
                    upper = forecast.confidence_upper[i]
                    
                    if lower <= actual <= upper:
                        within_intervals += 1
                    
                    total_width += (upper - lower)
            
            coverage = within_intervals / len(actual_values)
            avg_width = total_width / len(actual_values)
            
            # Reliability score (how close coverage is to stated confidence level)
            reliability = 1 - abs(coverage - forecast.confidence_level)
            
            # Store tracking data
            key = f"{forecast.metric_id}_{forecast.model_type.value}"
            self.confidence_tracker[key] = {
                "coverage": coverage,
                "average_width": avg_width,
                "reliability": reliability,
                "last_updated": datetime.utcnow()
            }
            
            return {
                "coverage": coverage,
                "width": avg_width,
                "reliability": reliability
            }
            
        except Exception as e:
            self.logger.error(f"Error tracking confidence intervals: {str(e)}")
            return {"coverage": 0.0, "width": 0.0, "reliability": 0.0}
    
    def setup_early_warning_system(self, metric_id: str, thresholds: Dict[str, float]) -> bool:
        """Setup early warning system for a metric."""
        try:
            self.early_warning_system[metric_id] = {
                "thresholds": thresholds,
                "alerts_sent": [],
                "last_check": datetime.utcnow(),
                "status": "active"
            }
            
            self.logger.info(f"Early warning system setup for metric {metric_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up early warning system: {str(e)}")
            return False
    
    def check_early_warnings(self, current_metrics: List[BusinessMetric]) -> List[Dict[str, Any]]:
        """Check for early warning triggers."""
        warnings = []
        
        try:
            for metric in current_metrics:
                if metric.id in self.early_warning_system:
                    warning_config = self.early_warning_system[metric.id]
                    thresholds = warning_config["thresholds"]
                    
                    for threshold_name, threshold_value in thresholds.items():
                        if metric.value >= threshold_value:
                            warning = {
                                "metric_id": metric.id,
                                "metric_name": metric.name,
                                "threshold_name": threshold_name,
                                "threshold_value": threshold_value,
                                "current_value": metric.value,
                                "severity": self._determine_warning_severity(
                                    metric.value, threshold_value
                                ),
                                "timestamp": datetime.utcnow(),
                                "recommended_actions": self._get_warning_actions(
                                    threshold_name, metric.category
                                )
                            }
                            warnings.append(warning)
                            
                            # Update alerts sent
                            warning_config["alerts_sent"].append(warning)
                            warning_config["last_check"] = datetime.utcnow()
            
            return warnings
            
        except Exception as e:
            self.logger.error(f"Error checking early warnings: {str(e)}")
            return []
    
    def _determine_warning_severity(self, current_value: float, threshold_value: float) -> str:
        """Determine warning severity based on threshold breach."""
        ratio = current_value / threshold_value if threshold_value > 0 else 1
        
        if ratio >= 2.0:
            return "critical"
        elif ratio >= 1.5:
            return "high"
        elif ratio >= 1.2:
            return "medium"
        else:
            return "low"
    
    def _get_warning_actions(self, threshold_name: str, category: MetricCategory) -> List[str]:
        """Get recommended actions for warning."""
        actions = []
        
        if threshold_name == "critical":
            actions.extend([
                "Immediate escalation required",
                "Activate emergency response team",
                "Implement containment measures"
            ])
        elif threshold_name == "high":
            actions.extend([
                "Alert management team",
                "Review recent changes",
                "Prepare mitigation plan"
            ])
        elif threshold_name == "medium":
            actions.extend([
                "Monitor closely",
                "Investigate root cause",
                "Consider preventive measures"
            ])
        
        # Category-specific actions
        if category == MetricCategory.FINANCIAL:
            actions.append("Review budget and spending")
        elif category == MetricCategory.PERFORMANCE:
            actions.append("Check system resources and optimization")
        elif category == MetricCategory.QUALITY:
            actions.append("Review quality assurance processes")
        
        return actions
    
    def get_prediction_performance_report(self, metric_id: str = None) -> Dict[str, Any]:
        """Generate comprehensive prediction performance report."""
        try:
            report = {
                "accuracy_metrics": {},
                "confidence_metrics": {},
                "model_comparison": {},
                "recommendations": []
            }
            
            # Filter by metric if specified
            accuracy_data = self.accuracy_tracker
            confidence_data = self.confidence_tracker
            
            if metric_id:
                accuracy_data = {k: v for k, v in accuracy_data.items() if k.startswith(metric_id)}
                confidence_data = {k: v for k, v in confidence_data.items() if k.startswith(metric_id)}
            
            # Accuracy metrics
            for key, accuracy in accuracy_data.items():
                report["accuracy_metrics"][key] = {
                    "mae": accuracy.mae,
                    "mape": accuracy.mape,
                    "rmse": accuracy.rmse,
                    "r2_score": accuracy.r2_score,
                    "sample_size": accuracy.sample_size
                }
            
            # Confidence metrics
            for key, confidence in confidence_data.items():
                report["confidence_metrics"][key] = confidence
            
            # Model comparison
            if len(accuracy_data) > 1:
                best_model = min(accuracy_data.items(), key=lambda x: x[1].mape)
                report["model_comparison"]["best_model"] = best_model[0]
                report["model_comparison"]["best_mape"] = best_model[1].mape
            
            # Recommendations
            report["recommendations"] = self._generate_performance_recommendations(
                accuracy_data, confidence_data
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_performance_recommendations(self, accuracy_data: Dict, confidence_data: Dict) -> List[str]:
        """Generate recommendations based on performance data."""
        recommendations = []
        
        try:
            # Check for poor performing models
            for key, accuracy in accuracy_data.items():
                if accuracy.mape > 20:  # High error rate
                    recommendations.append(
                        f"Model {key} has high error rate ({accuracy.mape:.1f}% MAPE). "
                        "Consider retraining or using different algorithm."
                    )
                
                if accuracy.r2_score < 0.5:  # Poor fit
                    recommendations.append(
                        f"Model {key} shows poor fit (R² = {accuracy.r2_score:.2f}). "
                        "Review feature engineering and data quality."
                    )
            
            # Check confidence interval performance
            for key, confidence in confidence_data.items():
                if confidence["coverage"] < 0.8:  # Poor coverage
                    recommendations.append(
                        f"Model {key} has poor confidence interval coverage "
                        f"({confidence['coverage']:.1%}). Consider adjusting interval calculation."
                    )
            
            # General recommendations
            if len(accuracy_data) == 0:
                recommendations.append("No accuracy data available. Implement model validation.")
            
            if len(confidence_data) == 0:
                recommendations.append("No confidence tracking data. Enable interval performance monitoring.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return ["Unable to generate recommendations due to error."]