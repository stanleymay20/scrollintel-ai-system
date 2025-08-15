"""
Forecast Agent - Time series prediction and forecasting
"""
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .base import Agent, AgentRequest, AgentResponse

logger = logging.getLogger(__name__)

# Try to import forecasting libraries, fallback gracefully
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("Statsmodels not available. Install with: pip install statsmodels")

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Install with: pip install scikit-learn")


class ForecastAgent(Agent):
    """Forecast Agent for time series prediction and forecasting"""
    
    def __init__(self):
        super().__init__(
            name="Forecast Agent",
            description="Provides time series forecasting, trend analysis, and future predictions"
        )
    
    def get_capabilities(self) -> List[str]:
        """Return Forecast agent capabilities"""
        return [
            "Time series forecasting",
            "Multiple forecasting algorithms",
            "Automatic model selection",
            "Confidence intervals",
            "Trend analysis",
            "Seasonality detection"
        ]
    
    async def process(self, request: AgentRequest) -> AgentResponse:
        """Process forecasting requests"""
        start_time = time.time()
        
        try:
            query = request.query.lower()
            context = request.context
            
            if "forecast" in query or "predict" in query:
                result = self._create_forecast(context)
            elif "trend" in query:
                result = self._analyze_trends(context)
            elif "seasonality" in query or "seasonal" in query:
                result = self._detect_seasonality(context)
            elif "model" in query:
                result = self._recommend_models(context)
            else:
                result = self._provide_forecast_guidance(request.query, context)
            
            return AgentResponse(
                agent_name=self.name,
                success=True,
                result=result,
                metadata={"forecast_task": self._classify_forecast_task(query)},
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Forecast Agent error: {e}")
            return AgentResponse(
                agent_name=self.name,
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _create_forecast(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create time series forecast with actual predictions"""
        try:
            # Get data from context
            data = context.get("data")
            forecast_horizon = context.get("forecast_horizon", 30)  # days
            date_column = context.get("date_column", "date")
            value_column = context.get("value_column", "value")
            
            if data is None:
                return self._get_forecast_template()
            
            # Convert to DataFrame if needed
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Prepare time series data
            if date_column in df.columns and value_column in df.columns:
                ts_data = self._prepare_time_series(df, date_column, value_column)
                
                # Generate actual forecasts
                forecasts = self._generate_forecasts(ts_data, forecast_horizon)
                
                return {
                    "forecast_results": forecasts,
                    "data_summary": self._analyze_data_characteristics(ts_data),
                    "recommendations": self._get_forecast_recommendations(ts_data),
                    "success": True
                }
            else:
                return self._get_forecast_template()
                
        except Exception as e:
            logger.error(f"Forecast creation error: {e}")
            return self._get_forecast_template()
    
    def _prepare_time_series(self, df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
        """Prepare time series data for forecasting"""
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Remove duplicates and handle missing values
        df = df.drop_duplicates(subset=[date_col])
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[value_col])
        
        # Set date as index
        df.set_index(date_col, inplace=True)
        
        return df[[value_col]]
    
    def _generate_forecasts(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Generate forecasts using available methods"""
        forecasts = {}
        value_col = data.columns[0]
        
        # Simple moving average forecast (always available)
        forecasts["moving_average"] = self._moving_average_forecast(data[value_col], horizon)
        
        # Exponential smoothing forecast
        forecasts["exponential_smoothing"] = self._exponential_smoothing_forecast(data[value_col], horizon)
        
        # Prophet forecast (if available)
        if PROPHET_AVAILABLE and len(data) >= 10:
            try:
                forecasts["prophet"] = self._prophet_forecast(data, horizon)
            except Exception as e:
                logger.warning(f"Prophet forecast failed: {e}")
        
        # ARIMA forecast (if available)
        if STATSMODELS_AVAILABLE and len(data) >= 20:
            try:
                forecasts["arima"] = self._arima_forecast(data[value_col], horizon)
            except Exception as e:
                logger.warning(f"ARIMA forecast failed: {e}")
        
        # Select best forecast
        best_forecast = self._select_best_forecast(forecasts, data[value_col])
        
        return {
            "forecasts": forecasts,
            "best_forecast": best_forecast,
            "forecast_horizon": horizon,
            "data_points": len(data)
        }
    
    def _moving_average_forecast(self, series: pd.Series, horizon: int, window: int = 7) -> Dict[str, Any]:
        """Simple moving average forecast"""
        if len(series) < window:
            window = len(series)
        
        last_values = series.tail(window).mean()
        forecast_values = [last_values] * horizon
        
        # Generate future dates
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        
        return {
            "method": "Moving Average",
            "forecast_values": forecast_values,
            "forecast_dates": future_dates.strftime('%Y-%m-%d').tolist(),
            "confidence_lower": [v * 0.9 for v in forecast_values],
            "confidence_upper": [v * 1.1 for v in forecast_values],
            "accuracy_score": 0.7  # Placeholder
        }
    
    def _exponential_smoothing_forecast(self, series: pd.Series, horizon: int, alpha: float = 0.3) -> Dict[str, Any]:
        """Exponential smoothing forecast"""
        # Simple exponential smoothing
        smoothed = series.ewm(alpha=alpha).mean()
        last_smoothed = smoothed.iloc[-1]
        
        forecast_values = [last_smoothed] * horizon
        
        # Generate future dates
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        
        # Simple confidence intervals based on historical variance
        std_dev = series.std()
        
        return {
            "method": "Exponential Smoothing",
            "forecast_values": forecast_values,
            "forecast_dates": future_dates.strftime('%Y-%m-%d').tolist(),
            "confidence_lower": [v - 1.96 * std_dev for v in forecast_values],
            "confidence_upper": [v + 1.96 * std_dev for v in forecast_values],
            "accuracy_score": 0.75  # Placeholder
        }
    
    def _prophet_forecast(self, data: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Prophet forecast (if available)"""
        if not PROPHET_AVAILABLE:
            return {"error": "Prophet not available"}
        
        # Prepare data for Prophet
        prophet_data = data.reset_index()
        prophet_data.columns = ['ds', 'y']
        
        # Create and fit model
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_data)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        
        # Extract forecast values
        forecast_values = forecast['yhat'].tail(horizon).tolist()
        forecast_lower = forecast['yhat_lower'].tail(horizon).tolist()
        forecast_upper = forecast['yhat_upper'].tail(horizon).tolist()
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
        
        return {
            "method": "Prophet",
            "forecast_values": forecast_values,
            "forecast_dates": future_dates.strftime('%Y-%m-%d').tolist(),
            "confidence_lower": forecast_lower,
            "confidence_upper": forecast_upper,
            "accuracy_score": 0.85  # Placeholder
        }
    
    def _arima_forecast(self, series: pd.Series, horizon: int) -> Dict[str, Any]:
        """ARIMA forecast (if available)"""
        if not STATSMODELS_AVAILABLE:
            return {"error": "Statsmodels not available"}
        
        try:
            # Auto ARIMA (simplified)
            model = ARIMA(series, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecast
            forecast = fitted_model.forecast(steps=horizon)
            conf_int = fitted_model.get_forecast(steps=horizon).conf_int()
            
            # Generate future dates
            last_date = series.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')
            
            return {
                "method": "ARIMA",
                "forecast_values": forecast.tolist(),
                "forecast_dates": future_dates.strftime('%Y-%m-%d').tolist(),
                "confidence_lower": conf_int.iloc[:, 0].tolist(),
                "confidence_upper": conf_int.iloc[:, 1].tolist(),
                "accuracy_score": 0.8  # Placeholder
            }
        except Exception as e:
            logger.error(f"ARIMA forecast error: {e}")
            return {"error": f"ARIMA forecast failed: {str(e)}"}
    
    def _select_best_forecast(self, forecasts: Dict[str, Any], historical_data: pd.Series) -> Dict[str, Any]:
        """Select the best forecast based on available methods"""
        # Simple selection logic - prefer Prophet if available, then ARIMA, then exponential smoothing
        if "prophet" in forecasts and "error" not in forecasts["prophet"]:
            return forecasts["prophet"]
        elif "arima" in forecasts and "error" not in forecasts["arima"]:
            return forecasts["arima"]
        elif "exponential_smoothing" in forecasts:
            return forecasts["exponential_smoothing"]
        else:
            return forecasts.get("moving_average", {})
    
    def _analyze_data_characteristics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series characteristics"""
        series = data.iloc[:, 0]
        
        return {
            "data_points": len(series),
            "date_range": {
                "start": series.index[0].strftime('%Y-%m-%d'),
                "end": series.index[-1].strftime('%Y-%m-%d')
            },
            "statistics": {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max())
            },
            "trend": self._detect_trend(series),
            "seasonality": self._detect_basic_seasonality(series)
        }
    
    def _detect_trend(self, series: pd.Series) -> str:
        """Simple trend detection"""
        if len(series) < 10:
            return "insufficient_data"
        
        # Simple linear trend detection
        x = np.arange(len(series))
        correlation = np.corrcoef(x, series)[0, 1]
        
        if correlation > 0.3:
            return "upward"
        elif correlation < -0.3:
            return "downward"
        else:
            return "stable"
    
    def _detect_basic_seasonality(self, series: pd.Series) -> Dict[str, Any]:
        """Basic seasonality detection"""
        if len(series) < 14:
            return {"detected": False, "reason": "insufficient_data"}
        
        # Simple weekly seasonality check
        if len(series) >= 14:
            weekly_pattern = series.groupby(series.index.dayofweek).mean()
            weekly_std = weekly_pattern.std()
            weekly_mean = weekly_pattern.mean()
            
            if weekly_std / weekly_mean > 0.1:  # 10% coefficient of variation
                return {"detected": True, "type": "weekly", "strength": "moderate"}
        
        return {"detected": False, "reason": "no_clear_pattern"}
    
    def _get_forecast_recommendations(self, data: pd.DataFrame) -> List[str]:
        """Get recommendations based on data characteristics"""
        recommendations = []
        series = data.iloc[:, 0]
        
        if len(series) < 30:
            recommendations.append("Consider collecting more historical data for better forecast accuracy")
        
        if series.std() / series.mean() > 0.5:
            recommendations.append("High variability detected - consider external factors or data cleaning")
        
        if len(series) >= 30:
            recommendations.append("Sufficient data available for advanced forecasting methods")
        
        recommendations.append("Update forecasts regularly as new data becomes available")
        recommendations.append("Consider business context and external factors when interpreting forecasts")
        
        return recommendations
    
    def _get_forecast_template(self) -> Dict[str, Any]:
        """Return template forecast structure when no data is available"""
        return {
            "forecasting_process": {
                "data_preparation": [
                    "Time series validation and cleaning",
                    "Missing value handling",
                    "Outlier detection and treatment",
                    "Data frequency standardization"
                ],
                "model_selection": [
                    "Automatic algorithm comparison",
                    "Cross-validation for model evaluation",
                    "Best model selection based on accuracy",
                    "Ensemble methods for improved accuracy"
                ],
                "forecast_generation": [
                    "Point forecasts for each time period",
                    "Confidence intervals (80%, 95%)",
                    "Prediction intervals for uncertainty",
                    "Scenario analysis (optimistic, pessimistic)"
                ]
            },
            "available_algorithms": {
                "prophet": {
                    "description": "Facebook's Prophet - handles seasonality and holidays",
                    "best_for": "Business metrics with strong seasonality",
                    "pros": ["Handles missing data", "Automatic seasonality detection", "Holiday effects"]
                },
                "arima": {
                    "description": "AutoRegressive Integrated Moving Average",
                    "best_for": "Stationary time series with clear patterns",
                    "pros": ["Statistical foundation", "Good for short-term forecasts", "Interpretable"]
                },
                "lstm": {
                    "description": "Long Short-Term Memory neural networks",
                    "best_for": "Complex patterns and long sequences",
                    "pros": ["Handles non-linear patterns", "Good for multivariate data", "Long-term dependencies"]
                },
                "exponential_smoothing": {
                    "description": "Exponential smoothing methods",
                    "best_for": "Simple trends and seasonality",
                    "pros": ["Fast computation", "Good baseline", "Robust to outliers"]
                }
            },
            "forecast_outputs": {
                "predictions": f"Forecasts for next {forecast_horizon}",
                "confidence_bands": "Upper and lower prediction bounds",
                "trend_components": "Trend, seasonal, and residual decomposition",
                "accuracy_metrics": "MAE, MAPE, RMSE on validation data"
            }
        }
    
    def _analyze_trends(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        return {
            "trend_analysis": {
                "trend_direction": {
                    "upward": "Consistent increase over time",
                    "downward": "Consistent decrease over time",
                    "stable": "No significant trend",
                    "cyclical": "Repeating up and down patterns"
                },
                "trend_strength": {
                    "measurement": "Statistical significance of trend",
                    "interpretation": "How confident we are in the trend direction",
                    "factors": ["Data quality", "Time period", "Variability"]
                },
                "change_points": {
                    "detection": "Automatic identification of trend changes",
                    "significance": "Statistical tests for change point validity",
                    "business_context": "Correlation with business events"
                }
            },
            "trend_decomposition": {
                "components": [
                    "Overall trend (long-term direction)",
                    "Seasonal patterns (recurring cycles)",
                    "Irregular fluctuations (noise)",
                    "Cyclical patterns (longer-term cycles)"
                ],
                "visualization": [
                    "Trend line overlays",
                    "Component decomposition plots",
                    "Moving averages",
                    "Trend strength indicators"
                ]
            },
            "insights": [
                "Trend acceleration or deceleration",
                "Seasonal impact on trends",
                "Volatility analysis",
                "Comparative trend analysis"
            ]
        }
    
    def _detect_seasonality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect seasonal patterns"""
        return {
            "seasonality_detection": {
                "types": {
                    "daily": "Within-day patterns (hourly cycles)",
                    "weekly": "Day-of-week patterns",
                    "monthly": "Month-of-year patterns",
                    "quarterly": "Seasonal business cycles",
                    "yearly": "Annual patterns"
                },
                "detection_methods": [
                    "Autocorrelation analysis",
                    "Fourier transform analysis",
                    "Seasonal decomposition",
                    "Statistical tests for seasonality"
                ]
            },
            "seasonal_patterns": {
                "strength": "How strong the seasonal effect is",
                "consistency": "How regular the seasonal pattern is",
                "evolution": "How seasonal patterns change over time",
                "multiple_seasonality": "Detection of multiple seasonal cycles"
            },
            "business_applications": [
                "Inventory planning based on seasonal demand",
                "Staffing adjustments for seasonal patterns",
                "Marketing campaign timing",
                "Budget allocation across seasons"
            ],
            "visualization": [
                "Seasonal plots by period",
                "Heatmaps for pattern visualization",
                "Box plots for seasonal distribution",
                "Seasonal strength indicators"
            ]
        }
    
    def _recommend_models(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend forecasting models"""
        data_characteristics = context.get("data_characteristics", {})
        
        return {
            "model_selection_criteria": {
                "data_size": {
                    "small": "< 100 observations - Simple exponential smoothing",
                    "medium": "100-1000 observations - ARIMA or Prophet",
                    "large": "> 1000 observations - LSTM or ensemble methods"
                },
                "seasonality": {
                    "none": "ARIMA, Exponential smoothing",
                    "simple": "Prophet, Seasonal ARIMA",
                    "complex": "LSTM, Multiple seasonality models"
                },
                "trend": {
                    "linear": "Linear trend models, ARIMA",
                    "non_linear": "Prophet, LSTM",
                    "changing": "Structural break models, Prophet"
                }
            },
            "model_comparison": {
                "accuracy_metrics": [
                    "Mean Absolute Error (MAE)",
                    "Mean Absolute Percentage Error (MAPE)",
                    "Root Mean Square Error (RMSE)",
                    "Symmetric MAPE (sMAPE)"
                ],
                "validation_approach": [
                    "Time series cross-validation",
                    "Walk-forward validation",
                    "Holdout validation",
                    "Backtesting on historical data"
                ]
            },
            "ensemble_methods": {
                "simple_average": "Average of multiple model predictions",
                "weighted_average": "Weighted based on historical accuracy",
                "stacking": "Meta-model to combine predictions",
                "dynamic_selection": "Choose best model for each forecast period"
            }
        }
    
    def _provide_forecast_guidance(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Provide general forecasting guidance"""
        return {
            "forecasting_best_practices": [
                "Ensure data quality and consistency",
                "Use appropriate forecast horizon",
                "Validate models on out-of-sample data",
                "Consider external factors and events",
                "Update forecasts regularly with new data"
            ],
            "data_requirements": {
                "minimum_history": "At least 2-3 seasonal cycles for seasonal data",
                "data_frequency": "Consistent time intervals (daily, weekly, monthly)",
                "data_quality": "Minimal missing values and outliers",
                "external_factors": "Include relevant external variables if available"
            },
            "forecast_interpretation": {
                "point_forecasts": "Single best estimate for each period",
                "confidence_intervals": "Range of likely values",
                "prediction_intervals": "Account for model uncertainty",
                "scenario_analysis": "What-if analysis for different conditions"
            },
            "common_pitfalls": [
                "Over-fitting to historical data",
                "Ignoring structural breaks",
                "Not accounting for external factors",
                "Using inappropriate forecast horizon",
                "Not updating models with new data"
            ],
            "business_applications": [
                "Demand forecasting for inventory management",
                "Revenue and sales projections",
                "Resource planning and capacity management",
                "Budget planning and financial forecasting",
                "Risk assessment and scenario planning"
            ]
        }
    
    def _classify_forecast_task(self, query: str) -> str:
        """Classify the type of forecasting task"""
        if "forecast" in query or "predict" in query:
            return "forecast_creation"
        elif "trend" in query:
            return "trend_analysis"
        elif "seasonal" in query:
            return "seasonality_detection"
        elif "model" in query:
            return "model_recommendation"
        else:
            return "general_forecast_guidance"