"""
ScrollForecast Engine for time series prediction and forecasting.
Implements requirement 2.3: Time series predictions using Prophet, ARIMA, or LSTM.
"""

import os
import json
import pickle
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from uuid import uuid4

# Core ML libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Time series libraries
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    Prophet = None

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    sm = None
    ARIMA = None
    seasonal_decompose = None
    adfuller = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None
    keras = None
    layers = None

# Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import io
import base64

import warnings
warnings.filterwarnings('ignore')

from .base_engine import BaseEngine, EngineCapability, EngineStatus

logger = logging.getLogger(__name__)


class ForecastModel:
    """Supported forecasting model types."""
    PROPHET = "prophet"
    ARIMA = "arima"
    LSTM = "lstm"


class SeasonalityType:
    """Types of seasonality patterns."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class ScrollForecastEngine(BaseEngine):
    """
    ScrollForecast engine for time series prediction and forecasting.
    
    Capabilities:
    - Multiple forecasting models (Prophet, ARIMA, LSTM)
    - Automated seasonal decomposition and trend analysis
    - Confidence interval calculation and uncertainty quantification
    - Automated model selection based on data characteristics
    - Forecast visualization with historical data comparison
    """
    
    def __init__(self):
        super().__init__(
            engine_id="scroll_forecast_engine",
            name="ScrollForecast Engine",
            capabilities=[
                EngineCapability.FORECASTING,
                EngineCapability.DATA_ANALYSIS,
                EngineCapability.VISUALIZATION
            ]
        )
        self.models_dir = Path("models/forecasts")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.trained_models = {}
        self.supported_models = []
        
        # Check available libraries
        if HAS_PROPHET:
            self.supported_models.append(ForecastModel.PROPHET)
        else:
            logger.warning("Prophet not available. Install with: pip install prophet")
            
        if HAS_STATSMODELS:
            self.supported_models.append(ForecastModel.ARIMA)
        else:
            logger.warning("Statsmodels not available. Install with: pip install statsmodels")
            
        if HAS_TENSORFLOW:
            self.supported_models.append(ForecastModel.LSTM)
        else:
            logger.warning("TensorFlow not available. Install with: pip install tensorflow")
    
    async def initialize(self) -> None:
        """Initialize the ScrollForecast engine."""
        logger.info("Initializing ScrollForecast engine...")
        
        # Ensure models directory exists
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load any existing trained models
        await self._load_existing_models()
        
        # Set status to ready
        self.status = EngineStatus.READY
        
        logger.info(f"ScrollForecast engine initialized with models: {self.supported_models}")
    
    async def process(self, input_data: Any, parameters: Dict[str, Any] = None) -> Any:
        """
        Process forecasting request.
        
        Args:
            input_data: Dictionary containing time series data and forecasting parameters
            parameters: Additional processing parameters
            
        Returns:
            Forecasting results with predictions and visualizations
        """
        try:
            action = input_data.get("action", "forecast")
            
            if action == "forecast":
                return await self._create_forecast(input_data, parameters)
            elif action == "analyze":
                return await self._analyze_time_series(input_data, parameters)
            elif action == "compare":
                return await self._compare_models(input_data, parameters)
            elif action == "decompose":
                return await self._decompose_time_series(input_data, parameters)
            else:
                raise ValueError(f"Unknown action: {action}")
                
        except Exception as e:
            logger.error(f"Error in ScrollForecast processing: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up ScrollForecast engine...")
        self.trained_models.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "healthy": True,
            "models_trained": len(self.trained_models),
            "supported_models": self.supported_models,
            "models_directory": str(self.models_dir),
            "libraries_available": {
                "prophet": HAS_PROPHET,
                "statsmodels": HAS_STATSMODELS,
                "tensorflow": HAS_TENSORFLOW
            }
        }
    
    async def _create_forecast(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create time series forecast using multiple models.
        
        Args:
            input_data: Contains time series data and forecasting parameters
            parameters: Additional forecasting parameters
            
        Returns:
            Forecasting results with predictions, metrics, and visualizations
        """
        start_time = datetime.utcnow()
        
        # Extract forecasting parameters
        data = input_data.get("data")
        date_column = input_data.get("date_column", "date")
        value_column = input_data.get("value_column", "value")
        forecast_periods = input_data.get("forecast_periods", 30)
        models_to_use = input_data.get("models", self.supported_models)
        confidence_level = input_data.get("confidence_level", 0.95)
        forecast_name = input_data.get("forecast_name", f"forecast_{uuid4().hex[:8]}")
        
        if not data:
            raise ValueError("Time series data is required")
        
        # Prepare data
        df = await self._prepare_time_series_data(data, date_column, value_column)
        
        # Analyze data characteristics
        data_analysis = await self._analyze_data_characteristics(df)
        
        # Auto-select best models if not specified
        if not models_to_use or models_to_use == ["auto"]:
            models_to_use = await self._auto_select_models(df, data_analysis)
        
        # Train and evaluate models
        results = {}
        best_model = None
        best_score = np.inf
        
        for model_type in models_to_use:
            if model_type not in self.supported_models:
                results[model_type] = {"error": f"Model {model_type} not available"}
                continue
                
            try:
                logger.info(f"Training {model_type} model...")
                model_result = await self._train_forecast_model(
                    df, model_type, forecast_periods, confidence_level
                )
                results[model_type] = model_result
                
                # Track best model based on validation error
                if "validation_mae" in model_result["metrics"]:
                    score = model_result["metrics"]["validation_mae"]
                    if score < best_score:
                        best_score = score
                        best_model = {
                            "type": model_type,
                            "model": model_result["model"],
                            "forecast": model_result["forecast"],
                            "metrics": model_result["metrics"]
                        }
                        
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                results[model_type] = {"error": str(e)}
        
        # Generate ensemble forecast if multiple models succeeded
        ensemble_forecast = None
        if len([r for r in results.values() if "forecast" in r]) > 1:
            ensemble_forecast = await self._create_ensemble_forecast(results)
        
        # Create visualizations
        visualizations = await self._create_forecast_visualizations(
            df, results, ensemble_forecast
        )
        
        # Save best model
        model_path = None
        if best_model:
            model_path = await self._save_forecast_model(
                best_model, forecast_name, df, data_analysis
            )
        
        forecast_duration = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "forecast_name": forecast_name,
            "data_analysis": data_analysis,
            "models_tested": models_to_use,
            "results": results,
            "best_model": {
                "type": best_model["type"] if best_model else None,
                "score": best_score if best_model else None,
                "metrics": best_model["metrics"] if best_model else None
            },
            "ensemble_forecast": ensemble_forecast,
            "visualizations": visualizations,
            "forecast_duration_seconds": forecast_duration,
            "model_path": model_path
        }
    
    async def _prepare_time_series_data(
        self, 
        data: Union[Dict, List, pd.DataFrame], 
        date_column: str, 
        value_column: str
    ) -> pd.DataFrame:
        """Prepare and validate time series data."""
        
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Ensure required columns exist
        if date_column not in df.columns:
            raise ValueError(f"Date column '{date_column}' not found in data")
        if value_column not in df.columns:
            raise ValueError(f"Value column '{value_column}' not found in data")
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Convert value column to numeric
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        
        # Remove rows with missing values
        df = df.dropna(subset=[date_column, value_column])
        
        # Sort by date
        df = df.sort_values(date_column).reset_index(drop=True)
        
        # Rename columns for consistency
        df = df.rename(columns={date_column: 'ds', value_column: 'y'})
        
        return df[['ds', 'y']]
    
    async def _analyze_data_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time series data characteristics."""
        
        analysis = {
            "data_points": len(df),
            "date_range": {
                "start": df['ds'].min().isoformat(),
                "end": df['ds'].max().isoformat(),
                "duration_days": (df['ds'].max() - df['ds'].min()).days
            },
            "value_stats": {
                "mean": float(df['y'].mean()),
                "std": float(df['y'].std()),
                "min": float(df['y'].min()),
                "max": float(df['y'].max()),
                "median": float(df['y'].median())
            }
        }
        
        # Detect frequency
        time_diffs = df['ds'].diff().dropna()
        most_common_diff = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else None
        
        if most_common_diff:
            if most_common_diff.days == 1:
                analysis["frequency"] = "daily"
            elif most_common_diff.days == 7:
                analysis["frequency"] = "weekly"
            elif most_common_diff.days >= 28 and most_common_diff.days <= 31:
                analysis["frequency"] = "monthly"
            else:
                analysis["frequency"] = f"{most_common_diff.days}_days"
        else:
            analysis["frequency"] = "irregular"
        
        # Check for stationarity (if statsmodels available)
        if HAS_STATSMODELS:
            try:
                adf_result = adfuller(df['y'].dropna())
                analysis["stationarity"] = {
                    "adf_statistic": float(adf_result[0]),
                    "p_value": float(adf_result[1]),
                    "is_stationary": adf_result[1] < 0.05
                }
            except Exception as e:
                logger.warning(f"Could not perform stationarity test: {e}")
                analysis["stationarity"] = {"error": str(e)}
        
        # Detect trend and seasonality patterns
        if len(df) >= 24:  # Need sufficient data for decomposition
            try:
                # Simple trend detection
                x = np.arange(len(df))
                y = df['y'].values
                trend_coef = np.polyfit(x, y, 1)[0]
                analysis["trend"] = {
                    "coefficient": float(trend_coef),
                    "direction": "increasing" if trend_coef > 0 else "decreasing" if trend_coef < 0 else "flat"
                }
                
                # Basic seasonality detection
                if analysis["frequency"] in ["daily", "weekly"]:
                    # Check for weekly patterns in daily data
                    df_with_weekday = df.copy()
                    df_with_weekday['weekday'] = df_with_weekday['ds'].dt.dayofweek
                    weekday_means = df_with_weekday.groupby('weekday')['y'].mean()
                    weekday_std = weekday_means.std()
                    analysis["seasonality"] = {
                        "weekly_variation": float(weekday_std),
                        "has_weekly_pattern": weekday_std > df['y'].std() * 0.1
                    }
                
            except Exception as e:
                logger.warning(f"Could not analyze trend/seasonality: {e}")
        
        return analysis
    
    async def _auto_select_models(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> List[str]:
        """Automatically select best models based on data characteristics."""
        
        selected_models = []
        data_points = len(df)
        
        # Prophet is good for data with seasonality and trends
        if HAS_PROPHET and data_points >= 10:
            selected_models.append(ForecastModel.PROPHET)
        
        # ARIMA is good for stationary data or data that can be made stationary
        if HAS_STATSMODELS and data_points >= 20:
            selected_models.append(ForecastModel.ARIMA)
        
        # LSTM is good for complex patterns and sufficient data
        if HAS_TENSORFLOW and data_points >= 50:
            selected_models.append(ForecastModel.LSTM)
        
        # If no models selected, use whatever is available
        if not selected_models:
            selected_models = self.supported_models[:1]  # Use first available
        
        return selected_models
    
    async def _train_forecast_model(
        self, 
        df: pd.DataFrame, 
        model_type: str, 
        forecast_periods: int, 
        confidence_level: float
    ) -> Dict[str, Any]:
        """Train a specific forecasting model."""
        
        if model_type == ForecastModel.PROPHET:
            return await self._train_prophet_model(df, forecast_periods, confidence_level)
        elif model_type == ForecastModel.ARIMA:
            return await self._train_arima_model(df, forecast_periods, confidence_level)
        elif model_type == ForecastModel.LSTM:
            return await self._train_lstm_model(df, forecast_periods, confidence_level)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    async def _train_prophet_model(
        self, 
        df: pd.DataFrame, 
        forecast_periods: int, 
        confidence_level: float
    ) -> Dict[str, Any]:
        """Train Prophet forecasting model."""
        
        if not HAS_PROPHET:
            raise RuntimeError("Prophet not available")
        
        # Split data for validation
        train_size = int(len(df) * 0.8)
        train_df = df[:train_size].copy()
        val_df = df[train_size:].copy()
        
        # Create and fit Prophet model
        model = Prophet(
            interval_width=confidence_level,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True if len(df) > 365 else False
        )
        
        model.fit(train_df)
        
        # Validate on held-out data
        val_periods = len(val_df)
        if val_periods > 0:
            val_future = model.make_future_dataframe(periods=val_periods, include_history=False)
            val_forecast = model.predict(val_future)
            
            # Calculate validation metrics
            val_mae = mean_absolute_error(val_df['y'], val_forecast['yhat'])
            val_mse = mean_squared_error(val_df['y'], val_forecast['yhat'])
            val_rmse = np.sqrt(val_mse)
        else:
            val_mae = val_mse = val_rmse = None
        
        # Create future forecast
        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)
        
        # Extract forecast results
        forecast_data = forecast.tail(forecast_periods)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
        
        return {
            "model": model,
            "forecast": forecast_data,
            "metrics": {
                "validation_mae": val_mae,
                "validation_mse": val_mse,
                "validation_rmse": val_rmse
            },
            "model_components": {
                "trend": forecast['trend'].tolist(),
                "seasonal": forecast.get('weekly', []),
                "yearly": forecast.get('yearly', []) if 'yearly' in forecast.columns else []
            }
        }
    
    async def _train_arima_model(
        self, 
        df: pd.DataFrame, 
        forecast_periods: int, 
        confidence_level: float
    ) -> Dict[str, Any]:
        """Train ARIMA forecasting model."""
        
        if not HAS_STATSMODELS:
            raise RuntimeError("Statsmodels not available")
        
        # Split data for validation
        train_size = int(len(df) * 0.8)
        train_data = df['y'][:train_size]
        val_data = df['y'][train_size:]
        
        # Auto-select ARIMA parameters (simple approach)
        best_aic = np.inf
        best_order = (1, 1, 1)
        
        # Try different parameter combinations
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        # Fit best model
        model = ARIMA(train_data, order=best_order)
        fitted_model = model.fit()
        
        # Validate on held-out data
        if len(val_data) > 0:
            val_forecast = fitted_model.forecast(steps=len(val_data))
            val_mae = mean_absolute_error(val_data, val_forecast)
            val_mse = mean_squared_error(val_data, val_forecast)
            val_rmse = np.sqrt(val_mse)
        else:
            val_mae = val_mse = val_rmse = None
        
        # Create forecast
        forecast_result = fitted_model.forecast(steps=forecast_periods, alpha=1-confidence_level)
        forecast_values = forecast_result[0] if isinstance(forecast_result, tuple) else forecast_result
        
        # Generate confidence intervals (simplified)
        forecast_std = np.std(train_data) * np.sqrt(np.arange(1, forecast_periods + 1))
        z_score = 1.96  # 95% confidence
        
        # Create forecast dates
        last_date = df['ds'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        
        forecast_data = []
        for i, date in enumerate(forecast_dates):
            forecast_data.append({
                'ds': date.isoformat(),
                'yhat': float(forecast_values[i]),
                'yhat_lower': float(forecast_values[i] - z_score * forecast_std[i]),
                'yhat_upper': float(forecast_values[i] + z_score * forecast_std[i])
            })
        
        return {
            "model": fitted_model,
            "forecast": forecast_data,
            "metrics": {
                "validation_mae": val_mae,
                "validation_mse": val_mse,
                "validation_rmse": val_rmse,
                "aic": float(fitted_model.aic),
                "bic": float(fitted_model.bic)
            },
            "model_params": {
                "order": best_order,
                "aic": float(fitted_model.aic)
            }
        }
    
    async def _train_lstm_model(
        self, 
        df: pd.DataFrame, 
        forecast_periods: int, 
        confidence_level: float
    ) -> Dict[str, Any]:
        """Train LSTM forecasting model."""
        
        if not HAS_TENSORFLOW:
            raise RuntimeError("TensorFlow not available")
        
        # Prepare data for LSTM
        values = df['y'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values)
        
        # Create sequences
        sequence_length = min(10, len(df) // 4)  # Use 10 or 1/4 of data length
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_values)):
            X.append(scaled_values[i-sequence_length:i, 0])
            y.append(scaled_values[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split for validation
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build LSTM model
        model = keras.Sequential([
            layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
            layers.Dropout(0.2),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(25),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
        
        # Validate
        if len(X_val) > 0:
            val_pred = model.predict(X_val)
            val_pred_scaled = scaler.inverse_transform(val_pred)
            val_actual_scaled = scaler.inverse_transform(y_val.reshape(-1, 1))
            
            val_mae = mean_absolute_error(val_actual_scaled, val_pred_scaled)
            val_mse = mean_squared_error(val_actual_scaled, val_pred_scaled)
            val_rmse = np.sqrt(val_mse)
        else:
            val_mae = val_mse = val_rmse = None
        
        # Generate forecast
        last_sequence = scaled_values[-sequence_length:].reshape(1, sequence_length, 1)
        forecast_scaled = []
        
        current_sequence = last_sequence.copy()
        for _ in range(forecast_periods):
            next_pred = model.predict(current_sequence, verbose=0)
            forecast_scaled.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Scale back to original values
        forecast_values = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        
        # Generate confidence intervals (simplified using prediction variance)
        forecast_std = np.std(df['y']) * 0.1  # Simple approximation
        z_score = 1.96
        
        # Create forecast dates
        last_date = df['ds'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        
        forecast_data = []
        for i, date in enumerate(forecast_dates):
            forecast_data.append({
                'ds': date.isoformat(),
                'yhat': float(forecast_values[i]),
                'yhat_lower': float(forecast_values[i] - z_score * forecast_std),
                'yhat_upper': float(forecast_values[i] + z_score * forecast_std)
            })
        
        # Create wrapper for consistency
        class LSTMWrapper:
            def __init__(self, model, scaler, sequence_length):
                self.model = model
                self.scaler = scaler
                self.sequence_length = sequence_length
        
        wrapped_model = LSTMWrapper(model, scaler, sequence_length)
        
        return {
            "model": wrapped_model,
            "forecast": forecast_data,
            "metrics": {
                "validation_mae": val_mae,
                "validation_mse": val_mse,
                "validation_rmse": val_rmse
            },
            "training_history": {
                "loss": history.history['loss'],
                "val_loss": history.history.get('val_loss', [])
            }
        }
    
    async def _create_ensemble_forecast(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble forecast from multiple models."""
        
        successful_results = {k: v for k, v in results.items() if "forecast" in v}
        
        if len(successful_results) < 2:
            return None
        
        # Get all forecasts
        forecasts = []
        for model_name, result in successful_results.items():
            forecast_df = pd.DataFrame(result["forecast"])
            forecast_df['model'] = model_name
            forecasts.append(forecast_df)
        
        # Combine forecasts
        combined_df = pd.concat(forecasts, ignore_index=True)
        
        # Calculate ensemble predictions (simple average)
        ensemble_forecast = []
        for ds in combined_df['ds'].unique():
            day_forecasts = combined_df[combined_df['ds'] == ds]
            
            ensemble_forecast.append({
                'ds': ds,
                'yhat': float(day_forecasts['yhat'].mean()),
                'yhat_lower': float(day_forecasts['yhat_lower'].mean()),
                'yhat_upper': float(day_forecasts['yhat_upper'].mean()),
                'models_used': day_forecasts['model'].tolist()
            })
        
        return {
            "forecast": ensemble_forecast,
            "models_combined": list(successful_results.keys()),
            "combination_method": "simple_average"
        }
    
    async def _create_forecast_visualizations(
        self, 
        df: pd.DataFrame, 
        results: Dict[str, Any], 
        ensemble_forecast: Optional[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Create forecast visualizations."""
        
        visualizations = {}
        
        try:
            # Create main forecast plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot historical data
            ax.plot(df['ds'], df['y'], label='Historical Data', color='black', linewidth=2)
            
            # Plot forecasts from each model
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            color_idx = 0
            
            for model_name, result in results.items():
                if "forecast" in result:
                    forecast_df = pd.DataFrame(result["forecast"])
                    forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
                    
                    color = colors[color_idx % len(colors)]
                    
                    # Plot forecast line
                    ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                           label=f'{model_name.upper()} Forecast', 
                           color=color, linestyle='--', linewidth=2)
                    
                    # Plot confidence intervals
                    ax.fill_between(forecast_df['ds'], 
                                   forecast_df['yhat_lower'], 
                                   forecast_df['yhat_upper'],
                                   alpha=0.2, color=color)
                    
                    color_idx += 1
            
            # Plot ensemble forecast if available
            if ensemble_forecast:
                ensemble_df = pd.DataFrame(ensemble_forecast["forecast"])
                ensemble_df['ds'] = pd.to_datetime(ensemble_df['ds'])
                
                ax.plot(ensemble_df['ds'], ensemble_df['yhat'], 
                       label='Ensemble Forecast', 
                       color='gold', linewidth=3)
                
                ax.fill_between(ensemble_df['ds'], 
                               ensemble_df['yhat_lower'], 
                               ensemble_df['yhat_upper'],
                               alpha=0.3, color='gold')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.set_title('Time Series Forecast Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            visualizations["forecast_comparison"] = f"data:image/png;base64,{plot_data}"
            
        except Exception as e:
            logger.error(f"Error creating forecast visualization: {e}")
            visualizations["error"] = str(e)
        
        return visualizations
    
    async def _analyze_time_series(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze time series data characteristics and patterns."""
        
        data = input_data.get("data")
        date_column = input_data.get("date_column", "date")
        value_column = input_data.get("value_column", "value")
        
        if not data:
            raise ValueError("Time series data is required")
        
        # Prepare data
        df = await self._prepare_time_series_data(data, date_column, value_column)
        
        # Analyze characteristics
        analysis = await self._analyze_data_characteristics(df)
        
        # Perform seasonal decomposition if possible
        decomposition = None
        if HAS_STATSMODELS and len(df) >= 24:
            try:
                # Determine period for decomposition
                if analysis.get("frequency") == "daily":
                    period = 7  # Weekly seasonality
                elif analysis.get("frequency") == "weekly":
                    period = 52  # Yearly seasonality
                elif analysis.get("frequency") == "monthly":
                    period = 12  # Yearly seasonality
                else:
                    period = min(12, len(df) // 2)  # Default period
                
                decomp_result = seasonal_decompose(df['y'], model='additive', period=period)
                
                decomposition = {
                    "trend": decomp_result.trend.dropna().tolist(),
                    "seasonal": decomp_result.seasonal.tolist(),
                    "residual": decomp_result.resid.dropna().tolist(),
                    "period": period
                }
                
            except Exception as e:
                logger.warning(f"Could not perform seasonal decomposition: {e}")
        
        return {
            "data_analysis": analysis,
            "decomposition": decomposition,
            "recommendations": await self._get_model_recommendations(analysis)
        }
    
    async def _decompose_time_series(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed time series decomposition."""
        
        data = input_data.get("data")
        date_column = input_data.get("date_column", "date")
        value_column = input_data.get("value_column", "value")
        decomposition_type = input_data.get("type", "additive")  # additive or multiplicative
        
        if not data:
            raise ValueError("Time series data is required")
        
        if not HAS_STATSMODELS:
            raise RuntimeError("Statsmodels required for decomposition")
        
        # Prepare data
        df = await self._prepare_time_series_data(data, date_column, value_column)
        
        if len(df) < 24:
            raise ValueError("Need at least 24 data points for decomposition")
        
        # Determine period
        analysis = await self._analyze_data_characteristics(df)
        if analysis.get("frequency") == "daily":
            period = 7
        elif analysis.get("frequency") == "weekly":
            period = 52
        elif analysis.get("frequency") == "monthly":
            period = 12
        else:
            period = min(12, len(df) // 2)
        
        # Perform decomposition
        decomp_result = seasonal_decompose(df['y'], model=decomposition_type, period=period)
        
        # Create visualization
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original data
        axes[0].plot(df['ds'], df['y'])
        axes[0].set_title('Original Time Series')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        axes[1].plot(df['ds'], decomp_result.trend)
        axes[1].set_title('Trend Component')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        axes[2].plot(df['ds'], decomp_result.seasonal)
        axes[2].set_title('Seasonal Component')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        axes[3].plot(df['ds'], decomp_result.resid)
        axes[3].set_title('Residual Component')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "decomposition": {
                "trend": decomp_result.trend.dropna().tolist(),
                "seasonal": decomp_result.seasonal.tolist(),
                "residual": decomp_result.resid.dropna().tolist(),
                "period": period,
                "type": decomposition_type
            },
            "visualization": f"data:image/png;base64,{plot_data}",
            "analysis": analysis
        }
    
    async def _compare_models(self, input_data: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance of multiple forecasting models."""
        
        model_names = input_data.get("model_names", list(self.trained_models.keys()))
        
        comparison = {}
        for model_name in model_names:
            if model_name in self.trained_models:
                model_info = self.trained_models[model_name]
                comparison[model_name] = {
                    "model_type": model_info["model_type"],
                    "metrics": model_info["metrics"],
                    "data_analysis": model_info["data_analysis"],
                    "trained_at": model_info["trained_at"].isoformat()
                }
        
        return {
            "comparison": comparison,
            "total_models": len(comparison)
        }
    
    async def _get_model_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Get model recommendations based on data analysis."""
        
        recommendations = []
        data_points = analysis["data_points"]
        
        if data_points < 20:
            recommendations.append("Consider collecting more data for better forecasting accuracy")
        
        if analysis.get("seasonality", {}).get("has_weekly_pattern"):
            recommendations.append("Prophet model recommended due to detected seasonality")
        
        if analysis.get("stationarity", {}).get("is_stationary"):
            recommendations.append("ARIMA model suitable due to stationary data")
        else:
            recommendations.append("Consider data differencing for ARIMA model")
        
        if data_points >= 50:
            recommendations.append("LSTM model can capture complex patterns with sufficient data")
        
        return recommendations
    
    async def _save_forecast_model(
        self, 
        model_info: Dict[str, Any], 
        forecast_name: str, 
        df: pd.DataFrame, 
        analysis: Dict[str, Any]
    ) -> str:
        """Save trained forecast model to disk."""
        
        model_path = self.models_dir / f"{forecast_name}_{model_info['type']}.pkl"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(model_info["model"], f)
        
        # Save metadata
        metadata = {
            "forecast_name": forecast_name,
            "model_type": model_info["type"],
            "metrics": model_info["metrics"],
            "data_analysis": analysis,
            "trained_at": datetime.utcnow().isoformat(),
            "data_points": len(df)
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in memory
        self.trained_models[forecast_name] = {
            "model_type": model_info["type"],
            "model_path": str(model_path),
            "metrics": model_info["metrics"],
            "data_analysis": analysis,
            "trained_at": datetime.utcnow()
        }
        
        return str(model_path)
    
    async def _load_existing_models(self) -> None:
        """Load existing trained models from disk."""
        
        for model_file in self.models_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem.split('_')[0]
                metadata_file = model_file.with_suffix('.json')
                
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        metadata["model_path"] = str(model_file)
                        metadata["trained_at"] = datetime.fromisoformat(metadata["trained_at"])
                        self.trained_models[model_name] = metadata
                        
            except Exception as e:
                logger.warning(f"Could not load forecast model {model_file}: {e}")