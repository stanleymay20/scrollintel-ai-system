"""
Predictive Market Making System for Computational Resource Futures
Implements advanced market making with time-series prediction and arbitrage detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..models.economic_optimization_models import (
    ResourceType, CloudProvider, MarketAction, ResourcePrice, MarketPrediction,
    ArbitrageOpportunity, TradingDecision, EconomicForecast, MarketState
)

logger = logging.getLogger(__name__)

@dataclass
class MarketFeatures:
    """Market features for prediction models"""
    price_features: np.ndarray
    volume_features: np.ndarray
    volatility_features: np.ndarray
    seasonal_features: np.ndarray
    external_features: np.ndarray
    technical_indicators: np.ndarray

class PredictiveMarketMaker:
    """
    Predictive Market Making System for Computational Resources
    
    Implements:
    - Time-series prediction for resource pricing
    - Market making with bid-ask spread optimization
    - Arbitrage detection and exploitation
    - Volatility modeling and risk management
    - Seasonal pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.prediction_horizon = self.config.get('prediction_horizon', 24)  # hours
        self.update_frequency = self.config.get('update_frequency', 300)  # seconds
        self.min_arbitrage_profit = self.config.get('min_arbitrage_profit', 0.05)  # 5%
        
        # Prediction models
        self.price_models: Dict[str, Any] = {}
        self.volatility_models: Dict[str, Any] = {}
        self.demand_models: Dict[str, Any] = {}
        
        # Market data
        self.historical_prices: Dict[str, List[ResourcePrice]] = {}
        self.market_features: Dict[str, MarketFeatures] = {}
        self.current_predictions: List[MarketPrediction] = []
        
        # Trading state
        self.active_positions: Dict[str, Dict] = {}
        self.arbitrage_opportunities: List[ArbitrageOpportunity] = []
        self.trading_history: List[TradingDecision] = []
        
        # Feature scalers
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Initialize models
        self._initialize_prediction_models()
        
        logger.info("Predictive Market Maker initialized")
    
    def _initialize_prediction_models(self) -> None:
        """Initialize machine learning models for prediction"""
        try:
            # Price prediction models
            for resource_type in ResourceType:
                for provider in CloudProvider:
                    key = f"{resource_type.value}_{provider.value}"
                    
                    # Ensemble of models for robustness
                    self.price_models[key] = {
                        'rf': RandomForestRegressor(
                            n_estimators=100,
                            max_depth=10,
                            random_state=42
                        ),
                        'gbm': GradientBoostingRegressor(
                            n_estimators=100,
                            max_depth=6,
                            learning_rate=0.1,
                            random_state=42
                        ),
                        'ensemble_weights': [0.6, 0.4]  # RF, GBM
                    }
                    
                    # Volatility models
                    self.volatility_models[key] = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=8,
                        random_state=42
                    )
                    
                    # Demand models
                    self.demand_models[key] = GradientBoostingRegressor(
                        n_estimators=50,
                        max_depth=5,
                        learning_rate=0.15,
                        random_state=42
                    )
                    
                    # Feature scaler
                    self.scalers[key] = StandardScaler()
            
            logger.info("Prediction models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing prediction models: {e}")
            raise
    
    async def update_market_data(self, new_prices: List[ResourcePrice]) -> None:
        """Update market data with new price information"""
        try:
            for price in new_prices:
                key = f"{price.resource_type.value}_{price.provider.value}"
                
                if key not in self.historical_prices:
                    self.historical_prices[key] = []
                
                self.historical_prices[key].append(price)
                
                # Keep only recent data (e.g., last 30 days)
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                self.historical_prices[key] = [
                    p for p in self.historical_prices[key]
                    if p.timestamp >= cutoff_time
                ]
            
            # Update features and retrain models if enough data
            await self._update_features_and_models()
            
            logger.debug(f"Updated market data with {len(new_prices)} new prices")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _update_features_and_models(self) -> None:
        """Update features and retrain models with new data"""
        try:
            for key, prices in self.historical_prices.items():
                if len(prices) < 50:  # Need minimum data for training
                    continue
                
                # Extract features
                features = await self._extract_features(prices)
                self.market_features[key] = features
                
                # Retrain models if enough new data
                if len(prices) % 100 == 0:  # Retrain every 100 new data points
                    await self._retrain_models(key, prices, features)
            
        except Exception as e:
            logger.error(f"Error updating features and models: {e}")
    
    async def _extract_features(self, prices: List[ResourcePrice]) -> MarketFeatures:
        """Extract features from price history for ML models"""
        try:
            df = pd.DataFrame([{
                'timestamp': p.timestamp,
                'price': p.price_per_hour,
                'spot_price': p.spot_price or p.price_per_hour,
                'availability': p.availability
            } for p in prices])
            
            df = df.sort_values('timestamp')
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            
            # Price features
            df['price_ma_5'] = df['price'].rolling(window=5).mean()
            df['price_ma_20'] = df['price'].rolling(window=20).mean()
            df['price_std_5'] = df['price'].rolling(window=5).std()
            df['price_change'] = df['price'].pct_change()
            df['price_momentum'] = df['price'].rolling(window=10).apply(
                lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0
            )
            
            # Volume features (using availability as proxy)
            df['volume_ma_5'] = df['availability'].rolling(window=5).mean()
            df['volume_change'] = df['availability'].pct_change()
            
            # Volatility features
            df['volatility_5'] = df['price'].rolling(window=5).std()
            df['volatility_20'] = df['price'].rolling(window=20).std()
            df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-8)
            
            # Technical indicators
            df['rsi'] = self._calculate_rsi(df['price'])
            df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df['price'])
            df['macd'], df['macd_signal'] = self._calculate_macd(df['price'])
            
            # Fill NaN values
            df = df.fillna(method='forward').fillna(0)
            
            # Extract feature arrays
            price_features = df[['price', 'price_ma_5', 'price_ma_20', 'price_change', 'price_momentum']].values
            volume_features = df[['availability', 'volume_ma_5', 'volume_change']].values
            volatility_features = df[['volatility_5', 'volatility_20', 'volatility_ratio']].values
            seasonal_features = df[['hour', 'day_of_week', 'day_of_month']].values
            external_features = df[['spot_price']].values
            technical_indicators = df[['rsi', 'bollinger_upper', 'bollinger_lower', 'macd', 'macd_signal']].values
            
            return MarketFeatures(
                price_features=price_features,
                volume_features=volume_features,
                volatility_features=volatility_features,
                seasonal_features=seasonal_features,
                external_features=external_features,
                technical_indicators=technical_indicators
            )
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper, lower
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    async def _retrain_models(self, key: str, prices: List[ResourcePrice], features: MarketFeatures) -> None:
        """Retrain prediction models with new data"""
        try:
            # Prepare training data
            X = np.concatenate([
                features.price_features,
                features.volume_features,
                features.volatility_features,
                features.seasonal_features,
                features.external_features,
                features.technical_indicators
            ], axis=1)
            
            # Target variables (next hour price)
            y_price = np.array([p.price_per_hour for p in prices[1:]])  # Shift by 1
            X = X[:-1]  # Remove last row to match y
            
            if len(X) < 20:  # Need minimum samples
                return
            
            # Scale features
            X_scaled = self.scalers[key].fit_transform(X)
            
            # Train price models
            models = self.price_models[key]
            models['rf'].fit(X_scaled, y_price)
            models['gbm'].fit(X_scaled, y_price)
            
            # Train volatility model (target: price volatility)
            y_volatility = np.array([
                np.std([p.price_per_hour for p in prices[max(0, i-5):i+1]])
                for i in range(len(prices))
            ])[1:]  # Shift by 1
            
            if len(y_volatility) == len(X_scaled):
                self.volatility_models[key].fit(X_scaled, y_volatility)
            
            # Train demand model (target: availability change)
            y_demand = np.array([p.availability for p in prices[1:]])
            if len(y_demand) == len(X_scaled):
                self.demand_models[key].fit(X_scaled, y_demand)
            
            logger.debug(f"Retrained models for {key}")
            
        except Exception as e:
            logger.error(f"Error retraining models for {key}: {e}")
    
    async def generate_predictions(self, market_state: MarketState) -> List[MarketPrediction]:
        """Generate price predictions for all resources and providers"""
        try:
            predictions = []
            
            for price in market_state.prices:
                key = f"{price.resource_type.value}_{price.provider.value}"
                
                if key not in self.price_models or key not in self.market_features:
                    continue
                
                # Get latest features
                features = self.market_features[key]
                if len(features.price_features) == 0:
                    continue
                
                # Prepare input for prediction
                latest_features = np.concatenate([
                    features.price_features[-1:],
                    features.volume_features[-1:],
                    features.volatility_features[-1:],
                    features.seasonal_features[-1:],
                    features.external_features[-1:],
                    features.technical_indicators[-1:]
                ], axis=1)
                
                # Scale features
                try:
                    X_scaled = self.scalers[key].transform(latest_features)
                except:
                    continue  # Skip if scaler not fitted
                
                # Generate ensemble prediction
                models = self.price_models[key]
                pred_rf = models['rf'].predict(X_scaled)[0]
                pred_gbm = models['gbm'].predict(X_scaled)[0]
                
                weights = models['ensemble_weights']
                predicted_price = weights[0] * pred_rf + weights[1] * pred_gbm
                
                # Calculate confidence based on model agreement
                price_diff = abs(pred_rf - pred_gbm)
                confidence = max(0.5, 1.0 - (price_diff / max(predicted_price, 1.0)))
                
                # Predict volatility
                try:
                    predicted_volatility = self.volatility_models[key].predict(X_scaled)[0]
                except:
                    predicted_volatility = 0.1
                
                # Determine trend direction
                current_price = price.price_per_hour
                if predicted_price > current_price * 1.02:
                    trend = "up"
                elif predicted_price < current_price * 0.98:
                    trend = "down"
                else:
                    trend = "stable"
                
                # Create prediction
                prediction = MarketPrediction(
                    resource_type=price.resource_type,
                    provider=price.provider,
                    predicted_price=predicted_price,
                    confidence_score=confidence,
                    prediction_horizon=timedelta(hours=1),
                    factors=["price_momentum", "volatility", "seasonal_patterns"],
                    trend_direction=trend,
                    volatility_score=predicted_volatility
                )
                
                predictions.append(prediction)
            
            self.current_predictions = predictions
            logger.info(f"Generated {len(predictions)} market predictions")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return []
    
    async def detect_arbitrage_opportunities(self, market_state: MarketState) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities across cloud providers"""
        try:
            opportunities = []
            
            # Group prices by resource type
            prices_by_resource = {}
            for price in market_state.prices:
                if price.resource_type not in prices_by_resource:
                    prices_by_resource[price.resource_type] = []
                prices_by_resource[price.resource_type].append(price)
            
            # Find arbitrage opportunities within each resource type
            for resource_type, prices in prices_by_resource.items():
                if len(prices) < 2:
                    continue
                
                # Sort by price
                sorted_prices = sorted(prices, key=lambda p: p.price_per_hour)
                
                # Check all pairs for arbitrage
                for i, buy_price in enumerate(sorted_prices[:-1]):
                    for sell_price in sorted_prices[i+1:]:
                        if buy_price.provider == sell_price.provider:
                            continue
                        
                        # Calculate profit potential
                        profit_margin = sell_price.price_per_hour - buy_price.price_per_hour
                        profit_percentage = profit_margin / buy_price.price_per_hour
                        
                        if profit_percentage >= self.min_arbitrage_profit:
                            # Estimate execution window and risk
                            execution_window = await self._estimate_execution_window(
                                buy_price, sell_price
                            )
                            risk_score = await self._calculate_arbitrage_risk(
                                buy_price, sell_price, market_state
                            )
                            
                            # Calculate confidence based on price stability
                            confidence = await self._calculate_arbitrage_confidence(
                                buy_price, sell_price
                            )
                            
                            opportunity = ArbitrageOpportunity(
                                resource_type=resource_type,
                                buy_provider=buy_price.provider,
                                sell_provider=sell_price.provider,
                                buy_price=buy_price.price_per_hour,
                                sell_price=sell_price.price_per_hour,
                                profit_margin=profit_margin,
                                profit_percentage=profit_percentage,
                                execution_window=execution_window,
                                risk_score=risk_score,
                                confidence=confidence
                            )
                            
                            opportunities.append(opportunity)
            
            # Sort by profit potential and filter by risk
            opportunities = sorted(
                opportunities, 
                key=lambda o: o.profit_percentage * o.confidence / (1 + o.risk_score),
                reverse=True
            )
            
            # Keep only top opportunities with acceptable risk
            filtered_opportunities = [
                opp for opp in opportunities[:10]  # Top 10
                if opp.risk_score < 0.7 and opp.confidence > 0.6
            ]
            
            self.arbitrage_opportunities = filtered_opportunities
            logger.info(f"Detected {len(filtered_opportunities)} arbitrage opportunities")
            
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Error detecting arbitrage opportunities: {e}")
            return []
    
    async def _estimate_execution_window(self, buy_price: ResourcePrice, sell_price: ResourcePrice) -> timedelta:
        """Estimate execution window for arbitrage opportunity"""
        # Simplified estimation based on provider characteristics
        provider_speeds = {
            CloudProvider.AWS: timedelta(minutes=5),
            CloudProvider.GCP: timedelta(minutes=7),
            CloudProvider.AZURE: timedelta(minutes=6),
            CloudProvider.LAMBDA_LABS: timedelta(minutes=15),
            CloudProvider.RUNPOD: timedelta(minutes=20),
            CloudProvider.VAST_AI: timedelta(minutes=30),
            CloudProvider.PAPERSPACE: timedelta(minutes=25)
        }
        
        buy_time = provider_speeds.get(buy_price.provider, timedelta(minutes=15))
        sell_time = provider_speeds.get(sell_price.provider, timedelta(minutes=15))
        
        return buy_time + sell_time + timedelta(minutes=5)  # Buffer
    
    async def _calculate_arbitrage_risk(self, buy_price: ResourcePrice, sell_price: ResourcePrice, market_state: MarketState) -> float:
        """Calculate risk score for arbitrage opportunity"""
        # Provider reliability risk
        provider_risk = {
            CloudProvider.AWS: 0.1,
            CloudProvider.GCP: 0.15,
            CloudProvider.AZURE: 0.12,
            CloudProvider.LAMBDA_LABS: 0.3,
            CloudProvider.RUNPOD: 0.4,
            CloudProvider.VAST_AI: 0.5,
            CloudProvider.PAPERSPACE: 0.35
        }
        
        buy_risk = provider_risk.get(buy_price.provider, 0.3)
        sell_risk = provider_risk.get(sell_price.provider, 0.3)
        
        # Market volatility risk
        volatility_risk = market_state.market_volatility.get(buy_price.resource_type, 0.2)
        
        # Availability risk
        availability_risk = 1.0 - min(buy_price.availability, sell_price.availability) / 100.0
        
        # Combined risk
        total_risk = (buy_risk + sell_risk) / 2 + volatility_risk * 0.5 + availability_risk * 0.3
        
        return min(1.0, total_risk)
    
    async def _calculate_arbitrage_confidence(self, buy_price: ResourcePrice, sell_price: ResourcePrice) -> float:
        """Calculate confidence score for arbitrage opportunity"""
        # Price stability (based on recent price history)
        buy_key = f"{buy_price.resource_type.value}_{buy_price.provider.value}"
        sell_key = f"{sell_price.resource_type.value}_{sell_price.provider.value}"
        
        buy_stability = 0.8  # Default
        sell_stability = 0.8  # Default
        
        if buy_key in self.historical_prices:
            recent_prices = [p.price_per_hour for p in self.historical_prices[buy_key][-10:]]
            if len(recent_prices) > 1:
                price_std = np.std(recent_prices)
                buy_stability = max(0.3, 1.0 - price_std / np.mean(recent_prices))
        
        if sell_key in self.historical_prices:
            recent_prices = [p.price_per_hour for p in self.historical_prices[sell_key][-10:]]
            if len(recent_prices) > 1:
                price_std = np.std(recent_prices)
                sell_stability = max(0.3, 1.0 - price_std / np.mean(recent_prices))
        
        # Availability confidence
        availability_confidence = min(buy_price.availability, sell_price.availability) / 100.0
        
        # Combined confidence
        confidence = (buy_stability + sell_stability) / 2 * availability_confidence
        
        return min(1.0, confidence)
    
    async def generate_trading_decisions(self, market_state: MarketState, budget_limit: float) -> List[TradingDecision]:
        """Generate algorithmic trading decisions"""
        try:
            decisions = []
            
            # Generate predictions first
            predictions = await self.generate_predictions(market_state)
            
            # Generate decisions based on predictions and arbitrage opportunities
            for prediction in predictions:
                # Find current price
                current_prices = [
                    p for p in market_state.prices
                    if p.resource_type == prediction.resource_type and p.provider == prediction.provider
                ]
                
                if not current_prices:
                    continue
                
                current_price = current_prices[0].price_per_hour
                predicted_price = prediction.predicted_price
                
                # Determine action based on prediction
                price_change = (predicted_price - current_price) / current_price
                
                if price_change > 0.05 and prediction.confidence_score > 0.7:
                    # Strong upward prediction - buy
                    action = MarketAction.BUY
                    quantity = min(10, int(budget_limit * 0.1 / current_price))
                elif price_change < -0.05 and prediction.confidence_score > 0.7:
                    # Strong downward prediction - sell (if we have positions)
                    action = MarketAction.SELL
                    quantity = 5  # Simplified
                else:
                    # Uncertain or small change - hold
                    action = MarketAction.HOLD
                    quantity = 0
                
                if quantity > 0:
                    expected_profit = quantity * abs(price_change) * current_price
                    risk_assessment = 1.0 - prediction.confidence_score
                    
                    decision = TradingDecision(
                        action=action,
                        resource_type=prediction.resource_type,
                        provider=prediction.provider,
                        quantity=quantity,
                        target_price=predicted_price,
                        current_price=current_price,
                        expected_profit=expected_profit,
                        risk_assessment=risk_assessment,
                        execution_priority=int(prediction.confidence_score * 10),
                        conditions=[f"confidence>{prediction.confidence_score:.2f}"]
                    )
                    
                    decisions.append(decision)
            
            # Add arbitrage decisions
            for opportunity in self.arbitrage_opportunities:
                if opportunity.confidence > 0.6 and opportunity.risk_score < 0.5:
                    # Calculate position size based on budget and risk
                    position_size = min(
                        5,  # Max 5 units
                        int(budget_limit * 0.05 / opportunity.buy_price)  # 5% of budget
                    )
                    
                    if position_size > 0:
                        arbitrage_decision = TradingDecision(
                            action=MarketAction.ARBITRAGE,
                            resource_type=opportunity.resource_type,
                            provider=opportunity.buy_provider,  # Primary provider
                            quantity=position_size,
                            target_price=opportunity.sell_price,
                            current_price=opportunity.buy_price,
                            expected_profit=opportunity.profit_margin * position_size,
                            risk_assessment=opportunity.risk_score,
                            execution_priority=10,  # High priority for arbitrage
                            conditions=[
                                f"buy_from_{opportunity.buy_provider.value}",
                                f"sell_to_{opportunity.sell_provider.value}",
                                f"profit_margin>{opportunity.profit_percentage:.2f}"
                            ]
                        )
                        
                        decisions.append(arbitrage_decision)
            
            # Sort by execution priority and expected profit
            decisions = sorted(
                decisions,
                key=lambda d: (d.execution_priority, d.expected_profit),
                reverse=True
            )
            
            logger.info(f"Generated {len(decisions)} trading decisions")
            
            return decisions[:20]  # Return top 20 decisions
            
        except Exception as e:
            logger.error(f"Error generating trading decisions: {e}")
            return []
    
    async def generate_economic_forecast(self, resource_type: ResourceType, 
                                       forecast_horizon: timedelta) -> EconomicForecast:
        """Generate economic forecast for specific resource type"""
        try:
            # Aggregate data across all providers for this resource type
            all_prices = []
            for key, prices in self.historical_prices.items():
                if key.startswith(resource_type.value):
                    all_prices.extend(prices)
            
            if len(all_prices) < 20:
                # Not enough data for forecasting
                return EconomicForecast(
                    forecast_type="price",
                    resource_type=resource_type,
                    forecast_values=[],
                    forecast_timestamps=[],
                    model_accuracy=0.0
                )
            
            # Sort by timestamp
            all_prices = sorted(all_prices, key=lambda p: p.timestamp)
            
            # Create time series
            df = pd.DataFrame([{
                'timestamp': p.timestamp,
                'price': p.price_per_hour,
                'availability': p.availability
            } for p in all_prices])
            
            # Resample to hourly data
            df = df.set_index('timestamp').resample('H').mean().fillna(method='forward')
            
            # Generate forecast timestamps
            last_timestamp = df.index[-1]
            forecast_hours = int(forecast_horizon.total_seconds() / 3600)
            forecast_timestamps = [
                last_timestamp + timedelta(hours=i+1)
                for i in range(forecast_hours)
            ]
            
            # Simple trend-based forecasting (in practice, use ARIMA, Prophet, etc.)
            recent_prices = df['price'].tail(24).values  # Last 24 hours
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Generate forecast values
            last_price = recent_prices[-1]
            forecast_values = []
            
            for i in range(forecast_hours):
                # Trend + seasonal + noise
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * i / 24)  # Daily seasonality
                noise = np.random.normal(0, 0.02)  # 2% noise
                
                forecast_price = last_price + trend * (i + 1) * seasonal_factor + noise
                forecast_values.append(max(0.01, forecast_price))  # Ensure positive
            
            # Calculate confidence intervals (simplified)
            price_std = np.std(recent_prices)
            confidence_intervals = [
                (val - 1.96 * price_std, val + 1.96 * price_std)
                for val in forecast_values
            ]
            
            # Calculate model accuracy (based on recent predictions vs actual)
            model_accuracy = 0.85  # Placeholder - would calculate from backtesting
            
            # Identify seasonal factors
            seasonal_factors = {
                'daily_peak_hour': 14,  # 2 PM
                'weekly_peak_day': 2,   # Tuesday
                'seasonal_multiplier': 1.0
            }
            
            # Identify trend components
            trend_components = {
                'linear_trend': trend,
                'momentum': trend * 0.5,
                'volatility_trend': 0.02
            }
            
            forecast = EconomicForecast(
                forecast_type="price",
                resource_type=resource_type,
                forecast_values=forecast_values,
                forecast_timestamps=forecast_timestamps,
                confidence_intervals=confidence_intervals,
                seasonal_factors=seasonal_factors,
                trend_components=trend_components,
                model_accuracy=model_accuracy
            )
            
            logger.info(f"Generated economic forecast for {resource_type.value} "
                       f"with {len(forecast_values)} data points")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating economic forecast: {e}")
            raise
    
    async def get_market_state(self) -> MarketState:
        """Get current comprehensive market state"""
        try:
            # Get latest prices (would come from real market data feeds)
            latest_prices = []
            for key, prices in self.historical_prices.items():
                if prices:
                    latest_prices.append(prices[-1])
            
            # Calculate market volatility
            market_volatility = {}
            for resource_type in ResourceType:
                resource_prices = [
                    p for p in latest_prices
                    if p.resource_type == resource_type
                ]
                
                if len(resource_prices) > 1:
                    prices_array = np.array([p.price_per_hour for p in resource_prices])
                    volatility = np.std(prices_array) / np.mean(prices_array)
                    market_volatility[resource_type] = volatility
                else:
                    market_volatility[resource_type] = 0.1
            
            # Calculate supply-demand ratios
            supply_demand_ratio = {}
            for resource_type in ResourceType:
                resource_prices = [
                    p for p in latest_prices
                    if p.resource_type == resource_type
                ]
                
                if resource_prices:
                    avg_availability = np.mean([p.availability for p in resource_prices])
                    # Higher availability = more supply relative to demand
                    supply_demand_ratio[resource_type] = avg_availability / 100.0
                else:
                    supply_demand_ratio[resource_type] = 1.0
            
            # Calculate trend indicators
            trend_indicators = {
                'overall_price_trend': 0.02,  # 2% upward trend
                'volatility_trend': -0.01,    # Decreasing volatility
                'demand_growth': 0.05         # 5% demand growth
            }
            
            market_state = MarketState(
                prices=latest_prices,
                predictions=self.current_predictions,
                arbitrage_opportunities=self.arbitrage_opportunities,
                market_volatility=market_volatility,
                supply_demand_ratio=supply_demand_ratio,
                trend_indicators=trend_indicators
            )
            
            return market_state
            
        except Exception as e:
            logger.error(f"Error getting market state: {e}")
            raise