"""
Algorithmic Trading Engine for GPU/Compute Resources
Implements sophisticated trading strategies across cloud providers with risk management
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
import json
from enum import Enum

from ..models.economic_optimization_models import (
    ResourceType, CloudProvider, MarketAction, TradingDecision, 
    ArbitrageOpportunity, ResourcePrice, MarketState, PerformanceMetrics
)

logger = logging.getLogger(__name__)

class TradingStrategy(str, Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    MARKET_MAKING = "market_making"
    TREND_FOLLOWING = "trend_following"

@dataclass
class TradingSignal:
    """Trading signal with strength and confidence"""
    strategy: TradingStrategy
    action: MarketAction
    resource_type: ResourceType
    provider: CloudProvider
    signal_strength: float  # -1 to 1
    confidence: float      # 0 to 1
    expected_return: float
    risk_score: float
    time_horizon: timedelta
    metadata: Dict[str, Any]

@dataclass
class Position:
    """Trading position in compute resources"""
    id: str
    resource_type: ResourceType
    provider: CloudProvider
    quantity: int
    entry_price: float
    current_price: float
    entry_time: datetime
    position_type: str  # "long", "short", "neutral"
    unrealized_pnl: float
    realized_pnl: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class RiskMetrics:
    """Risk management metrics"""
    var_95: float          # Value at Risk (95% confidence)
    expected_shortfall: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    alpha: float
    correlation_matrix: Dict[str, Dict[str, float]]

class AlgorithmicTradingEngine:
    """
    Algorithmic Trading Engine for Compute Resources
    
    Implements:
    - Multiple trading strategies (momentum, mean reversion, arbitrage)
    - Risk management and position sizing
    - Portfolio optimization across providers
    - Real-time signal generation and execution
    - Performance tracking and analytics
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_position_size = self.config.get('max_position_size', 100)
        self.risk_limit = self.config.get('risk_limit', 0.02)  # 2% VaR limit
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.05)  # 5% stop loss
        self.take_profit_pct = self.config.get('take_profit_pct', 0.10)  # 10% take profit
        
        # Trading state
        self.positions: Dict[str, Position] = {}
        self.trading_history: List[TradingDecision] = []
        self.signals_history: List[TradingSignal] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Market data
        self.price_history: Dict[str, List[ResourcePrice]] = {}
        self.returns_history: Dict[str, List[float]] = {}
        
        # Strategy parameters
        self.strategy_params = {
            TradingStrategy.MOMENTUM: {
                'lookback_period': 20,
                'momentum_threshold': 0.02,
                'signal_decay': 0.95
            },
            TradingStrategy.MEAN_REVERSION: {
                'lookback_period': 50,
                'std_threshold': 2.0,
                'reversion_speed': 0.1
            },
            TradingStrategy.ARBITRAGE: {
                'min_spread': 0.01,
                'execution_cost': 0.005,
                'max_holding_time': timedelta(hours=1)
            },
            TradingStrategy.PAIRS_TRADING: {
                'correlation_threshold': 0.8,
                'spread_threshold': 2.0,
                'lookback_period': 100
            }
        }
        
        logger.info("Algorithmic Trading Engine initialized")
    
    async def generate_trading_signals(self, market_state: MarketState) -> List[TradingSignal]:
        """Generate trading signals using multiple strategies"""
        try:
            signals = []
            
            # Update price history
            await self._update_price_history(market_state.prices)
            
            # Generate signals from each strategy
            momentum_signals = await self._momentum_strategy(market_state)
            mean_reversion_signals = await self._mean_reversion_strategy(market_state)
            arbitrage_signals = await self._arbitrage_strategy(market_state)
            pairs_signals = await self._pairs_trading_strategy(market_state)
            
            signals.extend(momentum_signals)
            signals.extend(mean_reversion_signals)
            signals.extend(arbitrage_signals)
            signals.extend(pairs_signals)
            
            # Filter and rank signals
            filtered_signals = await self._filter_signals(signals)
            
            # Store signals history
            self.signals_history.extend(filtered_signals)
            
            # Keep only recent signals (last 1000)
            if len(self.signals_history) > 1000:
                self.signals_history = self.signals_history[-1000:]
            
            logger.info(f"Generated {len(filtered_signals)} trading signals")
            
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return []
    
    async def _update_price_history(self, prices: List[ResourcePrice]) -> None:
        """Update price history for analysis"""
        try:
            for price in prices:
                key = f"{price.resource_type.value}_{price.provider.value}"
                
                if key not in self.price_history:
                    self.price_history[key] = []
                    self.returns_history[key] = []
                
                self.price_history[key].append(price)
                
                # Calculate returns
                if len(self.price_history[key]) > 1:
                    prev_price = self.price_history[key][-2].price_per_hour
                    current_price = price.price_per_hour
                    
                    if prev_price > 0:
                        return_pct = (current_price - prev_price) / prev_price
                        self.returns_history[key].append(return_pct)
                
                # Keep only recent history (last 500 data points)
                if len(self.price_history[key]) > 500:
                    self.price_history[key] = self.price_history[key][-500:]
                    self.returns_history[key] = self.returns_history[key][-500:]
            
        except Exception as e:
            logger.error(f"Error updating price history: {e}")
    
    async def _momentum_strategy(self, market_state: MarketState) -> List[TradingSignal]:
        """Generate momentum-based trading signals"""
        try:
            signals = []
            params = self.strategy_params[TradingStrategy.MOMENTUM]
            lookback = params['lookback_period']
            threshold = params['momentum_threshold']
            
            for price in market_state.prices:
                key = f"{price.resource_type.value}_{price.provider.value}"
                
                if key not in self.returns_history or len(self.returns_history[key]) < lookback:
                    continue
                
                # Calculate momentum indicators
                recent_returns = self.returns_history[key][-lookback:]
                
                # Simple momentum: average return over lookback period
                momentum = np.mean(recent_returns)
                
                # Momentum strength: consistency of direction
                positive_returns = sum(1 for r in recent_returns if r > 0)
                momentum_strength = abs(positive_returns / len(recent_returns) - 0.5) * 2
                
                # Volatility-adjusted momentum
                volatility = np.std(recent_returns)
                adjusted_momentum = momentum / max(volatility, 0.01)
                
                # Generate signal
                if abs(adjusted_momentum) > threshold:
                    action = MarketAction.BUY if adjusted_momentum > 0 else MarketAction.SELL
                    
                    # Calculate expected return and risk
                    expected_return = momentum * 10  # Scale for position sizing
                    risk_score = volatility
                    
                    # Confidence based on momentum strength and consistency
                    confidence = min(0.95, momentum_strength * (abs(adjusted_momentum) / threshold))
                    
                    signal = TradingSignal(
                        strategy=TradingStrategy.MOMENTUM,
                        action=action,
                        resource_type=price.resource_type,
                        provider=price.provider,
                        signal_strength=adjusted_momentum,
                        confidence=confidence,
                        expected_return=expected_return,
                        risk_score=risk_score,
                        time_horizon=timedelta(hours=4),
                        metadata={
                            'momentum': momentum,
                            'volatility': volatility,
                            'lookback_period': lookback
                        }
                    )
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
            return []
    
    async def _mean_reversion_strategy(self, market_state: MarketState) -> List[TradingSignal]:
        """Generate mean reversion trading signals"""
        try:
            signals = []
            params = self.strategy_params[TradingStrategy.MEAN_REVERSION]
            lookback = params['lookback_period']
            std_threshold = params['std_threshold']
            
            for price in market_state.prices:
                key = f"{price.resource_type.value}_{price.provider.value}"
                
                if key not in self.price_history or len(self.price_history[key]) < lookback:
                    continue
                
                # Calculate mean reversion indicators
                recent_prices = [p.price_per_hour for p in self.price_history[key][-lookback:]]
                
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)
                current_price = price.price_per_hour
                
                # Z-score (how many standard deviations from mean)
                if std_price > 0:
                    z_score = (current_price - mean_price) / std_price
                else:
                    continue
                
                # Generate signal if price is significantly away from mean
                if abs(z_score) > std_threshold:
                    # Mean reversion: buy when price is low, sell when high
                    action = MarketAction.BUY if z_score < -std_threshold else MarketAction.SELL
                    
                    # Signal strength based on how far from mean
                    signal_strength = -z_score / std_threshold  # Negative because we expect reversion
                    
                    # Expected return based on reversion to mean
                    expected_return = (mean_price - current_price) / current_price
                    if action == MarketAction.SELL:
                        expected_return = -expected_return
                    
                    # Risk based on volatility
                    risk_score = std_price / mean_price
                    
                    # Confidence based on statistical significance
                    confidence = min(0.9, abs(z_score) / (std_threshold * 2))
                    
                    signal = TradingSignal(
                        strategy=TradingStrategy.MEAN_REVERSION,
                        action=action,
                        resource_type=price.resource_type,
                        provider=price.provider,
                        signal_strength=signal_strength,
                        confidence=confidence,
                        expected_return=expected_return,
                        risk_score=risk_score,
                        time_horizon=timedelta(hours=2),
                        metadata={
                            'z_score': z_score,
                            'mean_price': mean_price,
                            'std_price': std_price,
                            'lookback_period': lookback
                        }
                    )
                    
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
            return []
    
    async def _arbitrage_strategy(self, market_state: MarketState) -> List[TradingSignal]:
        """Generate arbitrage trading signals"""
        try:
            signals = []
            params = self.strategy_params[TradingStrategy.ARBITRAGE]
            min_spread = params['min_spread']
            execution_cost = params['execution_cost']
            
            # Group prices by resource type
            prices_by_resource = {}
            for price in market_state.prices:
                if price.resource_type not in prices_by_resource:
                    prices_by_resource[price.resource_type] = []
                prices_by_resource[price.resource_type].append(price)
            
            # Find arbitrage opportunities
            for resource_type, prices in prices_by_resource.items():
                if len(prices) < 2:
                    continue
                
                # Sort by price
                sorted_prices = sorted(prices, key=lambda p: p.price_per_hour)
                
                for i in range(len(sorted_prices) - 1):
                    buy_price = sorted_prices[i]
                    
                    for j in range(i + 1, len(sorted_prices)):
                        sell_price = sorted_prices[j]
                        
                        if buy_price.provider == sell_price.provider:
                            continue
                        
                        # Calculate spread
                        spread = (sell_price.price_per_hour - buy_price.price_per_hour) / buy_price.price_per_hour
                        net_spread = spread - execution_cost
                        
                        if net_spread > min_spread:
                            # Generate buy signal for cheaper provider
                            buy_signal = TradingSignal(
                                strategy=TradingStrategy.ARBITRAGE,
                                action=MarketAction.BUY,
                                resource_type=resource_type,
                                provider=buy_price.provider,
                                signal_strength=net_spread,
                                confidence=0.9,  # High confidence for arbitrage
                                expected_return=net_spread,
                                risk_score=0.1,  # Low risk for arbitrage
                                time_horizon=timedelta(minutes=30),
                                metadata={
                                    'spread': spread,
                                    'net_spread': net_spread,
                                    'sell_provider': sell_price.provider.value,
                                    'sell_price': sell_price.price_per_hour
                                }
                            )
                            
                            signals.append(buy_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in arbitrage strategy: {e}")
            return []
    
    async def _pairs_trading_strategy(self, market_state: MarketState) -> List[TradingSignal]:
        """Generate pairs trading signals"""
        try:
            signals = []
            params = self.strategy_params[TradingStrategy.PAIRS_TRADING]
            correlation_threshold = params['correlation_threshold']
            spread_threshold = params['spread_threshold']
            lookback = params['lookback_period']
            
            # Find correlated pairs
            resource_keys = list(self.returns_history.keys())
            
            for i in range(len(resource_keys)):
                for j in range(i + 1, len(resource_keys)):
                    key1, key2 = resource_keys[i], resource_keys[j]
                    
                    # Need sufficient history
                    if (len(self.returns_history[key1]) < lookback or 
                        len(self.returns_history[key2]) < lookback):
                        continue
                    
                    # Calculate correlation
                    returns1 = self.returns_history[key1][-lookback:]
                    returns2 = self.returns_history[key2][-lookback:]
                    
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    
                    if abs(correlation) > correlation_threshold:
                        # Calculate price spread
                        prices1 = [p.price_per_hour for p in self.price_history[key1][-lookback:]]
                        prices2 = [p.price_per_hour for p in self.price_history[key2][-lookback:]]
                        
                        # Normalize prices to calculate spread
                        norm_prices1 = np.array(prices1) / prices1[0]
                        norm_prices2 = np.array(prices2) / prices2[0]
                        
                        spread = norm_prices1 - norm_prices2
                        spread_mean = np.mean(spread)
                        spread_std = np.std(spread)
                        
                        current_spread = spread[-1]
                        
                        if spread_std > 0:
                            z_score = (current_spread - spread_mean) / spread_std
                            
                            if abs(z_score) > spread_threshold:
                                # Parse resource info from keys
                                resource1_type, provider1 = key1.split('_', 1)
                                resource2_type, provider2 = key2.split('_', 1)
                                
                                resource1_type = ResourceType(resource1_type)
                                resource2_type = ResourceType(resource2_type)
                                provider1 = CloudProvider(provider1)
                                provider2 = CloudProvider(provider2)
                                
                                # Generate signals for pairs trade
                                if z_score > spread_threshold:
                                    # Spread too high: sell expensive, buy cheap
                                    if norm_prices1[-1] > norm_prices2[-1]:
                                        # Sell resource1, buy resource2
                                        sell_signal = TradingSignal(
                                            strategy=TradingStrategy.PAIRS_TRADING,
                                            action=MarketAction.SELL,
                                            resource_type=resource1_type,
                                            provider=provider1,
                                            signal_strength=-z_score / spread_threshold,
                                            confidence=min(0.8, abs(correlation)),
                                            expected_return=abs(z_score) * 0.01,
                                            risk_score=1 - abs(correlation),
                                            time_horizon=timedelta(hours=6),
                                            metadata={
                                                'pair_resource': resource2_type.value,
                                                'pair_provider': provider2.value,
                                                'correlation': correlation,
                                                'z_score': z_score
                                            }
                                        )
                                        signals.append(sell_signal)
                                
                                elif z_score < -spread_threshold:
                                    # Spread too low: buy expensive, sell cheap
                                    if norm_prices1[-1] < norm_prices2[-1]:
                                        # Buy resource1, sell resource2
                                        buy_signal = TradingSignal(
                                            strategy=TradingStrategy.PAIRS_TRADING,
                                            action=MarketAction.BUY,
                                            resource_type=resource1_type,
                                            provider=provider1,
                                            signal_strength=-z_score / spread_threshold,
                                            confidence=min(0.8, abs(correlation)),
                                            expected_return=abs(z_score) * 0.01,
                                            risk_score=1 - abs(correlation),
                                            time_horizon=timedelta(hours=6),
                                            metadata={
                                                'pair_resource': resource2_type.value,
                                                'pair_provider': provider2.value,
                                                'correlation': correlation,
                                                'z_score': z_score
                                            }
                                        )
                                        signals.append(buy_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in pairs trading strategy: {e}")
            return []
    
    async def _filter_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter and rank trading signals"""
        try:
            if not signals:
                return []
            
            # Filter by minimum confidence and signal strength
            filtered = [
                signal for signal in signals
                if signal.confidence > 0.5 and abs(signal.signal_strength) > 0.1
            ]
            
            # Remove conflicting signals (same resource/provider with opposite actions)
            signal_map = {}
            for signal in filtered:
                key = f"{signal.resource_type.value}_{signal.provider.value}"
                
                if key not in signal_map:
                    signal_map[key] = []
                signal_map[key].append(signal)
            
            # Resolve conflicts by keeping highest confidence signal
            resolved_signals = []
            for key, key_signals in signal_map.items():
                if len(key_signals) == 1:
                    resolved_signals.append(key_signals[0])
                else:
                    # Group by action
                    buy_signals = [s for s in key_signals if s.action == MarketAction.BUY]
                    sell_signals = [s for s in key_signals if s.action == MarketAction.SELL]
                    
                    # Keep best signal from each action type
                    if buy_signals:
                        best_buy = max(buy_signals, key=lambda s: s.confidence * abs(s.signal_strength))
                        resolved_signals.append(best_buy)
                    
                    if sell_signals:
                        best_sell = max(sell_signals, key=lambda s: s.confidence * abs(s.signal_strength))
                        resolved_signals.append(best_sell)
            
            # Rank by expected risk-adjusted return
            ranked_signals = sorted(
                resolved_signals,
                key=lambda s: (s.expected_return * s.confidence) / max(s.risk_score, 0.01),
                reverse=True
            )
            
            # Return top signals
            return ranked_signals[:20]
            
        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return signals
    
    async def execute_trading_decisions(self, 
                                      signals: List[TradingSignal],
                                      budget_limit: float,
                                      market_state: MarketState) -> List[TradingDecision]:
        """Execute trading decisions based on signals"""
        try:
            decisions = []
            remaining_budget = budget_limit
            
            # Calculate position sizes using risk management
            for signal in signals:
                if remaining_budget <= 0:
                    break
                
                # Find current price
                current_prices = [
                    p for p in market_state.prices
                    if (p.resource_type == signal.resource_type and 
                        p.provider == signal.provider)
                ]
                
                if not current_prices:
                    continue
                
                current_price = current_prices[0].price_per_hour
                
                # Calculate position size using Kelly criterion (simplified)
                win_prob = signal.confidence
                avg_win = abs(signal.expected_return)
                avg_loss = signal.risk_score
                
                if avg_loss > 0:
                    kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                    kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap at 25%
                else:
                    kelly_fraction = 0.1  # Default 10%
                
                # Calculate position size
                position_value = remaining_budget * kelly_fraction
                quantity = max(1, int(position_value / current_price))
                
                # Apply maximum position size limit
                quantity = min(quantity, self.max_position_size)
                
                actual_cost = quantity * current_price
                
                if actual_cost <= remaining_budget:
                    # Calculate stop loss and take profit
                    if signal.action == MarketAction.BUY:
                        stop_loss = current_price * (1 - self.stop_loss_pct)
                        take_profit = current_price * (1 + self.take_profit_pct)
                    else:
                        stop_loss = current_price * (1 + self.stop_loss_pct)
                        take_profit = current_price * (1 - self.take_profit_pct)
                    
                    decision = TradingDecision(
                        action=signal.action,
                        resource_type=signal.resource_type,
                        provider=signal.provider,
                        quantity=quantity,
                        target_price=current_price,
                        current_price=current_price,
                        expected_profit=quantity * signal.expected_return * current_price,
                        risk_assessment=signal.risk_score,
                        execution_priority=int(signal.confidence * 10),
                        conditions=[
                            f"strategy_{signal.strategy.value}",
                            f"confidence_{signal.confidence:.2f}",
                            f"stop_loss_{stop_loss:.4f}",
                            f"take_profit_{take_profit:.4f}"
                        ]
                    )
                    
                    decisions.append(decision)
                    remaining_budget -= actual_cost
                    
                    # Create position tracking
                    position_id = f"{signal.resource_type.value}_{signal.provider.value}_{datetime.utcnow().timestamp()}"
                    position = Position(
                        id=position_id,
                        resource_type=signal.resource_type,
                        provider=signal.provider,
                        quantity=quantity,
                        entry_price=current_price,
                        current_price=current_price,
                        entry_time=datetime.utcnow(),
                        position_type="long" if signal.action == MarketAction.BUY else "short",
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        stop_loss=stop_loss,
                        take_profit=take_profit
                    )
                    
                    self.positions[position_id] = position
            
            # Store trading history
            self.trading_history.extend(decisions)
            
            logger.info(f"Executed {len(decisions)} trading decisions")
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error executing trading decisions: {e}")
            return []
    
    async def update_positions(self, market_state: MarketState) -> None:
        """Update positions with current market prices"""
        try:
            for position_id, position in self.positions.items():
                # Find current price
                current_prices = [
                    p for p in market_state.prices
                    if (p.resource_type == position.resource_type and 
                        p.provider == position.provider)
                ]
                
                if current_prices:
                    new_price = current_prices[0].price_per_hour
                    position.current_price = new_price
                    
                    # Calculate unrealized P&L
                    if position.position_type == "long":
                        position.unrealized_pnl = (new_price - position.entry_price) * position.quantity
                    else:  # short
                        position.unrealized_pnl = (position.entry_price - new_price) * position.quantity
                    
                    # Check stop loss and take profit
                    should_close = False
                    
                    if position.position_type == "long":
                        if (position.stop_loss and new_price <= position.stop_loss) or \
                           (position.take_profit and new_price >= position.take_profit):
                            should_close = True
                    else:  # short
                        if (position.stop_loss and new_price >= position.stop_loss) or \
                           (position.take_profit and new_price <= position.take_profit):
                            should_close = True
                    
                    if should_close:
                        # Close position
                        position.realized_pnl = position.unrealized_pnl
                        position.unrealized_pnl = 0.0
                        
                        logger.info(f"Closed position {position_id} with P&L: {position.realized_pnl:.2f}")
            
            # Remove closed positions
            closed_positions = [
                pos_id for pos_id, pos in self.positions.items()
                if pos.realized_pnl != 0.0
            ]
            
            for pos_id in closed_positions:
                del self.positions[pos_id]
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            if not self.trading_history:
                return RiskMetrics(
                    var_95=0.0, expected_shortfall=0.0, max_drawdown=0.0,
                    sharpe_ratio=0.0, sortino_ratio=0.0, beta=0.0, alpha=0.0,
                    correlation_matrix={}
                )
            
            # Calculate returns from trading history
            returns = []
            for decision in self.trading_history[-100:]:  # Last 100 trades
                if decision.expected_profit != 0:
                    return_pct = decision.expected_profit / (decision.quantity * decision.current_price)
                    returns.append(return_pct)
            
            if not returns:
                returns = [0.0]
            
            returns = np.array(returns)
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(returns, 5)  # 5th percentile for 95% VaR
            
            # Expected Shortfall (Conditional VaR)
            tail_returns = returns[returns <= var_95]
            expected_shortfall = np.mean(tail_returns) if len(tail_returns) > 0 else var_95
            
            # Maximum Drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Sharpe Ratio (assuming risk-free rate = 0)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / max(std_return, 0.001)
            
            # Sortino Ratio (downside deviation)
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else std_return
            sortino_ratio = mean_return / max(downside_std, 0.001)
            
            # Beta and Alpha (simplified - using market proxy)
            # In practice, would use actual market benchmark
            market_returns = np.random.normal(0.001, 0.02, len(returns))  # Placeholder
            
            if len(returns) > 1 and len(market_returns) > 1:
                covariance = np.cov(returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / max(market_variance, 0.001)
                alpha = mean_return - beta * np.mean(market_returns)
            else:
                beta = 1.0
                alpha = 0.0
            
            # Correlation matrix (simplified)
            correlation_matrix = {}
            resource_returns = {}
            
            # Group returns by resource type
            for decision in self.trading_history[-50:]:
                key = decision.resource_type.value
                if key not in resource_returns:
                    resource_returns[key] = []
                
                if decision.expected_profit != 0:
                    return_pct = decision.expected_profit / (decision.quantity * decision.current_price)
                    resource_returns[key].append(return_pct)
            
            # Calculate correlations
            for resource1 in resource_returns:
                correlation_matrix[resource1] = {}
                for resource2 in resource_returns:
                    if len(resource_returns[resource1]) > 1 and len(resource_returns[resource2]) > 1:
                        corr = np.corrcoef(resource_returns[resource1], resource_returns[resource2])[0, 1]
                        correlation_matrix[resource1][resource2] = corr if not np.isnan(corr) else 0.0
                    else:
                        correlation_matrix[resource1][resource2] = 1.0 if resource1 == resource2 else 0.0
            
            return RiskMetrics(
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                alpha=alpha,
                correlation_matrix=correlation_matrix
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                var_95=0.0, expected_shortfall=0.0, max_drawdown=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, beta=0.0, alpha=0.0,
                correlation_matrix={}
            )
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            # Calculate total portfolio value
            total_value = 0.0
            total_pnl = 0.0
            
            for position in self.positions.values():
                position_value = position.quantity * position.current_price
                total_value += position_value
                total_pnl += position.unrealized_pnl
            
            # Add realized P&L from closed positions
            realized_pnl = sum(
                decision.expected_profit for decision in self.trading_history
                if hasattr(decision, 'realized_pnl')
            )
            
            # Calculate performance metrics
            risk_metrics = await self.calculate_risk_metrics()
            
            # Position breakdown
            positions_by_resource = {}
            positions_by_provider = {}
            
            for position in self.positions.values():
                # By resource type
                resource_key = position.resource_type.value
                if resource_key not in positions_by_resource:
                    positions_by_resource[resource_key] = {
                        'quantity': 0, 'value': 0.0, 'pnl': 0.0
                    }
                
                positions_by_resource[resource_key]['quantity'] += position.quantity
                positions_by_resource[resource_key]['value'] += position.quantity * position.current_price
                positions_by_resource[resource_key]['pnl'] += position.unrealized_pnl
                
                # By provider
                provider_key = position.provider.value
                if provider_key not in positions_by_provider:
                    positions_by_provider[provider_key] = {
                        'quantity': 0, 'value': 0.0, 'pnl': 0.0
                    }
                
                positions_by_provider[provider_key]['quantity'] += position.quantity
                positions_by_provider[provider_key]['value'] += position.quantity * position.current_price
                positions_by_provider[provider_key]['pnl'] += position.unrealized_pnl
            
            summary = {
                'total_positions': len(self.positions),
                'total_portfolio_value': total_value,
                'total_unrealized_pnl': total_pnl,
                'total_realized_pnl': realized_pnl,
                'total_pnl': total_pnl + realized_pnl,
                'positions_by_resource': positions_by_resource,
                'positions_by_provider': positions_by_provider,
                'risk_metrics': {
                    'var_95': risk_metrics.var_95,
                    'expected_shortfall': risk_metrics.expected_shortfall,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'sortino_ratio': risk_metrics.sortino_ratio
                },
                'recent_signals': len([s for s in self.signals_history[-24:] if s.confidence > 0.7]),
                'recent_trades': len(self.trading_history[-24:]),
                'timestamp': datetime.utcnow()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating portfolio summary: {e}")
            return {'error': str(e)}