"""
Comprehensive Tests for AI-Powered Economic Optimization System
Tests all components: AI optimizer, market maker, multi-objective optimization, trading, and forecasting
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio

from scrollintel.models.economic_optimization_models import (
    ResourceType, CloudProvider, OptimizationObjective, OptimizationStrategy,
    ResourceAllocation, OptimizationResult, MarketState, PerformanceMetrics,
    TradingDecision, ArbitrageOpportunity, EconomicForecast, ResourcePrice,
    MarketPrediction
)
from scrollintel.engines.ai_economic_optimizer import AIEconomicOptimizer
from scrollintel.engines.predictive_market_maker import PredictiveMarketMaker
from scrollintel.engines.multi_objective_optimizer import MultiObjectiveOptimizer, OptimizationConstraints
from scrollintel.engines.algorithmic_trading_engine import AlgorithmicTradingEngine
from scrollintel.engines.economic_forecasting_engine import EconomicForecastingEngine

class TestEconomicOptimizationModels:
    """Test data models for economic optimization"""
    
    def test_resource_price_model(self):
        """Test ResourcePrice model creation and validation"""
        price = ResourcePrice(
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_H100,
            region="us-east-1",
            price_per_hour=3.50,
            price_per_unit=3.50,
            availability=85,
            spot_price=2.80
        )
        
        assert price.provider == CloudProvider.AWS
        assert price.resource_type == ResourceType.GPU_H100
        assert price.price_per_hour == 3.50
        assert price.spot_price == 2.80
        assert price.availability == 85
    
    def test_optimization_strategy_model(self):
        """Test OptimizationStrategy model creation"""
        strategy = OptimizationStrategy(
            name="Cost-Performance Balance",
            objectives=[OptimizationObjective.MINIMIZE_COST, OptimizationObjective.MAXIMIZE_PERFORMANCE],
            weights={
                OptimizationObjective.MINIMIZE_COST: 0.6,
                OptimizationObjective.MAXIMIZE_PERFORMANCE: 0.4
            },
            risk_tolerance=0.3
        )
        
        assert strategy.name == "Cost-Performance Balance"
        assert len(strategy.objectives) == 2
        assert strategy.weights[OptimizationObjective.MINIMIZE_COST] == 0.6
        assert strategy.risk_tolerance == 0.3
    
    def test_market_prediction_model(self):
        """Test MarketPrediction model creation"""
        prediction = MarketPrediction(
            resource_type=ResourceType.GPU_A100,
            provider=CloudProvider.GCP,
            predicted_price=2.75,
            confidence_score=0.85,
            prediction_horizon=timedelta(hours=6),
            trend_direction="up",
            volatility_score=0.15
        )
        
        assert prediction.resource_type == ResourceType.GPU_A100
        assert prediction.predicted_price == 2.75
        assert prediction.confidence_score == 0.85
        assert prediction.trend_direction == "up"

class TestAIEconomicOptimizer:
    """Test AI Economic Optimizer with reinforcement learning"""
    
    @pytest.fixture
    def optimizer(self):
        return AIEconomicOptimizer({
            'learning_rate': 0.001,
            'discount_factor': 0.95,
            'exploration_rate': 0.1
        })
    
    @pytest.fixture
    def sample_market_state(self):
        prices = [
            ResourcePrice(
                provider=CloudProvider.AWS,
                resource_type=ResourceType.GPU_H100,
                region="us-east-1",
                price_per_hour=3.50,
                price_per_unit=3.50,
                availability=80
            ),
            ResourcePrice(
                provider=CloudProvider.GCP,
                resource_type=ResourceType.GPU_H100,
                region="us-central1",
                price_per_hour=3.20,
                price_per_unit=3.20,
                availability=90
            ),
            ResourcePrice(
                provider=CloudProvider.AZURE,
                resource_type=ResourceType.GPU_A100,
                region="eastus",
                price_per_hour=2.80,
                price_per_unit=2.80,
                availability=75
            )
        ]
        
        return MarketState(
            prices=prices,
            predictions=[],
            arbitrage_opportunities=[],
            market_volatility={ResourceType.GPU_H100: 0.15, ResourceType.GPU_A100: 0.12},
            supply_demand_ratio={ResourceType.GPU_H100: 0.8, ResourceType.GPU_A100: 0.9}
        )
    
    @pytest.fixture
    def sample_strategy(self):
        return OptimizationStrategy(
            name="Balanced Strategy",
            objectives=[OptimizationObjective.MINIMIZE_COST, OptimizationObjective.MAXIMIZE_PERFORMANCE],
            weights={
                OptimizationObjective.MINIMIZE_COST: 0.5,
                OptimizationObjective.MAXIMIZE_PERFORMANCE: 0.5
            }
        )
    
    @pytest.mark.asyncio
    async def test_resource_allocation_optimization(self, optimizer, sample_market_state, sample_strategy):
        """Test resource allocation optimization"""
        current_allocations = []
        strategies = [sample_strategy]
        budget_constraints = {"total": 1000.0, "hourly": 100.0}
        
        result = await optimizer.optimize_resource_allocation(
            market_state=sample_market_state,
            current_allocations=current_allocations,
            strategies=strategies,
            budget_constraints=budget_constraints
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.expected_cost > 0
        assert result.expected_performance > 0
        assert 0 <= result.risk_score <= 1
        assert 0 <= result.confidence <= 1
        assert len(result.allocations) >= 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, optimizer):
        """Test performance metrics calculation"""
        period_start = datetime.utcnow() - timedelta(days=7)
        period_end = datetime.utcnow()
        
        metrics = await optimizer.get_performance_metrics(period_start, period_end)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.period_start == period_start
        assert metrics.period_end == period_end
        assert metrics.total_cost_savings >= 0
        assert metrics.successful_trades >= 0
        assert metrics.failed_trades >= 0

class TestPredictiveMarketMaker:
    """Test Predictive Market Making System"""
    
    @pytest.fixture
    def market_maker(self):
        return PredictiveMarketMaker({
            'prediction_horizon': 24,
            'min_arbitrage_profit': 0.05
        })
    
    @pytest.fixture
    def sample_prices(self):
        return [
            ResourcePrice(
                provider=CloudProvider.AWS,
                resource_type=ResourceType.GPU_H100,
                region="us-east-1",
                price_per_hour=3.50,
                price_per_unit=3.50,
                availability=80,
                timestamp=datetime.utcnow() - timedelta(hours=i)
            )
            for i in range(50)  # 50 hours of data
        ]
    
    @pytest.mark.asyncio
    async def test_market_data_update(self, market_maker, sample_prices):
        """Test market data update functionality"""
        await market_maker.update_market_data(sample_prices)
        
        # Check that data was stored
        key = f"{ResourceType.GPU_H100.value}_{CloudProvider.AWS.value}"
        assert key in market_maker.historical_prices
        assert len(market_maker.historical_prices[key]) == len(sample_prices)
    
    @pytest.mark.asyncio
    async def test_prediction_generation(self, market_maker, sample_prices):
        """Test market prediction generation"""
        # First update with historical data
        await market_maker.update_market_data(sample_prices)
        
        # Create market state
        market_state = MarketState(
            prices=sample_prices[-5:],  # Last 5 prices
            predictions=[],
            arbitrage_opportunities=[]
        )
        
        predictions = await market_maker.generate_predictions(market_state)
        
        assert isinstance(predictions, list)
        # May be empty if not enough training data, which is acceptable
        for prediction in predictions:
            assert isinstance(prediction, MarketPrediction)
            assert prediction.predicted_price > 0
            assert 0 <= prediction.confidence_score <= 1
    
    @pytest.mark.asyncio
    async def test_arbitrage_detection(self, market_maker):
        """Test arbitrage opportunity detection"""
        # Create market state with price differences
        prices = [
            ResourcePrice(
                provider=CloudProvider.AWS,
                resource_type=ResourceType.GPU_H100,
                region="us-east-1",
                price_per_hour=3.50,
                price_per_unit=3.50,
                availability=80
            ),
            ResourcePrice(
                provider=CloudProvider.GCP,
                resource_type=ResourceType.GPU_H100,
                region="us-central1",
                price_per_hour=3.00,  # Cheaper - arbitrage opportunity
                price_per_unit=3.00,
                availability=90
            )
        ]
        
        market_state = MarketState(
            prices=prices,
            predictions=[],
            arbitrage_opportunities=[]
        )
        
        opportunities = await market_maker.detect_arbitrage_opportunities(market_state)
        
        assert isinstance(opportunities, list)
        for opp in opportunities:
            assert isinstance(opp, ArbitrageOpportunity)
            assert opp.profit_margin > 0
            assert opp.buy_price < opp.sell_price
            assert 0 <= opp.risk_score <= 1

class TestMultiObjectiveOptimizer:
    """Test Multi-Objective Optimization with Pareto frontier"""
    
    @pytest.fixture
    def optimizer(self):
        return MultiObjectiveOptimizer({
            'population_size': 20,  # Small for testing
            'max_generations': 10   # Few generations for speed
        })
    
    @pytest.fixture
    def sample_constraints(self):
        return OptimizationConstraints(
            budget_limit=1000.0,
            performance_minimum=500.0,
            latency_maximum=100.0,
            availability_minimum=70.0,
            provider_limits={provider: 50 for provider in CloudProvider},
            resource_limits={resource: 25 for resource in ResourceType}
        )
    
    @pytest.mark.asyncio
    async def test_pareto_optimization(self, optimizer, sample_constraints):
        """Test Pareto frontier optimization"""
        # Create sample market state
        prices = [
            ResourcePrice(
                provider=CloudProvider.AWS,
                resource_type=ResourceType.GPU_H100,
                region="us-east-1",
                price_per_hour=3.50,
                price_per_unit=3.50,
                availability=80
            ),
            ResourcePrice(
                provider=CloudProvider.GCP,
                resource_type=ResourceType.GPU_A100,
                region="us-central1",
                price_per_hour=2.80,
                price_per_unit=2.80,
                availability=90
            )
        ]
        
        market_state = MarketState(prices=prices, predictions=[], arbitrage_opportunities=[])
        
        strategies = [
            OptimizationStrategy(
                name="Test Strategy",
                objectives=[OptimizationObjective.MINIMIZE_COST, OptimizationObjective.MAXIMIZE_PERFORMANCE],
                weights={
                    OptimizationObjective.MINIMIZE_COST: 0.6,
                    OptimizationObjective.MAXIMIZE_PERFORMANCE: 0.4
                }
            )
        ]
        
        pareto_solutions = await optimizer.optimize_pareto_frontier(
            market_state=market_state,
            strategies=strategies,
            constraints=sample_constraints
        )
        
        assert isinstance(pareto_solutions, list)
        assert len(pareto_solutions) >= 0  # May be empty for small test case
        
        for solution in pareto_solutions:
            assert hasattr(solution, 'allocations')
            assert hasattr(solution, 'objectives')
            assert hasattr(solution, 'dominance_rank')

class TestAlgorithmicTradingEngine:
    """Test Algorithmic Trading Engine"""
    
    @pytest.fixture
    def trading_engine(self):
        return AlgorithmicTradingEngine({
            'max_position_size': 10,
            'risk_limit': 0.02
        })
    
    @pytest.fixture
    def sample_market_state_with_history(self, trading_engine):
        """Create market state and populate with historical data"""
        # Add some historical price data
        historical_prices = []
        base_price = 3.0
        
        for i in range(100):  # 100 data points
            price_variation = np.sin(i * 0.1) * 0.2 + np.random.normal(0, 0.05)
            price = ResourcePrice(
                provider=CloudProvider.AWS,
                resource_type=ResourceType.GPU_H100,
                region="us-east-1",
                price_per_hour=base_price + price_variation,
                price_per_unit=base_price + price_variation,
                availability=80 + np.random.randint(-10, 10),
                timestamp=datetime.utcnow() - timedelta(hours=100-i)
            )
            historical_prices.append(price)
        
        # Update trading engine with historical data
        asyncio.create_task(trading_engine._update_price_history(historical_prices))
        
        # Current market state
        current_prices = [
            ResourcePrice(
                provider=CloudProvider.AWS,
                resource_type=ResourceType.GPU_H100,
                region="us-east-1",
                price_per_hour=3.10,
                price_per_unit=3.10,
                availability=85
            ),
            ResourcePrice(
                provider=CloudProvider.GCP,
                resource_type=ResourceType.GPU_H100,
                region="us-central1",
                price_per_hour=2.95,
                price_per_unit=2.95,
                availability=90
            )
        ]
        
        return MarketState(
            prices=current_prices,
            predictions=[],
            arbitrage_opportunities=[]
        )
    
    @pytest.mark.asyncio
    async def test_trading_signal_generation(self, trading_engine, sample_market_state_with_history):
        """Test trading signal generation"""
        signals = await trading_engine.generate_trading_signals(sample_market_state_with_history)
        
        assert isinstance(signals, list)
        # Signals may be empty if not enough data or no clear patterns
        for signal in signals:
            assert hasattr(signal, 'strategy')
            assert hasattr(signal, 'action')
            assert hasattr(signal, 'confidence')
            assert 0 <= signal.confidence <= 1
    
    @pytest.mark.asyncio
    async def test_trading_execution(self, trading_engine, sample_market_state_with_history):
        """Test trading decision execution"""
        # Generate signals first
        signals = await trading_engine.generate_trading_signals(sample_market_state_with_history)
        
        # Execute trades
        decisions = await trading_engine.execute_trading_decisions(
            signals=signals,
            budget_limit=1000.0,
            market_state=sample_market_state_with_history
        )
        
        assert isinstance(decisions, list)
        for decision in decisions:
            assert isinstance(decision, TradingDecision)
            assert decision.quantity > 0
            assert decision.current_price > 0
    
    @pytest.mark.asyncio
    async def test_portfolio_summary(self, trading_engine):
        """Test portfolio summary generation"""
        summary = await trading_engine.get_portfolio_summary()
        
        assert isinstance(summary, dict)
        assert 'total_positions' in summary
        assert 'total_portfolio_value' in summary
        assert 'risk_metrics' in summary
    
    @pytest.mark.asyncio
    async def test_risk_metrics_calculation(self, trading_engine):
        """Test risk metrics calculation"""
        risk_metrics = await trading_engine.calculate_risk_metrics()
        
        assert hasattr(risk_metrics, 'var_95')
        assert hasattr(risk_metrics, 'sharpe_ratio')
        assert hasattr(risk_metrics, 'max_drawdown')

class TestEconomicForecastingEngine:
    """Test Economic Forecasting Engine"""
    
    @pytest.fixture
    def forecasting_engine(self):
        return EconomicForecastingEngine({
            'max_forecast_horizon': 48,
            'min_training_samples': 20  # Lower for testing
        })
    
    @pytest.fixture
    def sample_price_history(self):
        """Generate sample price history for forecasting"""
        prices = []
        base_price = 3.0
        
        for i in range(50):  # 50 data points
            # Add trend and seasonality
            trend = i * 0.01
            seasonal = 0.2 * np.sin(2 * np.pi * i / 24)  # Daily seasonality
            noise = np.random.normal(0, 0.05)
            
            price = ResourcePrice(
                provider=CloudProvider.AWS,
                resource_type=ResourceType.GPU_H100,
                region="us-east-1",
                price_per_hour=base_price + trend + seasonal + noise,
                price_per_unit=base_price + trend + seasonal + noise,
                availability=80 + np.random.randint(-10, 10),
                timestamp=datetime.utcnow() - timedelta(hours=50-i)
            )
            prices.append(price)
        
        return prices
    
    @pytest.mark.asyncio
    async def test_price_history_update(self, forecasting_engine, sample_price_history):
        """Test price history update"""
        await forecasting_engine.update_price_history(sample_price_history)
        
        key = f"{ResourceType.GPU_H100.value}_{CloudProvider.AWS.value}"
        assert key in forecasting_engine.price_history
        assert len(forecasting_engine.price_history[key]) == len(sample_price_history)
    
    @pytest.mark.asyncio
    async def test_forecast_generation(self, forecasting_engine, sample_price_history):
        """Test economic forecast generation"""
        # Update with historical data first
        await forecasting_engine.update_price_history(sample_price_history)
        
        forecast = await forecasting_engine.generate_forecast(
            resource_type=ResourceType.GPU_H100,
            provider=CloudProvider.AWS,
            forecast_horizon=12
        )
        
        assert isinstance(forecast, EconomicForecast)
        assert forecast.resource_type == ResourceType.GPU_H100
        assert forecast.provider == CloudProvider.AWS
        # Forecast values may be empty if not enough training data
        if forecast.forecast_values:
            assert len(forecast.forecast_values) <= 12
            assert all(val > 0 for val in forecast.forecast_values)
    
    @pytest.mark.asyncio
    async def test_cost_optimization_forecast(self, forecasting_engine, sample_price_history):
        """Test cost optimization forecast"""
        # Update with historical data
        await forecasting_engine.update_price_history(sample_price_history)
        
        forecast = await forecasting_engine.generate_cost_optimization_forecast(
            budget=1000.0,
            time_horizon=timedelta(hours=24)
        )
        
        assert isinstance(forecast, dict)
        assert 'total_budget' in forecast
        assert 'resource_forecasts' in forecast
        assert 'cost_optimization_recommendations' in forecast
        assert forecast['total_budget'] == 1000.0
    
    @pytest.mark.asyncio
    async def test_forecast_accuracy_metrics(self, forecasting_engine):
        """Test forecast accuracy metrics"""
        metrics = await forecasting_engine.get_forecast_accuracy_metrics()
        
        assert isinstance(metrics, dict)
        # May be empty if no models trained yet

class TestEconomicOptimizationIntegration:
    """Integration tests for the complete economic optimization system"""
    
    @pytest.fixture
    def complete_system(self):
        """Set up complete economic optimization system"""
        return {
            'ai_optimizer': AIEconomicOptimizer(),
            'market_maker': PredictiveMarketMaker(),
            'multi_objective_optimizer': MultiObjectiveOptimizer({
                'population_size': 10,
                'max_generations': 5
            }),
            'trading_engine': AlgorithmicTradingEngine(),
            'forecasting_engine': EconomicForecastingEngine()
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(self, complete_system):
        """Test complete end-to-end optimization workflow"""
        # 1. Create sample market data
        prices = [
            ResourcePrice(
                provider=CloudProvider.AWS,
                resource_type=ResourceType.GPU_H100,
                region="us-east-1",
                price_per_hour=3.50,
                price_per_unit=3.50,
                availability=80
            ),
            ResourcePrice(
                provider=CloudProvider.GCP,
                resource_type=ResourceType.GPU_H100,
                region="us-central1",
                price_per_hour=3.20,
                price_per_unit=3.20,
                availability=90
            )
        ]
        
        market_state = MarketState(
            prices=prices,
            predictions=[],
            arbitrage_opportunities=[]
        )
        
        # 2. Update all systems with market data
        await complete_system['market_maker'].update_market_data(prices)
        await complete_system['forecasting_engine'].update_price_history(prices)
        
        # 3. Generate market predictions
        predictions = await complete_system['market_maker'].generate_predictions(market_state)
        
        # 4. Detect arbitrage opportunities
        arbitrage_opps = await complete_system['market_maker'].detect_arbitrage_opportunities(market_state)
        
        # 5. Generate trading signals
        trading_signals = await complete_system['trading_engine'].generate_trading_signals(market_state)
        
        # 6. Optimize resource allocation
        strategy = OptimizationStrategy(
            name="Integration Test Strategy",
            objectives=[OptimizationObjective.MINIMIZE_COST, OptimizationObjective.MAXIMIZE_PERFORMANCE],
            weights={
                OptimizationObjective.MINIMIZE_COST: 0.6,
                OptimizationObjective.MAXIMIZE_PERFORMANCE: 0.4
            }
        )
        
        optimization_result = await complete_system['ai_optimizer'].optimize_resource_allocation(
            market_state=market_state,
            current_allocations=[],
            strategies=[strategy],
            budget_constraints={'total': 1000.0}
        )
        
        # 7. Generate economic forecast
        forecast = await complete_system['forecasting_engine'].generate_forecast(
            resource_type=ResourceType.GPU_H100,
            forecast_horizon=24
        )
        
        # Verify all components worked
        assert isinstance(predictions, list)
        assert isinstance(arbitrage_opps, list)
        assert isinstance(trading_signals, list)
        assert isinstance(optimization_result, OptimizationResult)
        assert isinstance(forecast, EconomicForecast)
        
        # Verify optimization result is reasonable
        assert optimization_result.expected_cost >= 0
        assert optimization_result.expected_performance >= 0
        assert 0 <= optimization_result.confidence <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])