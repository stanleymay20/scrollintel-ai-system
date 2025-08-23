"""
API Routes for AI-Powered Economic Optimization
Provides REST endpoints for economic optimization, market making, and trading functionality
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from ...models.economic_optimization_models import (
    ResourceType, CloudProvider, OptimizationObjective, OptimizationStrategy,
    ResourceAllocation, OptimizationResult, MarketState, PerformanceMetrics,
    TradingDecision, ArbitrageOpportunity, EconomicForecast, ResourcePrice
)
from ...engines.ai_economic_optimizer import AIEconomicOptimizer
from ...engines.predictive_market_maker import PredictiveMarketMaker
from ...engines.multi_objective_optimizer import MultiObjectiveOptimizer, OptimizationConstraints
from ...engines.algorithmic_trading_engine import AlgorithmicTradingEngine
from ...engines.economic_forecasting_engine import EconomicForecastingEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/economic-optimization", tags=["Economic Optimization"])

# Initialize engines (in production, these would be dependency-injected)
ai_optimizer = AIEconomicOptimizer()
market_maker = PredictiveMarketMaker()
multi_objective_optimizer = MultiObjectiveOptimizer()
trading_engine = AlgorithmicTradingEngine()
forecasting_engine = EconomicForecastingEngine()

@router.post("/optimize-allocation", response_model=OptimizationResult)
async def optimize_resource_allocation(
    market_state: MarketState = Body(...),
    current_allocations: List[ResourceAllocation] = Body(...),
    strategies: List[OptimizationStrategy] = Body(...),
    budget_constraints: Dict[str, float] = Body(...)
):
    """
    Optimize resource allocation using AI-powered reinforcement learning
    """
    try:
        result = await ai_optimizer.optimize_resource_allocation(
            market_state=market_state,
            current_allocations=current_allocations,
            strategies=strategies,
            budget_constraints=budget_constraints
        )
        
        logger.info(f"Resource allocation optimized: {len(result.allocations)} allocations")
        return result
        
    except Exception as e:
        logger.error(f"Error optimizing resource allocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pareto-optimization")
async def pareto_frontier_optimization(
    market_state: MarketState = Body(...),
    strategies: List[OptimizationStrategy] = Body(...),
    budget_limit: float = Body(...),
    performance_minimum: float = Body(default=0.0),
    latency_maximum: float = Body(default=1000.0)
):
    """
    Find Pareto-optimal solutions using multi-objective optimization
    """
    try:
        # Create constraints
        constraints = OptimizationConstraints(
            budget_limit=budget_limit,
            performance_minimum=performance_minimum,
            latency_maximum=latency_maximum,
            availability_minimum=50.0,
            provider_limits={provider: 100 for provider in CloudProvider},
            resource_limits={resource: 50 for resource in ResourceType}
        )
        
        # Find Pareto frontier
        pareto_solutions = await multi_objective_optimizer.optimize_pareto_frontier(
            market_state=market_state,
            strategies=strategies,
            constraints=constraints
        )
        
        # Analyze frontier
        analysis = await multi_objective_optimizer.analyze_pareto_frontier(pareto_solutions)
        
        return {
            "pareto_solutions": [
                {
                    "allocations": solution.allocations,
                    "objectives": solution.objectives,
                    "dominance_rank": solution.dominance_rank,
                    "crowding_distance": solution.crowding_distance,
                    "fitness_score": solution.fitness_score
                }
                for solution in pareto_solutions
            ],
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error in Pareto optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/market-predictions", response_model=List[Dict[str, Any]])
async def generate_market_predictions(
    market_state: MarketState = Body(...)
):
    """
    Generate predictive market analysis for computational resources
    """
    try:
        predictions = await market_maker.generate_predictions(market_state)
        
        return [
            {
                "resource_type": pred.resource_type.value,
                "provider": pred.provider.value,
                "predicted_price": pred.predicted_price,
                "confidence_score": pred.confidence_score,
                "prediction_horizon": pred.prediction_horizon.total_seconds() / 3600,
                "trend_direction": pred.trend_direction,
                "volatility_score": pred.volatility_score,
                "factors": pred.factors
            }
            for pred in predictions
        ]
        
    except Exception as e:
        logger.error(f"Error generating market predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/arbitrage-opportunities", response_model=List[Dict[str, Any]])
async def detect_arbitrage_opportunities(
    market_state: MarketState = Body(...)
):
    """
    Detect arbitrage opportunities across cloud providers
    """
    try:
        opportunities = await market_maker.detect_arbitrage_opportunities(market_state)
        
        return [
            {
                "resource_type": opp.resource_type.value,
                "buy_provider": opp.buy_provider.value,
                "sell_provider": opp.sell_provider.value,
                "buy_price": opp.buy_price,
                "sell_price": opp.sell_price,
                "profit_margin": opp.profit_margin,
                "profit_percentage": opp.profit_percentage,
                "execution_window": opp.execution_window.total_seconds() / 3600,
                "risk_score": opp.risk_score,
                "confidence": opp.confidence
            }
            for opp in opportunities
        ]
        
    except Exception as e:
        logger.error(f"Error detecting arbitrage opportunities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/trading-signals")
async def generate_trading_signals(
    market_state: MarketState = Body(...)
):
    """
    Generate algorithmic trading signals for compute resources
    """
    try:
        signals = await trading_engine.generate_trading_signals(market_state)
        
        return [
            {
                "strategy": signal.strategy.value,
                "action": signal.action.value,
                "resource_type": signal.resource_type.value,
                "provider": signal.provider.value,
                "signal_strength": signal.signal_strength,
                "confidence": signal.confidence,
                "expected_return": signal.expected_return,
                "risk_score": signal.risk_score,
                "time_horizon": signal.time_horizon.total_seconds() / 3600,
                "metadata": signal.metadata
            }
            for signal in signals
        ]
        
    except Exception as e:
        logger.error(f"Error generating trading signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute-trades", response_model=List[TradingDecision])
async def execute_trading_decisions(
    market_state: MarketState = Body(...),
    budget_limit: float = Body(...)
):
    """
    Execute algorithmic trading decisions based on market signals
    """
    try:
        # Generate signals first
        signals = await trading_engine.generate_trading_signals(market_state)
        
        # Execute trading decisions
        decisions = await trading_engine.execute_trading_decisions(
            signals=signals,
            budget_limit=budget_limit,
            market_state=market_state
        )
        
        logger.info(f"Executed {len(decisions)} trading decisions")
        return decisions
        
    except Exception as e:
        logger.error(f"Error executing trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/economic-forecast")
async def generate_economic_forecast(
    resource_type: ResourceType = Query(...),
    provider: Optional[CloudProvider] = Query(None),
    forecast_horizon: int = Query(24, ge=1, le=168),
    confidence_level: float = Query(0.95, ge=0.5, le=0.99)
):
    """
    Generate economic forecast for resource pricing
    """
    try:
        forecast = await forecasting_engine.generate_forecast(
            resource_type=resource_type,
            provider=provider,
            forecast_horizon=forecast_horizon,
            confidence_level=confidence_level
        )
        
        return {
            "forecast_type": forecast.forecast_type,
            "resource_type": forecast.resource_type.value,
            "provider": forecast.provider.value if forecast.provider else None,
            "forecast_values": forecast.forecast_values,
            "forecast_timestamps": [ts.isoformat() for ts in forecast.forecast_timestamps],
            "confidence_intervals": forecast.confidence_intervals,
            "seasonal_factors": forecast.seasonal_factors,
            "trend_components": forecast.trend_components,
            "model_accuracy": forecast.model_accuracy
        }
        
    except Exception as e:
        logger.error(f"Error generating economic forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cost-optimization-forecast")
async def generate_cost_optimization_forecast(
    budget: float = Body(...),
    time_horizon_hours: int = Body(24, ge=1, le=168)
):
    """
    Generate comprehensive cost optimization forecast
    """
    try:
        time_horizon = timedelta(hours=time_horizon_hours)
        
        forecast = await forecasting_engine.generate_cost_optimization_forecast(
            budget=budget,
            time_horizon=time_horizon
        )
        
        return forecast
        
    except Exception as e:
        logger.error(f"Error generating cost optimization forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/update-market-data")
async def update_market_data(
    new_prices: List[ResourcePrice] = Body(...)
):
    """
    Update market data with new pricing information
    """
    try:
        # Update all engines with new market data
        await market_maker.update_market_data(new_prices)
        await forecasting_engine.update_price_history(new_prices)
        
        return {
            "status": "success",
            "updated_prices": len(new_prices),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/portfolio-summary")
async def get_portfolio_summary():
    """
    Get comprehensive portfolio summary and performance metrics
    """
    try:
        summary = await trading_engine.get_portfolio_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-metrics")
async def get_performance_metrics(
    period_start: datetime = Query(...),
    period_end: datetime = Query(...)
):
    """
    Get performance metrics for the economic optimization system
    """
    try:
        metrics = await ai_optimizer.get_performance_metrics(
            period_start=period_start,
            period_end=period_end
        )
        
        return {
            "period_start": metrics.period_start.isoformat(),
            "period_end": metrics.period_end.isoformat(),
            "total_cost_savings": metrics.total_cost_savings,
            "cost_reduction_percentage": metrics.cost_reduction_percentage,
            "arbitrage_profits": metrics.arbitrage_profits,
            "prediction_accuracy": metrics.prediction_accuracy,
            "successful_trades": metrics.successful_trades,
            "failed_trades": metrics.failed_trades,
            "average_roi": metrics.average_roi,
            "sharpe_ratio": metrics.sharpe_ratio,
            "max_drawdown": metrics.max_drawdown
        }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risk-metrics")
async def get_risk_metrics():
    """
    Get comprehensive risk metrics for trading operations
    """
    try:
        risk_metrics = await trading_engine.calculate_risk_metrics()
        
        return {
            "var_95": risk_metrics.var_95,
            "expected_shortfall": risk_metrics.expected_shortfall,
            "max_drawdown": risk_metrics.max_drawdown,
            "sharpe_ratio": risk_metrics.sharpe_ratio,
            "sortino_ratio": risk_metrics.sortino_ratio,
            "beta": risk_metrics.beta,
            "alpha": risk_metrics.alpha,
            "correlation_matrix": risk_metrics.correlation_matrix
        }
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-state")
async def get_current_market_state():
    """
    Get current comprehensive market state
    """
    try:
        market_state = await market_maker.get_market_state()
        
        return {
            "prices": [
                {
                    "resource_type": p.resource_type.value,
                    "provider": p.provider.value,
                    "region": p.region,
                    "price_per_hour": p.price_per_hour,
                    "spot_price": p.spot_price,
                    "availability": p.availability,
                    "timestamp": p.timestamp.isoformat()
                }
                for p in market_state.prices
            ],
            "market_volatility": {
                rt.value: vol for rt, vol in market_state.market_volatility.items()
            },
            "supply_demand_ratio": {
                rt.value: ratio for rt, ratio in market_state.supply_demand_ratio.items()
            },
            "trend_indicators": market_state.trend_indicators,
            "timestamp": market_state.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting market state: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast-accuracy")
async def get_forecast_accuracy():
    """
    Get accuracy metrics for forecasting models
    """
    try:
        accuracy_metrics = await forecasting_engine.get_forecast_accuracy_metrics()
        return accuracy_metrics
        
    except Exception as e:
        logger.error(f"Error getting forecast accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-optimization-strategy", response_model=OptimizationStrategy)
async def create_optimization_strategy(
    name: str = Body(...),
    objectives: List[OptimizationObjective] = Body(...),
    weights: Dict[str, float] = Body(...),
    constraints: Dict[str, Any] = Body(default={}),
    risk_tolerance: float = Body(0.5, ge=0.0, le=1.0)
):
    """
    Create a new optimization strategy
    """
    try:
        # Convert string keys to enum keys for weights
        enum_weights = {}
        for obj_str, weight in weights.items():
            try:
                obj_enum = OptimizationObjective(obj_str)
                enum_weights[obj_enum] = weight
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid objective: {obj_str}")
        
        strategy = OptimizationStrategy(
            name=name,
            objectives=objectives,
            weights=enum_weights,
            constraints=constraints,
            risk_tolerance=risk_tolerance
        )
        
        return strategy
        
    except Exception as e:
        logger.error(f"Error creating optimization strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate-trading")
async def simulate_trading_strategy(
    market_state: MarketState = Body(...),
    budget: float = Body(...),
    simulation_days: int = Body(7, ge=1, le=30)
):
    """
    Simulate trading strategy performance over time
    """
    try:
        # This is a simplified simulation - in practice would be more sophisticated
        results = {
            "simulation_period_days": simulation_days,
            "initial_budget": budget,
            "final_portfolio_value": budget * 1.05,  # 5% return (placeholder)
            "total_return": 0.05,
            "total_trades": 25,
            "successful_trades": 18,
            "failed_trades": 7,
            "max_drawdown": -0.03,
            "sharpe_ratio": 1.2,
            "daily_returns": [0.002, 0.001, -0.001, 0.003, 0.002, 0.001, 0.002],  # Placeholder
            "risk_metrics": {
                "var_95": -0.02,
                "volatility": 0.15,
                "beta": 0.8
            }
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error simulating trading strategy: {e}")
        raise HTTPException(status_code=500, detail=str(e))