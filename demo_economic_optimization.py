"""
Demonstration of AI-Powered Economic Optimization System
Showcases all components: reinforcement learning, market making, multi-objective optimization, 
algorithmic trading, and economic forecasting
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import logging

from scrollintel.models.economic_optimization_models import (
    ResourceType, CloudProvider, OptimizationObjective, OptimizationStrategy,
    ResourceAllocation, OptimizationResult, MarketState, PerformanceMetrics,
    TradingDecision, ArbitrageOpportunity, EconomicForecast, ResourcePrice
)
from scrollintel.engines.ai_economic_optimizer import AIEconomicOptimizer
from scrollintel.engines.predictive_market_maker import PredictiveMarketMaker
from scrollintel.engines.multi_objective_optimizer import MultiObjectiveOptimizer, OptimizationConstraints
from scrollintel.engines.algorithmic_trading_engine import AlgorithmicTradingEngine
from scrollintel.engines.economic_forecasting_engine import EconomicForecastingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EconomicOptimizationDemo:
    """Comprehensive demonstration of the economic optimization system"""
    
    def __init__(self):
        # Initialize all engines
        self.ai_optimizer = AIEconomicOptimizer({
            'learning_rate': 0.001,
            'discount_factor': 0.95,
            'exploration_rate': 0.1
        })
        
        self.market_maker = PredictiveMarketMaker({
            'prediction_horizon': 24,
            'min_arbitrage_profit': 0.03
        })
        
        self.multi_objective_optimizer = MultiObjectiveOptimizer({
            'population_size': 50,
            'max_generations': 100
        })
        
        self.trading_engine = AlgorithmicTradingEngine({
            'max_position_size': 100,
            'risk_limit': 0.02,
            'stop_loss_pct': 0.05
        })
        
        self.forecasting_engine = EconomicForecastingEngine({
            'max_forecast_horizon': 168,
            'min_training_samples': 50
        })
        
        logger.info("Economic Optimization Demo initialized")
    
    def generate_realistic_market_data(self, days: int = 7) -> List[ResourcePrice]:
        """Generate realistic market data for demonstration"""
        prices = []
        
        # Base prices for different resource types and providers
        base_prices = {
            (ResourceType.GPU_H100, CloudProvider.AWS): 3.50,
            (ResourceType.GPU_H100, CloudProvider.GCP): 3.20,
            (ResourceType.GPU_H100, CloudProvider.AZURE): 3.40,
            (ResourceType.GPU_A100, CloudProvider.AWS): 2.80,
            (ResourceType.GPU_A100, CloudProvider.GCP): 2.60,
            (ResourceType.GPU_A100, CloudProvider.AZURE): 2.75,
            (ResourceType.GPU_V100, CloudProvider.AWS): 1.50,
            (ResourceType.GPU_V100, CloudProvider.GCP): 1.40,
            (ResourceType.CPU_COMPUTE, CloudProvider.AWS): 0.10,
            (ResourceType.CPU_COMPUTE, CloudProvider.GCP): 0.09,
        }
        
        regions = {
            CloudProvider.AWS: ["us-east-1", "us-west-2", "eu-west-1"],
            CloudProvider.GCP: ["us-central1", "us-west1", "europe-west1"],
            CloudProvider.AZURE: ["eastus", "westus2", "westeurope"]
        }
        
        # Generate hourly data for specified days
        hours = days * 24
        
        for hour in range(hours):
            timestamp = datetime.utcnow() - timedelta(hours=hours-hour)
            
            for (resource_type, provider), base_price in base_prices.items():
                # Add realistic price variations
                
                # Daily seasonality (higher prices during business hours)
                daily_factor = 1.0 + 0.1 * np.sin(2 * np.pi * timestamp.hour / 24 + np.pi/2)
                
                # Weekly seasonality (higher prices on weekdays)
                weekly_factor = 1.0 + 0.05 * (1 if timestamp.weekday() < 5 else -1)
                
                # Market volatility
                volatility = np.random.normal(0, 0.05)
                
                # Trend (slight upward trend over time)
                trend = 1.0 + (hour / hours) * 0.02
                
                # Calculate final price
                final_price = base_price * daily_factor * weekly_factor * trend * (1 + volatility)
                final_price = max(0.01, final_price)  # Ensure positive
                
                # Spot price (typically 60-80% of on-demand)
                spot_price = final_price * np.random.uniform(0.6, 0.8)
                
                # Availability (varies by provider and time)
                base_availability = {
                    CloudProvider.AWS: 85,
                    CloudProvider.GCP: 80,
                    CloudProvider.AZURE: 75
                }.get(provider, 80)
                
                availability = max(10, min(100, base_availability + np.random.randint(-15, 15)))
                
                # Create price entry for each region
                for region in regions[provider]:
                    price = ResourcePrice(
                        provider=provider,
                        resource_type=resource_type,
                        region=region,
                        price_per_hour=final_price,
                        price_per_unit=final_price,
                        availability=availability,
                        spot_price=spot_price,
                        timestamp=timestamp
                    )
                    prices.append(price)
        
        logger.info(f"Generated {len(prices)} realistic market data points for {days} days")
        return prices
    
    async def demonstrate_market_making(self, market_data: List[ResourcePrice]):
        """Demonstrate predictive market making capabilities"""
        print("\n" + "="*60)
        print("PREDICTIVE MARKET MAKING DEMONSTRATION")
        print("="*60)
        
        # Update market maker with historical data
        await self.market_maker.update_market_data(market_data)
        
        # Create current market state
        current_prices = market_data[-20:]  # Last 20 data points
        market_state = MarketState(
            prices=current_prices,
            predictions=[],
            arbitrage_opportunities=[]
        )
        
        # Generate market predictions
        print("\n1. Generating Market Predictions...")
        predictions = await self.market_maker.generate_predictions(market_state)
        
        print(f"Generated {len(predictions)} market predictions:")
        for pred in predictions[:5]:  # Show first 5
            print(f"  • {pred.resource_type.value} ({pred.provider.value}): "
                  f"${pred.predicted_price:.3f} (confidence: {pred.confidence_score:.2f}, "
                  f"trend: {pred.trend_direction})")
        
        # Detect arbitrage opportunities
        print("\n2. Detecting Arbitrage Opportunities...")
        arbitrage_opportunities = await self.market_maker.detect_arbitrage_opportunities(market_state)
        
        print(f"Found {len(arbitrage_opportunities)} arbitrage opportunities:")
        for opp in arbitrage_opportunities[:3]:  # Show first 3
            print(f"  • {opp.resource_type.value}: Buy from {opp.buy_provider.value} "
                  f"(${opp.buy_price:.3f}) → Sell to {opp.sell_provider.value} "
                  f"(${opp.sell_price:.3f}) = {opp.profit_percentage:.1f}% profit")
        
        # Generate trading decisions
        print("\n3. Generating Trading Decisions...")
        trading_decisions = await self.market_maker.generate_trading_decisions(
            market_state, budget_limit=5000.0
        )
        
        print(f"Generated {len(trading_decisions)} trading decisions:")
        for decision in trading_decisions[:3]:  # Show first 3
            print(f"  • {decision.action.value.upper()} {decision.quantity}x "
                  f"{decision.resource_type.value} ({decision.provider.value}) "
                  f"at ${decision.target_price:.3f} (expected profit: ${decision.expected_profit:.2f})")
        
        return predictions, arbitrage_opportunities, trading_decisions
    
    async def demonstrate_ai_optimization(self, market_data: List[ResourcePrice]):
        """Demonstrate AI-powered resource allocation optimization"""
        print("\n" + "="*60)
        print("AI-POWERED RESOURCE ALLOCATION OPTIMIZATION")
        print("="*60)
        
        # Create market state
        current_prices = market_data[-10:]
        market_state = MarketState(
            prices=current_prices,
            predictions=[],
            arbitrage_opportunities=[]
        )
        
        # Define optimization strategies
        strategies = [
            OptimizationStrategy(
                name="Cost-Focused Strategy",
                objectives=[OptimizationObjective.MINIMIZE_COST, OptimizationObjective.MAXIMIZE_PERFORMANCE],
                weights={
                    OptimizationObjective.MINIMIZE_COST: 0.7,
                    OptimizationObjective.MAXIMIZE_PERFORMANCE: 0.3
                },
                risk_tolerance=0.2
            ),
            OptimizationStrategy(
                name="Performance-Focused Strategy",
                objectives=[OptimizationObjective.MAXIMIZE_PERFORMANCE, OptimizationObjective.MINIMIZE_LATENCY],
                weights={
                    OptimizationObjective.MAXIMIZE_PERFORMANCE: 0.6,
                    OptimizationObjective.MINIMIZE_LATENCY: 0.4
                },
                risk_tolerance=0.5
            )
        ]
        
        print("\n1. Optimizing Resource Allocation with Reinforcement Learning...")
        
        for i, strategy in enumerate(strategies, 1):
            print(f"\nStrategy {i}: {strategy.name}")
            
            result = await self.ai_optimizer.optimize_resource_allocation(
                market_state=market_state,
                current_allocations=[],
                strategies=[strategy],
                budget_constraints={'total': 10000.0, 'hourly': 1000.0}
            )
            
            print(f"  • Expected Cost: ${result.expected_cost:.2f}")
            print(f"  • Expected Performance: {result.expected_performance:.1f}")
            print(f"  • Risk Score: {result.risk_score:.3f}")
            print(f"  • Confidence: {result.confidence:.3f}")
            print(f"  • Allocations: {len(result.allocations)}")
            
            # Show top allocations
            for j, alloc in enumerate(result.allocations[:3]):
                print(f"    {j+1}. {alloc.quantity}x {alloc.resource_type.value} "
                      f"({alloc.provider.value}) - ${alloc.total_cost:.2f}")
        
        return strategies
    
    async def demonstrate_multi_objective_optimization(self, market_data: List[ResourcePrice]):
        """Demonstrate multi-objective optimization with Pareto frontier"""
        print("\n" + "="*60)
        print("MULTI-OBJECTIVE OPTIMIZATION & PARETO FRONTIER")
        print("="*60)
        
        # Create market state
        current_prices = market_data[-15:]
        market_state = MarketState(
            prices=current_prices,
            predictions=[],
            arbitrage_opportunities=[]
        )
        
        # Define constraints
        constraints = OptimizationConstraints(
            budget_limit=8000.0,
            performance_minimum=1000.0,
            latency_maximum=50.0,
            availability_minimum=70.0,
            provider_limits={
                CloudProvider.AWS: 50,
                CloudProvider.GCP: 50,
                CloudProvider.AZURE: 30
            },
            resource_limits={
                ResourceType.GPU_H100: 20,
                ResourceType.GPU_A100: 30,
                ResourceType.GPU_V100: 40
            }
        )
        
        # Define multi-objective strategy
        strategy = OptimizationStrategy(
            name="Multi-Objective Balance",
            objectives=[
                OptimizationObjective.MINIMIZE_COST,
                OptimizationObjective.MAXIMIZE_PERFORMANCE,
                OptimizationObjective.MINIMIZE_LATENCY
            ],
            weights={
                OptimizationObjective.MINIMIZE_COST: 0.4,
                OptimizationObjective.MAXIMIZE_PERFORMANCE: 0.4,
                OptimizationObjective.MINIMIZE_LATENCY: 0.2
            }
        )
        
        print("\n1. Finding Pareto-Optimal Solutions...")
        pareto_solutions = await self.multi_objective_optimizer.optimize_pareto_frontier(
            market_state=market_state,
            strategies=[strategy],
            constraints=constraints
        )
        
        print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
        
        # Analyze Pareto frontier
        print("\n2. Analyzing Pareto Frontier...")
        analysis = await self.multi_objective_optimizer.analyze_pareto_frontier(pareto_solutions)
        
        if 'objective_ranges' in analysis:
            print("Objective Ranges:")
            for obj, ranges in analysis['objective_ranges'].items():
                print(f"  • {obj}: {ranges['min']:.2f} - {ranges['max']:.2f} "
                      f"(mean: {ranges['mean']:.2f})")
        
        if 'recommended_solutions' in analysis:
            print("\nRecommended Solutions:")
            for rec_type, solution in analysis['recommended_solutions'].items():
                if hasattr(solution, 'objectives'):
                    cost = solution.objectives.get('cost', 0)
                    performance = solution.objectives.get('performance', 0)
                    print(f"  • {rec_type.replace('_', ' ').title()}: "
                          f"Cost=${cost:.2f}, Performance={performance:.1f}")
        
        return pareto_solutions, analysis
    
    async def demonstrate_algorithmic_trading(self, market_data: List[ResourcePrice]):
        """Demonstrate algorithmic trading strategies"""
        print("\n" + "="*60)
        print("ALGORITHMIC TRADING STRATEGIES")
        print("="*60)
        
        # Update trading engine with historical data
        await self.trading_engine._update_price_history(market_data)
        
        # Create current market state
        current_prices = market_data[-5:]
        market_state = MarketState(
            prices=current_prices,
            predictions=[],
            arbitrage_opportunities=[]
        )
        
        print("\n1. Generating Trading Signals...")
        signals = await self.trading_engine.generate_trading_signals(market_state)
        
        print(f"Generated {len(signals)} trading signals:")
        
        # Group signals by strategy
        strategy_signals = {}
        for signal in signals:
            strategy = signal.strategy.value
            if strategy not in strategy_signals:
                strategy_signals[strategy] = []
            strategy_signals[strategy].append(signal)
        
        for strategy, strategy_signal_list in strategy_signals.items():
            print(f"\n  {strategy.replace('_', ' ').title()} Strategy:")
            for signal in strategy_signal_list[:2]:  # Show top 2 per strategy
                print(f"    • {signal.action.value.upper()} {signal.resource_type.value} "
                      f"({signal.provider.value}) - Strength: {signal.signal_strength:.3f}, "
                      f"Confidence: {signal.confidence:.3f}")
        
        print("\n2. Executing Trading Decisions...")
        decisions = await self.trading_engine.execute_trading_decisions(
            signals=signals,
            budget_limit=5000.0,
            market_state=market_state
        )
        
        print(f"Executed {len(decisions)} trading decisions:")
        for decision in decisions[:5]:  # Show first 5
            print(f"  • {decision.action.value.upper()} {decision.quantity}x "
                  f"{decision.resource_type.value} ({decision.provider.value}) "
                  f"at ${decision.current_price:.3f}")
        
        # Update positions and get portfolio summary
        await self.trading_engine.update_positions(market_state)
        
        print("\n3. Portfolio Summary...")
        portfolio = await self.trading_engine.get_portfolio_summary()
        
        print(f"  • Total Positions: {portfolio['total_positions']}")
        print(f"  • Portfolio Value: ${portfolio['total_portfolio_value']:.2f}")
        print(f"  • Unrealized P&L: ${portfolio['total_unrealized_pnl']:.2f}")
        print(f"  • Recent Signals: {portfolio['recent_signals']}")
        
        return signals, decisions, portfolio
    
    async def demonstrate_economic_forecasting(self, market_data: List[ResourcePrice]):
        """Demonstrate economic forecasting capabilities"""
        print("\n" + "="*60)
        print("ECONOMIC FORECASTING & COST OPTIMIZATION")
        print("="*60)
        
        # Update forecasting engine with historical data
        await self.forecasting_engine.update_price_history(market_data)
        
        print("\n1. Generating Economic Forecasts...")
        
        # Generate forecasts for different resource types
        resource_types = [ResourceType.GPU_H100, ResourceType.GPU_A100, ResourceType.GPU_V100]
        forecasts = {}
        
        for resource_type in resource_types:
            forecast = await self.forecasting_engine.generate_forecast(
                resource_type=resource_type,
                provider=CloudProvider.AWS,
                forecast_horizon=24
            )
            forecasts[resource_type] = forecast
            
            if forecast.forecast_values:
                current_avg = np.mean([p.price_per_hour for p in market_data[-24:] 
                                     if p.resource_type == resource_type and p.provider == CloudProvider.AWS])
                forecast_avg = np.mean(forecast.forecast_values)
                change_pct = ((forecast_avg - current_avg) / current_avg * 100) if current_avg > 0 else 0
                
                print(f"  • {resource_type.value} (AWS):")
                print(f"    Current Avg: ${current_avg:.3f}")
                print(f"    Forecast Avg: ${forecast_avg:.3f} ({change_pct:+.1f}%)")
                print(f"    Model Accuracy: {forecast.model_accuracy:.1f}%")
                
                if forecast.seasonal_factors:
                    print(f"    Seasonal Patterns: {list(forecast.seasonal_factors.keys())}")
        
        print("\n2. Cost Optimization Forecast...")
        cost_forecast = await self.forecasting_engine.generate_cost_optimization_forecast(
            budget=10000.0,
            time_horizon=timedelta(hours=48)
        )
        
        print(f"  • Budget: ${cost_forecast['total_budget']:.2f}")
        print(f"  • Time Horizon: {cost_forecast['time_horizon_hours']} hours")
        print(f"  • Expected Savings: ${cost_forecast['expected_savings']:.2f} "
              f"({cost_forecast['savings_percentage']:.1f}%)")
        
        if cost_forecast['cost_optimization_recommendations']:
            print("  • Recommendations:")
            for rec in cost_forecast['cost_optimization_recommendations'][:3]:
                print(f"    - {rec['resource_type']}: {rec['recommendation']} "
                      f"({rec['reason']})")
        
        return forecasts, cost_forecast
    
    async def demonstrate_performance_analytics(self):
        """Demonstrate performance analytics and metrics"""
        print("\n" + "="*60)
        print("PERFORMANCE ANALYTICS & METRICS")
        print("="*60)
        
        # Get performance metrics from AI optimizer
        period_start = datetime.utcnow() - timedelta(days=7)
        period_end = datetime.utcnow()
        
        print("\n1. AI Optimizer Performance...")
        ai_metrics = await self.ai_optimizer.get_performance_metrics(period_start, period_end)
        
        print(f"  • Cost Savings: ${ai_metrics.total_cost_savings:.2f}")
        print(f"  • Cost Reduction: {ai_metrics.cost_reduction_percentage:.1f}%")
        print(f"  • Prediction Accuracy: {ai_metrics.prediction_accuracy:.1f}%")
        print(f"  • Average ROI: {ai_metrics.average_roi:.3f}")
        
        # Get trading risk metrics
        print("\n2. Trading Risk Metrics...")
        risk_metrics = await self.trading_engine.calculate_risk_metrics()
        
        print(f"  • Value at Risk (95%): {risk_metrics.var_95:.3f}")
        print(f"  • Expected Shortfall: {risk_metrics.expected_shortfall:.3f}")
        print(f"  • Sharpe Ratio: {risk_metrics.sharpe_ratio:.3f}")
        print(f"  • Maximum Drawdown: {risk_metrics.max_drawdown:.3f}")
        
        # Get forecasting accuracy
        print("\n3. Forecasting Model Accuracy...")
        forecast_accuracy = await self.forecasting_engine.get_forecast_accuracy_metrics()
        
        if forecast_accuracy:
            if 'overall' in forecast_accuracy:
                overall = forecast_accuracy['overall']
                print(f"  • Overall Accuracy: {overall['accuracy_percentage']:.1f}%")
                print(f"  • Mean Absolute Error: {overall['mae']:.3f}")
                print(f"  • Number of Models: {overall['num_models']}")
            else:
                print("  • No trained models yet")
        
        return ai_metrics, risk_metrics, forecast_accuracy
    
    async def run_complete_demonstration(self):
        """Run the complete economic optimization demonstration"""
        print("🚀 SCROLLINTEL-G6 ECONOMIC OPTIMIZATION DEMONSTRATION")
        print("=" * 80)
        print("Showcasing AI-Powered Economic Optimization with Predictive Market Making")
        print("=" * 80)
        
        try:
            # Generate realistic market data
            print("\n📊 Generating realistic market data...")
            market_data = self.generate_realistic_market_data(days=10)
            
            # Run all demonstrations
            predictions, arbitrage_opps, trading_decisions = await self.demonstrate_market_making(market_data)
            strategies = await self.demonstrate_ai_optimization(market_data)
            pareto_solutions, pareto_analysis = await self.demonstrate_multi_objective_optimization(market_data)
            signals, decisions, portfolio = await self.demonstrate_algorithmic_trading(market_data)
            forecasts, cost_forecast = await self.demonstrate_economic_forecasting(market_data)
            ai_metrics, risk_metrics, forecast_accuracy = await self.demonstrate_performance_analytics()
            
            # Summary
            print("\n" + "="*60)
            print("DEMONSTRATION SUMMARY")
            print("="*60)
            
            print(f"\n✅ Market Making:")
            print(f"   • Generated {len(predictions)} price predictions")
            print(f"   • Found {len(arbitrage_opps)} arbitrage opportunities")
            print(f"   • Created {len(trading_decisions)} trading decisions")
            
            print(f"\n✅ AI Optimization:")
            print(f"   • Tested {len(strategies)} optimization strategies")
            print(f"   • Achieved cost reduction: {ai_metrics.cost_reduction_percentage:.1f}%")
            print(f"   • Prediction accuracy: {ai_metrics.prediction_accuracy:.1f}%")
            
            print(f"\n✅ Multi-Objective Optimization:")
            print(f"   • Found {len(pareto_solutions)} Pareto-optimal solutions")
            print(f"   • Analyzed trade-offs between cost, performance, and latency")
            
            print(f"\n✅ Algorithmic Trading:")
            print(f"   • Generated {len(signals)} trading signals")
            print(f"   • Executed {len(decisions)} trading decisions")
            print(f"   • Portfolio value: ${portfolio['total_portfolio_value']:.2f}")
            
            print(f"\n✅ Economic Forecasting:")
            print(f"   • Generated forecasts for {len(forecasts)} resource types")
            print(f"   • Expected savings: ${cost_forecast['expected_savings']:.2f}")
            print(f"   • Optimization recommendations: {len(cost_forecast['cost_optimization_recommendations'])}")
            
            print(f"\n🎯 Key Achievements:")
            print(f"   • Economic superintelligence with RL optimization")
            print(f"   • Predictive market making with arbitrage detection")
            print(f"   • Multi-objective Pareto frontier exploration")
            print(f"   • Algorithmic trading with risk management")
            print(f"   • Time-series forecasting for cost optimization")
            
            print(f"\n🏆 ScrollIntel-G6 Economic Optimization: DEMONSTRATION COMPLETE!")
            print("   Ready for production deployment with economic superintelligence.")
            
        except Exception as e:
            logger.error(f"Error in demonstration: {e}")
            print(f"\n❌ Demonstration failed: {e}")

async def main():
    """Main demonstration function"""
    demo = EconomicOptimizationDemo()
    await demo.run_complete_demonstration()

if __name__ == "__main__":
    asyncio.run(main())