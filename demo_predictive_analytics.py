"""
Demo script for predictive analytics engine.
Showcases forecasting, scenario modeling, and risk prediction capabilities.
"""
# import asyncio  # Not needed for synchronous execution
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Any
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from scrollintel.engines.predictive_engine import PredictiveEngine
from scrollintel.models.predictive_models import (
    BusinessMetric, ScenarioConfig, ScenarioResult, RiskPrediction,
    BusinessContext, ForecastModel, MetricCategory, RiskLevel
)

def generate_sample_data() -> Dict[str, List[BusinessMetric]]:
    """Generate sample business metrics for demonstration."""
    print("📊 Generating sample business metrics...")
    
    metrics_data = {}
    base_date = datetime.now() - timedelta(days=365)
    
    # Revenue metrics
    revenue_data = []
    base_revenue = 100000
    for i in range(365):
        # Add trend, seasonality, and noise
        trend = i * 50  # Growing trend
        seasonal = 10000 * np.sin(2 * np.pi * i / 365)  # Yearly seasonality
        weekly = 5000 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
        noise = np.random.normal(0, 2000)
        
        value = base_revenue + trend + seasonal + weekly + noise
        
        metric = BusinessMetric(
            id="revenue_daily",
            name="Daily Revenue",
            category=MetricCategory.FINANCIAL,
            value=max(value, 0),  # Ensure non-negative
            unit="USD",
            timestamp=base_date + timedelta(days=i),
            source="sales_system",
            context={"region": "global", "currency": "USD"}
        )
        revenue_data.append(metric)
    
    metrics_data["revenue_daily"] = revenue_data
    
    # Customer acquisition metrics
    customer_data = []
    base_customers = 50
    for i in range(365):
        # Seasonal customer acquisition pattern
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)
        trend_factor = 1 + (i / 365) * 0.5  # 50% growth over year
        noise = np.random.normal(1, 0.1)
        
        value = base_customers * seasonal_factor * trend_factor * noise
        
        metric = BusinessMetric(
            id="customers_acquired",
            name="Daily Customer Acquisitions",
            category=MetricCategory.CUSTOMER,
            value=max(int(value), 0),
            unit="count",
            timestamp=base_date + timedelta(days=i),
            source="crm_system",
            context={"channel": "all", "segment": "b2b"}
        )
        customer_data.append(metric)
    
    metrics_data["customers_acquired"] = customer_data
    
    # System performance metrics
    performance_data = []
    base_response_time = 200  # ms
    for i in range(365):
        # Performance degradation over time with occasional spikes
        degradation = i * 0.1
        spike_probability = 0.05
        spike = 500 if np.random.random() < spike_probability else 0
        noise = np.random.normal(0, 20)
        
        value = base_response_time + degradation + spike + noise
        
        metric = BusinessMetric(
            id="api_response_time",
            name="API Response Time",
            category=MetricCategory.TECHNICAL,
            value=max(value, 10),  # Minimum 10ms
            unit="milliseconds",
            timestamp=base_date + timedelta(days=i),
            source="monitoring_system",
            context={"service": "api_gateway", "region": "us-east-1"}
        )
        performance_data.append(metric)
    
    metrics_data["api_response_time"] = performance_data
    
    print(f"✅ Generated {len(metrics_data)} metric types with {sum(len(data) for data in metrics_data.values())} total data points")
    return metrics_data


def demo_forecasting(engine: PredictiveEngine, historical_data: Dict[str, List[BusinessMetric]]):
    """Demonstrate forecasting capabilities."""
    print("\n🔮 FORECASTING DEMONSTRATION")
    print("=" * 50)
    
    # Test different forecasting models
    models_to_test = [
        ForecastModel.LINEAR_REGRESSION,
        ForecastModel.ENSEMBLE,
        ForecastModel.PROPHET,
        ForecastModel.ARIMA
    ]
    
    for metric_id, data in historical_data.items():
        print(f"\n📈 Forecasting for {metric_id}:")
        latest_metric = data[-1]
        
        for model_type in models_to_test:
            try:
                forecast = engine.forecast_metrics(
                    metric=latest_metric,
                    horizon=30,  # 30-day forecast
                    historical_data=data,
                    model_type=model_type
                )
                
                avg_prediction = np.mean(forecast.predictions)
                confidence_range = np.mean(forecast.confidence_upper) - np.mean(forecast.confidence_lower)
                
                print(f"  {model_type.value:20} | Avg: {avg_prediction:8.1f} | Confidence: ±{confidence_range/2:.1f} | Level: {forecast.confidence_level:.2f}")
                
            except Exception as e:
                print(f"  {model_type.value:20} | Error: {str(e)[:50]}...")
    
    # Detailed forecast for revenue
    print(f"\n📊 Detailed Revenue Forecast (Next 30 days):")
    revenue_data = historical_data["revenue_daily"]
    latest_revenue = revenue_data[-1]
    
    forecast = engine.forecast_metrics(
        metric=latest_revenue,
        horizon=30,
        historical_data=revenue_data,
        model_type=ForecastModel.ENSEMBLE
    )
    
    print(f"Current Revenue: ${latest_revenue.value:,.2f}")
    print(f"30-day Average Forecast: ${np.mean(forecast.predictions):,.2f}")
    print(f"Confidence Range: ${np.mean(forecast.confidence_lower):,.2f} - ${np.mean(forecast.confidence_upper):,.2f}")
    print(f"Model Confidence: {forecast.confidence_level:.1%}")
    
    # Show first 7 days in detail
    print(f"\nNext 7 Days Detailed Forecast:")
    for i in range(min(7, len(forecast.predictions))):
        date = forecast.timestamps[i].strftime("%Y-%m-%d")
        pred = forecast.predictions[i]
        lower = forecast.confidence_lower[i]
        upper = forecast.confidence_upper[i]
        print(f"  {date}: ${pred:8,.0f} (${lower:8,.0f} - ${upper:8,.0f})")


def demo_scenario_modeling(engine: PredictiveEngine, historical_data: Dict[str, List[BusinessMetric]]):
    """Demonstrate scenario modeling capabilities."""
    print("\n🎯 SCENARIO MODELING DEMONSTRATION")
    print("=" * 50)
    
    # Scenario 1: Marketing campaign impact
    marketing_scenario = ScenarioConfig(
        id="marketing_boost",
        name="Marketing Campaign Impact",
        description="Simulate 20% increase in customer acquisition with seasonal adjustment",
        parameters={
            "percentage_change": 20,
            "seasonal_adjustment": 0.1,
            "trend_adjustment": 0.05
        },
        target_metrics=["customers_acquired", "revenue_daily"],
        time_horizon=90,
        created_by="demo_user",
        created_at=datetime.now()
    )
    
    print(f"🚀 Scenario: {marketing_scenario.name}")
    print(f"   Description: {marketing_scenario.description}")
    print(f"   Parameters: {marketing_scenario.parameters}")
    
    result = engine.model_scenario(marketing_scenario, historical_data)
    
    print(f"\n📊 Scenario Results:")
    print(f"   Confidence Score: {result.confidence_score:.1%}")
    print(f"   Impact Analysis:")
    
    for metric_id, impact in result.impact_analysis.items():
        direction = "📈" if impact > 0 else "📉"
        print(f"     {direction} {metric_id}: {impact:+.1f}%")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Scenario 2: Economic downturn
    downturn_scenario = ScenarioConfig(
        id="economic_downturn",
        name="Economic Downturn Impact",
        description="Simulate 15% decrease in revenue with increased volatility",
        parameters={
            "percentage_change": -15,
            "volatility_increase": 0.3,
            "trend_adjustment": -0.02
        },
        target_metrics=["revenue_daily", "customers_acquired"],
        time_horizon=180,
        created_by="demo_user",
        created_at=datetime.now()
    )
    
    print(f"\n⚠️  Scenario: {downturn_scenario.name}")
    result2 = engine.model_scenario(downturn_scenario, historical_data)
    
    print(f"📊 Impact Analysis:")
    for metric_id, impact in result2.impact_analysis.items():
        direction = "📈" if impact > 0 else "📉"
        print(f"   {direction} {metric_id}: {impact:+.1f}%")


def demo_risk_prediction(engine: PredictiveEngine, historical_data: Dict[str, List[BusinessMetric]]):
    """Demonstrate risk prediction capabilities."""
    print("\n⚠️  RISK PREDICTION DEMONSTRATION")
    print("=" * 50)
    
    # Create business context
    context = BusinessContext(
        industry="technology",
        company_size="medium",
        market_conditions={"growth_rate": 0.15, "competition": "high"},
        seasonal_factors={"q4_boost": 1.2, "summer_dip": 0.9},
        external_factors={"economic_outlook": "uncertain"},
        historical_patterns={"volatility": "medium"}
    )
    
    # Get current metrics (last data points)
    current_metrics = [data[-1] for data in historical_data.values()]
    
    # Predict risks
    risks = engine.predict_risks(context, current_metrics, historical_data)
    
    print(f"🔍 Identified {len(risks)} potential risks:")
    
    if not risks:
        print("   ✅ No significant risks detected")
        return
    
    # Group risks by level
    risk_groups = {}
    for risk in risks:
        level = risk.risk_level.value
        if level not in risk_groups:
            risk_groups[level] = []
        risk_groups[level].append(risk)
    
    # Display risks by severity
    level_icons = {
        "critical": "🚨",
        "high": "⚠️",
        "medium": "⚡",
        "low": "ℹ️"
    }
    
    for level in ["critical", "high", "medium", "low"]:
        if level in risk_groups:
            print(f"\n{level_icons[level]} {level.upper()} RISKS:")
            for risk in risk_groups[level]:
                print(f"   • {risk.description}")
                print(f"     Probability: {risk.probability:.1%} | Impact: {risk.impact_score:.2f}")
                print(f"     Metric: {risk.metric_id} | Type: {risk.risk_type}")
                if risk.mitigation_strategies:
                    print(f"     Mitigation: {risk.mitigation_strategies[0]}")
                print()


def demo_prediction_updates(engine: PredictiveEngine, historical_data: Dict[str, List[BusinessMetric]]):
    """Demonstrate prediction update capabilities."""
    print("\n🔄 PREDICTION UPDATES DEMONSTRATION")
    print("=" * 50)
    
    # Simulate new data arriving
    print("📥 Simulating new data arrival...")
    
    new_metrics = []
    for metric_id, data in historical_data.items():
        latest = data[-1]
        
        # Create new metric with some variation
        variation = np.random.normal(1, 0.1)  # 10% variation
        new_value = latest.value * variation
        
        new_metric = BusinessMetric(
            id=metric_id,
            name=latest.name,
            category=latest.category,
            value=new_value,
            unit=latest.unit,
            timestamp=datetime.now(),
            source=latest.source,
            context=latest.context
        )
        new_metrics.append(new_metric)
    
    # Update predictions
    updates = engine.update_predictions(new_metrics)
    
    print(f"📊 Generated {len(updates)} prediction updates:")
    
    for update in updates:
        change_direction = "📈" if update.change_magnitude > 0 else "📉"
        print(f"\n{change_direction} Metric: {update.metric_id}")
        print(f"   Change Magnitude: {update.change_magnitude:+.1%}")
        print(f"   Reason: {update.change_reason}")
        print(f"   Updated: {update.update_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Show prediction comparison
        old_avg = np.mean(update.previous_forecast.predictions)
        new_avg = np.mean(update.updated_forecast.predictions)
        print(f"   Previous Avg: {old_avg:.1f} → New Avg: {new_avg:.1f}")


def visualize_forecast(forecast, metric_name: str, historical_data: List[BusinessMetric]):
    """Create a simple visualization of the forecast."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Historical data
        hist_dates = [m.timestamp for m in historical_data[-60:]]  # Last 60 days
        hist_values = [m.value for m in historical_data[-60:]]
        
        plt.plot(hist_dates, hist_values, 'b-', label='Historical', linewidth=2)
        
        # Forecast
        plt.plot(forecast.timestamps, forecast.predictions, 'r--', label='Forecast', linewidth=2)
        
        # Confidence intervals
        plt.fill_between(forecast.timestamps, 
                        forecast.confidence_lower, 
                        forecast.confidence_upper, 
                        alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title(f'{metric_name} - Forecast ({forecast.model_type.value})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f'forecast_{metric_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"📊 Forecast visualization saved as {filename}")
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {str(e)}")


def main():
    """Main demonstration function."""
    print("🚀 SCROLLINTEL PREDICTIVE ANALYTICS DEMO")
    print("=" * 60)
    
    # Initialize engine
    print("🔧 Initializing Predictive Analytics Engine...")
    engine = PredictiveEngine()
    
    # Generate sample data
    historical_data = generate_sample_data()
    
    # Run demonstrations
    demo_forecasting(engine, historical_data)
    demo_scenario_modeling(engine, historical_data)
    demo_risk_prediction(engine, historical_data)
    demo_prediction_updates(engine, historical_data)
    
    # Create visualizations
    print("\n📊 CREATING VISUALIZATIONS")
    print("=" * 50)
    
    try:
        # Visualize revenue forecast
        revenue_data = historical_data["revenue_daily"]
        latest_revenue = revenue_data[-1]
        
        forecast = engine.forecast_metrics(
            metric=latest_revenue,
            horizon=30,
            historical_data=revenue_data,
            model_type=ForecastModel.ENSEMBLE
        )
        
        visualize_forecast(forecast, "Daily Revenue", revenue_data)
        
        # Visualize customer acquisition forecast
        customer_data = historical_data["customers_acquired"]
        latest_customers = customer_data[-1]
        
        customer_forecast = engine.forecast_metrics(
            metric=latest_customers,
            horizon=30,
            historical_data=customer_data,
            model_type=ForecastModel.ENSEMBLE
        )
        
        visualize_forecast(customer_forecast, "Customer Acquisitions", customer_data)
        
    except Exception as e:
        print(f"⚠️  Visualization error: {str(e)}")
    
    # Summary
    print("\n✅ DEMO COMPLETE")
    print("=" * 50)
    print("🎯 Demonstrated capabilities:")
    print("   • Multi-model forecasting (Linear, ARIMA, Prophet, LSTM, Ensemble)")
    print("   • Scenario modeling and what-if analysis")
    print("   • Risk prediction with early warning systems")
    print("   • Real-time prediction updates")
    print("   • Confidence intervals and accuracy tracking")
    print("   • Business context integration")
    print("\n📊 Check generated visualization files for forecast charts!")


if __name__ == "__main__":
    print("🔍 Starting demo execution...")
    try:
        main()
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()