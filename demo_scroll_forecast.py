"""
Demo script for ScrollForecast Engine.
Shows basic functionality without requiring external forecasting libraries.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from scrollintel.engines.scroll_forecast_engine import ScrollForecastEngine


async def demo_scroll_forecast():
    """Demonstrate ScrollForecast engine capabilities."""
    
    print("🔮 ScrollForecast Engine Demo")
    print("=" * 50)
    
    # Initialize engine
    print("\n1. Initializing ScrollForecast Engine...")
    engine = ScrollForecastEngine()
    await engine.initialize()
    
    # Check engine status
    print("\n2. Engine Status:")
    status = engine.get_status()
    print(f"   - Healthy: {status['healthy']}")
    print(f"   - Supported Models: {status['supported_models']}")
    print(f"   - Libraries Available: {status['libraries_available']}")
    
    # Create sample time series data
    print("\n3. Creating Sample Time Series Data...")
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create realistic sales data with trend and seasonality
    base_sales = 1000
    trend = np.linspace(0, 200, len(dates))  # Growing trend
    
    # Weekly seasonality (higher sales on weekends)
    weekly_pattern = []
    for date in dates:
        if date.weekday() in [5, 6]:  # Saturday, Sunday
            weekly_pattern.append(150)
        elif date.weekday() in [0, 4]:  # Monday, Friday
            weekly_pattern.append(50)
        else:
            weekly_pattern.append(0)
    
    # Add some noise
    noise = np.random.normal(0, 30, len(dates))
    
    # Combine all components
    sales = base_sales + trend + np.array(weekly_pattern) + noise
    
    # Create data in the format expected by the engine
    time_series_data = [
        {"date": date.strftime('%Y-%m-%d'), "sales": max(0, sale)}
        for date, sale in zip(dates, sales)
    ]
    
    print(f"   - Created {len(time_series_data)} data points")
    print(f"   - Date range: {time_series_data[0]['date']} to {time_series_data[-1]['date']}")
    print(f"   - Sales range: {min(d['sales'] for d in time_series_data):.0f} to {max(d['sales'] for d in time_series_data):.0f}")
    
    # Test time series analysis
    print("\n4. Analyzing Time Series Data...")
    try:
        analysis_request = {
            "action": "analyze",
            "data": time_series_data,
            "date_column": "date",
            "value_column": "sales"
        }
        
        analysis_result = await engine.process(analysis_request, {})
        
        print("   ✅ Analysis completed successfully!")
        print(f"   - Data points: {analysis_result['data_analysis']['data_points']}")
        print(f"   - Frequency: {analysis_result['data_analysis']['frequency']}")
        print(f"   - Date range: {analysis_result['data_analysis']['date_range']['start']} to {analysis_result['data_analysis']['date_range']['end']}")
        
        value_stats = analysis_result['data_analysis']['value_stats']
        print(f"   - Mean sales: {value_stats['mean']:.2f}")
        print(f"   - Std deviation: {value_stats['std']:.2f}")
        
        if 'trend' in analysis_result['data_analysis']:
            trend_info = analysis_result['data_analysis']['trend']
            print(f"   - Trend direction: {trend_info['direction']}")
        
        print(f"   - Recommendations: {len(analysis_result['recommendations'])} suggestions")
        for i, rec in enumerate(analysis_result['recommendations'][:3], 1):
            print(f"     {i}. {rec}")
        
    except Exception as e:
        print(f"   ❌ Analysis failed: {e}")
    
    # Test forecasting (will show what happens when no models are available)
    print("\n5. Testing Forecast Creation...")
    try:
        forecast_request = {
            "action": "forecast",
            "data": time_series_data[:100],  # Use subset for faster processing
            "date_column": "date",
            "value_column": "sales",
            "forecast_periods": 10,
            "forecast_name": "demo_sales_forecast"
        }
        
        forecast_result = await engine.process(forecast_request, {})
        
        print("   ✅ Forecast request processed!")
        print(f"   - Models tested: {forecast_result['models_tested']}")
        print(f"   - Results: {list(forecast_result['results'].keys())}")
        
        # Check if any models succeeded
        successful_models = [
            model for model, result in forecast_result['results'].items()
            if 'forecast' in result and 'error' not in result
        ]
        
        if successful_models:
            print(f"   - Successful models: {successful_models}")
            best_model = forecast_result['best_model']
            if best_model['type']:
                print(f"   - Best model: {best_model['type']} (score: {best_model['score']:.4f})")
        else:
            print("   - No models succeeded (this is expected without forecasting libraries)")
            for model, result in forecast_result['results'].items():
                if 'error' in result:
                    print(f"     {model}: {result['error']}")
        
    except Exception as e:
        print(f"   ❌ Forecast failed: {e}")
    
    # Test model comparison
    print("\n6. Testing Model Comparison...")
    try:
        if engine.trained_models:
            compare_request = {
                "action": "compare",
                "model_names": list(engine.trained_models.keys())
            }
            
            compare_result = await engine.process(compare_request, {})
            print(f"   ✅ Compared {compare_result['total_models']} models")
        else:
            print("   ℹ️  No trained models available for comparison")
    
    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
    
    # Test engine metrics
    print("\n7. Engine Performance Metrics:")
    metrics = engine.get_metrics()
    print(f"   - Engine ID: {metrics['engine_id']}")
    print(f"   - Status: {metrics['status']}")
    print(f"   - Usage count: {metrics['usage_count']}")
    print(f"   - Error count: {metrics['error_count']}")
    print(f"   - Error rate: {metrics['error_rate']:.2%}")
    
    # Health check
    print("\n8. Health Check:")
    health = await engine.health_check()
    print(f"   - Engine healthy: {health}")
    
    # Cleanup
    print("\n9. Cleaning up...")
    await engine.cleanup()
    print("   ✅ Engine cleanup completed")
    
    print("\n" + "=" * 50)
    print("🎉 ScrollForecast Engine Demo Complete!")
    print("\nNote: To enable full forecasting capabilities, install:")
    print("  pip install prophet statsmodels")
    print("  (TensorFlow is already available for LSTM models)")


if __name__ == "__main__":
    asyncio.run(demo_scroll_forecast())