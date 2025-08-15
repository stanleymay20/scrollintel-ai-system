#!/usr/bin/env python3
"""
Simple test for predictive analytics engine.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("🔧 Starting simple predictive analytics test...")

try:
    from scrollintel.engines.predictive_engine import PredictiveEngine
    print("✅ Successfully imported PredictiveEngine")
    
    engine = PredictiveEngine()
    print("✅ Successfully created PredictiveEngine instance")
    
    from scrollintel.models.predictive_models import (
        BusinessMetric, MetricCategory
    )
    print("✅ Successfully imported models")
    
    from datetime import datetime
    import numpy as np
    
    # Create a simple test metric
    test_metric = BusinessMetric(
        id="test_metric",
        name="Test Metric",
        category=MetricCategory.FINANCIAL,
        value=100.0,
        unit="USD",
        timestamp=datetime.now(),
        source="test",
        context={}
    )
    print("✅ Created test metric")
    
    # Create some historical data
    historical_data = []
    for i in range(30):
        metric = BusinessMetric(
            id="test_metric",
            name="Test Metric",
            category=MetricCategory.FINANCIAL,
            value=100.0 + i * 2 + np.random.normal(0, 5),
            unit="USD",
            timestamp=datetime.now(),
            source="test",
            context={}
        )
        historical_data.append(metric)
    
    print(f"✅ Created {len(historical_data)} historical data points")
    
    # Test forecasting
    from scrollintel.models.predictive_models import ForecastModel
    
    forecast = engine.forecast_metrics(
        metric=test_metric,
        horizon=7,
        historical_data=historical_data,
        model_type=ForecastModel.LINEAR_REGRESSION
    )
    
    print(f"✅ Generated forecast with {len(forecast.predictions)} predictions")
    print(f"   Model: {forecast.model_type.value}")
    print(f"   Confidence: {forecast.confidence_level:.2f}")
    print(f"   Average prediction: {np.mean(forecast.predictions):.2f}")
    
    print("\n🎉 Simple test completed successfully!")
    
except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()