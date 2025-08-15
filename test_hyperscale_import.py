#!/usr/bin/env python3
"""Test hyperscale monitoring import"""

try:
    print("Testing models import...")
    from scrollintel.models.hyperscale_monitoring_models import (
        GlobalMetrics, RegionalMetrics, ServiceMetrics, PredictiveAlert,
        SystemIncident, AutomatedResponse, ExecutiveDashboardMetrics,
        CapacityForecast, PerformanceBaseline, GlobalInfrastructureHealth,
        BusinessImpactMetrics, ScalingEvent, AlertRule, MonitoringDashboard,
        SeverityLevel, SystemStatus, IncidentStatus
    )
    print("✅ Models import successful")
    
    print("Testing class definition...")
    class TestHyperscaleMonitoringEngine:
        def __init__(self):
            self.test = "working"
    
    print("✅ Class definition successful")
    
    print("Testing full engine import...")
    exec(open('scrollintel/engines/hyperscale_monitoring_engine.py').read())
    print("✅ Engine execution successful")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()