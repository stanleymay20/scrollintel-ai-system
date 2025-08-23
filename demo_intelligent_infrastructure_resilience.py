"""
Demo: Intelligent Infrastructure Resilience System
Demonstrates auto-tuning infrastructure achieving 99.99% uptime with performance optimization
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
import json

from security.infrastructure.intelligent_infrastructure_resilience import (
    IntelligentInfrastructureResilience,
    InfrastructureStatus,
    OptimizationAction
)
from security.infrastructure.zero_downtime_deployment import (
    ZeroDowntimeDeployment,
    DeploymentConfig,
    DeploymentStrategy
)
from security.infrastructure.predictive_capacity_planning import (
    PredictiveCapacityPlanning,
    ResourceType
)
from security.infrastructure.multi_cloud_cost_optimizer import (
    MultiCloudCostOptimizer,
    CloudProvider
)
from security.infrastructure.disaster_recovery_system import (
    DisasterRecoverySystem,
    DisasterType
)
from security.infrastructure.configuration_drift_detection import (
    ConfigurationDriftDetection,
    ConfigurationType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_intelligent_infrastructure_resilience():
    """Demonstrate intelligent infrastructure resilience capabilities"""
    print("=" * 80)
    print("INTELLIGENT INFRASTRUCTURE RESILIENCE SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize systems
    infrastructure = IntelligentInfrastructureResilience()
    deployment_system = ZeroDowntimeDeployment()
    capacity_planner = PredictiveCapacityPlanning()
    cost_optimizer = MultiCloudCostOptimizer()
    disaster_recovery = DisasterRecoverySystem()
    drift_detection = ConfigurationDriftDetection()
    
    print("\n1. INFRASTRUCTURE METRICS COLLECTION")
    print("-" * 50)
    
    # Collect initial metrics
    metrics = await infrastructure.collect_infrastructure_metrics()
    print(f"✓ Collected infrastructure metrics:")
    print(f"  - CPU Usage: {metrics.cpu_usage:.1f}%")
    print(f"  - Memory Usage: {metrics.memory_usage:.1f}%")
    print(f"  - Disk Usage: {metrics.disk_usage:.1f}%")
    print(f"  - Response Time: {metrics.response_time:.1f}ms")
    print(f"  - Error Rate: {metrics.error_rate:.2f}%")
    print(f"  - Availability: {metrics.availability:.2f}%")
    print(f"  - Cost per Hour: ${metrics.cost_per_hour:.2f}")
    
    print("\n2. AUTO-TUNING INFRASTRUCTURE")
    print("-" * 50)
    
    # Simulate high load scenario
    print("Simulating high load scenario...")
    optimization_result = await infrastructure.auto_tune_infrastructure()
    
    print(f"✓ Auto-tuning completed:")
    print(f"  - Actions Taken: {len(optimization_result.get('actions_taken', []))}")
    for action in optimization_result.get('actions_taken', []):
        print(f"    • {action}")
    print(f"  - Performance Improvement: {optimization_result.get('performance_improvement', 0):.2f}%")
    print(f"  - Cost Savings: ${optimization_result.get('cost_savings', 0):.2f}")
    
    print("\n3. PREDICTIVE CAPACITY PLANNING")
    print("-" * 50)
    
    # Generate historical data for capacity planning
    print("Generating historical metrics for capacity planning...")
    for i in range(200):
        await capacity_planner.collect_resource_metrics()
        if i % 50 == 0:
            print(f"  Generated {i + 1} historical data points...")
    
    # Train prediction models
    print("Training predictive models...")
    training_results = await capacity_planner.train_prediction_models()
    
    print("✓ Model training results:")
    for resource_type, result in training_results.items():
        if 'accuracy' in result:
            print(f"  - {resource_type.value}: {result['accuracy']:.1f}% accuracy")
        else:
            print(f"  - {resource_type.value}: Training failed")
    
    # Generate capacity forecast
    print("Generating 90-day capacity forecast...")
    capacity_plan = await capacity_planner.generate_capacity_forecast(forecast_days=90)
    
    print(f"✓ Capacity forecast generated:")
    print(f"  - Plan ID: {capacity_plan.plan_id}")
    print(f"  - Forecast Horizon: {capacity_plan.forecast_horizon_days} days")
    print(f"  - Total Forecasts: {len(capacity_plan.forecasts)}")
    print(f"  - Predicted Monthly Cost: ${capacity_plan.total_predicted_cost:.2f}")
    print(f"  - Cost Optimizations: {len(capacity_plan.cost_optimization_opportunities)}")
    print(f"  - Scaling Recommendations: {len(capacity_plan.scaling_recommendations)}")
    print(f"  - Accuracy Assessment: {capacity_plan.accuracy_assessment.value}")
    
    # Show sample forecasts
    if capacity_plan.forecasts:
        print("\n  Sample forecasts:")
        for i, forecast in enumerate(capacity_plan.forecasts[:5]):
            print(f"    Day {i+1} - {forecast.resource_type.value}: {forecast.predicted_value:.1f} "
                  f"(confidence: {forecast.accuracy_score:.1f}%)")
    
    print("\n4. MULTI-CLOUD COST OPTIMIZATION")
    print("-" * 50)
    
    # Discover cloud resources
    print("Discovering multi-cloud resources...")
    discovery_results = await cost_optimizer.discover_cloud_resources()
    
    print("✓ Resource discovery completed:")
    for provider, count in discovery_results.items():
        print(f"  - {provider.value}: {count} resources")
    
    # Analyze cost optimizations
    print("Analyzing cost optimization opportunities...")
    cost_report = await cost_optimizer.analyze_cost_optimizations()
    
    print(f"✓ Cost optimization analysis:")
    print(f"  - Report ID: {cost_report.report_id}")
    print(f"  - Total Monthly Cost: ${cost_report.total_monthly_cost:.2f}")
    print(f"  - Potential Savings: ${cost_report.potential_monthly_savings:.2f}")
    print(f"  - Savings Percentage: {cost_report.savings_percentage:.1f}%")
    print(f"  - Optimization Opportunities: {len(cost_report.optimizations)}")
    
    # Show top optimizations
    if cost_report.optimizations:
        print("\n  Top optimization opportunities:")
        sorted_optimizations = sorted(cost_report.optimizations, 
                                    key=lambda x: x.potential_savings, reverse=True)
        for i, opt in enumerate(sorted_optimizations[:3]):
            print(f"    {i+1}. {opt.strategy.value}: ${opt.potential_savings:.2f}/month "
                  f"({opt.savings_percentage:.1f}% savings)")
    
    print("\n5. ZERO-DOWNTIME DEPLOYMENT")
    print("-" * 50)
    
    # Configure deployment
    deployment_config = DeploymentConfig(
        strategy=DeploymentStrategy.BLUE_GREEN,
        image_tag="v2.1.0",
        replicas=3,
        health_check_path="/health",
        health_check_timeout=30,
        rollback_threshold=5.0,
        canary_percentage=25,
        max_surge=1,
        max_unavailable=0
    )
    
    print(f"Executing {deployment_config.strategy.value} deployment...")
    deployment_result = await deployment_system.deploy("web-app", deployment_config)
    
    print(f"✓ Deployment completed:")
    print(f"  - Deployment ID: {deployment_result.deployment_id}")
    print(f"  - Status: {deployment_result.status.value}")
    print(f"  - Strategy: {deployment_result.strategy.value}")
    print(f"  - Duration: {deployment_result.duration_seconds:.1f} seconds")
    print(f"  - Rollback Triggered: {deployment_result.rollback_triggered}")
    print(f"  - Health Checks: {len(deployment_result.health_checks)}")
    
    if deployment_result.metrics_before and deployment_result.metrics_after:
        print(f"  - Success Rate: {deployment_result.metrics_before.success_rate:.1f}% → "
              f"{deployment_result.metrics_after.success_rate:.1f}%")
    
    print("\n6. DISASTER RECOVERY TESTING")
    print("-" * 50)
    
    # Create test disaster event
    event_id = disaster_recovery.create_disaster_event(
        DisasterType.HARDWARE_FAILURE,
        ["database_primary"],
        "high"
    )
    
    print(f"Created disaster event: {event_id}")
    
    # Execute disaster recovery
    print("Executing disaster recovery procedures...")
    recovery_result = await disaster_recovery.initiate_disaster_recovery(event_id)
    
    print(f"✓ Disaster recovery completed:")
    print(f"  - Event ID: {recovery_result['event_id']}")
    print(f"  - Recovery Time: {recovery_result['recovery_time_minutes']} minutes")
    print(f"  - RTO Target Met: {recovery_result['rto_target_met']}")
    print(f"  - Status: {recovery_result['status']}")
    
    # Test disaster recovery for specific system
    print("\nTesting disaster recovery for database system...")
    test_result = await disaster_recovery.test_disaster_recovery("database_primary")
    
    if 'error' not in test_result:
        print(f"✓ DR test completed:")
        print(f"  - Test Duration: {test_result['test_duration_minutes']} minutes")
        print(f"  - RTO Target Met: {test_result['rto_met']}")
        print(f"  - Validation Passed: {test_result['validation_passed']}")
    
    print("\n7. CONFIGURATION DRIFT DETECTION")
    print("-" * 50)
    
    # Get drift statistics
    drift_stats = await drift_detection.get_drift_statistics()
    
    print(f"✓ Configuration drift monitoring:")
    print(f"  - Total Baselines: {drift_stats.get('total_baselines', 0)}")
    print(f"  - Total Detections: {drift_stats.get('total_detections', 0)}")
    print(f"  - Successful Remediations: {drift_stats.get('successful_remediations', 0)}")
    print(f"  - Remediation Success Rate: {drift_stats.get('remediation_success_rate', 0):.1f}%")
    print(f"  - Auto-remediation: {'Enabled' if drift_stats.get('auto_remediation_enabled') else 'Disabled'}")
    print(f"  - Monitoring: {'Enabled' if drift_stats.get('monitoring_enabled') else 'Disabled'}")
    
    # Show configuration type breakdown
    type_breakdown = drift_stats.get('type_breakdown', {})
    if type_breakdown:
        print("\n  Configuration types monitored:")
        for config_type, count in type_breakdown.items():
            print(f"    - {config_type}: {count} baselines")
    
    print("\n8. INFRASTRUCTURE STATUS SUMMARY")
    print("-" * 50)
    
    # Get overall infrastructure status
    status = infrastructure.get_infrastructure_status()
    
    print(f"✓ Infrastructure Status: {status['status'].upper()}")
    print(f"  - Uptime: {status['uptime_percentage']:.2f}%")
    print(f"  - Response Time: {status['response_time_ms']:.1f}ms")
    print(f"  - Error Rate: {status['error_rate_percentage']:.2f}%")
    print(f"  - CPU Usage: {status['cpu_usage']:.1f}%")
    print(f"  - Memory Usage: {status['memory_usage']:.1f}%")
    print(f"  - Cost per Hour: ${status['cost_per_hour']:.2f}")
    print(f"  - Optimization Actions: {status['optimization_actions_count']}")
    
    # Check if uptime target is met
    uptime_target_met = status['uptime_percentage'] >= 99.99
    print(f"  - 99.99% Uptime Target: {'✓ MET' if uptime_target_met else '✗ NOT MET'}")
    
    print("\n9. REAL-TIME COST SAVINGS")
    print("-" * 50)
    
    # Get real-time savings
    savings = await cost_optimizer.get_real_time_savings()
    
    print(f"✓ Cost optimization results:")
    print(f"  - Monthly Savings: ${savings['monthly_savings']:.2f}")
    print(f"  - Annual Savings: ${savings['annual_savings']:.2f}")
    print(f"  - Optimizations Implemented: {savings['optimizations_implemented']}")
    
    # Calculate savings percentage
    if cost_report.total_monthly_cost > 0:
        actual_savings_percentage = (savings['monthly_savings'] / cost_report.total_monthly_cost) * 100
        target_met = actual_savings_percentage >= 30.0
        print(f"  - Actual Savings Percentage: {actual_savings_percentage:.1f}%")
        print(f"  - 30% Savings Target: {'✓ MET' if target_met else '✗ NOT MET'}")
    
    print("\n" + "=" * 80)
    print("INTELLIGENT INFRASTRUCTURE RESILIENCE DEMO COMPLETED")
    print("=" * 80)
    
    # Summary of achievements
    print("\nKEY ACHIEVEMENTS:")
    print(f"✓ Auto-tuning infrastructure with {status['uptime_percentage']:.2f}% uptime")
    print(f"✓ Zero-downtime deployment in {deployment_result.duration_seconds:.1f} seconds")
    print(f"✓ 90-day capacity forecasting with {capacity_plan.accuracy_assessment.value} accuracy")
    print(f"✓ Multi-cloud cost optimization: ${savings['monthly_savings']:.2f}/month savings")
    print(f"✓ Disaster recovery: {recovery_result['recovery_time_minutes']}-minute RTO")
    print(f"✓ Configuration drift detection with auto-remediation")
    
    return {
        "infrastructure_status": status,
        "deployment_result": deployment_result,
        "capacity_plan": capacity_plan,
        "cost_report": cost_report,
        "recovery_result": recovery_result,
        "drift_stats": drift_stats,
        "cost_savings": savings
    }

async def demo_performance_scenarios():
    """Demonstrate performance under various scenarios"""
    print("\n" + "=" * 80)
    print("PERFORMANCE SCENARIO TESTING")
    print("=" * 80)
    
    infrastructure = IntelligentInfrastructureResilience()
    
    scenarios = [
        {"name": "Normal Load", "cpu": 45, "memory": 55, "response_time": 150, "error_rate": 0.1},
        {"name": "High Load", "cpu": 85, "memory": 80, "response_time": 400, "error_rate": 2.0},
        {"name": "Critical Load", "cpu": 95, "memory": 90, "response_time": 800, "error_rate": 5.0},
        {"name": "Recovery", "cpu": 60, "memory": 65, "response_time": 200, "error_rate": 0.5}
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']} Scenario")
        print("-" * 30)
        
        # Simulate scenario metrics
        with patch.object(infrastructure, 'collect_infrastructure_metrics') as mock_collect:
            from security.infrastructure.intelligent_infrastructure_resilience import InfrastructureMetrics
            
            mock_metrics = InfrastructureMetrics(
                timestamp=datetime.now(),
                cpu_usage=scenario['cpu'],
                memory_usage=scenario['memory'],
                disk_usage=70.0,
                network_io=1000.0,
                response_time=scenario['response_time'],
                error_rate=scenario['error_rate'],
                throughput=1000.0,
                availability=100.0 - scenario['error_rate'],
                cost_per_hour=15.0
            )
            mock_collect.return_value = mock_metrics
            
            # Test auto-tuning response
            result = await infrastructure.auto_tune_infrastructure()
            
            print(f"Metrics: CPU {scenario['cpu']}%, Memory {scenario['memory']}%, "
                  f"Response {scenario['response_time']}ms, Errors {scenario['error_rate']}%")
            print(f"Actions: {', '.join(result.get('actions_taken', ['None']))}")
            print(f"Performance Improvement: {result.get('performance_improvement', 0):.1f}%")
        
        await asyncio.sleep(1)  # Brief pause between scenarios

if __name__ == "__main__":
    async def main():
        try:
            # Run main demo
            demo_results = await demo_intelligent_infrastructure_resilience()
            
            # Run performance scenarios
            await demo_performance_scenarios()
            
            print(f"\nDemo completed successfully!")
            print(f"Results saved to demo_results.json")
            
            # Save results to file
            with open("demo_results.json", "w") as f:
                # Convert datetime objects to strings for JSON serialization
                serializable_results = {}
                for key, value in demo_results.items():
                    if hasattr(value, '__dict__'):
                        serializable_results[key] = str(value)
                    else:
                        serializable_results[key] = value
                
                json.dump(serializable_results, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    # Import patch for testing
    from unittest.mock import patch
    
    # Run the demo
    asyncio.run(main())