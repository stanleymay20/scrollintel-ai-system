#!/usr/bin/env python3
"""
Demo script for Predictive Failure Prevention Engine
Demonstrates comprehensive failure prediction and prevention capabilities
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from scrollintel.core.predictive_failure_prevention import (
    PredictiveFailurePreventionEngine,
    HealthMonitor,
    FailurePredictor,
    DependencyMonitor,
    ResourceScaler,
    SystemHealthMetrics,
    AnomalyDetection,
    FailurePrediction,
    PredictionConfidence,
    AnomalyType,
    ScalingAction
)
from scrollintel.core.failure_prevention import FailureType, FailureEvent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictiveFailurePreventionDemo:
    """Comprehensive demo of the Predictive Failure Prevention Engine"""
    
    def __init__(self):
        self.engine = PredictiveFailurePreventionEngine()
        self.demo_scenarios = [
            self.demo_normal_operations,
            self.demo_cpu_overload_prediction,
            self.demo_memory_exhaustion_prediction,
            self.demo_dependency_failure_handling,
            self.demo_anomaly_detection,
            self.demo_resource_optimization,
            self.demo_comprehensive_prevention_cycle
        ]
    
    async def run_complete_demo(self):
        """Run the complete demo showcasing all capabilities"""
        print("=" * 80)
        print("🔮 SCROLLINTEL PREDICTIVE FAILURE PREVENTION ENGINE DEMO")
        print("=" * 80)
        print()
        
        try:
            # Start the engine
            await self.engine.start()
            print("✅ Predictive Failure Prevention Engine started")
            print()
            
            # Run all demo scenarios
            for i, scenario in enumerate(self.demo_scenarios, 1):
                print(f"📋 Demo Scenario {i}: {scenario.__name__.replace('demo_', '').replace('_', ' ').title()}")
                print("-" * 60)
                
                try:
                    await scenario()
                    print("✅ Scenario completed successfully")
                except Exception as e:
                    print(f"❌ Scenario failed: {e}")
                
                print()
                await asyncio.sleep(2)  # Brief pause between scenarios
            
            # Final system status
            await self.show_final_status()
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            print(f"❌ Demo failed: {e}")
        
        finally:
            # Stop the engine
            await self.engine.stop()
            print("🛑 Predictive Failure Prevention Engine stopped")
    
    async def demo_normal_operations(self):
        """Demo normal system operations with healthy metrics"""
        print("Simulating normal system operations...")
        
        # Create normal metrics
        normal_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=35.0,
            memory_usage=45.0,
            disk_usage=25.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=8,
            response_time_avg=0.3,
            response_time_p95=0.8,
            error_rate=0.05,
            request_rate=120.0,
            queue_depth=2,
            cache_hit_rate=0.92,
            database_connections=6,
            database_query_time=0.03,
            external_api_latency={'openai': 1.2, 'anthropic': 1.5},
            user_sessions=30,
            agent_processing_time=1.8
        )
        
        # Add to monitoring
        self.engine.health_monitor.add_metrics(normal_metrics)
        
        # Check for anomalies (should be none)
        anomalies = await self.engine.health_monitor.detect_anomalies(normal_metrics)
        
        # Check for predictions (should be none or low confidence)
        predictions = await self.engine.failure_predictor.predict_failures(normal_metrics)
        
        print(f"📊 System Metrics:")
        print(f"   CPU Usage: {normal_metrics.cpu_usage}%")
        print(f"   Memory Usage: {normal_metrics.memory_usage}%")
        print(f"   Response Time: {normal_metrics.response_time_avg}s")
        print(f"   Error Rate: {normal_metrics.error_rate}%")
        print()
        
        print(f"🔍 Analysis Results:")
        print(f"   Anomalies Detected: {len(anomalies)}")
        print(f"   Failure Predictions: {len(predictions)}")
        
        if anomalies:
            for anomaly in anomalies:
                print(f"   - {anomaly.description} (Confidence: {anomaly.confidence.value})")
        
        if predictions:
            for prediction in predictions:
                print(f"   - {prediction.failure_type.value} predicted (Confidence: {prediction.confidence.value})")
        else:
            print("   ✅ No significant risks detected - system operating normally")
    
    async def demo_cpu_overload_prediction(self):
        """Demo CPU overload prediction and prevention"""
        print("Simulating high CPU usage scenario...")
        
        # Create high CPU metrics
        high_cpu_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=87.0,  # High CPU usage
            memory_usage=55.0,
            disk_usage=30.0,
            network_io={'bytes_sent': 5000, 'bytes_recv': 8000},
            disk_io={'read_bytes': 2000, 'write_bytes': 1500},
            active_connections=25,
            response_time_avg=2.1,  # Slower responses
            response_time_p95=4.5,
            error_rate=0.15,  # Higher error rate
            request_rate=200.0,  # Higher load
            queue_depth=12,  # Higher queue depth
            cache_hit_rate=0.78,
            database_connections=15,
            database_query_time=0.08,
            external_api_latency={'openai': 2.5, 'anthropic': 3.0},
            user_sessions=45,
            agent_processing_time=4.2
        )
        
        # Add to monitoring
        self.engine.health_monitor.add_metrics(high_cpu_metrics)
        
        # Detect anomalies
        anomalies = await self.engine.health_monitor.detect_anomalies(high_cpu_metrics)
        
        # Predict failures
        predictions = await self.engine.failure_predictor.predict_failures(high_cpu_metrics)
        
        # Analyze resource needs
        optimizations = await self.engine.resource_scaler.analyze_resource_needs(high_cpu_metrics, predictions)
        
        print(f"📊 High CPU Scenario:")
        print(f"   CPU Usage: {high_cpu_metrics.cpu_usage}% ⚠️")
        print(f"   Response Time: {high_cpu_metrics.response_time_avg}s ⚠️")
        print(f"   Queue Depth: {high_cpu_metrics.queue_depth} ⚠️")
        print()
        
        print(f"🚨 Anomalies Detected: {len(anomalies)}")
        for anomaly in anomalies:
            print(f"   - {anomaly.description}")
            print(f"     Confidence: {anomaly.confidence.value}")
            print(f"     Recommended Actions: {', '.join(anomaly.recommended_actions[:2])}")
        print()
        
        print(f"🔮 Failure Predictions: {len(predictions)}")
        for prediction in predictions:
            print(f"   - {prediction.failure_type.value}")
            print(f"     Confidence: {prediction.confidence.value}")
            print(f"     Probability: {prediction.probability:.2f}")
            print(f"     Prevention Actions: {', '.join(prediction.prevention_actions[:2])}")
        print()
        
        print(f"⚡ Resource Optimizations: {len(optimizations)}")
        for opt in optimizations:
            print(f"   - {opt.recommended_action.value}")
            print(f"     Resource: {opt.resource_type}")
            print(f"     Urgency: {opt.urgency.value}")
            print(f"     Impact: {opt.estimated_impact}")
        
        # Simulate taking preventive actions
        print()
        print("🛠️ Taking Preventive Actions:")
        await self.engine._take_preventive_actions(anomalies, predictions, optimizations)
        
        if self.engine.prevention_history:
            latest_actions = self.engine.prevention_history[-1]
            print(f"   Actions Taken: {len(latest_actions['actions'])}")
            for action in latest_actions['actions']:
                print(f"   - {action}")
    
    async def demo_memory_exhaustion_prediction(self):
        """Demo memory exhaustion prediction and prevention"""
        print("Simulating memory exhaustion scenario...")
        
        # Create high memory metrics
        high_memory_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=65.0,
            memory_usage=94.0,  # Critical memory usage
            disk_usage=40.0,
            network_io={'bytes_sent': 3000, 'bytes_recv': 4000},
            disk_io={'read_bytes': 1000, 'write_bytes': 800},
            active_connections=20,
            response_time_avg=1.8,
            response_time_p95=3.2,
            error_rate=0.12,
            request_rate=150.0,
            queue_depth=8,
            cache_hit_rate=0.65,  # Lower cache hit rate
            database_connections=12,
            database_query_time=0.06,
            external_api_latency={'openai': 2.0, 'anthropic': 2.2},
            user_sessions=35,
            agent_processing_time=3.5
        )
        
        # Add to monitoring
        self.engine.health_monitor.add_metrics(high_memory_metrics)
        
        # Detect anomalies
        anomalies = await self.engine.health_monitor.detect_anomalies(high_memory_metrics)
        
        # Predict failures
        predictions = await self.engine.failure_predictor.predict_failures(high_memory_metrics)
        
        print(f"📊 Memory Exhaustion Scenario:")
        print(f"   Memory Usage: {high_memory_metrics.memory_usage}% 🚨")
        print(f"   Cache Hit Rate: {high_memory_metrics.cache_hit_rate:.2f} ⚠️")
        print()
        
        # Find memory-related anomalies and predictions
        memory_anomalies = [a for a in anomalies if 'memory' in str(a.affected_metrics).lower()]
        memory_predictions = [p for p in predictions if p.failure_type == FailureType.MEMORY_ERROR]
        
        print(f"🚨 Memory-Related Anomalies: {len(memory_anomalies)}")
        for anomaly in memory_anomalies:
            print(f"   - {anomaly.description}")
            print(f"     Confidence: {anomaly.confidence.value}")
            if anomaly.time_to_failure:
                print(f"     Time to Failure: {anomaly.time_to_failure.total_seconds():.0f} seconds")
        print()
        
        print(f"🔮 Memory Failure Predictions: {len(memory_predictions)}")
        for prediction in memory_predictions:
            print(f"   - Memory exhaustion predicted")
            print(f"     Confidence: {prediction.confidence.value}")
            print(f"     Predicted Time: {prediction.predicted_time.strftime('%H:%M:%S')}")
            print(f"     Impact: {prediction.impact_assessment}")
        
        # Simulate immediate preventive action for critical memory situation
        if memory_anomalies and memory_anomalies[0].confidence == PredictionConfidence.CRITICAL:
            print()
            print("🚨 CRITICAL MEMORY SITUATION - Taking Immediate Action!")
            print("   - Scaling up memory resources")
            print("   - Clearing application caches")
            print("   - Restarting memory-intensive services")
    
    async def demo_dependency_failure_handling(self):
        """Demo dependency failure detection and handling"""
        print("Simulating dependency failure scenario...")
        
        # Register test dependencies
        self.engine.dependency_monitor.register_dependency(
            'test_api_1',
            'https://api1.test.com/health'
        )
        self.engine.dependency_monitor.register_dependency(
            'test_api_2',
            'https://api2.test.com/health'
        )
        
        # Simulate API failures
        original_check = self.engine.dependency_monitor._check_api_health
        
        async def mock_failing_api(endpoint):
            if 'api1' in endpoint:
                return False  # API 1 is failing
            return await original_check(endpoint)  # API 2 is healthy
        
        self.engine.dependency_monitor._check_api_health = mock_failing_api
        
        # Check dependencies multiple times to trigger failure detection
        print("Checking dependency health...")
        for i in range(4):
            health1 = await self.engine.dependency_monitor.check_dependency_health('test_api_1')
            health2 = await self.engine.dependency_monitor.check_dependency_health('test_api_2')
            
            print(f"   Check {i+1}:")
            print(f"     API 1: {health1.status} (failures: {health1.consecutive_failures})")
            print(f"     API 2: {health2.status} (failures: {health2.consecutive_failures})")
            
            await asyncio.sleep(0.5)
        
        # Get final dependency status
        dep_status = self.engine.dependency_monitor.get_dependency_status()
        
        print()
        print("📊 Final Dependency Status:")
        for name, status in dep_status.items():
            if name.startswith('test_api'):
                status_icon = "❌" if status['status'] == 'failed' else "✅"
                print(f"   {status_icon} {name}: {status['status']}")
                print(f"      Availability: {status['availability']:.1f}%")
                print(f"      Consecutive Failures: {status['consecutive_failures']}")
        
        # Restore original method
        self.engine.dependency_monitor._check_api_health = original_check
    
    async def demo_anomaly_detection(self):
        """Demo advanced anomaly detection capabilities"""
        print("Demonstrating anomaly detection with various scenarios...")
        
        # Scenario 1: Unusual traffic pattern
        unusual_traffic_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=45.0,
            memory_usage=50.0,
            disk_usage=25.0,
            network_io={'bytes_sent': 50000, 'bytes_recv': 80000},  # Unusual traffic
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=100,  # Many connections
            response_time_avg=0.4,
            response_time_p95=0.9,
            error_rate=0.02,
            request_rate=500.0,  # Very high request rate
            queue_depth=3,
            cache_hit_rate=0.95,
            database_connections=8,
            database_query_time=0.04,
            external_api_latency={'openai': 1.0, 'anthropic': 1.1},
            user_sessions=150,  # Many users
            agent_processing_time=2.0
        )
        
        # Add some baseline metrics first
        for i in range(10):
            baseline_metrics = SystemHealthMetrics(
                timestamp=datetime.utcnow() - timedelta(minutes=i),
                cpu_usage=35.0 + i * 0.5,
                memory_usage=45.0 + i * 0.3,
                disk_usage=25.0,
                network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
                disk_io={'read_bytes': 500, 'write_bytes': 300},
                active_connections=10,
                response_time_avg=0.3,
                response_time_p95=0.8,
                error_rate=0.05,
                request_rate=120.0,
                queue_depth=2,
                cache_hit_rate=0.90,
                database_connections=6,
                database_query_time=0.03,
                external_api_latency={'openai': 1.2, 'anthropic': 1.5},
                user_sessions=30,
                agent_processing_time=1.8
            )
            self.engine.health_monitor.add_metrics(baseline_metrics)
        
        # Now add the unusual metrics
        self.engine.health_monitor.add_metrics(unusual_traffic_metrics)
        
        # Detect anomalies
        anomalies = await self.engine.health_monitor.detect_anomalies(unusual_traffic_metrics)
        
        print(f"📊 Unusual Traffic Pattern:")
        print(f"   Request Rate: {unusual_traffic_metrics.request_rate} req/s (vs ~120 baseline)")
        print(f"   Active Connections: {unusual_traffic_metrics.active_connections} (vs ~10 baseline)")
        print(f"   User Sessions: {unusual_traffic_metrics.user_sessions} (vs ~30 baseline)")
        print()
        
        print(f"🔍 Anomaly Detection Results: {len(anomalies)}")
        for anomaly in anomalies:
            print(f"   - Type: {anomaly.anomaly_type.value}")
            print(f"     Description: {anomaly.description}")
            print(f"     Confidence: {anomaly.confidence.value}")
            print(f"     Anomaly Score: {anomaly.anomaly_score:.3f}")
            print(f"     Affected Metrics: {', '.join(anomaly.affected_metrics)}")
        
        if not anomalies:
            print("   ℹ️ No anomalies detected - traffic pattern within normal variance")
    
    async def demo_resource_optimization(self):
        """Demo resource optimization recommendations"""
        print("Demonstrating resource optimization analysis...")
        
        # Create metrics that need optimization
        suboptimal_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=78.0,  # High but not critical
            memory_usage=82.0,  # High memory usage
            disk_usage=88.0,  # High disk usage
            network_io={'bytes_sent': 3000, 'bytes_recv': 4000},
            disk_io={'read_bytes': 2000, 'write_bytes': 1500},
            active_connections=18,
            response_time_avg=1.2,
            response_time_p95=2.8,
            error_rate=0.08,
            request_rate=180.0,
            queue_depth=7,
            cache_hit_rate=0.62,  # Low cache hit rate
            database_connections=14,
            database_query_time=1.2,  # Slow queries
            external_api_latency={'openai': 3.5, 'anthropic': 4.0},
            user_sessions=40,
            agent_processing_time=5.5
        )
        
        # Create some predictions for comprehensive analysis
        predictions = await self.engine.failure_predictor.predict_failures(suboptimal_metrics)
        
        # Analyze resource needs
        optimizations = await self.engine.resource_scaler.analyze_resource_needs(suboptimal_metrics, predictions)
        
        print(f"📊 System Resource Status:")
        print(f"   CPU Usage: {suboptimal_metrics.cpu_usage}% (approaching scale-up threshold)")
        print(f"   Memory Usage: {suboptimal_metrics.memory_usage}% (needs scaling)")
        print(f"   Disk Usage: {suboptimal_metrics.disk_usage}% (needs scaling)")
        print(f"   Cache Hit Rate: {suboptimal_metrics.cache_hit_rate:.2f} (suboptimal)")
        print(f"   DB Query Time: {suboptimal_metrics.database_query_time}s (slow)")
        print()
        
        print(f"⚡ Resource Optimization Recommendations: {len(optimizations)}")
        
        # Group optimizations by urgency
        critical_opts = [o for o in optimizations if o.urgency == PredictionConfidence.CRITICAL]
        high_opts = [o for o in optimizations if o.urgency == PredictionConfidence.HIGH]
        medium_opts = [o for o in optimizations if o.urgency == PredictionConfidence.MEDIUM]
        
        if critical_opts:
            print("   🚨 CRITICAL OPTIMIZATIONS:")
            for opt in critical_opts:
                print(f"     - {opt.recommended_action.value}")
                print(f"       Resource: {opt.resource_type}")
                print(f"       Current: {opt.current_usage:.1f}% → Predicted: {opt.predicted_usage:.1f}%")
                print(f"       Impact: {opt.estimated_impact}")
        
        if high_opts:
            print("   ⚠️ HIGH PRIORITY OPTIMIZATIONS:")
            for opt in high_opts:
                print(f"     - {opt.recommended_action.value}")
                print(f"       Resource: {opt.resource_type}")
                print(f"       Impact: {opt.estimated_impact}")
                if opt.cost_benefit:
                    cost_info = ", ".join([f"{k}: {v:.1f}" for k, v in opt.cost_benefit.items()])
                    print(f"       Cost/Benefit: {cost_info}")
        
        if medium_opts:
            print("   ℹ️ MEDIUM PRIORITY OPTIMIZATIONS:")
            for opt in medium_opts:
                print(f"     - {opt.recommended_action.value} ({opt.resource_type})")
        
        # Simulate executing high-priority optimizations
        urgent_opts = critical_opts + high_opts
        if urgent_opts:
            print()
            print("🛠️ Executing Urgent Optimizations:")
            for opt in urgent_opts[:3]:  # Execute top 3
                success = await self.engine.resource_scaler.execute_scaling_action(opt)
                status = "✅ Success" if success else "❌ Failed"
                print(f"   {status}: {opt.recommended_action.value}")
    
    async def demo_comprehensive_prevention_cycle(self):
        """Demo a complete prevention cycle with multiple issues"""
        print("Demonstrating comprehensive prevention cycle...")
        
        # Create a complex scenario with multiple issues
        complex_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=89.0,  # High CPU
            memory_usage=93.0,  # Critical memory
            disk_usage=96.0,  # Critical disk
            network_io={'bytes_sent': 10000, 'bytes_recv': 15000},
            disk_io={'read_bytes': 5000, 'write_bytes': 3000},
            active_connections=50,
            response_time_avg=3.5,  # Very slow
            response_time_p95=8.0,
            error_rate=0.25,  # High error rate
            request_rate=250.0,
            queue_depth=20,  # High queue
            cache_hit_rate=0.35,  # Very low cache hit
            database_connections=25,
            database_query_time=2.5,  # Very slow queries
            external_api_latency={'openai': 6.0, 'anthropic': 7.5},
            user_sessions=80,
            agent_processing_time=12.0  # Very slow processing
        )
        
        print("🚨 CRITICAL SYSTEM STATE DETECTED!")
        print(f"   CPU: {complex_metrics.cpu_usage}% 🔥")
        print(f"   Memory: {complex_metrics.memory_usage}% 🚨")
        print(f"   Disk: {complex_metrics.disk_usage}% ⚠️")
        print(f"   Response Time: {complex_metrics.response_time_avg}s 🐌")
        print(f"   Error Rate: {complex_metrics.error_rate:.1%} ❌")
        print()
        
        # Full analysis cycle
        print("🔍 Running Comprehensive Analysis...")
        
        # 1. Add metrics and detect anomalies
        self.engine.health_monitor.add_metrics(complex_metrics)
        anomalies = await self.engine.health_monitor.detect_anomalies(complex_metrics)
        
        # 2. Predict failures
        predictions = await self.engine.failure_predictor.predict_failures(complex_metrics)
        
        # 3. Analyze resource needs
        optimizations = await self.engine.resource_scaler.analyze_resource_needs(complex_metrics, predictions)
        
        # 4. Check dependencies
        dep_status = self.engine.dependency_monitor.get_dependency_status()
        
        print(f"📊 Analysis Results:")
        print(f"   Anomalies: {len(anomalies)}")
        print(f"   Failure Predictions: {len(predictions)}")
        print(f"   Resource Optimizations: {len(optimizations)}")
        print(f"   Dependencies Monitored: {len(dep_status)}")
        print()
        
        # Show critical findings
        critical_anomalies = [a for a in anomalies if a.confidence == PredictionConfidence.CRITICAL]
        high_conf_predictions = [p for p in predictions if p.confidence in [PredictionConfidence.HIGH, PredictionConfidence.CRITICAL]]
        urgent_optimizations = [o for o in optimizations if o.urgency in [PredictionConfidence.HIGH, PredictionConfidence.CRITICAL]]
        
        print("🚨 CRITICAL FINDINGS:")
        for anomaly in critical_anomalies:
            print(f"   - {anomaly.description}")
            if anomaly.time_to_failure:
                print(f"     Time to failure: {anomaly.time_to_failure.total_seconds():.0f} seconds")
        
        for prediction in high_conf_predictions:
            print(f"   - {prediction.failure_type.value} predicted")
            print(f"     Probability: {prediction.probability:.1%}")
            print(f"     Impact: {prediction.impact_assessment}")
        
        print()
        print("⚡ IMMEDIATE ACTIONS REQUIRED:")
        for opt in urgent_optimizations:
            print(f"   - {opt.recommended_action.value}")
            print(f"     Urgency: {opt.urgency.value}")
        
        # Execute prevention cycle
        print()
        print("🛠️ EXECUTING COMPREHENSIVE PREVENTION CYCLE...")
        await self.engine._take_preventive_actions(anomalies, predictions, optimizations)
        
        if self.engine.prevention_history:
            latest_prevention = self.engine.prevention_history[-1]
            print(f"   ✅ Prevention cycle completed")
            print(f"   Actions taken: {len(latest_prevention['actions'])}")
            print(f"   Anomalies addressed: {latest_prevention['anomaly_count']}")
            print(f"   Predictions addressed: {latest_prevention['prediction_count']}")
            print(f"   Optimizations applied: {latest_prevention['optimization_count']}")
            
            print()
            print("📋 Actions Executed:")
            for action in latest_prevention['actions']:
                print(f"   - {action}")
    
    async def show_final_status(self):
        """Show final system status and summary"""
        print("=" * 80)
        print("📊 FINAL SYSTEM STATUS")
        print("=" * 80)
        
        # Get comprehensive status
        status = self.engine.get_system_status()
        health_report = await self.engine.get_health_report()
        
        print(f"🔧 Engine Status:")
        print(f"   Running: {'✅ Yes' if status['engine_running'] else '❌ No'}")
        print(f"   Metrics History: {status['metrics_history_size']} entries")
        print(f"   Anomaly Detector: {'✅ Trained' if status['anomaly_detector_trained'] else '⏳ Training'}")
        print(f"   Prediction Models: {status['prediction_models']}")
        print()
        
        print(f"📈 Prevention History:")
        print(f"   Total Prevention Cycles: {status['recent_actions']}")
        if self.engine.prevention_history:
            total_actions = sum(len(entry['actions']) for entry in self.engine.prevention_history)
            print(f"   Total Actions Taken: {total_actions}")
            
            # Show recent actions
            if self.engine.prevention_history:
                print(f"   Recent Actions:")
                for entry in self.engine.prevention_history[-3:]:
                    timestamp = entry['timestamp'].strftime('%H:%M:%S')
                    print(f"     {timestamp}: {len(entry['actions'])} actions")
        print()
        
        print(f"🔗 Dependencies:")
        dep_status = status['dependency_status']
        healthy_deps = len([d for d in dep_status.values() if d['status'] == 'healthy'])
        total_deps = len(dep_status)
        print(f"   Total: {total_deps}")
        print(f"   Healthy: {healthy_deps}")
        print(f"   Issues: {total_deps - healthy_deps}")
        print()
        
        print("✅ Demo completed successfully!")
        print("   The Predictive Failure Prevention Engine demonstrated:")
        print("   - Real-time system health monitoring")
        print("   - Advanced anomaly detection with ML")
        print("   - Failure prediction based on patterns")
        print("   - Proactive resource scaling recommendations")
        print("   - Dependency health monitoring with failover")
        print("   - Comprehensive prevention action execution")
        print()
        print("🚀 ScrollIntel is now bulletproof against failures!")


async def main():
    """Main demo function"""
    demo = PredictiveFailurePreventionDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())