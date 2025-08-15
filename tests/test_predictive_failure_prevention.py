"""
Tests for Predictive Failure Prevention Engine
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from scrollintel.core.predictive_failure_prevention import (
    HealthMonitor,
    FailurePredictor,
    DependencyMonitor,
    ResourceScaler,
    PredictiveFailurePreventionEngine,
    SystemHealthMetrics,
    AnomalyDetection,
    FailurePrediction,
    DependencyHealth,
    ResourceOptimization,
    PredictionConfidence,
    AnomalyType,
    ScalingAction,
    predictive_engine
)
from scrollintel.core.failure_prevention import FailureType, FailureEvent


class TestHealthMonitor:
    """Test the HealthMonitor component"""
    
    @pytest.fixture
    def health_monitor(self):
        return HealthMonitor(history_size=100)
    
    @pytest.fixture
    def sample_metrics(self):
        return SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=45.0,
            memory_usage=60.0,
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
    
    @pytest.mark.asyncio
    async def test_collect_comprehensive_metrics(self, health_monitor):
        """Test comprehensive metrics collection"""
        metrics = await health_monitor.collect_comprehensive_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, SystemHealthMetrics)
        assert metrics.cpu_usage >= 0
        assert metrics.memory_usage >= 0
        assert metrics.disk_usage >= 0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_add_metrics(self, health_monitor, sample_metrics):
        """Test adding metrics to history"""
        initial_size = len(health_monitor.metrics_history)
        health_monitor.add_metrics(sample_metrics)
        
        assert len(health_monitor.metrics_history) == initial_size + 1
        assert health_monitor.baseline_metrics is not None
        assert 'cpu_usage' in health_monitor.baseline_metrics
    
    def test_train_anomaly_detector(self, health_monitor, sample_metrics):
        """Test anomaly detector training"""
        # Add enough metrics for training
        for i in range(60):
            metrics = SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=45.0 + i * 0.5,
                memory_usage=60.0 + i * 0.3,
                disk_usage=30.0,
                network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
                disk_io={'read_bytes': 500, 'write_bytes': 300},
                active_connections=10,
                response_time_avg=0.5,
                response_time_p95=1.2,
                error_rate=0.1,
                request_rate=100.0,
                queue_depth=5,
                cache_hit_rate=0.85,
                database_connections=8,
                database_query_time=0.05,
                external_api_latency={'openai': 1.5},
                user_sessions=25,
                agent_processing_time=2.0
            )
            health_monitor.add_metrics(metrics)
        
        assert health_monitor.is_trained
    
    @pytest.mark.asyncio
    async def test_detect_anomalies_threshold(self, health_monitor, sample_metrics):
        """Test threshold-based anomaly detection"""
        # Create metrics with high CPU usage
        high_cpu_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=90.0,  # High CPU
            memory_usage=60.0,
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
        
        anomalies = await health_monitor.detect_anomalies(high_cpu_metrics)
        
        assert len(anomalies) > 0
        cpu_anomaly = next((a for a in anomalies if 'cpu_usage' in a.affected_metrics), None)
        assert cpu_anomaly is not None
        assert cpu_anomaly.anomaly_type == AnomalyType.CAPACITY_THRESHOLD
        assert cpu_anomaly.confidence == PredictionConfidence.HIGH
    
    @pytest.mark.asyncio
    async def test_detect_critical_memory_anomaly(self, health_monitor):
        """Test critical memory anomaly detection"""
        critical_memory_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=45.0,
            memory_usage=95.0,  # Critical memory usage
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
        
        anomalies = await health_monitor.detect_anomalies(critical_memory_metrics)
        
        memory_anomaly = next((a for a in anomalies if 'memory_usage' in a.affected_metrics), None)
        assert memory_anomaly is not None
        assert memory_anomaly.confidence == PredictionConfidence.CRITICAL
        assert memory_anomaly.time_to_failure is not None
        assert memory_anomaly.time_to_failure.total_seconds() <= 120  # 2 minutes


class TestFailurePredictor:
    """Test the FailurePredictor component"""
    
    @pytest.fixture
    def failure_predictor(self):
        return FailurePredictor()
    
    @pytest.fixture
    def sample_metrics(self):
        return SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=45.0,
            memory_usage=60.0,
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
    
    def test_extract_prediction_features(self, failure_predictor, sample_metrics):
        """Test feature extraction for prediction"""
        features = failure_predictor._extract_prediction_features(sample_metrics)
        
        assert isinstance(features, list)
        assert len(features) > 10  # Should have multiple features
        assert all(isinstance(f, (int, float)) for f in features)
    
    @pytest.mark.asyncio
    async def test_add_training_data(self, failure_predictor, sample_metrics):
        """Test adding training data"""
        initial_size = len(failure_predictor.feature_history)
        
        await failure_predictor.add_training_data(sample_metrics)
        
        assert len(failure_predictor.feature_history) == initial_size + 1
    
    @pytest.mark.asyncio
    async def test_add_training_data_with_failure(self, failure_predictor, sample_metrics):
        """Test adding training data with failure event"""
        failure_event = FailureEvent(
            failure_type=FailureType.CPU_OVERLOAD,
            timestamp=datetime.utcnow(),
            error_message="CPU overload detected",
            stack_trace="",
            context={}
        )
        
        initial_failure_count = len(failure_predictor.failure_history)
        
        await failure_predictor.add_training_data(sample_metrics, failure_event)
        
        assert len(failure_predictor.failure_history) == initial_failure_count + 1
    
    @pytest.mark.asyncio
    async def test_pattern_based_prediction_cpu_overload(self, failure_predictor):
        """Test pattern-based CPU overload prediction"""
        high_cpu_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=85.0,  # High CPU usage
            memory_usage=60.0,
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
        
        predictions = await failure_predictor.predict_failures(high_cpu_metrics)
        
        cpu_prediction = next((p for p in predictions if p.failure_type == FailureType.CPU_OVERLOAD), None)
        assert cpu_prediction is not None
        assert cpu_prediction.probability > 0
        assert 'Scale up CPU' in cpu_prediction.prevention_actions
    
    @pytest.mark.asyncio
    async def test_pattern_based_prediction_memory_exhaustion(self, failure_predictor):
        """Test pattern-based memory exhaustion prediction"""
        high_memory_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=45.0,
            memory_usage=90.0,  # High memory usage
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
        
        predictions = await failure_predictor.predict_failures(high_memory_metrics)
        
        memory_prediction = next((p for p in predictions if p.failure_type == FailureType.MEMORY_ERROR), None)
        assert memory_prediction is not None
        assert memory_prediction.confidence in [PredictionConfidence.MEDIUM, PredictionConfidence.HIGH, PredictionConfidence.CRITICAL]
        assert 'Scale up memory' in memory_prediction.prevention_actions
    
    def test_probability_to_confidence(self, failure_predictor):
        """Test probability to confidence conversion"""
        assert failure_predictor._probability_to_confidence(0.9) == PredictionConfidence.CRITICAL
        assert failure_predictor._probability_to_confidence(0.7) == PredictionConfidence.HIGH
        assert failure_predictor._probability_to_confidence(0.5) == PredictionConfidence.MEDIUM
        assert failure_predictor._probability_to_confidence(0.3) == PredictionConfidence.LOW


class TestDependencyMonitor:
    """Test the DependencyMonitor component"""
    
    @pytest.fixture
    def dependency_monitor(self):
        return DependencyMonitor()
    
    def test_register_dependency(self, dependency_monitor):
        """Test registering a new dependency"""
        initial_count = len(dependency_monitor.dependencies)
        
        dependency_monitor.register_dependency(
            'test_service',
            'https://test.example.com/health'
        )
        
        assert len(dependency_monitor.dependencies) == initial_count + 1
        assert 'test_service' in dependency_monitor.dependencies
    
    @pytest.mark.asyncio
    async def test_check_api_health_success(self, dependency_monitor):
        """Test successful API health check"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await dependency_monitor._check_api_health('https://test.example.com')
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_api_health_failure(self, dependency_monitor):
        """Test failed API health check"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await dependency_monitor._check_api_health('https://test.example.com')
            assert result is False
    
    @pytest.mark.asyncio
    async def test_check_dependency_health(self, dependency_monitor):
        """Test checking dependency health"""
        # Register a test dependency
        dependency_monitor.register_dependency(
            'test_api',
            'https://api.test.com/health'
        )
        
        with patch.object(dependency_monitor, '_check_api_health', return_value=True):
            health = await dependency_monitor.check_dependency_health('test_api')
            
            assert health.service_name == 'test_api'
            assert health.status == 'healthy'
            assert health.consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_dependency_failure_detection(self, dependency_monitor):
        """Test dependency failure detection and status update"""
        dependency_monitor.register_dependency(
            'failing_service',
            'https://failing.example.com'
        )
        
        with patch.object(dependency_monitor, '_check_api_health', return_value=False):
            # Simulate multiple failures
            for _ in range(3):
                health = await dependency_monitor.check_dependency_health('failing_service')
            
            assert health.status == 'failed'
            assert health.consecutive_failures >= 3
    
    def test_get_dependency_status(self, dependency_monitor):
        """Test getting dependency status summary"""
        status = dependency_monitor.get_dependency_status()
        
        assert isinstance(status, dict)
        # Should have default dependencies
        assert len(status) > 0
        
        for name, dep_status in status.items():
            assert 'status' in dep_status
            assert 'response_time' in dep_status
            assert 'availability' in dep_status


class TestResourceScaler:
    """Test the ResourceScaler component"""
    
    @pytest.fixture
    def resource_scaler(self):
        return ResourceScaler()
    
    @pytest.fixture
    def sample_metrics(self):
        return SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=45.0,
            memory_usage=60.0,
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
    
    @pytest.mark.asyncio
    async def test_analyze_current_resources_cpu_scale_up(self, resource_scaler):
        """Test CPU scale up recommendation"""
        high_cpu_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=80.0,  # Above scale up threshold
            memory_usage=60.0,
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
        
        optimizations = await resource_scaler._analyze_current_resources(high_cpu_metrics)
        
        cpu_optimization = next((o for o in optimizations if o.resource_type == 'cpu'), None)
        assert cpu_optimization is not None
        assert cpu_optimization.recommended_action == ScalingAction.SCALE_UP_CPU
    
    @pytest.mark.asyncio
    async def test_analyze_current_resources_memory_scale_up(self, resource_scaler):
        """Test memory scale up recommendation"""
        high_memory_metrics = SystemHealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=45.0,
            memory_usage=85.0,  # Above scale up threshold
            disk_usage=30.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            disk_io={'read_bytes': 500, 'write_bytes': 300},
            active_connections=10,
            response_time_avg=0.5,
            response_time_p95=1.2,
            error_rate=0.1,
            request_rate=100.0,
            queue_depth=5,
            cache_hit_rate=0.85,
            database_connections=8,
            database_query_time=0.05,
            external_api_latency={'openai': 1.5},
            user_sessions=25,
            agent_processing_time=2.0
        )
        
        optimizations = await resource_scaler._analyze_current_resources(high_memory_metrics)
        
        memory_optimization = next((o for o in optimizations if o.resource_type == 'memory'), None)
        assert memory_optimization is not None
        assert memory_optimization.recommended_action == ScalingAction.SCALE_UP_MEMORY
    
    @pytest.mark.asyncio
    async def test_execute_scaling_action(self, resource_scaler):
        """Test executing a scaling action"""
        optimization = ResourceOptimization(
            resource_type='cpu',
            current_usage=80.0,
            predicted_usage=85.0,
            recommended_action=ScalingAction.SCALE_UP_CPU,
            urgency=PredictionConfidence.HIGH,
            estimated_impact='Improved performance',
            cost_benefit={'cost_increase': 0.3, 'performance_gain': 0.8}
        )
        
        with patch.object(resource_scaler, '_scale_up_cpu', return_value=None) as mock_scale:
            result = await resource_scaler.execute_scaling_action(optimization)
            
            assert result is True
            mock_scale.assert_called_once()
            assert len(resource_scaler.scaling_history) > 0


class TestPredictiveFailurePreventionEngine:
    """Test the main PredictiveFailurePreventionEngine"""
    
    @pytest.fixture
    def engine(self):
        return PredictiveFailurePreventionEngine()
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine.health_monitor is not None
        assert engine.failure_predictor is not None
        assert engine.dependency_monitor is not None
        assert engine.resource_scaler is not None
        assert not engine.running
    
    def test_get_system_status(self, engine):
        """Test getting system status"""
        status = engine.get_system_status()
        
        assert isinstance(status, dict)
        assert 'engine_running' in status
        assert 'metrics_history_size' in status
        assert 'dependency_status' in status
        assert 'timestamp' in status
    
    @pytest.mark.asyncio
    async def test_get_health_report(self, engine):
        """Test getting comprehensive health report"""
        with patch.object(engine.health_monitor, 'collect_comprehensive_metrics') as mock_collect:
            mock_metrics = SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=45.0,
                memory_usage=60.0,
                disk_usage=30.0,
                network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
                disk_io={'read_bytes': 500, 'write_bytes': 300},
                active_connections=10,
                response_time_avg=0.5,
                response_time_p95=1.2,
                error_rate=0.1,
                request_rate=100.0,
                queue_depth=5,
                cache_hit_rate=0.85,
                database_connections=8,
                database_query_time=0.05,
                external_api_latency={'openai': 1.5},
                user_sessions=25,
                agent_processing_time=2.0
            )
            mock_collect.return_value = mock_metrics
            
            report = await engine.get_health_report()
            
            assert isinstance(report, dict)
            assert 'current_metrics' in report
            assert 'system_status' in report
            assert 'failure_predictions' in report
            assert 'dependency_health' in report
    
    @pytest.mark.asyncio
    async def test_take_preventive_actions_anomaly(self, engine):
        """Test taking preventive actions for anomalies"""
        anomaly = AnomalyDetection(
            anomaly_type=AnomalyType.CAPACITY_THRESHOLD,
            confidence=PredictionConfidence.CRITICAL,
            timestamp=datetime.utcnow(),
            affected_metrics=['cpu_usage'],
            anomaly_score=-0.9,
            description="Critical CPU usage",
            recommended_actions=['Scale up CPU resources', 'Optimize processes'],
            predicted_impact="System failure imminent",
            time_to_failure=timedelta(minutes=2)
        )
        
        with patch.object(engine, '_execute_action', return_value=None) as mock_execute:
            await engine._take_preventive_actions([anomaly], [], [])
            
            # Should execute actions for critical anomaly
            assert mock_execute.call_count >= 1
            assert len(engine.prevention_history) > 0
    
    @pytest.mark.asyncio
    async def test_take_preventive_actions_prediction(self, engine):
        """Test taking preventive actions for predictions"""
        prediction = FailurePrediction(
            failure_type=FailureType.MEMORY_ERROR,
            confidence=PredictionConfidence.HIGH,
            predicted_time=datetime.utcnow() + timedelta(minutes=5),
            affected_components=['application_server'],
            root_cause_analysis={'memory_usage': 90.0},
            prevention_actions=['Scale up memory', 'Clear caches'],
            impact_assessment='Application crashes likely',
            probability=0.8
        )
        
        with patch.object(engine, '_execute_action', return_value=None) as mock_execute:
            await engine._take_preventive_actions([], [prediction], [])
            
            # Should execute actions for high-confidence prediction
            assert mock_execute.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_execute_action_scale_up_cpu(self, engine):
        """Test executing scale up CPU action"""
        with patch.object(engine.resource_scaler, '_scale_up_cpu', return_value=None) as mock_scale:
            await engine._execute_action('Scale up CPU resources')
            mock_scale.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_action_clear_cache(self, engine):
        """Test executing clear cache action"""
        with patch.object(engine.resource_scaler, '_clear_cache', return_value=None) as mock_clear:
            await engine._execute_action('Clear caches')
            mock_clear.assert_called_once()


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_prediction_cycle(self):
        """Test a complete prediction and prevention cycle"""
        engine = PredictiveFailurePreventionEngine()
        
        # Mock high resource usage metrics
        with patch.object(engine.health_monitor, 'collect_comprehensive_metrics') as mock_collect:
            high_usage_metrics = SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=88.0,  # High CPU
                memory_usage=92.0,  # High memory
                disk_usage=30.0,
                network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
                disk_io={'read_bytes': 500, 'write_bytes': 300},
                active_connections=10,
                response_time_avg=2.5,  # Slow response
                response_time_p95=5.0,
                error_rate=0.3,  # High error rate
                request_rate=100.0,
                queue_depth=15,  # High queue depth
                cache_hit_rate=0.45,  # Low cache hit rate
                database_connections=8,
                database_query_time=1.5,  # Slow queries
                external_api_latency={'openai': 4.0},  # High latency
                user_sessions=25,
                agent_processing_time=8.0  # Slow processing
            )
            mock_collect.return_value = high_usage_metrics
            
            # Add metrics to history
            engine.health_monitor.add_metrics(high_usage_metrics)
            
            # Detect anomalies
            anomalies = await engine.health_monitor.detect_anomalies(high_usage_metrics)
            
            # Should detect multiple anomalies
            assert len(anomalies) > 0
            
            # Predict failures
            predictions = await engine.failure_predictor.predict_failures(high_usage_metrics)
            
            # Should predict multiple potential failures
            assert len(predictions) > 0
            
            # Analyze resource needs
            optimizations = await engine.resource_scaler.analyze_resource_needs(high_usage_metrics, predictions)
            
            # Should recommend multiple optimizations
            assert len(optimizations) > 0
            
            # Verify we have CPU and memory optimizations
            cpu_opt = next((o for o in optimizations if o.resource_type == 'cpu'), None)
            memory_opt = next((o for o in optimizations if o.resource_type == 'memory'), None)
            
            assert cpu_opt is not None
            assert memory_opt is not None
            assert cpu_opt.recommended_action == ScalingAction.SCALE_UP_CPU
            assert memory_opt.recommended_action == ScalingAction.SCALE_UP_MEMORY
    
    @pytest.mark.asyncio
    async def test_dependency_failure_handling(self):
        """Test handling of dependency failures"""
        engine = PredictiveFailurePreventionEngine()
        
        # Register a test dependency
        engine.dependency_monitor.register_dependency(
            'critical_api',
            'https://critical.example.com/health'
        )
        
        # Mock dependency failure
        with patch.object(engine.dependency_monitor, '_check_api_health', return_value=False):
            # Check dependency multiple times to trigger failure
            for _ in range(3):
                await engine.dependency_monitor.check_dependency_health('critical_api')
            
            # Verify dependency is marked as failed
            status = engine.dependency_monitor.get_dependency_status()
            assert status['critical_api']['status'] == 'failed'
            assert status['critical_api']['consecutive_failures'] >= 3
    
    @pytest.mark.asyncio
    async def test_prevention_engine_lifecycle(self):
        """Test starting and stopping the prevention engine"""
        engine = PredictiveFailurePreventionEngine()
        
        # Initially not running
        assert not engine.running
        
        # Start engine
        await engine.start()
        assert engine.running
        assert engine.dependency_monitor.monitoring_active
        
        # Stop engine
        await engine.stop()
        assert not engine.running
        assert not engine.dependency_monitor.monitoring_active


@pytest.mark.asyncio
async def test_global_engine_functions():
    """Test global convenience functions"""
    from scrollintel.core.predictive_failure_prevention import (
        start_predictive_prevention,
        stop_predictive_prevention,
        get_prevention_status,
        get_health_report
    )
    
    # Test status function
    status = get_prevention_status()
    assert isinstance(status, dict)
    assert 'engine_running' in status
    
    # Test health report function
    with patch('scrollintel.core.predictive_failure_prevention.predictive_engine.get_health_report') as mock_report:
        mock_report.return_value = {'test': 'data'}
        report = await get_health_report()
        assert report == {'test': 'data'}


if __name__ == "__main__":
    pytest.main([__file__])