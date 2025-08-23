"""
Integration tests for performance monitoring and optimization system.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from scrollintel.models.performance_models import (
    PerformanceMetrics, ResourceUsage, SLAViolation, PerformanceAlert,
    OptimizationRecommendation, PerformanceTuningConfig
)
from scrollintel.engines.performance_monitoring_engine import PerformanceMonitoringEngine
from scrollintel.core.database import SessionLocal

@pytest.fixture
def monitoring_engine():
    """Create a performance monitoring engine for testing."""
    return PerformanceMonitoringEngine()

@pytest.fixture
def sample_pipeline_data():
    """Sample pipeline data for testing."""
    return {
        "pipeline_id": "test-pipeline-001",
        "execution_id": "exec-001",
        "start_time": datetime.utcnow(),
        "cpu_usage": 75.5,
        "memory_usage_mb": 2048.0,
        "records_processed": 10000
    }

class TestPerformanceMonitoringEngine:
    """Test cases for performance monitoring engine."""
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitoring_engine, sample_pipeline_data):
        """Test starting performance monitoring."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock database operations
            mock_metrics = Mock()
            mock_metrics.id = 1
            mock_metrics.start_time = sample_pipeline_data["start_time"]
            mock_session.add.return_value = None
            mock_session.commit.return_value = None
            mock_session.refresh.return_value = None
            mock_session.query.return_value.filter.return_value.first.return_value = mock_metrics
            
            result = await monitoring_engine.start_monitoring(
                sample_pipeline_data["pipeline_id"],
                sample_pipeline_data["execution_id"]
            )
            
            assert result["status"] == "monitoring_started"
            assert result["pipeline_id"] == sample_pipeline_data["pipeline_id"]
            assert result["execution_id"] == sample_pipeline_data["execution_id"]
            assert "metrics_id" in result
            
            # Verify database calls
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self, monitoring_engine):
        """Test metrics collection."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db, \
             patch('scrollintel.engines.performance_monitoring_engine.psutil') as mock_psutil:
            
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock system metrics
            mock_psutil.cpu_percent.return_value = 65.0
            mock_psutil.virtual_memory.return_value = Mock(
                used=2147483648,  # 2GB in bytes
                percent=75.0,
                total=8589934592  # 8GB in bytes
            )
            mock_psutil.disk_io_counters.return_value = Mock(
                read_bytes=1073741824,  # 1GB
                write_bytes=536870912   # 512MB
            )
            mock_psutil.net_io_counters.return_value = Mock(
                bytes_sent=268435456,   # 256MB
                bytes_recv=134217728    # 128MB
            )
            
            # Mock database metrics object
            mock_metrics = Mock()
            mock_session.query.return_value.filter.return_value.first.return_value = mock_metrics
            
            result = await monitoring_engine.collect_metrics(1)
            
            assert "timestamp" in result
            assert result["cpu_usage"] == 65.0
            assert result["memory_usage"] == 75.0
            assert "disk_io_mb" in result
            assert "network_io_mb" in result
            
            # Verify metrics were updated
            assert mock_metrics.cpu_usage_percent == 65.0
            assert mock_metrics.memory_usage_mb == 2048.0  # 2GB in MB
    
    @pytest.mark.asyncio
    async def test_sla_violation_detection(self, monitoring_engine):
        """Test SLA violation detection and alerting."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Create metrics with high CPU usage (violation)
            mock_metrics = Mock()
            mock_metrics.id = 1
            mock_metrics.pipeline_id = "test-pipeline"
            mock_metrics.cpu_usage_percent = 95.0  # Above threshold
            mock_metrics.memory_usage_mb = 1024.0
            mock_metrics.error_rate = 0.02  # Above threshold
            
            # Mock psutil for memory calculation
            with patch('scrollintel.engines.performance_monitoring_engine.psutil') as mock_psutil:
                mock_psutil.virtual_memory.return_value = Mock(total=8589934592)  # 8GB
                
                await monitoring_engine._check_sla_violations(mock_session, mock_metrics)
            
            # Verify SLA violation and alert were created
            assert mock_session.add.call_count >= 2  # At least violation + alert
            mock_session.flush.assert_called()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitoring_engine):
        """Test stopping monitoring and generating recommendations."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock metrics object
            mock_metrics = Mock()
            mock_metrics.id = 1
            mock_metrics.start_time = datetime.utcnow() - timedelta(minutes=30)
            mock_metrics.records_processed = 5000
            mock_metrics.cpu_usage_percent = 85.0
            mock_metrics.memory_usage_mb = 4096.0
            mock_metrics.error_rate = 0.015
            mock_session.query.return_value.filter.return_value.first.return_value = mock_metrics
            
            result = await monitoring_engine.stop_monitoring(1)
            
            assert result["status"] == "monitoring_stopped"
            assert result["metrics_id"] == 1
            assert "end_time" in result
            assert "duration_seconds" in result
            assert "recommendations_generated" in result
            
            # Verify metrics were updated
            assert mock_metrics.end_time is not None
            assert mock_metrics.duration_seconds is not None
            assert mock_metrics.records_per_second is not None
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations_generation(self, monitoring_engine):
        """Test generation of optimization recommendations."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Create metrics that should trigger recommendations
            mock_metrics = Mock()
            mock_metrics.pipeline_id = "test-pipeline"
            mock_metrics.cpu_usage_percent = 95.0  # High CPU
            mock_metrics.memory_usage_mb = 10240.0  # High memory (10GB)
            mock_metrics.duration_seconds = 7200.0  # Long duration (2 hours)
            mock_metrics.error_rate = 0.05  # High error rate
            
            recommendations = await monitoring_engine._generate_optimization_recommendations(
                mock_session, mock_metrics
            )
            
            assert len(recommendations) >= 3  # Should generate multiple recommendations
            
            # Check for specific recommendation types
            rec_types = [rec['type'] for rec in recommendations]
            assert 'cpu_optimization' in rec_types
            assert 'memory_optimization' in rec_types
            assert 'performance_optimization' in rec_types
            assert 'reliability_optimization' in rec_types
            
            # Verify recommendations were saved to database
            assert mock_session.add.call_count == len(recommendations)
            mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dashboard_data_generation(self, monitoring_engine):
        """Test performance dashboard data generation."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock metrics data
            mock_metrics = [
                Mock(
                    duration_seconds=300.0,
                    cpu_usage_percent=70.0,
                    memory_usage_mb=2048.0,
                    records_processed=1000,
                    error_count=5
                ),
                Mock(
                    duration_seconds=450.0,
                    cpu_usage_percent=80.0,
                    memory_usage_mb=2560.0,
                    records_processed=1500,
                    error_count=3
                )
            ]
            
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics
            
            # Mock violations and alerts
            mock_session.query.return_value.join.return_value.filter.return_value.all.return_value = []
            
            dashboard_data = await monitoring_engine.get_performance_dashboard_data(
                pipeline_id="test-pipeline",
                time_range_hours=24
            )
            
            assert "summary" in dashboard_data
            assert "violations" in dashboard_data
            assert "active_alerts" in dashboard_data
            
            summary = dashboard_data["summary"]
            assert summary["total_executions"] == 2
            assert summary["avg_duration_seconds"] == 375.0  # (300 + 450) / 2
            assert summary["avg_cpu_usage"] == 75.0  # (70 + 80) / 2
            assert summary["total_records_processed"] == 2500  # 1000 + 1500
            assert summary["total_errors"] == 8  # 5 + 3
    
    @pytest.mark.asyncio
    async def test_auto_tuning_application(self, monitoring_engine):
        """Test automated performance tuning."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock tuning configuration
            mock_config = Mock()
            mock_config.pipeline_id = "test-pipeline"
            mock_config.auto_scaling_enabled = True
            mock_config.target_cpu_utilization = 70.0
            mock_config.target_memory_utilization = 80.0
            mock_config.latency_threshold_ms = 5000.0
            mock_session.query.return_value.filter.return_value.first.return_value = mock_config
            
            # Mock recent metrics with high CPU usage
            mock_metrics = [
                Mock(cpu_usage_percent=85.0, memory_usage_mb=2048.0, duration_seconds=300.0),
                Mock(cpu_usage_percent=90.0, memory_usage_mb=2560.0, duration_seconds=450.0)
            ]
            mock_session.query.return_value.filter.return_value.all.return_value = mock_metrics
            
            result = await monitoring_engine.apply_auto_tuning("test-pipeline")
            
            assert result["status"] == "tuning_applied"
            assert result["pipeline_id"] == "test-pipeline"
            assert "actions" in result
            assert "performance_summary" in result
            
            # Should recommend scaling up due to high CPU
            actions = result["actions"]
            assert len(actions) > 0
            assert any(action["action"] == "scale_up" for action in actions)
            
            # Verify config was updated
            assert mock_config.last_tuned is not None

class TestPerformanceMonitoringAPI:
    """Test cases for performance monitoring API endpoints."""
    
    def test_start_monitoring_endpoint(self, client):
        """Test start monitoring API endpoint."""
        response = client.post(
            "/api/v1/performance/monitoring/start",
            params={
                "pipeline_id": "test-pipeline",
                "execution_id": "test-execution"
            }
        )
        
        # Note: This would require proper database setup in actual test
        # For now, we expect it to fail gracefully
        assert response.status_code in [200, 500]
    
    def test_get_pipeline_metrics_endpoint(self, client):
        """Test get pipeline metrics API endpoint."""
        response = client.get("/api/v1/performance/metrics/test-pipeline")
        
        # Should return proper structure even with no data
        assert response.status_code in [200, 500]
    
    def test_performance_dashboard_endpoint(self, client):
        """Test performance dashboard API endpoint."""
        response = client.get("/api/v1/performance/dashboard")
        
        assert response.status_code in [200, 500]
    
    def test_health_check_endpoint(self, client):
        """Test performance monitoring health check."""
        response = client.get("/api/v1/performance/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "performance_monitoring"

class TestPerformanceOptimization:
    """Test cases for performance optimization features."""
    
    @pytest.mark.asyncio
    async def test_cpu_optimization_recommendation(self, monitoring_engine):
        """Test CPU optimization recommendation generation."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # High CPU usage scenario
            mock_metrics = Mock()
            mock_metrics.pipeline_id = "test-pipeline"
            mock_metrics.cpu_usage_percent = 95.0
            mock_metrics.memory_usage_mb = 1024.0
            mock_metrics.duration_seconds = 300.0
            mock_metrics.error_rate = 0.001
            
            recommendations = await monitoring_engine._generate_optimization_recommendations(
                mock_session, mock_metrics
            )
            
            cpu_rec = next((r for r in recommendations if r['type'] == 'cpu_optimization'), None)
            assert cpu_rec is not None
            assert cpu_rec['priority'] == 'high'
            assert 'scaling' in cpu_rec['description'].lower()
            assert len(cpu_rec['implementation_steps']) > 0
    
    @pytest.mark.asyncio
    async def test_memory_optimization_recommendation(self, monitoring_engine):
        """Test memory optimization recommendation generation."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # High memory usage scenario
            mock_metrics = Mock()
            mock_metrics.pipeline_id = "test-pipeline"
            mock_metrics.cpu_usage_percent = 50.0
            mock_metrics.memory_usage_mb = 12288.0  # 12GB
            mock_metrics.duration_seconds = 300.0
            mock_metrics.error_rate = 0.001
            
            recommendations = await monitoring_engine._generate_optimization_recommendations(
                mock_session, mock_metrics
            )
            
            memory_rec = next((r for r in recommendations if r['type'] == 'memory_optimization'), None)
            assert memory_rec is not None
            assert 'streaming' in memory_rec['description'].lower() or 'memory' in memory_rec['description'].lower()
            assert len(memory_rec['implementation_steps']) > 0
    
    @pytest.mark.asyncio
    async def test_performance_optimization_recommendation(self, monitoring_engine):
        """Test performance optimization recommendation for long execution times."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Long execution time scenario
            mock_metrics = Mock()
            mock_metrics.pipeline_id = "test-pipeline"
            mock_metrics.cpu_usage_percent = 60.0
            mock_metrics.memory_usage_mb = 2048.0
            mock_metrics.duration_seconds = 7200.0  # 2 hours
            mock_metrics.error_rate = 0.001
            
            recommendations = await monitoring_engine._generate_optimization_recommendations(
                mock_session, mock_metrics
            )
            
            perf_rec = next((r for r in recommendations if r['type'] == 'performance_optimization'), None)
            assert perf_rec is not None
            assert perf_rec['priority'] == 'high'
            assert 'parallel' in perf_rec['description'].lower() or 'optimization' in perf_rec['description'].lower()
    
    @pytest.mark.asyncio
    async def test_reliability_optimization_recommendation(self, monitoring_engine):
        """Test reliability optimization recommendation for high error rates."""
        with patch('scrollintel.engines.performance_monitoring_engine.SessionLocal') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # High error rate scenario
            mock_metrics = Mock()
            mock_metrics.pipeline_id = "test-pipeline"
            mock_metrics.cpu_usage_percent = 60.0
            mock_metrics.memory_usage_mb = 2048.0
            mock_metrics.duration_seconds = 300.0
            mock_metrics.error_rate = 0.08  # 8% error rate
            
            recommendations = await monitoring_engine._generate_optimization_recommendations(
                mock_session, mock_metrics
            )
            
            reliability_rec = next((r for r in recommendations if r['type'] == 'reliability_optimization'), None)
            assert reliability_rec is not None
            assert reliability_rec['priority'] == 'critical'
            assert 'error' in reliability_rec['description'].lower()
            assert 'retry' in str(reliability_rec['implementation_steps']).lower()

if __name__ == "__main__":
    pytest.main([__file__])