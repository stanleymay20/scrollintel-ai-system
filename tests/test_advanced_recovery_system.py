"""
Tests for Advanced Recovery and Self-Healing System.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from scrollintel.core.advanced_recovery_system import (
    AdvancedRecoverySystem,
    DependencyNode,
    PerformanceMetrics,
    MaintenanceTask,
    RecoveryAction,
    SystemHealth,
    advanced_recovery_system
)


class TestAdvancedRecoverySystem:
    """Test the Advanced Recovery System."""
    
    @pytest.fixture
    def recovery_system(self):
        """Create a test recovery system."""
        system = AdvancedRecoverySystem()
        system.auto_recovery_enabled = True
        return system
    
    @pytest.mark.asyncio
    async def test_health_check_database(self, recovery_system):
        """Test database health check."""
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            mock_memory.return_value.percent = 50.0
            mock_cpu.return_value = 40.0
            
            health_score = await recovery_system.perform_health_check('database')
            
            assert 0.0 <= health_score <= 1.0
            assert health_score > 0.5  # Should be healthy with low resource usage
    
    @pytest.mark.asyncio
    async def test_health_check_service(self, recovery_system):
        """Test service health check."""
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_cpu.return_value = 30.0
            mock_memory.return_value.percent = 40.0
            
            health_score = await recovery_system.perform_health_check('ai_services')
            
            assert 0.0 <= health_score <= 1.0
            assert health_score > 0.7  # Should be healthy
    
    @pytest.mark.asyncio
    async def test_health_check_filesystem(self, recovery_system):
        """Test filesystem health check."""
        with patch('psutil.disk_usage') as mock_disk:
            mock_disk.return_value.percent = 60.0
            
            health_score = await recovery_system.perform_health_check('file_system')
            
            assert 0.0 <= health_score <= 1.0
            assert health_score > 0.5  # Should be healthy with 60% disk usage
    
    @pytest.mark.asyncio
    async def test_autonomous_repair_success(self, recovery_system):
        """Test successful autonomous repair."""
        # Mock a failing node
        node = recovery_system.dependency_graph['database']
        node.health_score = 0.3
        node.failure_count = 2
        node.last_check = None  # Reset cooldown
        
        with patch.object(recovery_system, '_diagnose_failure', return_value='database_connection_failure'), \
             patch.object(recovery_system, '_execute_recovery_action', return_value=True), \
             patch.object(recovery_system, 'perform_health_check', return_value=0.8):
            
            success = await recovery_system.autonomous_system_repair('database')
            
            assert success is True
            # Recovery attempts should be reset to 0 on successful recovery
            assert node.recovery_attempts == 0
    
    @pytest.mark.asyncio
    async def test_autonomous_repair_failure(self, recovery_system):
        """Test failed autonomous repair."""
        # Mock a failing node
        node = recovery_system.dependency_graph['ai_services']
        node.health_score = 0.2
        node.failure_count = 3
        
        with patch.object(recovery_system, '_diagnose_failure', return_value='service_timeout'), \
             patch.object(recovery_system, '_execute_recovery_action', return_value=False):
            
            success = await recovery_system.autonomous_system_repair('ai_services')
            
            assert success is False
            assert node.recovery_attempts > 0
    
    @pytest.mark.asyncio
    async def test_recovery_action_restart_service(self, recovery_system):
        """Test restart service recovery action."""
        with patch.object(recovery_system, '_clear_cache', return_value=True):
            success = await recovery_system._execute_recovery_action(
                'web_server', RecoveryAction.RESTART_SERVICE
            )
            
            assert success is True
    
    @pytest.mark.asyncio
    async def test_recovery_action_clear_cache(self, recovery_system):
        """Test clear cache recovery action."""
        with patch('gc.collect'):
            success = await recovery_system._execute_recovery_action(
                'ai_services', RecoveryAction.CLEAR_CACHE
            )
            
            assert success is True
    
    @pytest.mark.asyncio
    async def test_recovery_action_clean_resources(self, recovery_system):
        """Test clean resources recovery action."""
        with patch('os.path.exists', return_value=True), \
             patch('os.walk', return_value=[('temp', [], ['old_file.tmp'])]), \
             patch('os.path.getmtime', return_value=0), \
             patch('os.remove'):
            
            success = await recovery_system._execute_recovery_action(
                'file_system', RecoveryAction.CLEAN_RESOURCES
            )
            
            assert success is True
    
    @pytest.mark.asyncio
    async def test_intelligent_dependency_management(self, recovery_system):
        """Test intelligent dependency management."""
        with patch.object(recovery_system, 'perform_health_check') as mock_health_check:
            mock_health_check.side_effect = [0.9, 0.8, 0.3, 0.7, 0.6]  # Various health scores
            
            with patch.object(recovery_system, 'autonomous_system_repair', return_value=True):
                status = await recovery_system.intelligent_dependency_management()
                
                assert isinstance(status, dict)
                assert len(status) > 0
                
                # Check that all nodes are included
                for node_name in recovery_system.dependency_graph.keys():
                    assert node_name in status
                    assert 'health_score' in status[node_name]
                    assert 'status' in status[node_name]
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, recovery_system):
        """Test performance metrics collection."""
        with patch('psutil.cpu_percent', return_value=45.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_io_counters') as mock_disk_io, \
             patch('psutil.net_io_counters') as mock_net_io:
            
            mock_memory.return_value.percent = 60.0
            mock_disk_io.return_value.read_bytes = 1000000
            mock_disk_io.return_value.write_bytes = 500000
            mock_net_io.return_value.bytes_sent = 2000000
            mock_net_io.return_value.bytes_recv = 1500000
            
            metrics = await recovery_system._collect_performance_metrics()
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.cpu_usage == 45.0
            assert metrics.memory_usage == 60.0
            assert metrics.disk_io > 0
            assert metrics.network_io > 0
    
    @pytest.mark.asyncio
    async def test_self_optimizing_performance_tuning(self, recovery_system):
        """Test self-optimizing performance tuning."""
        # Add some performance history
        for i in range(15):
            metrics = PerformanceMetrics(
                cpu_usage=50 + i * 2,  # Increasing CPU usage
                memory_usage=60 + i,   # Increasing memory usage
                disk_io=10.0,
                network_io=5.0,
                response_time=1.0 + i * 0.1,  # Increasing response time
                throughput=100.0,
                error_rate=0.01
            )
            recovery_system.performance_history.append(metrics)
        
        with patch.object(recovery_system, '_collect_performance_metrics') as mock_collect, \
             patch.object(recovery_system, '_apply_performance_optimization', return_value=True):
            
            mock_collect.return_value = PerformanceMetrics(
                cpu_usage=80.0, memory_usage=75.0, disk_io=15.0,
                network_io=8.0, response_time=2.0, throughput=90.0, error_rate=0.02
            )
            
            result = await recovery_system.self_optimizing_performance_tuning()
            
            assert isinstance(result, dict)
            assert 'current_metrics' in result
            assert 'optimizations_applied' in result
            assert 'performance_trend' in result
    
    @pytest.mark.asyncio
    async def test_performance_optimization_cpu(self, recovery_system):
        """Test CPU optimization."""
        success = await recovery_system._optimize_cpu_usage()
        assert success is True
    
    @pytest.mark.asyncio
    async def test_performance_optimization_memory(self, recovery_system):
        """Test memory optimization."""
        with patch('gc.collect'):
            success = await recovery_system._optimize_memory_usage()
            assert success is True
    
    @pytest.mark.asyncio
    async def test_performance_optimization_disk(self, recovery_system):
        """Test disk optimization."""
        with patch.object(recovery_system, '_clean_resources', return_value=True):
            success = await recovery_system._optimize_disk_usage()
            assert success is True
    
    @pytest.mark.asyncio
    async def test_predictive_maintenance(self, recovery_system):
        """Test predictive maintenance."""
        # Add performance history to enable predictions
        for i in range(25):
            metrics = PerformanceMetrics(
                cpu_usage=60 + i * 1.5,  # Increasing trend
                memory_usage=50 + i * 1.2,
                disk_io=10.0,
                network_io=5.0,
                response_time=1.0 + i * 0.05,
                throughput=100.0,
                error_rate=0.01
            )
            recovery_system.performance_history.append(metrics)
        
        with patch('psutil.disk_usage') as mock_disk:
            mock_disk.return_value.percent = 88.0  # High disk usage
            
            result = await recovery_system.predictive_maintenance()
            
            assert isinstance(result, dict)
            assert 'scheduled_tasks' in result
            assert 'completed_tasks' in result
            assert 'predicted_issues' in result
            assert 'preventive_actions' in result
    
    @pytest.mark.asyncio
    async def test_predict_system_issues(self, recovery_system):
        """Test system issue prediction."""
        # Add performance history with concerning trends
        for i in range(25):
            metrics = PerformanceMetrics(
                cpu_usage=70 + i * 2,  # Rapidly increasing CPU
                memory_usage=60 + i * 1.5,  # Increasing memory
                disk_io=10.0,
                network_io=5.0,
                response_time=1.0 + i * 0.1,  # Increasing response time
                throughput=100.0,
                error_rate=0.01
            )
            recovery_system.performance_history.append(metrics)
        
        with patch('psutil.disk_usage') as mock_disk:
            mock_disk.return_value.percent = 90.0  # Critical disk usage
            
            issues = await recovery_system._predict_system_issues()
            
            assert isinstance(issues, list)
            assert len(issues) > 0
            
            # Should predict CPU and disk issues
            issue_types = [issue['type'] for issue in issues]
            assert 'cpu_overload' in issue_types
            assert 'disk_space_critical' in issue_types
    
    @pytest.mark.asyncio
    async def test_maintenance_task_creation(self, recovery_system):
        """Test maintenance task creation."""
        issue = {
            'type': 'cpu_overload',
            'severity': 'high',
            'estimated_time': '2-4 hours',
            'confidence': 0.8
        }
        
        task = await recovery_system._create_maintenance_task(issue)
        
        assert isinstance(task, MaintenanceTask)
        assert task.task_type == 'cpu_overload'
        assert task.priority >= 6  # High severity should have high priority
        assert task.scheduled_time is not None
    
    @pytest.mark.asyncio
    async def test_maintenance_task_execution(self, recovery_system):
        """Test maintenance task execution."""
        task = MaintenanceTask(
            task_id='test_task',
            task_type='memory_leak',
            priority=8,
            estimated_duration=timedelta(minutes=30),
            required_resources={'cpu': 0.1, 'memory': 0.05},
            dependencies=[]
        )
        
        with patch.object(recovery_system, '_optimize_memory_usage', return_value=True):
            success = await recovery_system._execute_maintenance_task(task)
            
            assert success is True
            assert task.completed is True
            assert task.success is True
    
    def test_get_health_status(self, recovery_system):
        """Test health status conversion."""
        assert recovery_system._get_health_status(0.9) == SystemHealth.HEALTHY.value
        assert recovery_system._get_health_status(0.6) == SystemHealth.DEGRADED.value
        assert recovery_system._get_health_status(0.3) == SystemHealth.CRITICAL.value
        assert recovery_system._get_health_status(0.1) == SystemHealth.FAILING.value
    
    def test_get_system_status(self, recovery_system):
        """Test system status retrieval."""
        status = recovery_system.get_system_status()
        
        assert isinstance(status, dict)
        assert 'timestamp' in status
        assert 'auto_recovery_enabled' in status
        assert 'monitoring_active' in status
        assert 'dependency_health' in status
        assert 'maintenance_tasks' in status
        assert 'performance_thresholds' in status
    
    @pytest.mark.asyncio
    async def test_cascade_failure_prevention(self, recovery_system):
        """Test cascade failure prevention."""
        # Set up a scenario where database is failing
        recovery_system.dependency_graph['database'].health_score = 0.2
        
        with patch.object(recovery_system, '_strengthen_node') as mock_strengthen:
            await recovery_system._prevent_cascade_failures()
            
            # Should strengthen nodes that depend on the failing database
            mock_strengthen.assert_called()
    
    @pytest.mark.asyncio
    async def test_strengthen_node(self, recovery_system):
        """Test node strengthening."""
        with patch.object(recovery_system, '_scale_resources', return_value=True), \
             patch.object(recovery_system, '_clear_cache', return_value=True), \
             patch.object(recovery_system, '_optimize_performance', return_value=True):
            
            await recovery_system._strengthen_node('ai_services')
            # Should complete without errors
    
    def test_performance_trend_analysis(self, recovery_system):
        """Test performance trend analysis."""
        # Add performance history with trends
        for i in range(25):
            metrics = PerformanceMetrics(
                cpu_usage=50 + i * 1.5,  # Increasing
                memory_usage=60 - i * 0.5,  # Decreasing
                disk_io=10.0,
                network_io=5.0,
                response_time=1.0,  # Stable
                throughput=100.0,
                error_rate=0.01
            )
            recovery_system.performance_history.append(metrics)
        
        trends = recovery_system._analyze_performance_trends()
        
        assert isinstance(trends, dict)
        # Should detect increasing CPU trend
        assert trends.get('cpu_optimization') is True
    
    def test_performance_trend_summary(self, recovery_system):
        """Test performance trend summary."""
        # Add some performance history
        for i in range(15):
            metrics = PerformanceMetrics(
                cpu_usage=50 + i,
                memory_usage=60,
                disk_io=10.0,
                network_io=5.0,
                response_time=1.0,
                throughput=100.0,
                error_rate=0.01
            )
            recovery_system.performance_history.append(metrics)
        
        summary = recovery_system._get_performance_trend_summary()
        
        assert isinstance(summary, dict)
        assert 'cpu_trend' in summary
        assert 'memory_trend' in summary
        assert 'response_time_trend' in summary
        assert 'error_rate_trend' in summary
    
    @pytest.mark.asyncio
    async def test_preventive_actions(self, recovery_system):
        """Test preventive actions."""
        predicted_issues = [
            {'type': 'cpu_overload', 'confidence': 0.8},
            {'type': 'memory_leak', 'confidence': 0.9},
            {'type': 'disk_space_critical', 'confidence': 0.95}
        ]
        
        with patch.object(recovery_system, '_optimize_cpu_usage', return_value=True), \
             patch.object(recovery_system, '_optimize_memory_usage', return_value=True), \
             patch.object(recovery_system, '_clean_resources', return_value=True):
            
            actions = await recovery_system._take_preventive_actions(predicted_issues)
            
            assert isinstance(actions, list)
            assert len(actions) == 3  # Should take action for all high-confidence issues
    
    def test_monitoring_lifecycle(self, recovery_system):
        """Test monitoring start and stop."""
        # Stop any existing monitoring
        recovery_system.stop_monitoring()
        
        # Start monitoring
        recovery_system._start_monitoring()
        assert recovery_system.maintenance_scheduler_active is True
        
        # Stop monitoring
        recovery_system.stop_monitoring()
        assert recovery_system.maintenance_scheduler_active is False


class TestGlobalRecoverySystem:
    """Test the global recovery system instance."""
    
    def test_global_instance_exists(self):
        """Test that global instance exists and is properly initialized."""
        assert advanced_recovery_system is not None
        assert isinstance(advanced_recovery_system, AdvancedRecoverySystem)
        assert len(advanced_recovery_system.dependency_graph) > 0
    
    def test_global_instance_dependency_graph(self):
        """Test that dependency graph is properly initialized."""
        graph = advanced_recovery_system.dependency_graph
        
        # Check core dependencies exist
        assert 'database' in graph
        assert 'file_system' in graph
        assert 'ai_services' in graph
        assert 'visualization_engine' in graph
        assert 'web_server' in graph
        
        # Check dependency relationships
        assert 'database' in graph['ai_services'].dependencies
        assert 'ai_services' in graph['database'].dependents
    
    @pytest.mark.asyncio
    async def test_global_instance_health_check(self):
        """Test health check on global instance."""
        health_score = await advanced_recovery_system.perform_health_check('database')
        assert 0.0 <= health_score <= 1.0
    
    def test_global_instance_recovery_patterns(self):
        """Test that recovery patterns are properly configured."""
        patterns = advanced_recovery_system.recovery_patterns
        
        assert 'database_connection_failure' in patterns
        assert 'memory_leak' in patterns
        assert 'high_cpu_usage' in patterns
        assert 'disk_full' in patterns
        
        # Check that patterns contain valid recovery actions
        for pattern_name, actions in patterns.items():
            assert isinstance(actions, list)
            assert len(actions) > 0
            for action in actions:
                assert isinstance(action, RecoveryAction)
    
    def test_global_instance_optimization_rules(self):
        """Test that optimization rules are properly configured."""
        rules = advanced_recovery_system.optimization_rules
        
        assert 'cpu_optimization' in rules
        assert 'memory_optimization' in rules
        assert 'disk_optimization' in rules
        assert 'network_optimization' in rules
        assert 'cache_optimization' in rules
        assert 'database_optimization' in rules
        
        # Check that rules are callable
        for rule_name, rule_func in rules.items():
            assert callable(rule_func)