"""
Tests for ScrollIntel Real-Time Monitoring System
Comprehensive test suite for monitoring, analytics, and business impact tracking
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import json

from scrollintel.core.real_time_monitoring import (
    RealTimeAgentMonitor,
    BusinessImpactTracker,
    ExecutiveReportingEngine,
    AutomatedAlertingSystem,
    AgentPerformanceMetrics,
    BusinessImpactMetrics,
    SystemHealthMetrics
)
from scrollintel.core.monitoring import metrics_collector
from scrollintel.core.analytics import event_tracker
from scrollintel.core.alerting import alert_manager, AlertSeverity

class TestRealTimeAgentMonitor:
    """Test real-time agent monitoring functionality"""
    
    @pytest.fixture
    def monitor(self):
        return RealTimeAgentMonitor()
    
    @pytest.mark.asyncio
    async def test_register_agent(self, monitor):
        """Test agent registration"""
        agent_id = "test-agent-001"
        agent_type = "data_scientist"
        
        await monitor.register_agent(agent_id, agent_type)
        
        assert agent_id in monitor.agent_metrics
        agent = monitor.agent_metrics[agent_id]
        assert agent.agent_id == agent_id
        assert agent.agent_type == agent_type
        assert agent.status == "active"
        assert agent.success_rate == 100.0
    
    @pytest.mark.asyncio
    async def test_update_agent_metrics(self, monitor):
        """Test updating agent metrics"""
        agent_id = "test-agent-002"
        
        # Register agent first
        await monitor.register_agent(agent_id, "ml_engineer")
        
        # Update metrics
        metrics_update = {
            "cpu_usage": 45.5,
            "memory_usage": 62.3,
            "success_rate": 95.8,
            "avg_response_time": 1.2,
            "request_count": 150,
            "business_value_generated": 5000.0
        }
        
        await monitor.update_agent_metrics(agent_id, metrics_update)
        
        agent = monitor.agent_metrics[agent_id]
        assert agent.cpu_usage == 45.5
        assert agent.memory_usage == 62.3
        assert agent.success_rate == 95.8
        assert agent.avg_response_time == 1.2
        assert agent.request_count == 150
        assert agent.business_value_generated == 5000.0
    
    @pytest.mark.asyncio
    async def test_performance_threshold_alerts(self, monitor):
        """Test performance threshold monitoring"""
        agent_id = "test-agent-003"
        await monitor.register_agent(agent_id, "bi_agent")
        
        # Update with metrics that exceed thresholds
        high_usage_metrics = {
            "cpu_usage": 95.0,  # Above 80% threshold
            "memory_usage": 90.0,  # Above 85% threshold
            "avg_response_time": 5.5,  # Above 2.0s threshold
            "success_rate": 85.0  # Below 95% threshold
        }
        
        with patch.object(monitor.logger, 'warning') as mock_warning:
            await monitor.update_agent_metrics(agent_id, high_usage_metrics)
            
            # Should log performance issues
            mock_warning.assert_called()
            call_args = mock_warning.call_args[0]
            assert "performance issues detected" in call_args[0]
    
    def test_get_agent_metrics(self, monitor):
        """Test retrieving agent metrics"""
        # Test non-existent agent
        assert monitor.get_agent_metrics("non-existent") is None
        
        # Test existing agent
        agent_id = "test-agent-004"
        monitor.agent_metrics[agent_id] = AgentPerformanceMetrics(
            agent_id=agent_id,
            agent_type="qa_agent",
            status="active",
            cpu_usage=25.0,
            memory_usage=40.0,
            request_count=100,
            success_rate=98.5,
            avg_response_time=0.8,
            error_count=2,
            last_activity=datetime.utcnow(),
            uptime_seconds=3600.0,
            throughput_per_minute=50.0,
            business_value_generated=2500.0,
            cost_savings=1200.0
        )
        
        metrics = monitor.get_agent_metrics(agent_id)
        assert metrics is not None
        assert metrics.agent_id == agent_id
        assert metrics.success_rate == 98.5
    
    def test_get_all_agent_metrics(self, monitor):
        """Test retrieving all agent metrics"""
        # Add multiple agents
        for i in range(3):
            agent_id = f"test-agent-{i+10}"
            monitor.agent_metrics[agent_id] = AgentPerformanceMetrics(
                agent_id=agent_id,
                agent_type=f"agent_type_{i}",
                status="active",
                cpu_usage=20.0 + i * 10,
                memory_usage=30.0 + i * 10,
                request_count=100 + i * 50,
                success_rate=95.0 + i,
                avg_response_time=1.0 + i * 0.5,
                error_count=i,
                last_activity=datetime.utcnow(),
                uptime_seconds=3600.0,
                throughput_per_minute=40.0 + i * 10,
                business_value_generated=1000.0 + i * 500,
                cost_savings=500.0 + i * 200
            )
        
        all_metrics = monitor.get_all_agent_metrics()
        assert len(all_metrics) == 3
        assert all(isinstance(m, AgentPerformanceMetrics) for m in all_metrics)

class TestBusinessImpactTracker:
    """Test business impact tracking and ROI calculations"""
    
    @pytest.fixture
    def tracker(self):
        return BusinessImpactTracker()
    
    @pytest.mark.asyncio
    async def test_calculate_roi_metrics(self, tracker):
        """Test ROI metrics calculation"""
        with patch.object(tracker, '_calculate_cost_savings') as mock_cost_savings, \
             patch.object(tracker, '_calculate_revenue_impact') as mock_revenue, \
             patch.object(tracker, '_calculate_productivity_gain') as mock_productivity, \
             patch.object(tracker, '_calculate_decision_accuracy') as mock_accuracy, \
             patch.object(tracker, '_calculate_time_to_insight_reduction') as mock_time_reduction, \
             patch.object(tracker, '_calculate_user_satisfaction') as mock_satisfaction, \
             patch.object(tracker, '_calculate_competitive_advantage') as mock_competitive, \
             patch.object(tracker, '_calculate_total_costs') as mock_costs:
            
            # Mock return values
            mock_cost_savings.return_value = 50000.0
            mock_revenue.return_value = 100000.0
            mock_productivity.return_value = 25.0
            mock_accuracy.return_value = 30.0
            mock_time_reduction.return_value = 60.0
            mock_satisfaction.return_value = 92.5
            mock_competitive.return_value = 88.0
            mock_costs.return_value = 7000.0
            
            metrics = await tracker.calculate_roi_metrics()
            
            assert isinstance(metrics, BusinessImpactMetrics)
            assert metrics.cost_savings_30d == 50000.0
            assert metrics.revenue_impact == 100000.0
            assert metrics.productivity_gain == 25.0
            assert metrics.decision_accuracy_improvement == 30.0
            assert metrics.time_to_insight_reduction == 60.0
            assert metrics.user_satisfaction_score == 92.5
            assert metrics.competitive_advantage_score == 88.0
            
            # ROI should be calculated correctly
            expected_roi = ((175000.0 - 7000.0) / 7000.0) * 100  # ~2400%
            assert abs(metrics.total_roi - expected_roi) < 1.0
    
    @pytest.mark.asyncio
    async def test_calculate_cost_savings(self, tracker):
        """Test cost savings calculation"""
        # Mock analytics engine
        mock_agent_stats = {
            'agent_usage': [
                {
                    'agent_type': 'data_scientist',
                    'requests': 100,
                    'avg_duration': 300  # 5 minutes vs 1 hour baseline
                },
                {
                    'agent_type': 'ml_engineer', 
                    'requests': 50,
                    'avg_duration': 600  # 10 minutes vs 1 hour baseline
                }
            ]
        }
        
        with patch('scrollintel.core.real_time_monitoring.analytics_engine') as mock_analytics:
            mock_analytics.get_agent_usage_stats.return_value = mock_agent_stats
            
            savings = await tracker._calculate_cost_savings(hours=24)
            
            # Should calculate savings based on time saved vs baseline
            assert savings > 0
            # 100 requests * (3600-300)/3600 * 150 + 50 requests * (3600-600)/3600 * 150
            expected_savings = (100 * (3300/3600) * 150) + (50 * (3000/3600) * 150)
            assert abs(savings - expected_savings) < 100
    
    @pytest.mark.asyncio
    async def test_calculate_productivity_gain(self, tracker):
        """Test productivity gain calculation"""
        with patch('scrollintel.core.real_time_monitoring.analytics_engine') as mock_analytics:
            mock_analytics.get_analytics_summary.return_value = Mock(
                total_users=1000,
                active_users_30d=800
            )
            
            productivity_gain = await tracker._calculate_productivity_gain()
            
            # Should return percentage improvement
            assert isinstance(productivity_gain, float)
            assert productivity_gain > 0
            assert productivity_gain <= 100  # Reasonable upper bound

class TestExecutiveReportingEngine:
    """Test executive reporting and dashboard generation"""
    
    @pytest.fixture
    def reporting_engine(self):
        return ExecutiveReportingEngine()
    
    @pytest.mark.asyncio
    async def test_generate_executive_dashboard(self, reporting_engine):
        """Test executive dashboard generation"""
        with patch('scrollintel.core.real_time_monitoring.business_impact_tracker') as mock_tracker, \
             patch.object(reporting_engine, '_calculate_system_health') as mock_health, \
             patch.object(reporting_engine, '_get_agent_performance_summary') as mock_agent_summary, \
             patch.object(reporting_engine, '_calculate_key_performance_indicators') as mock_kpis, \
             patch.object(reporting_engine, '_get_unique_capabilities') as mock_capabilities, \
             patch.object(reporting_engine, '_get_market_differentiation') as mock_differentiation:
            
            # Mock business metrics
            mock_business_metrics = BusinessImpactMetrics(
                timestamp=datetime.utcnow(),
                total_roi=250.0,
                cost_savings_24h=5000.0,
                cost_savings_7d=25000.0,
                cost_savings_30d=100000.0,
                revenue_impact=50000.0,
                productivity_gain=25.0,
                decision_accuracy_improvement=30.0,
                time_to_insight_reduction=60.0,
                user_satisfaction_score=92.5,
                competitive_advantage_score=88.0
            )
            mock_tracker.calculate_roi_metrics.return_value = mock_business_metrics
            
            # Mock system health
            mock_health.return_value = SystemHealthMetrics(
                timestamp=datetime.utcnow(),
                overall_health_score=95.5,
                uptime_percentage=99.95,
                availability_score=99.95,
                performance_score=92.0,
                security_score=96.5,
                agent_health_score=94.2,
                data_quality_score=94.2,
                user_experience_score=91.8
            )
            
            # Mock other components
            mock_agent_summary.return_value = {"total_agents": 5, "active_agents": 5}
            mock_kpis.return_value = {"requests_per_minute": 125.5}
            mock_capabilities.return_value = ["Real-time orchestration"]
            mock_differentiation.return_value = {"performance_advantage": "10x faster"}
            
            dashboard = await reporting_engine.generate_executive_dashboard()
            
            assert "timestamp" in dashboard
            assert "executive_summary" in dashboard
            assert "business_impact" in dashboard
            assert "system_health" in dashboard
            assert "agent_performance" in dashboard
            assert "key_performance_indicators" in dashboard
            assert "competitive_positioning" in dashboard
            
            # Check executive summary
            summary = dashboard["executive_summary"]
            assert summary["total_roi"] == 250.0
            assert summary["monthly_cost_savings"] == 100000.0
            assert summary["system_health"] == 95.5
    
    @pytest.mark.asyncio
    async def test_calculate_system_health(self, reporting_engine):
        """Test system health calculation"""
        with patch.object(reporting_engine, '_calculate_performance_score') as mock_perf, \
             patch.object(reporting_engine, '_calculate_availability_score') as mock_avail, \
             patch.object(reporting_engine, '_calculate_security_score') as mock_security, \
             patch.object(reporting_engine, '_calculate_agent_health_score') as mock_agent, \
             patch.object(reporting_engine, '_calculate_data_quality_score') as mock_data, \
             patch.object(reporting_engine, '_calculate_user_experience_score') as mock_ux:
            
            # Mock component scores
            mock_perf.return_value = 90.0
            mock_avail.return_value = 99.5
            mock_security.return_value = 95.0
            mock_agent.return_value = 92.0
            mock_data.return_value = 94.0
            mock_ux.return_value = 88.0
            
            health_metrics = await reporting_engine._calculate_system_health()
            
            assert isinstance(health_metrics, SystemHealthMetrics)
            assert health_metrics.performance_score == 90.0
            assert health_metrics.availability_score == 99.5
            assert health_metrics.security_score == 95.0
            assert health_metrics.agent_health_score == 92.0
            
            # Overall health should be weighted average
            expected_overall = (90.0 * 0.25 + 99.5 * 0.20 + 95.0 * 0.15 + 
                              92.0 * 0.20 + 94.0 * 0.10 + 88.0 * 0.10)
            assert abs(health_metrics.overall_health_score - expected_overall) < 0.1

class TestAutomatedAlertingSystem:
    """Test automated alerting functionality"""
    
    @pytest.fixture
    def alerting_system(self):
        return AutomatedAlertingSystem()
    
    def test_setup_monitoring_alert_rules(self, alerting_system):
        """Test alert rules setup"""
        rules = alerting_system.alert_rules
        
        assert len(rules) > 0
        
        # Check for key alert rules
        rule_names = [rule["name"] for rule in rules]
        assert "Agent Performance Degradation" in rule_names
        assert "Agent Failure Rate High" in rule_names
        assert "Business Value Decline" in rule_names
        assert "ROI Below Target" in rule_names
        assert "System Health Critical" in rule_names
    
    @pytest.mark.asyncio
    async def test_check_business_impact_alerts(self, alerting_system):
        """Test business impact alerting"""
        mock_metrics = BusinessImpactMetrics(
            timestamp=datetime.utcnow(),
            total_roi=150.0,  # Below 200% target
            cost_savings_24h=1000.0,
            cost_savings_7d=5000.0,
            cost_savings_30d=20000.0,
            revenue_impact=10000.0,
            productivity_gain=15.0,
            decision_accuracy_improvement=20.0,
            time_to_insight_reduction=40.0,
            user_satisfaction_score=85.0,
            competitive_advantage_score=75.0
        )
        
        with patch('scrollintel.core.real_time_monitoring.alert_manager') as mock_alert_manager:
            await alerting_system._check_business_impact_alerts(mock_metrics)
            
            # Should evaluate metrics against thresholds
            mock_alert_manager.evaluate_metrics.assert_called_once()
            call_args = mock_alert_manager.evaluate_metrics.call_args[0][0]
            assert "total_roi" in call_args
            assert call_args["total_roi"] == 150.0
    
    @pytest.mark.asyncio
    async def test_check_agent_performance_alerts(self, alerting_system):
        """Test agent performance alerting"""
        mock_agent_metrics = [
            AgentPerformanceMetrics(
                agent_id="agent-1",
                agent_type="data_scientist",
                status="active",
                cpu_usage=75.0,
                memory_usage=80.0,
                request_count=100,
                success_rate=88.0,  # Below 90% threshold
                avg_response_time=6.0,  # Above 5s threshold
                error_count=12,
                last_activity=datetime.utcnow(),
                uptime_seconds=3600.0,
                throughput_per_minute=40.0,
                business_value_generated=2000.0,
                cost_savings=1000.0
            ),
            AgentPerformanceMetrics(
                agent_id="agent-2",
                agent_type="ml_engineer",
                status="active",
                cpu_usage=45.0,
                memory_usage=60.0,
                request_count=80,
                success_rate=96.0,
                avg_response_time=2.5,
                error_count=3,
                last_activity=datetime.utcnow(),
                uptime_seconds=7200.0,
                throughput_per_minute=35.0,
                business_value_generated=1800.0,
                cost_savings=900.0
            )
        ]
        
        with patch('scrollintel.core.real_time_monitoring.alert_manager') as mock_alert_manager:
            await alerting_system._check_agent_performance_alerts(mock_agent_metrics)
            
            # Should evaluate average metrics
            mock_alert_manager.evaluate_metrics.assert_called_once()
            call_args = mock_alert_manager.evaluate_metrics.call_args[0][0]
            
            expected_avg_response_time = (6.0 + 2.5) / 2
            expected_avg_success_rate = (88.0 + 96.0) / 2
            
            assert abs(call_args["agent_avg_response_time"] - expected_avg_response_time) < 0.1
            assert abs(call_args["agent_success_rate"] - expected_avg_success_rate) < 0.1

class TestIntegration:
    """Integration tests for the complete monitoring system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring workflow"""
        # Initialize components
        monitor = RealTimeAgentMonitor()
        tracker = BusinessImpactTracker()
        reporting = ExecutiveReportingEngine()
        
        # Register and update agent
        agent_id = "integration-test-agent"
        await monitor.register_agent(agent_id, "test_agent")
        
        metrics_update = {
            "cpu_usage": 55.0,
            "memory_usage": 70.0,
            "success_rate": 94.5,
            "avg_response_time": 1.8,
            "request_count": 200,
            "business_value_generated": 8000.0,
            "cost_savings": 3500.0
        }
        
        await monitor.update_agent_metrics(agent_id, metrics_update)
        
        # Verify agent metrics
        agent_metrics = monitor.get_agent_metrics(agent_id)
        assert agent_metrics is not None
        assert agent_metrics.success_rate == 94.5
        assert agent_metrics.business_value_generated == 8000.0
        
        # Calculate business impact (with mocked dependencies)
        with patch('scrollintel.core.real_time_monitoring.analytics_engine') as mock_analytics:
            mock_analytics.get_agent_usage_stats.return_value = {'agent_usage': []}
            mock_analytics.get_analytics_summary.return_value = Mock(
                total_users=500,
                active_users_30d=400
            )
            
            business_metrics = await tracker.calculate_roi_metrics()
            assert isinstance(business_metrics, BusinessImpactMetrics)
            assert business_metrics.total_roi >= 0
        
        # Generate executive dashboard (with mocked dependencies)
        with patch('scrollintel.core.real_time_monitoring.business_impact_tracker', tracker), \
             patch('scrollintel.core.real_time_monitoring.real_time_monitor', monitor):
            
            dashboard = await reporting.generate_executive_dashboard()
            
            assert "executive_summary" in dashboard
            assert "business_impact" in dashboard
            assert "agent_performance" in dashboard
            
            # Verify agent data is included
            agent_summary = dashboard["agent_performance"]
            assert agent_summary["total_agents"] >= 1
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test monitoring system performance under load"""
        monitor = RealTimeAgentMonitor()
        
        # Register multiple agents
        num_agents = 100
        for i in range(num_agents):
            await monitor.register_agent(f"load-test-agent-{i}", f"agent_type_{i % 5}")
        
        # Update metrics for all agents
        start_time = datetime.utcnow()
        
        for i in range(num_agents):
            metrics_update = {
                "cpu_usage": 30.0 + (i % 50),
                "memory_usage": 40.0 + (i % 40),
                "success_rate": 90.0 + (i % 10),
                "avg_response_time": 1.0 + (i % 3),
                "request_count": 50 + i,
                "business_value_generated": 1000.0 + i * 10
            }
            await monitor.update_agent_metrics(f"load-test-agent-{i}", metrics_update)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should process 100 agents quickly (under 5 seconds)
        assert processing_time < 5.0
        
        # Verify all agents are tracked
        all_metrics = monitor.get_all_agent_metrics()
        assert len(all_metrics) == num_agents
        
        # Verify metrics are correct
        for i, metrics in enumerate(all_metrics):
            assert metrics.agent_id == f"load-test-agent-{i}"
            assert metrics.success_rate >= 90.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])