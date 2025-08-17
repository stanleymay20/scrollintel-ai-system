"""
Comprehensive tests for Agent Registry and Management System

Tests the advanced agent registry functionality including capability matching,
performance-based selection, health monitoring, and automatic failover.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import uuid

from scrollintel.core.agent_registry import (
    AgentRegistry,
    AgentRegistrationRequest,
    AgentSelectionCriteria,
    AdvancedCapabilityMatcher,
    PerformanceBasedSelector,
    AgentHealthMonitor,
    CapabilityType
)
from scrollintel.core.realtime_messaging import RealTimeMessagingSystem, EventType, MessagePriority
from scrollintel.models.agent_steering_models import Agent, AgentStatus, AgentPerformanceMetric


class TestAdvancedCapabilityMatcher:
    """Test the advanced capability matching engine"""
    
    def setup_method(self):
        self.matcher = AdvancedCapabilityMatcher()
    
    def test_direct_capability_match(self):
        """Test direct capability matching"""
        agent_capabilities = [
            {"name": "data_analysis", "performance_score": 85.0},
            {"name": "machine_learning", "performance_score": 90.0}
        ]
        
        required_capabilities = ["data_analysis", "machine_learning"]
        
        score = self.matcher.calculate_capability_match_score(
            agent_capabilities, required_capabilities
        )
        
        assert score > 80.0  # Should be high score for direct match
    
    def test_synonym_capability_match(self):
        """Test capability matching with synonyms"""
        agent_capabilities = [
            {"name": "analytics", "performance_score": 80.0},  # Synonym for data_analysis
            {"name": "ml", "performance_score": 85.0}  # Synonym for machine_learning
        ]
        
        required_capabilities = ["data_analysis", "machine_learning"]
        
        score = self.matcher.calculate_capability_match_score(
            agent_capabilities, required_capabilities
        )
        
        assert score > 70.0  # Should match through synonyms
    
    def test_partial_capability_match(self):
        """Test partial capability matching"""
        agent_capabilities = [
            {"name": "data_analysis", "performance_score": 85.0}
        ]
        
        required_capabilities = ["data_analysis", "machine_learning"]
        
        score = self.matcher.calculate_capability_match_score(
            agent_capabilities, required_capabilities
        )
        
        assert 30.0 < score < 70.0  # Partial match should give medium score
    
    def test_preferred_capabilities_bonus(self):
        """Test that preferred capabilities increase score"""
        agent_capabilities = [
            {"name": "data_analysis", "performance_score": 85.0},
            {"name": "visualization", "performance_score": 80.0}
        ]
        
        required_capabilities = ["data_analysis"]
        preferred_capabilities = ["visualization"]
        
        score_with_preferred = self.matcher.calculate_capability_match_score(
            agent_capabilities, required_capabilities, preferred_capabilities
        )
        
        score_without_preferred = self.matcher.calculate_capability_match_score(
            agent_capabilities, required_capabilities
        )
        
        assert score_with_preferred > score_without_preferred
    
    def test_business_domain_matching(self):
        """Test business domain expertise matching"""
        agent_capabilities = [
            {"name": "financial_modeling", "performance_score": 90.0},
            {"name": "data_analysis", "performance_score": 85.0}
        ]
        
        required_capabilities = ["data_analysis"]
        
        score_with_domain = self.matcher.calculate_capability_match_score(
            agent_capabilities, required_capabilities, business_domain="finance"
        )
        
        score_without_domain = self.matcher.calculate_capability_match_score(
            agent_capabilities, required_capabilities
        )
        
        assert score_with_domain >= score_without_domain


class TestPerformanceBasedSelector:
    """Test the performance-based agent selection engine"""
    
    def setup_method(self):
        self.selector = PerformanceBasedSelector()
    
    def test_selection_score_calculation(self):
        """Test comprehensive selection score calculation"""
        agent = {
            "id": str(uuid.uuid4()),
            "name": "test_agent",
            "success_rate": 95.0,
            "average_response_time": 1.5,
            "current_load": 30.0,
            "last_heartbeat": datetime.utcnow().isoformat()
        }
        
        criteria = AgentSelectionCriteria(
            required_capabilities=["data_analysis"],
            max_response_time=5.0,
            min_success_rate=90.0,
            max_load_threshold=80.0
        )
        
        capability_match_score = 85.0
        
        result = self.selector.calculate_selection_score(
            agent, criteria, capability_match_score
        )
        
        assert "total_score" in result
        assert "component_scores" in result
        assert "weights_used" in result
        assert 0 <= result["total_score"] <= 100
    
    def test_high_load_penalty(self):
        """Test that high load reduces selection score"""
        high_load_agent = {
            "id": str(uuid.uuid4()),
            "success_rate": 95.0,
            "average_response_time": 1.0,
            "current_load": 90.0,  # High load
            "last_heartbeat": datetime.utcnow().isoformat()
        }
        
        low_load_agent = {
            "id": str(uuid.uuid4()),
            "success_rate": 95.0,
            "average_response_time": 1.0,
            "current_load": 20.0,  # Low load
            "last_heartbeat": datetime.utcnow().isoformat()
        }
        
        criteria = AgentSelectionCriteria(required_capabilities=["data_analysis"])
        capability_match_score = 85.0
        
        high_load_result = self.selector.calculate_selection_score(
            high_load_agent, criteria, capability_match_score
        )
        
        low_load_result = self.selector.calculate_selection_score(
            low_load_agent, criteria, capability_match_score
        )
        
        assert low_load_result["total_score"] > high_load_result["total_score"]
    
    def test_selection_outcome_recording(self):
        """Test recording selection outcomes for learning"""
        agent_id = str(uuid.uuid4())
        criteria = AgentSelectionCriteria(required_capabilities=["data_analysis"])
        
        # Record successful outcome
        self.selector.record_selection_outcome(agent_id, criteria, True)
        
        assert len(self.selector.selection_history) == 1
        assert self.selector.selection_history[0]["outcome_success"] is True
        assert self.selector.selection_history[0]["selected_agent_id"] == agent_id
    
    def test_adaptive_learning_adjustment(self):
        """Test adaptive learning adjustments to scores"""
        agent_id = str(uuid.uuid4())
        criteria = AgentSelectionCriteria(required_capabilities=["data_analysis"])
        
        # Record multiple successful outcomes
        for _ in range(5):
            self.selector.record_selection_outcome(agent_id, criteria, True)
        
        agent = {
            "id": agent_id,
            "success_rate": 80.0,
            "average_response_time": 2.0,
            "current_load": 50.0,
            "last_heartbeat": datetime.utcnow().isoformat()
        }
        
        result = self.selector.calculate_selection_score(agent, criteria, 75.0)
        
        # Score should be boosted due to good historical performance
        assert result["total_score"] > 70.0


class TestAgentHealthMonitor:
    """Test the advanced agent health monitoring system"""
    
    def setup_method(self):
        self.messaging_system = Mock(spec=RealTimeMessagingSystem)
        self.health_monitor = AgentHealthMonitor(self.messaging_system)
    
    @pytest.mark.asyncio
    async def test_basic_health_check(self):
        """Test basic HTTP health check"""
        agent = Mock()
        agent.health_check_url = "http://test-agent/health"
        agent.name = "test_agent"
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "healthy"})
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await self.health_monitor._basic_health_check(agent)
            
            assert result["is_healthy"] is True
            assert result["status_code"] == 200
            assert "response_time" in result
    
    @pytest.mark.asyncio
    async def test_comprehensive_health_check(self):
        """Test comprehensive health check with additional endpoints"""
        agent = Mock()
        agent.health_check_url = "http://test-agent/health"
        agent.endpoint_url = "http://test-agent"
        agent.name = "test_agent"
        
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock health check response
            health_response = AsyncMock()
            health_response.status = 200
            health_response.json = AsyncMock(return_value={"status": "healthy"})
            
            # Mock capabilities check response
            capabilities_response = AsyncMock()
            capabilities_response.status = 200
            capabilities_response.json = AsyncMock(return_value={"capabilities": ["data_analysis"]})
            
            # Mock status check response
            status_response = AsyncMock()
            status_response.status = 200
            status_response.json = AsyncMock(return_value={
                "load": 25.0,
                "memory_usage": 60.0,
                "cpu_usage": 40.0
            })
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.side_effect = [
                health_response, capabilities_response, status_response
            ]
            
            result = await self.health_monitor._comprehensive_health_check(agent)
            
            assert result["is_healthy"] is True
            assert result["health_data"]["capabilities_check"] is True
            assert result["health_data"]["load_check"] is True
            assert result["health_data"]["current_load"] == 25.0
    
    def test_health_trend_analysis(self):
        """Test health trend analysis for predictive failure detection"""
        agent_id = str(uuid.uuid4())
        
        # Add health history with degrading performance
        base_time = datetime.utcnow()
        for i in range(10):
            health_entry = {
                "timestamp": base_time - timedelta(minutes=i*5),
                "is_healthy": i < 8,  # Last 2 entries are unhealthy
                "response_time": 1.0 + (i * 0.2),  # Increasing response time
                "health_data": {}
            }
            self.health_monitor.health_history[agent_id].append(health_entry)
        
        trend_analysis = self.health_monitor._analyze_health_trends(agent_id)
        
        assert "failure_risk" in trend_analysis
        assert "response_time_trend" in trend_analysis
        assert "failure_rate" in trend_analysis
        assert trend_analysis["failure_risk"] > 0.0
    
    def test_consecutive_failure_counting(self):
        """Test counting consecutive failures"""
        agent_id = str(uuid.uuid4())
        
        # Add health history with consecutive failures
        base_time = datetime.utcnow()
        for i in range(5):
            health_entry = {
                "timestamp": base_time - timedelta(minutes=i*5),
                "is_healthy": i >= 3,  # Last 3 are healthy, first 2 are failures
                "response_time": 1.0,
                "health_data": {}
            }
            self.health_monitor.health_history[agent_id].append(health_entry)
        
        consecutive_failures = self.health_monitor._count_consecutive_failures(agent_id)
        
        assert consecutive_failures == 2
    
    def test_failover_group_configuration(self):
        """Test failover group configuration"""
        group_name = "critical_agents"
        agent_ids = [str(uuid.uuid4()) for _ in range(3)]
        
        self.health_monitor.configure_failover_group(group_name, agent_ids)
        
        assert group_name in self.health_monitor.failover_groups
        assert self.health_monitor.failover_groups[group_name] == agent_ids
    
    def test_monitoring_stats(self):
        """Test monitoring statistics collection"""
        # Add some test data
        self.health_monitor.monitoring_stats["checks_performed"] = 100
        self.health_monitor.monitoring_stats["failures_detected"] = 5
        self.health_monitor.circuit_breakers["agent1"] = {"opened_at": datetime.utcnow()}
        
        stats = self.health_monitor.get_monitoring_stats()
        
        assert stats["checks_performed"] == 100
        assert stats["failures_detected"] == 5
        assert stats["active_circuit_breakers"] == 1


@pytest.mark.asyncio
class TestAgentRegistry:
    """Test the main Agent Registry functionality"""
    
    def setup_method(self):
        self.messaging_system = Mock(spec=RealTimeMessagingSystem)
        self.agent_registry = AgentRegistry(self.messaging_system)
    
    async def test_agent_registration(self):
        """Test agent registration process"""
        registration_request = AgentRegistrationRequest(
            name="test_agent",
            type="data_analyst",
            version="1.0.0",
            capabilities=[{"name": "data_analysis", "performance_score": 85.0}],
            endpoint_url="http://test-agent:8000",
            health_check_url="http://test-agent:8000/health",
            resource_requirements={"cpu": 2, "memory": "4GB"},
            configuration={"max_tasks": 10}
        )
        
        with patch('scrollintel.models.database_utils.get_sync_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            mock_session.query.return_value.filter.return_value.first.return_value = None
            
            # Mock agent creation
            mock_agent = Mock()
            mock_agent.id = uuid.uuid4()
            mock_session.add = Mock()
            mock_session.commit = Mock()
            mock_session.refresh = Mock()
            mock_session.refresh.side_effect = lambda agent: setattr(agent, 'id', mock_agent.id)
            
            agent_id = await self.agent_registry.register_agent(registration_request)
            
            assert agent_id is not None
            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()
    
    async def test_agent_deregistration(self):
        """Test agent deregistration process"""
        agent_id = str(uuid.uuid4())
        
        with patch('scrollintel.models.database_utils.get_sync_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            mock_agent = Mock()
            mock_agent.name = "test_agent"
            mock_session.query.return_value.filter.return_value.first.return_value = mock_agent
            
            success = await self.agent_registry.deregister_agent(agent_id)
            
            assert success is True
            assert mock_agent.status == AgentStatus.INACTIVE
            mock_session.commit.assert_called_once()
    
    async def test_best_agent_selection(self):
        """Test intelligent agent selection"""
        criteria = AgentSelectionCriteria(
            required_capabilities=["data_analysis"],
            max_load_threshold=80.0,
            min_success_rate=90.0
        )
        
        # Mock available agents
        mock_agents = [
            {
                "id": str(uuid.uuid4()),
                "name": "agent1",
                "capabilities": [{"name": "data_analysis", "performance_score": 85.0}],
                "current_load": 30.0,
                "success_rate": 95.0,
                "average_response_time": 1.5,
                "last_heartbeat": datetime.utcnow().isoformat()
            },
            {
                "id": str(uuid.uuid4()),
                "name": "agent2",
                "capabilities": [{"name": "data_analysis", "performance_score": 80.0}],
                "current_load": 60.0,
                "success_rate": 92.0,
                "average_response_time": 2.0,
                "last_heartbeat": datetime.utcnow().isoformat()
            }
        ]
        
        with patch.object(self.agent_registry, 'get_available_agents', return_value=mock_agents):
            selected_agent = await self.agent_registry.select_best_agent(criteria)
            
            assert selected_agent is not None
            assert "selection_score" in selected_agent
            assert "capability_match_score" in selected_agent
            assert "score_breakdown" in selected_agent
    
    async def test_multiple_agent_selection(self):
        """Test selecting multiple agents with different strategies"""
        criteria = AgentSelectionCriteria(
            required_capabilities=["data_analysis"],
            max_load_threshold=80.0
        )
        
        # Mock available agents
        mock_agents = [
            {
                "id": str(uuid.uuid4()),
                "name": f"agent{i}",
                "type": "data_analyst" if i % 2 == 0 else "ml_engineer",
                "capabilities": [{"name": "data_analysis", "performance_score": 85.0 - i}],
                "current_load": 20.0 + (i * 10),
                "success_rate": 95.0 - i,
                "average_response_time": 1.0 + (i * 0.2),
                "last_heartbeat": datetime.utcnow().isoformat()
            }
            for i in range(5)
        ]
        
        with patch.object(self.agent_registry, 'get_available_agents', return_value=mock_agents):
            # Test performance strategy
            selected_agents = await self.agent_registry.select_multiple_agents(
                criteria, 3, "performance"
            )
            assert len(selected_agents) == 3
            
            # Test load balanced strategy
            selected_agents = await self.agent_registry.select_multiple_agents(
                criteria, 3, "load_balanced"
            )
            assert len(selected_agents) == 3
            
            # Test diverse strategy
            selected_agents = await self.agent_registry.select_multiple_agents(
                criteria, 3, "diverse"
            )
            assert len(selected_agents) == 3
    
    async def test_agent_configuration_update(self):
        """Test updating agent configuration"""
        agent_id = str(uuid.uuid4())
        configuration_updates = {
            "configuration": {"max_tasks": 20},
            "capabilities": [{"name": "advanced_analytics", "performance_score": 90.0}],
            "version": "2.0.0"
        }
        
        with patch('scrollintel.models.database_utils.get_sync_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            mock_agent = Mock()
            mock_agent.name = "test_agent"
            mock_agent.configuration = {}
            mock_agent.resource_requirements = {}
            mock_session.query.return_value.filter.return_value.first.return_value = mock_agent
            
            success = await self.agent_registry.update_agent_configuration(
                agent_id, configuration_updates
            )
            
            assert success is True
            assert mock_agent.version == "2.0.0"
            mock_session.commit.assert_called_once()
    
    async def test_agent_scaling(self):
        """Test agent scaling operations"""
        agent_id = str(uuid.uuid4())
        
        with patch('scrollintel.models.database_utils.get_sync_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            mock_agent = Mock()
            mock_agent.name = "test_agent"
            mock_agent.max_concurrent_tasks = 10
            mock_session.query.return_value.filter.return_value.first.return_value = mock_agent
            
            # Test scale up
            success = await self.agent_registry.scale_agent(agent_id, "scale_up")
            assert success is True
            assert mock_agent.max_concurrent_tasks == 20
            
            # Test scale down
            success = await self.agent_registry.scale_agent(agent_id, "scale_down")
            assert success is True
            assert mock_agent.max_concurrent_tasks == 10
            
            # Test set capacity
            success = await self.agent_registry.scale_agent(agent_id, "set_capacity", 15)
            assert success is True
            assert mock_agent.max_concurrent_tasks == 15
    
    async def test_maintenance_mode(self):
        """Test putting agent in and out of maintenance mode"""
        agent_id = str(uuid.uuid4())
        
        with patch('scrollintel.models.database_utils.get_sync_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            mock_agent = Mock()
            mock_agent.name = "test_agent"
            mock_agent.status = AgentStatus.ACTIVE
            mock_agent.configuration = {}
            mock_agent.id = uuid.UUID(agent_id)
            mock_session.query.return_value.filter.return_value.first.return_value = mock_agent
            
            # Mock task reassignment
            with patch.object(self.agent_registry, '_reassign_agent_tasks'):
                # Test entering maintenance
                success = await self.agent_registry.put_agent_in_maintenance(
                    agent_id, "Scheduled update"
                )
                assert success is True
                assert mock_agent.status == AgentStatus.MAINTENANCE
                
                # Test exiting maintenance
                success = await self.agent_registry.remove_agent_from_maintenance(agent_id)
                assert success is True
                assert mock_agent.status == AgentStatus.ACTIVE
    
    async def test_registry_statistics(self):
        """Test comprehensive registry statistics"""
        with patch('scrollintel.models.database_utils.get_sync_db') as mock_db:
            mock_session = Mock()
            mock_db.return_value.__enter__.return_value = mock_session
            
            # Mock database queries
            mock_session.query.return_value.count.return_value = 10
            mock_session.query.return_value.filter.return_value.count.return_value = 8
            mock_session.query.return_value.all.return_value = []
            mock_session.query.return_value.filter.return_value.with_entities.return_value.scalar.return_value = 1.5
            
            stats = await self.agent_registry.get_registry_stats()
            
            assert "agent_counts" in stats
            assert "capability_distribution" in stats
            assert "performance_metrics" in stats
            assert "health_monitoring" in stats
            assert "selection_algorithm" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])