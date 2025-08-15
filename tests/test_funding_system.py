"""
Tests for Unlimited Funding Access System

Tests the funding engine, coordination system, and API endpoints
to ensure reliable $25B+ funding management.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.funding_engine import (
    UnlimitedFundingEngine, FundingSource, FundingRequest,
    FundingSourceType, FundingStatus
)
from scrollintel.core.funding_coordinator import (
    FundingCoordinator, CoordinationStrategy, FundingPlan
)


class TestUnlimitedFundingEngine:
    """Test the core funding engine functionality"""
    
    @pytest.fixture
    def funding_engine(self):
        """Create a fresh funding engine for testing"""
        return UnlimitedFundingEngine()
    
    @pytest.mark.asyncio
    async def test_initialize_funding_sources(self, funding_engine):
        """Test funding source initialization"""
        success = await funding_engine.initialize_funding_sources()
        
        assert success is True
        assert len(funding_engine.funding_sources) > 0
        
        # Check total commitment meets target
        total_commitment = sum(
            source.total_commitment 
            for source in funding_engine.funding_sources.values()
        )
        assert total_commitment >= funding_engine.total_commitment_target
        
        # Verify source types are represented
        source_types = {source.source_type for source in funding_engine.funding_sources.values()}
        assert FundingSourceType.VENTURE_CAPITAL in source_types
        assert FundingSourceType.PRIVATE_EQUITY in source_types
        assert FundingSourceType.SOVEREIGN_WEALTH in source_types
    
    @pytest.mark.asyncio
    async def test_validate_funding_security(self, funding_engine):
        """Test funding source security validation"""
        await funding_engine.initialize_funding_sources()
        
        # Test with valid source
        source_id = list(funding_engine.funding_sources.keys())[0]
        is_valid = await funding_engine.validate_funding_security(source_id)
        assert is_valid is True
        
        # Test with invalid source ID
        is_valid = await funding_engine.validate_funding_security("invalid_id")
        assert is_valid is False
        
        # Test with low security source
        low_security_source = FundingSource(
            id="test_low_security",
            name="Low Security Test",
            source_type=FundingSourceType.CRYPTOCURRENCY,
            total_commitment=1000000,
            available_amount=1000000,
            deployed_amount=0,
            status=FundingStatus.ACTIVE,
            security_level=3,  # Below threshold
            response_time_hours=24,
            terms={},
            contact_info={}
        )
        funding_engine.funding_sources["test_low_security"] = low_security_source
        
        is_valid = await funding_engine.validate_funding_security("test_low_security")
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_activate_backup_sources(self, funding_engine):
        """Test backup source activation"""
        await funding_engine.initialize_funding_sources()
        
        # Get a primary source
        primary_source_id = list(funding_engine.funding_sources.keys())[0]
        primary_source = funding_engine.funding_sources[primary_source_id]
        
        # Set up backup sources
        backup_sources = [
            source_id for source_id, source in funding_engine.funding_sources.items()
            if source_id != primary_source_id and source.source_type == primary_source.source_type
        ]
        
        if backup_sources:
            primary_source.backup_sources = backup_sources[:1]
            
            activated = await funding_engine.activate_backup_sources(primary_source_id)
            assert len(activated) > 0
            
            # Verify backup source is now active
            backup_id = activated[0]
            backup_source = funding_engine.funding_sources[backup_id]
            assert backup_source.status == FundingStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_monitor_funding_availability(self, funding_engine):
        """Test real-time funding availability monitoring"""
        await funding_engine.initialize_funding_sources()
        
        availability = await funding_engine.monitor_funding_availability()
        
        assert "TOTAL_AVAILABLE" in availability
        assert availability["TOTAL_AVAILABLE"] > 0
        
        # Check individual source availability
        for source_name, amount in availability.items():
            if source_name != "TOTAL_AVAILABLE":
                assert isinstance(amount, (int, float))
                assert amount >= 0
    
    @pytest.mark.asyncio
    async def test_allocate_funding(self, funding_engine):
        """Test funding allocation process"""
        await funding_engine.initialize_funding_sources()
        
        # Create a funding request
        request = FundingRequest(
            id="test_request",
            amount_requested=1000000,  # $1M
            purpose="Test allocation",
            urgency_level=5,
            required_by=datetime.now() + timedelta(days=30)
        )
        
        success, sources = await funding_engine.allocate_funding(request)
        
        assert success is True
        assert len(sources) > 0
        assert request.allocated_amount == request.amount_requested
        assert request.status == "allocated"
        
        # Verify source amounts were updated
        for source_id in sources:
            source = funding_engine.funding_sources[source_id]
            assert source.deployed_amount > 0
    
    @pytest.mark.asyncio
    async def test_request_funding(self, funding_engine):
        """Test funding request submission"""
        await funding_engine.initialize_funding_sources()
        
        request_id = await funding_engine.request_funding(
            amount=5000000,  # $5M
            purpose="Test funding request",
            urgency=8  # High urgency
        )
        
        assert request_id is not None
        assert request_id in funding_engine.funding_requests
        
        request = funding_engine.funding_requests[request_id]
        assert request.amount_requested == 5000000
        assert request.purpose == "Test funding request"
        assert request.urgency_level == 8
        
        # High urgency requests should be processed immediately
        assert request.status in ["allocated", "partial_allocation"]
    
    @pytest.mark.asyncio
    async def test_get_funding_status(self, funding_engine):
        """Test funding status reporting"""
        await funding_engine.initialize_funding_sources()
        
        status = await funding_engine.get_funding_status()
        
        assert "total_commitment" in status
        assert "total_available" in status
        assert "total_deployed" in status
        assert "utilization_rate" in status
        assert "source_count" in status
        assert "source_breakdown" in status
        assert "target_achievement" in status
        
        assert status["total_commitment"] >= funding_engine.total_commitment_target
        assert status["source_count"] > 0
        assert status["target_achievement"] >= 100.0


class TestFundingCoordinator:
    """Test the funding coordination system"""
    
    @pytest.fixture
    def coordinator(self):
        """Create a fresh funding coordinator for testing"""
        return FundingCoordinator()
    
    @pytest.fixture
    async def initialized_system(self, coordinator):
        """Initialize the funding system for testing"""
        from scrollintel.engines.funding_engine import funding_engine
        await funding_engine.initialize_funding_sources()
        return coordinator
    
    @pytest.mark.asyncio
    async def test_create_funding_plan(self, initialized_system):
        """Test funding plan creation"""
        coordinator = initialized_system
        
        plan_id = await coordinator.create_funding_plan(
            amount=10000000,  # $10M
            purpose="Test funding plan",
            strategy=CoordinationStrategy.DIVERSIFIED,
            timeline_days=60
        )
        
        assert plan_id is not None
        assert plan_id in coordinator.active_plans
        
        plan = coordinator.active_plans[plan_id]
        assert plan.total_amount == 10000000
        assert plan.purpose == "Test funding plan"
        assert plan.strategy == CoordinationStrategy.DIVERSIFIED
        assert len(plan.source_allocations) > 0
        assert len(plan.backup_sources) > 0
        assert "overall_risk" in plan.risk_assessment
    
    @pytest.mark.asyncio
    async def test_diversified_strategy(self, initialized_system):
        """Test diversified funding strategy"""
        coordinator = initialized_system
        
        plan_id = await coordinator.create_funding_plan(
            amount=20000000,  # $20M
            purpose="Diversified strategy test",
            strategy=CoordinationStrategy.DIVERSIFIED
        )
        
        plan = coordinator.active_plans[plan_id]
        
        # Should spread across multiple sources
        assert len(plan.source_allocations) >= 3
        
        # No single source should have more than 50% allocation
        max_allocation = max(plan.source_allocations.values())
        assert max_allocation <= plan.total_amount * 0.5
    
    @pytest.mark.asyncio
    async def test_concentrated_strategy(self, initialized_system):
        """Test concentrated funding strategy"""
        coordinator = initialized_system
        
        plan_id = await coordinator.create_funding_plan(
            amount=15000000,  # $15M
            purpose="Concentrated strategy test",
            strategy=CoordinationStrategy.CONCENTRATED
        )
        
        plan = coordinator.active_plans[plan_id]
        
        # Should use fewer, higher-quality sources
        assert len(plan.source_allocations) <= 5
        
        # Should prioritize high-security sources
        from scrollintel.engines.funding_engine import funding_engine
        for source_id in plan.source_allocations.keys():
            source = funding_engine.funding_sources[source_id]
            assert source.security_level >= 8
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, initialized_system):
        """Test funding plan risk assessment"""
        coordinator = initialized_system
        
        plan_id = await coordinator.create_funding_plan(
            amount=5000000,  # $5M
            purpose="Risk assessment test",
            strategy=CoordinationStrategy.RISK_BALANCED
        )
        
        plan = coordinator.active_plans[plan_id]
        risks = plan.risk_assessment
        
        assert "concentration_risk" in risks
        assert "source_type_risk" in risks
        assert "security_risk" in risks
        assert "timing_risk" in risks
        assert "overall_risk" in risks
        
        # All risk values should be between 0 and 1
        for risk_type, risk_value in risks.items():
            assert 0 <= risk_value <= 1
        
        # Risk-balanced strategy should have lower overall risk
        assert risks["overall_risk"] <= 0.5
    
    @pytest.mark.asyncio
    async def test_execute_funding_plan(self, initialized_system):
        """Test funding plan execution"""
        coordinator = initialized_system
        
        plan_id = await coordinator.create_funding_plan(
            amount=2000000,  # $2M
            purpose="Execution test",
            strategy=CoordinationStrategy.SPEED_OPTIMIZED
        )
        
        success = await coordinator.execute_funding_plan(plan_id)
        assert success is True
        
        # Verify funding requests were created
        from scrollintel.engines.funding_engine import funding_engine
        plan = coordinator.active_plans[plan_id]
        
        # Should have created requests for each source allocation
        relevant_requests = [
            req for req in funding_engine.funding_requests.values()
            if plan_id in req.purpose
        ]
        assert len(relevant_requests) > 0
    
    @pytest.mark.asyncio
    async def test_get_coordination_status(self, initialized_system):
        """Test coordination status reporting"""
        coordinator = initialized_system
        
        # Create a few plans
        await coordinator.create_funding_plan(1000000, "Plan 1", CoordinationStrategy.DIVERSIFIED)
        await coordinator.create_funding_plan(2000000, "Plan 2", CoordinationStrategy.CONCENTRATED)
        
        status = await coordinator.get_coordination_status()
        
        assert "active_plans" in status
        assert "total_planned_amount" in status
        assert "strategy_breakdown" in status
        assert "average_risk_level" in status
        
        assert status["active_plans"] == 2
        assert status["total_planned_amount"] == 3000000
        
        # Check strategy breakdown
        strategy_breakdown = status["strategy_breakdown"]
        assert "diversified" in strategy_breakdown
        assert "concentrated" in strategy_breakdown


class TestFundingIntegration:
    """Test integration between funding engine and coordinator"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_funding_flow(self):
        """Test complete funding flow from request to allocation"""
        from scrollintel.engines.funding_engine import funding_engine
        from scrollintel.core.funding_coordinator import funding_coordinator
        
        # Initialize system
        await funding_engine.initialize_funding_sources()
        
        # Create and execute funding plan
        plan_id = await funding_coordinator.create_funding_plan(
            amount=8000000,  # $8M
            purpose="End-to-end test",
            strategy=CoordinationStrategy.DIVERSIFIED
        )
        
        # Execute plan
        success = await funding_coordinator.execute_funding_plan(plan_id)
        assert success is True
        
        # Verify funding was allocated
        status = await funding_engine.get_funding_status()
        assert status["total_deployed"] >= 8000000
        
        # Verify utilization increased
        assert status["utilization_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_backup_activation_integration(self):
        """Test backup source activation in integrated system"""
        from scrollintel.engines.funding_engine import funding_engine
        
        await funding_engine.initialize_funding_sources()
        
        # Simulate primary source failure
        primary_source_id = list(funding_engine.funding_sources.keys())[0]
        primary_source = funding_engine.funding_sources[primary_source_id]
        primary_source.status = FundingStatus.UNAVAILABLE
        
        # Activate backups
        activated = await funding_engine.activate_backup_sources(primary_source_id)
        
        # Verify backup sources are available
        if activated:
            for backup_id in activated:
                backup_source = funding_engine.funding_sources[backup_id]
                assert backup_source.status == FundingStatus.ACTIVE
                assert backup_source.available_amount > 0
    
    @pytest.mark.asyncio
    async def test_emergency_funding_scenario(self):
        """Test emergency funding activation"""
        from scrollintel.engines.funding_engine import funding_engine
        
        await funding_engine.initialize_funding_sources()
        
        # Submit emergency funding request
        request_id = await funding_engine.request_funding(
            amount=50000000,  # $50M emergency
            purpose="Critical system failure recovery",
            urgency=10,  # Maximum urgency
            required_by=datetime.now() + timedelta(hours=1)
        )
        
        # Should be processed immediately due to high urgency
        request = funding_engine.funding_requests[request_id]
        assert request.status in ["allocated", "partial_allocation"]
        
        # Should have significant allocation
        assert request.allocated_amount > 0
        
        # If partial, backup sources should be activated
        if request.status == "partial_allocation":
            # Monitor availability to trigger backup activation
            await funding_engine.monitor_funding_availability()
    
    @pytest.mark.asyncio
    async def test_system_resilience(self):
        """Test system resilience under stress"""
        from scrollintel.engines.funding_engine import funding_engine
        from scrollintel.core.funding_coordinator import funding_coordinator
        
        await funding_engine.initialize_funding_sources()
        
        # Create multiple large funding plans simultaneously
        plan_tasks = []
        for i in range(5):
            task = coordinator.create_funding_plan(
                amount=5000000,  # $5M each
                purpose=f"Stress test plan {i}",
                strategy=CoordinationStrategy.DIVERSIFIED
            )
            plan_tasks.append(task)
        
        plan_ids = await asyncio.gather(*plan_tasks)
        assert len(plan_ids) == 5
        
        # Execute all plans
        execution_tasks = []
        for plan_id in plan_ids:
            task = funding_coordinator.execute_funding_plan(plan_id)
            execution_tasks.append(task)
        
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        
        # At least some should succeed
        successful_executions = sum(1 for result in results if result is True)
        assert successful_executions > 0
        
        # System should remain stable
        status = await funding_engine.get_funding_status()
        assert status["total_commitment"] >= funding_engine.total_commitment_target
        assert len(funding_engine.funding_sources) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])