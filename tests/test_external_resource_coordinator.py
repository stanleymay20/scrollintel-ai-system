"""
Tests for External Resource Coordinator

Comprehensive tests for external partner coordination, resource request management,
and partnership activation protocols.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.external_resource_coordinator import (
    ExternalResourceCoordinator, PartnerRegistry, RequestManager,
    ProtocolEngine, PerformanceTracker, CommunicationHub,
    PartnerType, RequestStatus, ActivationLevel
)
from scrollintel.models.resource_mobilization_models import (
    ExternalPartner, ExternalResourceRequest, CoordinationProtocol,
    ResourceType, ResourcePriority, ResourceRequirement
)
from scrollintel.models.crisis_models_simple import Crisis, CrisisType, SeverityLevel


class TestExternalResourceCoordinator:
    """Test cases for ExternalResourceCoordinator"""
    
    @pytest.fixture
    def coordinator(self):
        """Create ExternalResourceCoordinator instance for testing"""
        return ExternalResourceCoordinator()
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis for testing"""
        return Crisis(
            id="test_crisis_001",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.CRITICAL,
            description="Test security breach requiring external resources"
        )
    
    @pytest.fixture
    def sample_requirements(self):
        """Create sample resource requirements"""
        return [
            ResourceRequirement(
                id="req_001",
                crisis_id="test_crisis_001",
                resource_type=ResourceType.EXTERNAL_SERVICES,
                required_capabilities=["forensic_analysis", "incident_response"],
                quantity_needed=5.0,
                priority=ResourcePriority.CRITICAL,
                duration_needed=timedelta(hours=24),
                budget_limit=50000.0,
                justification="Critical forensic analysis for security breach"
            ),
            ResourceRequirement(
                id="req_002",
                crisis_id="test_crisis_001",
                resource_type=ResourceType.HUMAN_RESOURCES,
                required_capabilities=["security_consulting", "crisis_management"],
                quantity_needed=10.0,
                priority=ResourcePriority.HIGH,
                duration_needed=timedelta(hours=48),
                budget_limit=75000.0,
                justification="Expert security consultants for breach response"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_coordinate_with_partners(self, coordinator, sample_crisis, sample_requirements):
        """Test comprehensive partner coordination"""
        # Perform coordination
        result = await coordinator.coordinate_with_partners(sample_crisis, sample_requirements)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'crisis_id' in result
        assert 'coordination_plan' in result
        assert 'activation_results' in result
        assert 'request_results' in result
        assert 'response_monitoring' in result
        assert 'total_partners_contacted' in result
        assert 'total_requests_submitted' in result
        assert 'coordination_status' in result
        
        # Verify coordination was initiated
        assert result['crisis_id'] == sample_crisis.id
        assert result['coordination_status'] == 'active'
        assert result['total_partners_contacted'] >= 0
        assert result['total_requests_submitted'] >= 0
        
        # Verify coordination event was recorded
        assert len(coordinator.coordination_history) > 0
        last_event = coordinator.coordination_history[-1]
        assert last_event.event_type == 'coordination_initiated'
    
    @pytest.mark.asyncio
    async def test_manage_resource_requests(self, coordinator):
        """Test resource request management"""
        # Create sample requests
        requests = [
            ExternalResourceRequest(
                id="req_001",
                crisis_id="test_crisis",
                partner_id="partner_001",
                resource_type=ResourceType.EXTERNAL_SERVICES,
                quantity_requested=3.0,
                urgency_level=ResourcePriority.HIGH,
                request_status=RequestStatus.PENDING.value
            ),
            ExternalResourceRequest(
                id="req_002",
                crisis_id="test_crisis",
                partner_id="partner_002",
                resource_type=ResourceType.HUMAN_RESOURCES,
                quantity_requested=5.0,
                urgency_level=ResourcePriority.CRITICAL,
                request_status=RequestStatus.IN_PROGRESS.value
            )
        ]
        
        # Add requests to active requests
        for request in requests:
            coordinator.active_requests[request.id] = request
        
        # Manage requests
        result = await coordinator.manage_resource_requests(requests)
        
        # Verify management results
        assert isinstance(result, dict)
        assert 'total_requests' in result
        assert 'request_status_summary' in result
        assert 'successful_requests' in result
        assert 'failed_requests' in result
        assert 'pending_requests' in result
        assert 'management_actions' in result
        assert 'next_steps' in result
        
        # Verify request counts
        assert result['total_requests'] == len(requests)
        assert len(result['management_actions']) == len(requests)
        assert len(result['next_steps']) > 0
    
    @pytest.mark.asyncio
    async def test_activate_partnership_protocols(self, coordinator):
        """Test partnership activation"""
        # Get a sample partner
        partners = await coordinator.partner_registry.get_all_partners()
        partner_id = partners[0].id if partners else "test_partner"
        
        activation_level = ActivationLevel.FULL_DEPLOYMENT
        crisis_context = {
            "crisis_id": "test_crisis",
            "severity": "critical",
            "urgency": "immediate"
        }
        
        # Activate partnership
        result = await coordinator.activate_partnership_protocols(
            partner_id, activation_level, crisis_context
        )
        
        # Verify activation result
        assert isinstance(result, dict)
        assert 'partner_id' in result
        assert 'partner_name' in result
        assert 'activation_level' in result
        assert 'activation_successful' in result
        assert 'activation_steps' in result
        assert 'communication_setup' in result
        assert 'verification_result' in result
        assert 'next_actions' in result
        
        # Verify activation details
        assert result['partner_id'] == partner_id
        assert result['activation_level'] == activation_level.value
        assert isinstance(result['activation_successful'], bool)
        assert len(result['activation_steps']) > 0
        assert len(result['next_actions']) > 0
        
        # Verify coordination event was recorded
        activation_events = [e for e in coordinator.coordination_history if e.event_type == 'partnership_activated']
        assert len(activation_events) > 0
    
    @pytest.mark.asyncio
    async def test_monitor_partner_performance(self, coordinator):
        """Test partner performance monitoring"""
        # Get sample partner IDs
        partners = await coordinator.partner_registry.get_all_partners()
        partner_ids = [p.id for p in partners[:2]]
        
        time_window = timedelta(days=30)
        
        # Monitor performance
        performance_data = await coordinator.monitor_partner_performance(partner_ids, time_window)
        
        # Verify performance data
        assert isinstance(performance_data, dict)
        assert len(performance_data) == len(partner_ids)
        
        for partner_id in partner_ids:
            assert partner_id in performance_data
            performance = performance_data[partner_id]
            
            # Verify performance structure
            assert hasattr(performance, 'partner_id')
            assert hasattr(performance, 'total_requests')
            assert hasattr(performance, 'successful_requests')
            assert hasattr(performance, 'failed_requests')
            assert hasattr(performance, 'reliability_score')
            assert hasattr(performance, 'quality_score')
            assert hasattr(performance, 'cost_efficiency')
            
            # Verify data types and ranges
            assert performance.partner_id == partner_id
            assert performance.total_requests >= 0
            assert performance.successful_requests >= 0
            assert performance.failed_requests >= 0
            assert 0 <= performance.reliability_score <= 1
            assert 0 <= performance.quality_score <= 1
            assert 0 <= performance.cost_efficiency <= 1


class TestPartnerRegistry:
    """Test cases for PartnerRegistry"""
    
    @pytest.fixture
    def registry(self):
        """Create PartnerRegistry instance for testing"""
        return PartnerRegistry()
    
    @pytest.mark.asyncio
    async def test_get_all_partners(self, registry):
        """Test getting all partners"""
        partners = await registry.get_all_partners()
        
        assert isinstance(partners, list)
        assert len(partners) > 0
        
        for partner in partners:
            assert isinstance(partner, ExternalPartner)
            assert partner.id is not None
            assert partner.name is not None
            assert partner.partner_type is not None
    
    @pytest.mark.asyncio
    async def test_get_partner(self, registry):
        """Test getting specific partner"""
        # Get all partners first
        all_partners = await registry.get_all_partners()
        test_partner_id = all_partners[0].id
        
        # Get specific partner
        partner = await registry.get_partner(test_partner_id)
        
        assert partner is not None
        assert partner.id == test_partner_id
        assert isinstance(partner, ExternalPartner)
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_partner(self, registry):
        """Test getting non-existent partner"""
        partner = await registry.get_partner("nonexistent_id")
        assert partner is None
    
    @pytest.mark.asyncio
    async def test_register_partner(self, registry):
        """Test registering new partner"""
        new_partner = ExternalPartner(
            name="Test Partner",
            partner_type="vendor",
            contact_info={"primary_email": "test@partner.com"},
            available_resources=[ResourceType.EXTERNAL_SERVICES],
            service_capabilities=["testing", "validation"],
            response_time_sla=timedelta(hours=4),
            reliability_score=0.85
        )
        
        # Register partner
        success = await registry.register_partner(new_partner)
        assert success is True
        
        # Verify registration
        retrieved_partner = await registry.get_partner(new_partner.id)
        assert retrieved_partner is not None
        assert retrieved_partner.name == "Test Partner"
        assert retrieved_partner.partner_type == "vendor"


class TestRequestManager:
    """Test cases for RequestManager"""
    
    @pytest.fixture
    def manager(self):
        """Create RequestManager instance for testing"""
        return RequestManager()
    
    @pytest.fixture
    def sample_request(self):
        """Create sample resource request"""
        return ExternalResourceRequest(
            crisis_id="test_crisis",
            partner_id="test_partner",
            resource_type=ResourceType.EXTERNAL_SERVICES,
            requested_capabilities=["consulting", "analysis"],
            quantity_requested=3.0,
            urgency_level=ResourcePriority.HIGH,
            duration_needed=timedelta(hours=12),
            budget_approved=15000.0
        )
    
    @pytest.mark.asyncio
    async def test_submit_request(self, manager, sample_request):
        """Test request submission"""
        result = await manager.submit_request(sample_request)
        
        # Verify submission result
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'request_id' in result
        assert 'submission_timestamp' in result
        
        # Verify successful submission
        assert result['success'] is True
        assert result['request_id'] == sample_request.id
        
        # Verify request is stored
        assert sample_request.id in manager.requests
        stored_request = manager.requests[sample_request.id]
        assert stored_request.request_status == RequestStatus.SUBMITTED.value
        
        # Verify history is recorded
        assert len(manager.request_history) > 0
        last_history = manager.request_history[-1]
        assert last_history['request_id'] == sample_request.id
        assert last_history['action'] == 'submitted'
    
    @pytest.mark.asyncio
    async def test_get_request_status(self, manager, sample_request):
        """Test getting request status"""
        # Submit request first
        await manager.submit_request(sample_request)
        
        # Get status
        status = await manager.get_request_status(sample_request.id)
        
        # Verify status
        assert isinstance(status, str)
        assert status in [s.value for s in RequestStatus]
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_request_status(self, manager):
        """Test getting status of non-existent request"""
        status = await manager.get_request_status("nonexistent_id")
        assert status == RequestStatus.FAILED.value


class TestProtocolEngine:
    """Test cases for ProtocolEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create ProtocolEngine instance for testing"""
        return ProtocolEngine()
    
    @pytest.fixture
    def sample_partner(self):
        """Create sample partner for testing"""
        return ExternalPartner(
            name="Test Partner",
            partner_type="vendor",
            response_time_sla=timedelta(hours=2),
            reliability_score=0.9
        )
    
    @pytest.mark.asyncio
    async def test_get_protocol(self, engine):
        """Test getting coordination protocol"""
        partner_id = "test_partner"
        protocol = await engine.get_protocol(partner_id)
        
        # Initially should be None
        assert protocol is None
    
    @pytest.mark.asyncio
    async def test_create_default_protocol(self, engine, sample_partner):
        """Test creating default protocol"""
        protocol = await engine.create_default_protocol(sample_partner)
        
        # Verify protocol structure
        assert isinstance(protocol, CoordinationProtocol)
        assert protocol.partner_id == sample_partner.id
        assert protocol.protocol_name is not None
        assert len(protocol.activation_triggers) > 0
        assert len(protocol.communication_channels) > 0
        assert len(protocol.escalation_procedures) > 0
        assert len(protocol.resource_request_process) > 0
        assert isinstance(protocol.quality_standards, dict)
        assert len(protocol.performance_metrics) > 0
        
        # Verify protocol is stored
        stored_protocol = await engine.get_protocol(sample_partner.id)
        assert stored_protocol is not None
        assert stored_protocol.partner_id == sample_partner.id


class TestPerformanceTracker:
    """Test cases for PerformanceTracker"""
    
    @pytest.fixture
    def tracker(self):
        """Create PerformanceTracker instance for testing"""
        return PerformanceTracker()
    
    @pytest.mark.asyncio
    async def test_analyze_performance(self, tracker):
        """Test performance analysis"""
        partner_id = "test_partner"
        time_window = timedelta(days=30)
        
        performance = await tracker.analyze_performance(partner_id, time_window)
        
        # Verify performance structure
        assert performance.partner_id == partner_id
        assert performance.total_requests >= 0
        assert performance.successful_requests >= 0
        assert performance.failed_requests >= 0
        assert performance.total_requests == performance.successful_requests + performance.failed_requests
        assert 0 <= performance.reliability_score <= 1
        assert 0 <= performance.quality_score <= 1
        assert 0 <= performance.cost_efficiency <= 1
        assert isinstance(performance.performance_trends, dict)
        assert isinstance(performance.feedback_summary, dict)


class TestCommunicationHub:
    """Test cases for CommunicationHub"""
    
    @pytest.fixture
    def hub(self):
        """Create CommunicationHub instance for testing"""
        return CommunicationHub()
    
    @pytest.fixture
    def sample_partner(self):
        """Create sample partner for testing"""
        return ExternalPartner(
            name="Test Partner",
            partner_type="vendor",
            contact_info={
                "primary_email": "test@partner.com",
                "primary_phone": "+1-555-TEST",
                "primary_api": "https://api.partner.com"
            }
        )
    
    @pytest.mark.asyncio
    async def test_establish_channels(self, hub, sample_partner):
        """Test establishing communication channels"""
        channels = ["email", "phone", "api"]
        
        result = await hub.establish_channels(sample_partner, channels)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'partner_id' in result
        assert 'established_channels' in result
        assert 'failed_channels' in result
        assert 'total_channels' in result
        assert 'establishment_timestamp' in result
        
        # Verify channel establishment
        assert result['partner_id'] == sample_partner.id
        assert len(result['established_channels']) > 0
        assert result['total_channels'] == len(result['established_channels'])
        
        # Verify channel details
        for channel in result['established_channels']:
            assert 'channel' in channel
            assert 'status' in channel
            assert 'contact_info' in channel
            assert 'established_at' in channel
            assert channel['status'] == 'active'
    
    @pytest.mark.asyncio
    async def test_establish_unsupported_channels(self, hub, sample_partner):
        """Test establishing unsupported channels"""
        channels = ["email", "unsupported_channel", "phone"]
        
        result = await hub.establish_channels(sample_partner, channels)
        
        # Verify some channels failed
        assert len(result['failed_channels']) > 0
        assert len(result['established_channels']) > 0
        
        # Verify failed channel details
        failed_channel = result['failed_channels'][0]
        assert 'channel' in failed_channel
        assert 'error' in failed_channel
        assert failed_channel['channel'] == 'unsupported_channel'


class TestIntegration:
    """Integration tests for external resource coordination system"""
    
    @pytest.fixture
    def coordinator(self):
        """Create ExternalResourceCoordinator for integration testing"""
        return ExternalResourceCoordinator()
    
    @pytest.mark.asyncio
    async def test_full_coordination_workflow(self, coordinator):
        """Test complete coordination workflow"""
        # Create test data
        crisis = Crisis(
            id="integration_test_crisis",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            description="Integration test system outage"
        )
        
        requirements = [
            ResourceRequirement(
                crisis_id=crisis.id,
                resource_type=ResourceType.EXTERNAL_SERVICES,
                required_capabilities=["emergency_response", "system_recovery"],
                quantity_needed=5.0,
                priority=ResourcePriority.CRITICAL,
                duration_needed=timedelta(hours=12),
                budget_limit=25000.0
            )
        ]
        
        # Step 1: Coordinate with partners
        coordination_result = await coordinator.coordinate_with_partners(crisis, requirements)
        assert coordination_result['coordination_status'] == 'active'
        
        # Step 2: Get submitted requests
        submitted_requests = coordination_result['request_results']['submitted_requests']
        
        # Step 3: Manage requests
        if submitted_requests:
            management_result = await coordinator.manage_resource_requests(submitted_requests)
            assert management_result['total_requests'] == len(submitted_requests)
        
        # Step 4: Monitor performance
        partners = await coordinator.partner_registry.get_all_partners()
        if partners:
            partner_ids = [p.id for p in partners[:1]]
            performance_data = await coordinator.monitor_partner_performance(partner_ids)
            assert len(performance_data) == len(partner_ids)
        
        # Verify coordination history
        assert len(coordinator.coordination_history) > 0
    
    @pytest.mark.asyncio
    async def test_partner_activation_workflow(self, coordinator):
        """Test partner activation workflow"""
        # Get available partners
        partners = await coordinator.partner_registry.get_all_partners()
        assert len(partners) > 0
        
        partner = partners[0]
        
        # Test different activation levels
        activation_levels = [
            ActivationLevel.ALERT,
            ActivationLevel.ACTIVATED,
            ActivationLevel.FULL_DEPLOYMENT
        ]
        
        for level in activation_levels:
            crisis_context = {
                "crisis_id": f"test_crisis_{level.value}",
                "activation_level": level.value
            }
            
            result = await coordinator.activate_partnership_protocols(
                partner.id, level, crisis_context
            )
            
            # Verify activation
            assert result['partner_id'] == partner.id
            assert result['activation_level'] == level.value
            assert isinstance(result['activation_successful'], bool)
        
        # Verify multiple activations recorded
        activation_events = [
            e for e in coordinator.coordination_history 
            if e.event_type == 'partnership_activated'
        ]
        assert len(activation_events) == len(activation_levels)
    
    @pytest.mark.asyncio
    async def test_request_lifecycle_management(self, coordinator):
        """Test complete request lifecycle management"""
        # Create and submit requests
        requests = []
        for i in range(3):
            request = ExternalResourceRequest(
                id=f"lifecycle_test_req_{i}",
                crisis_id="lifecycle_test_crisis",
                partner_id=f"partner_{i}",
                resource_type=ResourceType.EXTERNAL_SERVICES,
                quantity_requested=float(i + 1),
                urgency_level=ResourcePriority.HIGH
            )
            
            # Submit request
            submission_result = await coordinator.request_manager.submit_request(request)
            assert submission_result['success'] is True
            
            requests.append(request)
            coordinator.active_requests[request.id] = request
        
        # Manage all requests
        management_result = await coordinator.manage_resource_requests(requests)
        
        # Verify management
        assert management_result['total_requests'] == len(requests)
        assert len(management_result['management_actions']) == len(requests)
        
        # Check individual request statuses
        for request in requests:
            status = await coordinator.request_manager.get_request_status(request.id)
            assert status in [s.value for s in RequestStatus]


if __name__ == "__main__":
    pytest.main([__file__])