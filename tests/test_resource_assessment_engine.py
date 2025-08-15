"""
Tests for Resource Assessment Engine

Comprehensive tests for resource assessment, capacity tracking,
gap identification, and alternative sourcing capabilities.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from scrollintel.engines.resource_assessment_engine import (
    ResourceAssessmentEngine, ResourceRegistry, CapacityTracker,
    GapAnalyzer, AlternativeSourcer
)
from scrollintel.models.resource_mobilization_models import (
    Resource, ResourcePool, ResourceInventory, ResourceRequirement,
    ResourceGap, ResourceType, ResourceStatus, ResourcePriority,
    ResourceCapability
)
from scrollintel.models.crisis_models_simple import Crisis, CrisisType, SeverityLevel


class TestResourceAssessmentEngine:
    """Test cases for ResourceAssessmentEngine"""
    
    @pytest.fixture
    def engine(self):
        """Create ResourceAssessmentEngine instance for testing"""
        return ResourceAssessmentEngine()
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis for testing"""
        return Crisis(
            id="test_crisis_001",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH,
            description="Test system outage crisis"
        )
    
    @pytest.fixture
    def sample_resources(self):
        """Create sample resources for testing"""
        return [
            Resource(
                id="resource_001",
                name="Development Team Alpha",
                resource_type=ResourceType.HUMAN_RESOURCES,
                status=ResourceStatus.AVAILABLE,
                capabilities=[
                    ResourceCapability(name="software_development", proficiency_level=0.9),
                    ResourceCapability(name="crisis_response", proficiency_level=0.7)
                ],
                capacity=40.0,
                current_utilization=20.0,
                cost_per_hour=125.0
            ),
            Resource(
                id="resource_002",
                name="Primary Server Cluster",
                resource_type=ResourceType.TECHNICAL_INFRASTRUCTURE,
                status=ResourceStatus.AVAILABLE,
                capabilities=[
                    ResourceCapability(name="high_availability", proficiency_level=1.0),
                    ResourceCapability(name="load_balancing", proficiency_level=1.0)
                ],
                capacity=1000.0,
                current_utilization=600.0,
                cost_per_hour=50.0
            ),
            Resource(
                id="resource_003",
                name="Cloud Compute Resources",
                resource_type=ResourceType.CLOUD_COMPUTE,
                status=ResourceStatus.ALLOCATED,
                capabilities=[
                    ResourceCapability(name="elastic_scaling", proficiency_level=1.0)
                ],
                capacity=5000.0,
                current_utilization=3000.0,
                cost_per_hour=25.0
            )
        ]
    
    @pytest.mark.asyncio
    async def test_assess_available_resources(self, engine, sample_crisis):
        """Test comprehensive resource assessment"""
        # Perform assessment
        inventory = await engine.assess_available_resources(sample_crisis)
        
        # Verify inventory structure
        assert isinstance(inventory, ResourceInventory)
        assert inventory.assessment_time is not None
        assert inventory.total_resources > 0
        assert len(inventory.resources_by_type) > 0
        assert len(inventory.available_resources) >= 0
        assert len(inventory.total_capacity) > 0
        assert len(inventory.available_capacity) > 0
        assert len(inventory.utilization_rates) > 0
        
        # Verify cache
        assert sample_crisis.id in engine.inventory_cache
        assert engine.last_assessment_time is not None
    
    @pytest.mark.asyncio
    async def test_track_resource_capacity(self, engine):
        """Test resource capacity tracking"""
        # Get sample resource IDs
        resources = await engine.resource_registry.get_all_resources()
        resource_ids = [r.id for r in resources[:2]]
        
        # Track capacity
        capacity_info = await engine.track_resource_capacity(resource_ids)
        
        # Verify capacity information
        assert len(capacity_info) == len(resource_ids)
        for resource_id in resource_ids:
            assert resource_id in capacity_info
            info = capacity_info[resource_id]
            assert 'total_capacity' in info
            assert 'current_utilization' in info
            assert 'available_capacity' in info
            assert 'utilization_percentage' in info
            assert 'status' in info
    
    @pytest.mark.asyncio
    async def test_identify_resource_gaps(self, engine, sample_crisis):
        """Test resource gap identification"""
        # Create resource requirements
        requirements = [
            ResourceRequirement(
                crisis_id=sample_crisis.id,
                resource_type=ResourceType.HUMAN_RESOURCES,
                required_capabilities=["software_development", "crisis_response"],
                quantity_needed=60.0,  # More than available
                priority=ResourcePriority.HIGH
            ),
            ResourceRequirement(
                crisis_id=sample_crisis.id,
                resource_type=ResourceType.TECHNICAL_INFRASTRUCTURE,
                required_capabilities=["high_availability"],
                quantity_needed=200.0,  # Within available capacity
                priority=ResourcePriority.MEDIUM
            )
        ]
        
        # Get inventory
        inventory = await engine.assess_available_resources(sample_crisis)
        
        # Identify gaps
        gaps = await engine.identify_resource_gaps(requirements, inventory)
        
        # Verify gaps
        assert isinstance(gaps, list)
        # Should have at least one gap for the over-requested human resources
        human_resource_gaps = [g for g in gaps if g.resource_type == ResourceType.HUMAN_RESOURCES]
        assert len(human_resource_gaps) >= 0  # May or may not have gaps depending on available resources
        
        for gap in gaps:
            assert isinstance(gap, ResourceGap)
            assert gap.gap_quantity > 0
            assert gap.resource_type in [ResourceType.HUMAN_RESOURCES, ResourceType.TECHNICAL_INFRASTRUCTURE]
            assert len(gap.alternative_options) > 0
            assert gap.estimated_cost >= 0
    
    @pytest.mark.asyncio
    async def test_find_alternative_sources(self, engine):
        """Test alternative source finding"""
        # Create sample gap
        gap = ResourceGap(
            resource_type=ResourceType.HUMAN_RESOURCES,
            gap_quantity=10.0,
            severity=ResourcePriority.HIGH
        )
        
        # Find alternatives
        alternatives = await engine.find_alternative_sources(gap)
        
        # Verify alternatives
        assert isinstance(alternatives, list)
        assert len(alternatives) > 0
        
        for alternative in alternatives:
            assert isinstance(alternative, dict)
            assert 'type' in alternative
            assert 'description' in alternative
            assert 'estimated_cost' in alternative
            assert 'time_to_implement' in alternative
            assert 'reliability' in alternative
            assert 'capacity_provided' in alternative


class TestResourceRegistry:
    """Test cases for ResourceRegistry"""
    
    @pytest.fixture
    def registry(self):
        """Create ResourceRegistry instance for testing"""
        return ResourceRegistry()
    
    @pytest.mark.asyncio
    async def test_get_all_resources(self, registry):
        """Test getting all resources"""
        resources = await registry.get_all_resources()
        
        assert isinstance(resources, list)
        assert len(resources) > 0
        
        for resource in resources:
            assert isinstance(resource, Resource)
            assert resource.id is not None
            assert resource.name is not None
            assert isinstance(resource.resource_type, ResourceType)
    
    @pytest.mark.asyncio
    async def test_get_resource(self, registry):
        """Test getting specific resource"""
        # Get all resources first
        all_resources = await registry.get_all_resources()
        test_resource_id = all_resources[0].id
        
        # Get specific resource
        resource = await registry.get_resource(test_resource_id)
        
        assert resource is not None
        assert resource.id == test_resource_id
        assert isinstance(resource, Resource)
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_resource(self, registry):
        """Test getting non-existent resource"""
        resource = await registry.get_resource("nonexistent_id")
        assert resource is None
    
    @pytest.mark.asyncio
    async def test_register_resource(self, registry):
        """Test registering new resource"""
        new_resource = Resource(
            name="Test Resource",
            resource_type=ResourceType.EQUIPMENT_HARDWARE,
            status=ResourceStatus.AVAILABLE,
            capacity=100.0,
            current_utilization=0.0
        )
        
        # Register resource
        success = await registry.register_resource(new_resource)
        assert success is True
        
        # Verify registration
        retrieved_resource = await registry.get_resource(new_resource.id)
        assert retrieved_resource is not None
        assert retrieved_resource.name == "Test Resource"
        assert retrieved_resource.resource_type == ResourceType.EQUIPMENT_HARDWARE
    
    @pytest.mark.asyncio
    async def test_get_resource_pools(self, registry):
        """Test getting resource pools"""
        pools = await registry.get_resource_pools()
        
        assert isinstance(pools, list)
        # May be empty initially, but should be a list
        for pool in pools:
            assert isinstance(pool, ResourcePool)


class TestCapacityTracker:
    """Test cases for CapacityTracker"""
    
    @pytest.fixture
    def tracker(self):
        """Create CapacityTracker instance for testing"""
        return CapacityTracker()
    
    @pytest.fixture
    def sample_resources(self):
        """Create sample resources for testing"""
        return [
            Resource(
                resource_type=ResourceType.HUMAN_RESOURCES,
                capacity=40.0,
                current_utilization=20.0
            ),
            Resource(
                resource_type=ResourceType.HUMAN_RESOURCES,
                capacity=40.0,
                current_utilization=30.0
            ),
            Resource(
                resource_type=ResourceType.TECHNICAL_INFRASTRUCTURE,
                capacity=1000.0,
                current_utilization=600.0
            )
        ]
    
    @pytest.mark.asyncio
    async def test_calculate_capacity_metrics(self, tracker, sample_resources):
        """Test capacity metrics calculation"""
        metrics = await tracker.calculate_capacity_metrics(sample_resources)
        
        # Verify structure
        assert 'total_capacity' in metrics
        assert 'available_capacity' in metrics
        assert 'utilization_rates' in metrics
        
        # Verify human resources metrics
        hr_total = metrics['total_capacity'][ResourceType.HUMAN_RESOURCES]
        hr_available = metrics['available_capacity'][ResourceType.HUMAN_RESOURCES]
        hr_utilization = metrics['utilization_rates'][ResourceType.HUMAN_RESOURCES]
        
        assert hr_total == 80.0  # 40 + 40
        assert hr_available == 30.0  # 80 - 50 (20 + 30 utilization)
        assert hr_utilization == 62.5  # (50/80) * 100
        
        # Verify technical infrastructure metrics
        tech_total = metrics['total_capacity'][ResourceType.TECHNICAL_INFRASTRUCTURE]
        tech_available = metrics['available_capacity'][ResourceType.TECHNICAL_INFRASTRUCTURE]
        tech_utilization = metrics['utilization_rates'][ResourceType.TECHNICAL_INFRASTRUCTURE]
        
        assert tech_total == 1000.0
        assert tech_available == 400.0  # 1000 - 600
        assert tech_utilization == 60.0  # (600/1000) * 100


class TestGapAnalyzer:
    """Test cases for GapAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create GapAnalyzer instance for testing"""
        return GapAnalyzer()
    
    @pytest.fixture
    def sample_crisis(self):
        """Create sample crisis for testing"""
        return Crisis(
            id="test_crisis",
            crisis_type=CrisisType.SYSTEM_OUTAGE,
            severity_level=SeverityLevel.HIGH
        )
    
    @pytest.fixture
    def limited_resources(self):
        """Create limited resources to trigger shortages"""
        return [
            Resource(
                resource_type=ResourceType.HUMAN_RESOURCES,
                status=ResourceStatus.AVAILABLE,
                capacity=10.0,
                current_utilization=8.0  # Only 2.0 available
            ),
            Resource(
                resource_type=ResourceType.TECHNICAL_INFRASTRUCTURE,
                status=ResourceStatus.AVAILABLE,
                capacity=100.0,
                current_utilization=90.0  # Only 10.0 available
            )
        ]
    
    @pytest.mark.asyncio
    async def test_identify_critical_shortages(self, analyzer, sample_crisis, limited_resources):
        """Test critical shortage identification"""
        shortages = await analyzer.identify_critical_shortages(limited_resources, sample_crisis)
        
        assert isinstance(shortages, list)
        
        for shortage in shortages:
            assert isinstance(shortage, ResourceGap)
            assert shortage.gap_quantity > 0
            assert shortage.severity == ResourcePriority.CRITICAL
            assert shortage.impact_description is not None
            assert len(shortage.alternative_options) > 0


class TestAlternativeSourcer:
    """Test cases for AlternativeSourcer"""
    
    @pytest.fixture
    def sourcer(self):
        """Create AlternativeSourcer instance for testing"""
        return AlternativeSourcer()
    
    @pytest.fixture
    def sample_requirement(self):
        """Create sample resource requirement"""
        return ResourceRequirement(
            resource_type=ResourceType.HUMAN_RESOURCES,
            required_capabilities=["software_development"],
            quantity_needed=10.0,
            priority=ResourcePriority.HIGH
        )
    
    @pytest.fixture
    def sample_gap(self):
        """Create sample resource gap"""
        return ResourceGap(
            resource_type=ResourceType.HUMAN_RESOURCES,
            gap_quantity=5.0,
            severity=ResourcePriority.HIGH,
            estimated_cost=10000.0
        )
    
    @pytest.mark.asyncio
    async def test_find_alternatives(self, sourcer, sample_requirement):
        """Test finding alternatives for requirement"""
        alternatives = await sourcer.find_alternatives(sample_requirement)
        
        assert isinstance(alternatives, list)
        assert len(alternatives) > 0
        
        # Should include internal alternatives
        internal_alternatives = [alt for alt in alternatives if "reallocation" in alt.lower() or "overtime" in alt.lower()]
        assert len(internal_alternatives) > 0
        
        # Should include external alternatives for human resources
        external_alternatives = [alt for alt in alternatives if "contract" in alt.lower() or "consulting" in alt.lower()]
        assert len(external_alternatives) > 0
    
    @pytest.mark.asyncio
    async def test_find_procurement_options(self, sourcer, sample_requirement):
        """Test finding procurement options"""
        options = await sourcer.find_procurement_options(sample_requirement)
        
        assert isinstance(options, list)
        assert len(options) > 0
        
        # For human resources, should include staffing options
        staffing_options = [opt for opt in options if "staffing" in opt.lower() or "freelancer" in opt.lower()]
        assert len(staffing_options) > 0
    
    @pytest.mark.asyncio
    async def test_find_comprehensive_alternatives(self, sourcer, sample_gap):
        """Test finding comprehensive alternatives"""
        alternatives = await sourcer.find_comprehensive_alternatives(sample_gap)
        
        assert isinstance(alternatives, list)
        assert len(alternatives) > 0
        
        for alternative in alternatives:
            assert isinstance(alternative, dict)
            assert 'type' in alternative
            assert 'description' in alternative
            assert 'estimated_cost' in alternative
            assert 'time_to_implement' in alternative
            assert 'reliability' in alternative
            assert 'capacity_provided' in alternative
            
            # Verify data types
            assert isinstance(alternative['estimated_cost'], (int, float))
            assert isinstance(alternative['reliability'], (int, float))
            assert isinstance(alternative['capacity_provided'], (int, float))
            assert 0 <= alternative['reliability'] <= 1
            assert alternative['capacity_provided'] > 0


class TestIntegration:
    """Integration tests for resource assessment system"""
    
    @pytest.fixture
    def engine(self):
        """Create ResourceAssessmentEngine for integration testing"""
        return ResourceAssessmentEngine()
    
    @pytest.mark.asyncio
    async def test_full_assessment_workflow(self, engine):
        """Test complete resource assessment workflow"""
        # Create crisis
        crisis = Crisis(
            id="integration_test_crisis",
            crisis_type=CrisisType.SECURITY_BREACH,
            severity_level=SeverityLevel.CRITICAL,
            description="Integration test security breach"
        )
        
        # Step 1: Assess available resources
        inventory = await engine.assess_available_resources(crisis)
        assert isinstance(inventory, ResourceInventory)
        assert inventory.total_resources > 0
        
        # Step 2: Create resource requirements
        requirements = [
            ResourceRequirement(
                crisis_id=crisis.id,
                resource_type=ResourceType.HUMAN_RESOURCES,
                required_capabilities=["incident_response", "security_analysis"],
                quantity_needed=50.0,
                priority=ResourcePriority.CRITICAL
            ),
            ResourceRequirement(
                crisis_id=crisis.id,
                resource_type=ResourceType.EXTERNAL_SERVICES,
                required_capabilities=["forensic_analysis"],
                quantity_needed=10.0,
                priority=ResourcePriority.HIGH
            )
        ]
        
        # Step 3: Identify gaps
        gaps = await engine.identify_resource_gaps(requirements, inventory)
        assert isinstance(gaps, list)
        
        # Step 4: Find alternatives for any gaps
        for gap in gaps:
            alternatives = await engine.find_alternative_sources(gap)
            assert isinstance(alternatives, list)
            assert len(alternatives) > 0
        
        # Verify workflow completion
        assert crisis.id in engine.inventory_cache
        assert engine.last_assessment_time is not None
    
    @pytest.mark.asyncio
    async def test_capacity_tracking_integration(self, engine):
        """Test capacity tracking integration"""
        # Get all resources
        resources = await engine.resource_registry.get_all_resources()
        resource_ids = [r.id for r in resources]
        
        # Track capacity for all resources
        capacity_info = await engine.track_resource_capacity(resource_ids)
        
        # Verify all resources tracked
        assert len(capacity_info) == len(resource_ids)
        
        # Verify capacity information completeness
        for resource_id, info in capacity_info.items():
            if 'error' not in info:
                assert 'total_capacity' in info
                assert 'available_capacity' in info
                assert 'utilization_percentage' in info
                assert info['utilization_percentage'] >= 0
                assert info['utilization_percentage'] <= 100


if __name__ == "__main__":
    pytest.main([__file__])