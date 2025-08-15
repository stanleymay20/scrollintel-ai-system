"""
Resource Assessment Engine for Crisis Leadership Excellence

This engine provides comprehensive resource assessment capabilities including
rapid inventory, capacity tracking, gap identification, and alternative sourcing
for crisis response situations.
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from abc import ABC, abstractmethod

from ..models.resource_mobilization_models import (
    Resource, ResourcePool, ResourceInventory, ResourceRequirement,
    ResourceGap, ResourceType, ResourceStatus, ResourcePriority,
    ResourceCapability, ExternalPartner
)
from ..models.crisis_models_simple import Crisis

logger = logging.getLogger(__name__)


class ResourceAssessmentEngine:
    """
    Comprehensive resource assessment system for crisis response.
    
    Provides rapid inventory, capacity tracking, gap identification,
    and alternative sourcing capabilities.
    """
    
    def __init__(self):
        self.resource_registry = ResourceRegistry()
        self.capacity_tracker = CapacityTracker()
        self.gap_analyzer = GapAnalyzer()
        self.alternative_sourcer = AlternativeSourcer()
        self.inventory_cache = {}
        self.last_assessment_time = None
    
    async def assess_available_resources(self, crisis: Crisis) -> ResourceInventory:
        """
        Perform rapid inventory of available resources and capabilities
        
        Args:
            crisis: Crisis requiring resource assessment
            
        Returns:
            ResourceInventory: Complete inventory of available resources
        """
        logger.info(f"Starting resource assessment for crisis {crisis.id}")
        
        try:
            # Get all resources from registry
            all_resources = await self.resource_registry.get_all_resources()
            
            # Filter available resources
            available_resources = [
                resource for resource in all_resources
                if resource.status == ResourceStatus.AVAILABLE
            ]
            
            # Get allocated resources
            allocated_resources = [
                resource for resource in all_resources
                if resource.status == ResourceStatus.ALLOCATED
            ]
            
            # Get unavailable resources
            unavailable_resources = [
                resource for resource in all_resources
                if resource.status in [ResourceStatus.UNAVAILABLE, ResourceStatus.MAINTENANCE]
            ]
            
            # Get resource pools
            resource_pools = await self.resource_registry.get_resource_pools()
            
            # Calculate capacity metrics
            capacity_metrics = await self.capacity_tracker.calculate_capacity_metrics(all_resources)
            
            # Identify critical shortages
            critical_shortages = await self.gap_analyzer.identify_critical_shortages(
                available_resources, crisis
            )
            
            # Get upcoming maintenance
            upcoming_maintenance = await self._get_upcoming_maintenance(all_resources)
            
            # Calculate cost summary
            cost_summary = await self._calculate_cost_summary(all_resources)
            
            # Create inventory
            inventory = ResourceInventory(
                assessment_time=datetime.utcnow(),
                total_resources=len(all_resources),
                resources_by_type=self._count_resources_by_type(all_resources),
                available_resources=available_resources,
                allocated_resources=allocated_resources,
                unavailable_resources=unavailable_resources,
                resource_pools=resource_pools,
                total_capacity=capacity_metrics['total_capacity'],
                available_capacity=capacity_metrics['available_capacity'],
                utilization_rates=capacity_metrics['utilization_rates'],
                critical_shortages=critical_shortages,
                upcoming_maintenance=upcoming_maintenance,
                cost_summary=cost_summary
            )
            
            # Cache the inventory
            self.inventory_cache[crisis.id] = inventory
            self.last_assessment_time = datetime.utcnow()
            
            logger.info(f"Resource assessment completed for crisis {crisis.id}")
            return inventory
            
        except Exception as e:
            logger.error(f"Error during resource assessment: {str(e)}")
            raise
    
    async def track_resource_capacity(self, resource_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Track capacity and availability of specific resources
        
        Args:
            resource_ids: List of resource IDs to track
            
        Returns:
            Dict mapping resource IDs to capacity information
        """
        logger.info(f"Tracking capacity for {len(resource_ids)} resources")
        
        capacity_info = {}
        
        for resource_id in resource_ids:
            try:
                resource = await self.resource_registry.get_resource(resource_id)
                if resource:
                    capacity_info[resource_id] = {
                        'total_capacity': resource.capacity,
                        'current_utilization': resource.current_utilization,
                        'available_capacity': resource.capacity - resource.current_utilization,
                        'utilization_percentage': (resource.current_utilization / resource.capacity) * 100 if resource.capacity > 0 else 0,
                        'status': resource.status.value,
                        'last_updated': resource.updated_at.isoformat()
                    }
                else:
                    capacity_info[resource_id] = {'error': 'Resource not found'}
                    
            except Exception as e:
                logger.error(f"Error tracking capacity for resource {resource_id}: {str(e)}")
                capacity_info[resource_id] = {'error': str(e)}
        
        return capacity_info
    
    async def identify_resource_gaps(
        self, 
        requirements: List[ResourceRequirement], 
        available_inventory: ResourceInventory
    ) -> List[ResourceGap]:
        """
        Identify gaps between required and available resources
        
        Args:
            requirements: List of resource requirements
            available_inventory: Current resource inventory
            
        Returns:
            List of identified resource gaps
        """
        logger.info(f"Identifying resource gaps for {len(requirements)} requirements")
        
        gaps = []
        
        for requirement in requirements:
            try:
                # Find matching available resources
                matching_resources = await self._find_matching_resources(
                    requirement, available_inventory.available_resources
                )
                
                # Calculate gap
                available_quantity = sum(
                    resource.capacity - resource.current_utilization 
                    for resource in matching_resources
                )
                
                if available_quantity < requirement.quantity_needed:
                    gap_quantity = requirement.quantity_needed - available_quantity
                    
                    # Identify missing capabilities
                    available_capabilities = set()
                    for resource in matching_resources:
                        available_capabilities.update(
                            cap.name for cap in resource.capabilities
                        )
                    
                    missing_capabilities = [
                        cap for cap in requirement.required_capabilities
                        if cap not in available_capabilities
                    ]
                    
                    # Create gap record
                    gap = ResourceGap(
                        requirement_id=requirement.id,
                        resource_type=requirement.resource_type,
                        gap_quantity=gap_quantity,
                        gap_capabilities=missing_capabilities,
                        severity=requirement.priority,
                        impact_description=f"Shortage of {gap_quantity} units of {requirement.resource_type.value}",
                        alternative_options=await self.alternative_sourcer.find_alternatives(requirement),
                        procurement_options=await self.alternative_sourcer.find_procurement_options(requirement),
                        estimated_cost=await self._estimate_gap_cost(requirement, gap_quantity),
                        time_to_acquire=await self._estimate_acquisition_time(requirement)
                    )
                    
                    gaps.append(gap)
                    
            except Exception as e:
                logger.error(f"Error identifying gap for requirement {requirement.id}: {str(e)}")
        
        logger.info(f"Identified {len(gaps)} resource gaps")
        return gaps
    
    async def find_alternative_sources(self, gap: ResourceGap) -> List[Dict[str, Any]]:
        """
        Find alternative sources for resource gaps
        
        Args:
            gap: Resource gap to find alternatives for
            
        Returns:
            List of alternative sourcing options
        """
        return await self.alternative_sourcer.find_comprehensive_alternatives(gap)
    
    def _count_resources_by_type(self, resources: List[Resource]) -> Dict[ResourceType, int]:
        """Count resources by type"""
        counts = {}
        for resource in resources:
            counts[resource.resource_type] = counts.get(resource.resource_type, 0) + 1
        return counts
    
    async def _find_matching_resources(
        self, 
        requirement: ResourceRequirement, 
        available_resources: List[Resource]
    ) -> List[Resource]:
        """Find resources that match the requirement"""
        matching_resources = []
        
        for resource in available_resources:
            # Check resource type match
            if resource.resource_type != requirement.resource_type:
                continue
            
            # Check capability match
            resource_capabilities = {cap.name for cap in resource.capabilities}
            required_capabilities = set(requirement.required_capabilities)
            
            if required_capabilities.issubset(resource_capabilities):
                matching_resources.append(resource)
        
        return matching_resources
    
    async def _get_upcoming_maintenance(self, resources: List[Resource]) -> List[Dict[str, Any]]:
        """Get upcoming maintenance schedules"""
        upcoming = []
        now = datetime.utcnow()
        
        for resource in resources:
            if resource.next_maintenance and resource.next_maintenance > now:
                days_until = (resource.next_maintenance - now).days
                if days_until <= 30:  # Next 30 days
                    upcoming.append({
                        'resource_id': resource.id,
                        'resource_name': resource.name,
                        'maintenance_date': resource.next_maintenance.isoformat(),
                        'days_until': days_until,
                        'resource_type': resource.resource_type.value
                    })
        
        return sorted(upcoming, key=lambda x: x['days_until'])
    
    async def _calculate_cost_summary(self, resources: List[Resource]) -> Dict[str, float]:
        """Calculate cost summary for resources"""
        return {
            'total_hourly_cost': sum(resource.cost_per_hour for resource in resources),
            'available_hourly_cost': sum(
                resource.cost_per_hour for resource in resources
                if resource.status == ResourceStatus.AVAILABLE
            ),
            'allocated_hourly_cost': sum(
                resource.cost_per_hour for resource in resources
                if resource.status == ResourceStatus.ALLOCATED
            )
        }
    
    async def _estimate_gap_cost(self, requirement: ResourceRequirement, gap_quantity: float) -> float:
        """Estimate cost to fill resource gap"""
        # Base cost estimation - would be more sophisticated in production
        base_hourly_rates = {
            ResourceType.HUMAN_RESOURCES: 100.0,
            ResourceType.TECHNICAL_INFRASTRUCTURE: 50.0,
            ResourceType.CLOUD_COMPUTE: 25.0,
            ResourceType.EXTERNAL_SERVICES: 150.0,
            ResourceType.EQUIPMENT_HARDWARE: 75.0
        }
        
        hourly_rate = base_hourly_rates.get(requirement.resource_type, 100.0)
        duration_hours = requirement.duration_needed.total_seconds() / 3600
        
        return gap_quantity * hourly_rate * duration_hours
    
    async def _estimate_acquisition_time(self, requirement: ResourceRequirement) -> timedelta:
        """Estimate time to acquire missing resources"""
        # Base acquisition times by resource type
        acquisition_times = {
            ResourceType.HUMAN_RESOURCES: timedelta(days=7),
            ResourceType.TECHNICAL_INFRASTRUCTURE: timedelta(days=3),
            ResourceType.CLOUD_COMPUTE: timedelta(hours=1),
            ResourceType.EXTERNAL_SERVICES: timedelta(hours=4),
            ResourceType.EQUIPMENT_HARDWARE: timedelta(days=5)
        }
        
        base_time = acquisition_times.get(requirement.resource_type, timedelta(days=1))
        
        # Adjust based on priority
        if requirement.priority == ResourcePriority.EMERGENCY:
            return base_time * 0.25  # Rush delivery
        elif requirement.priority == ResourcePriority.CRITICAL:
            return base_time * 0.5
        elif requirement.priority == ResourcePriority.HIGH:
            return base_time * 0.75
        
        return base_time


class ResourceRegistry:
    """Registry for managing resource information"""
    
    def __init__(self):
        self.resources = {}
        self.resource_pools = {}
        self._initialize_sample_resources()
    
    async def get_all_resources(self) -> List[Resource]:
        """Get all registered resources"""
        return list(self.resources.values())
    
    async def get_resource(self, resource_id: str) -> Optional[Resource]:
        """Get specific resource by ID"""
        return self.resources.get(resource_id)
    
    async def get_resource_pools(self) -> List[ResourcePool]:
        """Get all resource pools"""
        return list(self.resource_pools.values())
    
    async def register_resource(self, resource: Resource) -> bool:
        """Register a new resource"""
        try:
            self.resources[resource.id] = resource
            logger.info(f"Registered resource {resource.id}: {resource.name}")
            return True
        except Exception as e:
            logger.error(f"Error registering resource: {str(e)}")
            return False
    
    def _initialize_sample_resources(self):
        """Initialize sample resources for demonstration"""
        # Sample human resources
        dev_team = Resource(
            name="Development Team",
            resource_type=ResourceType.HUMAN_RESOURCES,
            status=ResourceStatus.AVAILABLE,
            capabilities=[
                ResourceCapability(name="software_development", proficiency_level=0.9),
                ResourceCapability(name="system_debugging", proficiency_level=0.8),
                ResourceCapability(name="crisis_response", proficiency_level=0.7)
            ],
            capacity=40.0,  # 40 hours per week
            current_utilization=20.0,
            location="Remote",
            cost_per_hour=125.0
        )
        
        ops_team = Resource(
            name="Operations Team",
            resource_type=ResourceType.HUMAN_RESOURCES,
            status=ResourceStatus.AVAILABLE,
            capabilities=[
                ResourceCapability(name="infrastructure_management", proficiency_level=0.9),
                ResourceCapability(name="incident_response", proficiency_level=0.95),
                ResourceCapability(name="system_monitoring", proficiency_level=0.85)
            ],
            capacity=24.0,  # 24/7 coverage
            current_utilization=12.0,
            location="Data Center",
            cost_per_hour=100.0
        )
        
        # Sample technical infrastructure
        primary_servers = Resource(
            name="Primary Server Cluster",
            resource_type=ResourceType.TECHNICAL_INFRASTRUCTURE,
            status=ResourceStatus.AVAILABLE,
            capabilities=[
                ResourceCapability(name="high_availability", proficiency_level=1.0),
                ResourceCapability(name="load_balancing", proficiency_level=1.0),
                ResourceCapability(name="auto_scaling", proficiency_level=0.8)
            ],
            capacity=1000.0,  # Processing units
            current_utilization=600.0,
            location="Primary Data Center",
            cost_per_hour=50.0
        )
        
        # Sample cloud compute
        cloud_resources = Resource(
            name="AWS Cloud Compute",
            resource_type=ResourceType.CLOUD_COMPUTE,
            status=ResourceStatus.AVAILABLE,
            capabilities=[
                ResourceCapability(name="elastic_scaling", proficiency_level=1.0),
                ResourceCapability(name="global_deployment", proficiency_level=1.0),
                ResourceCapability(name="managed_services", proficiency_level=0.9)
            ],
            capacity=10000.0,  # Compute units
            current_utilization=3000.0,
            location="Global",
            cost_per_hour=25.0
        )
        
        # Register resources
        for resource in [dev_team, ops_team, primary_servers, cloud_resources]:
            self.resources[resource.id] = resource


class CapacityTracker:
    """Tracks resource capacity and utilization"""
    
    async def calculate_capacity_metrics(self, resources: List[Resource]) -> Dict[str, Dict[ResourceType, float]]:
        """Calculate comprehensive capacity metrics"""
        total_capacity = {}
        available_capacity = {}
        utilization_rates = {}
        
        # Group resources by type
        resources_by_type = {}
        for resource in resources:
            if resource.resource_type not in resources_by_type:
                resources_by_type[resource.resource_type] = []
            resources_by_type[resource.resource_type].append(resource)
        
        # Calculate metrics for each type
        for resource_type, type_resources in resources_by_type.items():
            total_cap = sum(r.capacity for r in type_resources)
            total_util = sum(r.current_utilization for r in type_resources)
            available_cap = total_cap - total_util
            
            total_capacity[resource_type] = total_cap
            available_capacity[resource_type] = available_cap
            utilization_rates[resource_type] = (total_util / total_cap) * 100 if total_cap > 0 else 0
        
        return {
            'total_capacity': total_capacity,
            'available_capacity': available_capacity,
            'utilization_rates': utilization_rates
        }


class GapAnalyzer:
    """Analyzes resource gaps and shortages"""
    
    async def identify_critical_shortages(
        self, 
        available_resources: List[Resource], 
        crisis: Crisis
    ) -> List[ResourceGap]:
        """Identify critical resource shortages for crisis response"""
        critical_shortages = []
        
        # Define critical resource requirements based on crisis type
        critical_requirements = self._get_critical_requirements_for_crisis(crisis)
        
        for req_type, min_capacity in critical_requirements.items():
            # Calculate available capacity for this resource type
            available_capacity = sum(
                resource.capacity - resource.current_utilization
                for resource in available_resources
                if resource.resource_type == req_type and resource.status == ResourceStatus.AVAILABLE
            )
            
            if available_capacity < min_capacity:
                shortage = ResourceGap(
                    resource_type=req_type,
                    gap_quantity=min_capacity - available_capacity,
                    severity=ResourcePriority.CRITICAL,
                    impact_description=f"Critical shortage of {req_type.value} for crisis response",
                    alternative_options=["Emergency procurement", "External contractor", "Resource reallocation"],
                    time_to_acquire=timedelta(hours=4)
                )
                critical_shortages.append(shortage)
        
        return critical_shortages
    
    def _get_critical_requirements_for_crisis(self, crisis: Crisis) -> Dict[ResourceType, float]:
        """Get critical resource requirements based on crisis type"""
        # Base requirements for different crisis types
        requirements = {
            'system_outage': {
                ResourceType.HUMAN_RESOURCES: 20.0,  # 20 person-hours
                ResourceType.TECHNICAL_INFRASTRUCTURE: 500.0,  # 500 processing units
                ResourceType.CLOUD_COMPUTE: 2000.0  # 2000 compute units
            },
            'security_breach': {
                ResourceType.HUMAN_RESOURCES: 40.0,  # 40 person-hours
                ResourceType.TECHNICAL_INFRASTRUCTURE: 300.0,  # 300 processing units
                ResourceType.EXTERNAL_SERVICES: 10.0  # 10 service units
            },
            'financial_crisis': {
                ResourceType.HUMAN_RESOURCES: 60.0,  # 60 person-hours
                ResourceType.FINANCIAL_CAPITAL: 100000.0,  # $100k
                ResourceType.EXTERNAL_SERVICES: 20.0  # 20 service units
            }
        }
        
        crisis_type_key = crisis.crisis_type.value if hasattr(crisis.crisis_type, 'value') else str(crisis.crisis_type)
        return requirements.get(crisis_type_key, {
            ResourceType.HUMAN_RESOURCES: 10.0,
            ResourceType.TECHNICAL_INFRASTRUCTURE: 100.0
        })


class AlternativeSourcer:
    """Finds alternative sources for resource gaps"""
    
    def __init__(self):
        self.external_partners = self._initialize_external_partners()
    
    async def find_alternatives(self, requirement: ResourceRequirement) -> List[str]:
        """Find alternative options for resource requirement"""
        alternatives = []
        
        # Internal alternatives
        alternatives.extend([
            "Resource reallocation from non-critical projects",
            "Overtime authorization for existing team",
            "Cross-training team members"
        ])
        
        # External alternatives
        if requirement.resource_type == ResourceType.HUMAN_RESOURCES:
            alternatives.extend([
                "Contract temporary staff",
                "Engage consulting firm",
                "Partner with external team"
            ])
        elif requirement.resource_type == ResourceType.TECHNICAL_INFRASTRUCTURE:
            alternatives.extend([
                "Cloud migration",
                "Equipment rental",
                "Partner data center usage"
            ])
        elif requirement.resource_type == ResourceType.CLOUD_COMPUTE:
            alternatives.extend([
                "Multi-cloud deployment",
                "Reserved instance activation",
                "Spot instance utilization"
            ])
        
        return alternatives
    
    async def find_procurement_options(self, requirement: ResourceRequirement) -> List[str]:
        """Find procurement options for resource requirement"""
        options = []
        
        if requirement.resource_type == ResourceType.HUMAN_RESOURCES:
            options.extend([
                "Emergency staffing agency",
                "Freelancer platforms",
                "Professional services firms"
            ])
        elif requirement.resource_type == ResourceType.EQUIPMENT_HARDWARE:
            options.extend([
                "Emergency equipment rental",
                "Expedited purchase orders",
                "Equipment leasing"
            ])
        elif requirement.resource_type == ResourceType.EXTERNAL_SERVICES:
            options.extend([
                "On-demand service providers",
                "Emergency service contracts",
                "Partner service agreements"
            ])
        
        return options
    
    async def find_comprehensive_alternatives(self, gap: ResourceGap) -> List[Dict[str, Any]]:
        """Find comprehensive alternative sourcing options"""
        alternatives = []
        
        # Internal reallocation
        alternatives.append({
            'type': 'internal_reallocation',
            'description': 'Reallocate resources from lower priority projects',
            'estimated_cost': 0.0,
            'time_to_implement': timedelta(hours=2),
            'reliability': 0.8,
            'capacity_provided': gap.gap_quantity * 0.7
        })
        
        # External procurement
        alternatives.append({
            'type': 'external_procurement',
            'description': 'Procure resources from external vendors',
            'estimated_cost': gap.estimated_cost * 1.5,
            'time_to_implement': gap.time_to_acquire,
            'reliability': 0.9,
            'capacity_provided': gap.gap_quantity
        })
        
        # Partner coordination
        alternatives.append({
            'type': 'partner_coordination',
            'description': 'Coordinate with external partners',
            'estimated_cost': gap.estimated_cost * 1.2,
            'time_to_implement': timedelta(hours=4),
            'reliability': 0.85,
            'capacity_provided': gap.gap_quantity * 0.8
        })
        
        return alternatives
    
    def _initialize_external_partners(self) -> List[ExternalPartner]:
        """Initialize sample external partners"""
        return [
            ExternalPartner(
                name="Emergency IT Services",
                partner_type="vendor",
                available_resources=[ResourceType.HUMAN_RESOURCES, ResourceType.TECHNICAL_INFRASTRUCTURE],
                service_capabilities=["24/7 support", "emergency response", "system recovery"],
                response_time_sla=timedelta(hours=2),
                reliability_score=0.9
            ),
            ExternalPartner(
                name="Cloud Solutions Partner",
                partner_type="cloud_provider",
                available_resources=[ResourceType.CLOUD_COMPUTE, ResourceType.DATA_STORAGE],
                service_capabilities=["elastic scaling", "disaster recovery", "global deployment"],
                response_time_sla=timedelta(minutes=30),
                reliability_score=0.95
            )
        ]