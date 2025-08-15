"""
API Routes for Resource Assessment System

Provides REST API endpoints for resource assessment, capacity tracking,
gap identification, and alternative sourcing during crisis situations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

from ...engines.resource_assessment_engine import ResourceAssessmentEngine
from ...models.resource_mobilization_models import (
    ResourceInventory, ResourceRequirement, ResourceGap, Resource,
    ResourceType, ResourcePriority
)
from ...models.crisis_models_simple import Crisis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resource-assessment", tags=["resource-assessment"])

# Global engine instance
resource_assessment_engine = ResourceAssessmentEngine()


@router.post("/assess/{crisis_id}", response_model=Dict[str, Any])
async def assess_resources_for_crisis(crisis_id: str):
    """
    Perform comprehensive resource assessment for a crisis
    
    Args:
        crisis_id: ID of the crisis requiring resource assessment
        
    Returns:
        Complete resource inventory and assessment results
    """
    try:
        logger.info(f"Starting resource assessment for crisis {crisis_id}")
        
        # Create mock crisis for demonstration
        crisis = Crisis(
            id=crisis_id,
            description=f"Crisis requiring resource assessment: {crisis_id}"
        )
        
        # Perform resource assessment
        inventory = await resource_assessment_engine.assess_available_resources(crisis)
        
        # Convert to response format
        response = {
            "crisis_id": crisis_id,
            "assessment_time": inventory.assessment_time.isoformat(),
            "total_resources": inventory.total_resources,
            "resources_by_type": {
                resource_type.value: count 
                for resource_type, count in inventory.resources_by_type.items()
            },
            "available_resources_count": len(inventory.available_resources),
            "allocated_resources_count": len(inventory.allocated_resources),
            "unavailable_resources_count": len(inventory.unavailable_resources),
            "resource_pools_count": len(inventory.resource_pools),
            "capacity_summary": {
                "total_capacity": {
                    resource_type.value: capacity 
                    for resource_type, capacity in inventory.total_capacity.items()
                },
                "available_capacity": {
                    resource_type.value: capacity 
                    for resource_type, capacity in inventory.available_capacity.items()
                },
                "utilization_rates": {
                    resource_type.value: rate 
                    for resource_type, rate in inventory.utilization_rates.items()
                }
            },
            "critical_shortages": [
                {
                    "id": gap.id,
                    "resource_type": gap.resource_type.value,
                    "gap_quantity": gap.gap_quantity,
                    "severity": gap.severity.value,
                    "impact_description": gap.impact_description,
                    "alternative_options": gap.alternative_options,
                    "estimated_cost": gap.estimated_cost,
                    "time_to_acquire_hours": gap.time_to_acquire.total_seconds() / 3600
                }
                for gap in inventory.critical_shortages
            ],
            "upcoming_maintenance": inventory.upcoming_maintenance,
            "cost_summary": inventory.cost_summary,
            "assessment_status": "completed"
        }
        
        logger.info(f"Resource assessment completed for crisis {crisis_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in resource assessment for crisis {crisis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resource assessment failed: {str(e)}")


@router.get("/capacity/{resource_id}", response_model=Dict[str, Any])
async def get_resource_capacity(resource_id: str):
    """
    Get capacity information for a specific resource
    
    Args:
        resource_id: ID of the resource to check capacity for
        
    Returns:
        Resource capacity and utilization information
    """
    try:
        logger.info(f"Getting capacity information for resource {resource_id}")
        
        capacity_info = await resource_assessment_engine.track_resource_capacity([resource_id])
        
        if resource_id not in capacity_info:
            raise HTTPException(status_code=404, detail="Resource not found")
        
        return {
            "resource_id": resource_id,
            "capacity_info": capacity_info[resource_id],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting capacity for resource {resource_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get resource capacity: {str(e)}")


@router.post("/capacity/batch", response_model=Dict[str, Any])
async def get_batch_resource_capacity(resource_ids: List[str]):
    """
    Get capacity information for multiple resources
    
    Args:
        resource_ids: List of resource IDs to check capacity for
        
    Returns:
        Capacity information for all requested resources
    """
    try:
        logger.info(f"Getting capacity information for {len(resource_ids)} resources")
        
        capacity_info = await resource_assessment_engine.track_resource_capacity(resource_ids)
        
        return {
            "requested_resources": len(resource_ids),
            "capacity_info": capacity_info,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting batch capacity information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get batch capacity: {str(e)}")


@router.post("/gaps/identify", response_model=Dict[str, Any])
async def identify_resource_gaps(
    crisis_id: str,
    requirements: List[Dict[str, Any]]
):
    """
    Identify resource gaps based on requirements
    
    Args:
        crisis_id: ID of the crisis
        requirements: List of resource requirements
        
    Returns:
        Identified resource gaps and recommendations
    """
    try:
        logger.info(f"Identifying resource gaps for crisis {crisis_id}")
        
        # Convert requirements to ResourceRequirement objects
        resource_requirements = []
        for req_data in requirements:
            requirement = ResourceRequirement(
                crisis_id=crisis_id,
                resource_type=ResourceType(req_data.get('resource_type', 'human_resources')),
                required_capabilities=req_data.get('required_capabilities', []),
                quantity_needed=req_data.get('quantity_needed', 1.0),
                priority=ResourcePriority(req_data.get('priority', 2)),
                justification=req_data.get('justification', ''),
                requested_by=req_data.get('requested_by', 'system')
            )
            resource_requirements.append(requirement)
        
        # Get current inventory
        crisis = Crisis(id=crisis_id)
        inventory = await resource_assessment_engine.assess_available_resources(crisis)
        
        # Identify gaps
        gaps = await resource_assessment_engine.identify_resource_gaps(
            resource_requirements, inventory
        )
        
        # Format response
        response = {
            "crisis_id": crisis_id,
            "requirements_analyzed": len(resource_requirements),
            "gaps_identified": len(gaps),
            "gaps": [
                {
                    "id": gap.id,
                    "requirement_id": gap.requirement_id,
                    "resource_type": gap.resource_type.value,
                    "gap_quantity": gap.gap_quantity,
                    "gap_capabilities": gap.gap_capabilities,
                    "severity": gap.severity.value,
                    "impact_description": gap.impact_description,
                    "alternative_options": gap.alternative_options,
                    "procurement_options": gap.procurement_options,
                    "estimated_cost": gap.estimated_cost,
                    "time_to_acquire_hours": gap.time_to_acquire.total_seconds() / 3600,
                    "identified_at": gap.identified_at.isoformat()
                }
                for gap in gaps
            ],
            "total_estimated_cost": sum(gap.estimated_cost for gap in gaps),
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Identified {len(gaps)} resource gaps for crisis {crisis_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error identifying resource gaps for crisis {crisis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gap identification failed: {str(e)}")


@router.get("/gaps/{gap_id}/alternatives", response_model=Dict[str, Any])
async def get_gap_alternatives(gap_id: str):
    """
    Get alternative sourcing options for a resource gap
    
    Args:
        gap_id: ID of the resource gap
        
    Returns:
        Alternative sourcing options and recommendations
    """
    try:
        logger.info(f"Getting alternatives for resource gap {gap_id}")
        
        # Create mock gap for demonstration
        gap = ResourceGap(
            id=gap_id,
            resource_type=ResourceType.HUMAN_RESOURCES,
            gap_quantity=5.0,
            severity=ResourcePriority.HIGH
        )
        
        alternatives = await resource_assessment_engine.find_alternative_sources(gap)
        
        response = {
            "gap_id": gap_id,
            "alternatives_found": len(alternatives),
            "alternatives": alternatives,
            "recommendation": "Consider internal reallocation first, then external procurement",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Found {len(alternatives)} alternatives for gap {gap_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error getting alternatives for gap {gap_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alternatives: {str(e)}")


@router.get("/inventory/summary", response_model=Dict[str, Any])
async def get_inventory_summary():
    """
    Get summary of current resource inventory
    
    Returns:
        High-level summary of resource inventory
    """
    try:
        logger.info("Getting resource inventory summary")
        
        # Create mock crisis for inventory assessment
        crisis = Crisis(id="inventory_summary")
        inventory = await resource_assessment_engine.assess_available_resources(crisis)
        
        response = {
            "total_resources": inventory.total_resources,
            "resources_by_type": {
                resource_type.value: count 
                for resource_type, count in inventory.resources_by_type.items()
            },
            "availability_summary": {
                "available": len(inventory.available_resources),
                "allocated": len(inventory.allocated_resources),
                "unavailable": len(inventory.unavailable_resources)
            },
            "capacity_utilization": {
                resource_type.value: f"{rate:.1f}%" 
                for resource_type, rate in inventory.utilization_rates.items()
            },
            "critical_shortages_count": len(inventory.critical_shortages),
            "upcoming_maintenance_count": len(inventory.upcoming_maintenance),
            "total_hourly_cost": inventory.cost_summary.get('total_hourly_cost', 0.0),
            "last_updated": inventory.assessment_time.isoformat()
        }
        
        logger.info("Resource inventory summary generated")
        return response
        
    except Exception as e:
        logger.error(f"Error getting inventory summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get inventory summary: {str(e)}")


@router.get("/resources/search", response_model=Dict[str, Any])
async def search_resources(
    resource_type: Optional[str] = Query(None, description="Filter by resource type"),
    capability: Optional[str] = Query(None, description="Filter by capability"),
    status: Optional[str] = Query(None, description="Filter by status"),
    location: Optional[str] = Query(None, description="Filter by location"),
    limit: int = Query(50, description="Maximum number of results")
):
    """
    Search for resources based on criteria
    
    Args:
        resource_type: Filter by resource type
        capability: Filter by capability
        status: Filter by status
        location: Filter by location
        limit: Maximum number of results
        
    Returns:
        List of matching resources
    """
    try:
        logger.info(f"Searching resources with filters: type={resource_type}, capability={capability}, status={status}, location={location}")
        
        # Get all resources
        all_resources = await resource_assessment_engine.resource_registry.get_all_resources()
        
        # Apply filters
        filtered_resources = all_resources
        
        if resource_type:
            try:
                filter_type = ResourceType(resource_type)
                filtered_resources = [r for r in filtered_resources if r.resource_type == filter_type]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")
        
        if capability:
            filtered_resources = [
                r for r in filtered_resources 
                if any(cap.name == capability for cap in r.capabilities)
            ]
        
        if status:
            filtered_resources = [r for r in filtered_resources if r.status.value == status]
        
        if location:
            filtered_resources = [r for r in filtered_resources if location.lower() in r.location.lower()]
        
        # Limit results
        filtered_resources = filtered_resources[:limit]
        
        # Format response
        response = {
            "total_found": len(filtered_resources),
            "filters_applied": {
                "resource_type": resource_type,
                "capability": capability,
                "status": status,
                "location": location
            },
            "resources": [
                {
                    "id": resource.id,
                    "name": resource.name,
                    "resource_type": resource.resource_type.value,
                    "status": resource.status.value,
                    "capabilities": [cap.name for cap in resource.capabilities],
                    "capacity": resource.capacity,
                    "current_utilization": resource.current_utilization,
                    "available_capacity": resource.capacity - resource.current_utilization,
                    "location": resource.location,
                    "cost_per_hour": resource.cost_per_hour
                }
                for resource in filtered_resources
            ],
            "search_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Found {len(filtered_resources)} resources matching search criteria")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching resources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Resource search failed: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint for resource assessment system
    
    Returns:
        System health status
    """
    try:
        # Basic health checks
        engine_status = "healthy" if resource_assessment_engine else "unhealthy"
        
        # Check if we can access resources
        try:
            resources = await resource_assessment_engine.resource_registry.get_all_resources()
            resource_count = len(resources)
            registry_status = "healthy"
        except Exception:
            resource_count = 0
            registry_status = "unhealthy"
        
        return {
            "status": "healthy" if engine_status == "healthy" and registry_status == "healthy" else "unhealthy",
            "engine_status": engine_status,
            "registry_status": registry_status,
            "resource_count": resource_count,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }