"""
API Routes for Resource Allocation Optimizer

Provides REST API endpoints for optimal resource distribution, allocation tracking,
and utilization monitoring during crisis situations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from ...engines.resource_allocation_optimizer import (
    ResourceAllocationOptimizer, OptimizationStrategy, AllocationConstraint
)
from ...engines.resource_assessment_engine import ResourceAssessmentEngine
from ...models.resource_mobilization_models import (
    ResourceRequirement, ResourceType, ResourcePriority, AllocationStatus
)
from ...models.crisis_models_simple import Crisis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resource-allocation", tags=["resource-allocation"])

# Global engine instances
allocation_optimizer = ResourceAllocationOptimizer()
resource_assessor = ResourceAssessmentEngine()


@router.post("/optimize/{crisis_id}", response_model=Dict[str, Any])
async def optimize_resource_allocation(
    crisis_id: str,
    requirements: List[Dict[str, Any]],
    strategy: str = Query("balanced_optimization", description="Optimization strategy"),
    constraints: Optional[Dict[str, Any]] = None
):
    """
    Create optimal resource allocation plan for crisis
    
    Args:
        crisis_id: ID of the crisis requiring resource allocation
        requirements: List of resource requirements
        strategy: Optimization strategy to use
        constraints: Additional constraints for allocation
        
    Returns:
        Comprehensive optimization result with allocation plan
    """
    try:
        logger.info(f"Starting resource allocation optimization for crisis {crisis_id}")
        
        # Validate strategy
        try:
            opt_strategy = OptimizationStrategy(strategy)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid optimization strategy: {strategy}")
        
        # Create crisis object
        crisis = Crisis(
            id=crisis_id,
            description=f"Crisis requiring resource allocation: {crisis_id}"
        )
        
        # Convert requirements to ResourceRequirement objects
        resource_requirements = []
        for req_data in requirements:
            try:
                requirement = ResourceRequirement(
                    crisis_id=crisis_id,
                    resource_type=ResourceType(req_data.get('resource_type', 'human_resources')),
                    required_capabilities=req_data.get('required_capabilities', []),
                    quantity_needed=float(req_data.get('quantity_needed', 1.0)),
                    priority=ResourcePriority(req_data.get('priority', 2)),
                    duration_needed=timedelta(hours=req_data.get('duration_hours', 8)),
                    location_preference=req_data.get('location_preference', ''),
                    budget_limit=float(req_data.get('budget_limit', 0.0)),
                    justification=req_data.get('justification', ''),
                    requested_by=req_data.get('requested_by', 'api_user')
                )
                resource_requirements.append(requirement)
            except (ValueError, TypeError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid requirement data: {str(e)}")
        
        # Get available resources
        inventory = await resource_assessor.assess_available_resources(crisis)
        available_resources = inventory.available_resources
        
        # Perform optimization
        result = await allocation_optimizer.optimize_resource_allocation(
            crisis=crisis,
            requirements=resource_requirements,
            available_resources=available_resources,
            strategy=opt_strategy,
            constraints=constraints or {}
        )
        
        # Format response
        response = {
            "crisis_id": crisis_id,
            "optimization_strategy": strategy,
            "optimization_completed": True,
            "optimization_score": result.optimization_score,
            "total_cost": result.total_cost,
            "total_time_hours": result.total_time.total_seconds() / 3600,
            "allocation_plan": {
                "plan_id": result.allocation_plan.id,
                "plan_name": result.allocation_plan.plan_name,
                "total_allocations": len(result.allocation_plan.allocations),
                "plan_status": result.allocation_plan.plan_status,
                "implementation_timeline": {
                    k: v.isoformat() if isinstance(v, datetime) else str(v)
                    for k, v in result.allocation_plan.implementation_timeline.items()
                },
                "allocations": [
                    {
                        "allocation_id": alloc.id,
                        "requirement_id": alloc.requirement_id,
                        "resource_id": alloc.resource_id,
                        "allocated_quantity": alloc.allocated_quantity,
                        "priority": alloc.allocation_priority.value,
                        "estimated_duration_hours": alloc.estimated_duration.total_seconds() / 3600,
                        "cost_estimate": alloc.cost_estimate,
                        "status": alloc.status.value,
                        "allocated_by": alloc.allocated_by
                    }
                    for alloc in result.allocation_plan.allocations
                ],
                "risk_factors": result.allocation_plan.risk_factors,
                "contingency_plans": result.allocation_plan.contingency_plans
            },
            "resource_utilization": result.resource_utilization,
            "unmet_requirements": [
                {
                    "requirement_id": req.id,
                    "resource_type": req.resource_type.value,
                    "quantity_needed": req.quantity_needed,
                    "priority": req.priority.value,
                    "justification": req.justification
                }
                for req in result.unmet_requirements
            ],
            "optimization_metrics": result.optimization_metrics,
            "recommendations": result.recommendations,
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Resource allocation optimization completed for crisis {crisis_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in resource allocation optimization for crisis {crisis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@router.get("/track/{allocation_id}", response_model=Dict[str, Any])
async def track_allocation_progress(allocation_id: str):
    """
    Track progress of specific resource allocation
    
    Args:
        allocation_id: ID of allocation to track
        
    Returns:
        Allocation progress and performance information
    """
    try:
        logger.info(f"Tracking allocation progress for {allocation_id}")
        
        progress = await allocation_optimizer.track_allocation_progress(allocation_id)
        
        response = {
            "allocation_id": allocation_id,
            "tracking_data": progress,
            "tracking_timestamp": datetime.utcnow().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error tracking allocation {allocation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to track allocation: {str(e)}")


@router.put("/adjust/{allocation_id}", response_model=Dict[str, Any])
async def adjust_allocation(
    allocation_id: str,
    adjustments: Dict[str, Any]
):
    """
    Adjust existing resource allocation
    
    Args:
        allocation_id: ID of allocation to adjust
        adjustments: Dictionary of adjustments to make
        
    Returns:
        Adjustment result and updated allocation information
    """
    try:
        logger.info(f"Adjusting allocation {allocation_id}")
        
        success = await allocation_optimizer.adjust_allocation(allocation_id, adjustments)
        
        if success:
            # Get updated tracking information
            updated_progress = await allocation_optimizer.track_allocation_progress(allocation_id)
            
            response = {
                "allocation_id": allocation_id,
                "adjustment_successful": True,
                "adjustments_applied": adjustments,
                "updated_allocation": updated_progress,
                "adjustment_timestamp": datetime.utcnow().isoformat()
            }
        else:
            response = {
                "allocation_id": allocation_id,
                "adjustment_successful": False,
                "error": "Failed to apply adjustments",
                "adjustment_timestamp": datetime.utcnow().isoformat()
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error adjusting allocation {allocation_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to adjust allocation: {str(e)}")


@router.post("/utilization/monitor", response_model=Dict[str, Any])
async def monitor_resource_utilization(
    resource_ids: List[str],
    time_window_hours: float = Query(24.0, description="Time window for monitoring in hours")
):
    """
    Monitor resource utilization and performance
    
    Args:
        resource_ids: List of resource IDs to monitor
        time_window_hours: Time window for monitoring in hours
        
    Returns:
        Resource utilization information and performance metrics
    """
    try:
        logger.info(f"Monitoring utilization for {len(resource_ids)} resources")
        
        time_window = timedelta(hours=time_window_hours)
        utilization_data = await allocation_optimizer.monitor_resource_utilization(
            resource_ids, time_window
        )
        
        # Format response
        response = {
            "monitored_resources": len(resource_ids),
            "time_window_hours": time_window_hours,
            "utilization_data": {
                resource_id: {
                    "resource_id": util.resource_id,
                    "utilization_period": {
                        "start": util.utilization_period['start'].isoformat(),
                        "end": util.utilization_period['end'].isoformat()
                    },
                    "planned_utilization": util.planned_utilization,
                    "actual_utilization": util.actual_utilization,
                    "efficiency_score": util.efficiency_score,
                    "performance_metrics": util.performance_metrics,
                    "issues_encountered": util.issues_encountered,
                    "optimization_recommendations": util.optimization_recommendations,
                    "cost_efficiency": util.cost_efficiency,
                    "recorded_at": util.recorded_at.isoformat()
                }
                for resource_id, util in utilization_data.items()
            },
            "summary": {
                "average_utilization": sum(u.actual_utilization for u in utilization_data.values()) / len(utilization_data) if utilization_data else 0,
                "average_efficiency": sum(u.efficiency_score for u in utilization_data.values()) / len(utilization_data) if utilization_data else 0,
                "total_issues": sum(len(u.issues_encountered) for u in utilization_data.values()),
                "resources_needing_optimization": len([u for u in utilization_data.values() if u.efficiency_score < 0.8])
            },
            "monitoring_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Utilization monitoring completed for {len(resource_ids)} resources")
        return response
        
    except Exception as e:
        logger.error(f"Error monitoring resource utilization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Utilization monitoring failed: {str(e)}")


@router.post("/utilization/optimize", response_model=Dict[str, Any])
async def optimize_utilization(
    allocation_plan_data: Dict[str, Any]
):
    """
    Optimize resource utilization for allocation plan
    
    Args:
        allocation_plan_data: Allocation plan data to optimize
        
    Returns:
        Utilization optimization recommendations and improvements
    """
    try:
        logger.info("Optimizing resource utilization")
        
        # Create mock allocation plan from data
        from ...models.resource_mobilization_models import AllocationPlan
        allocation_plan = AllocationPlan(
            id=allocation_plan_data.get('plan_id', 'optimization_plan'),
            crisis_id=allocation_plan_data.get('crisis_id', 'unknown'),
            plan_name=allocation_plan_data.get('plan_name', 'Utilization Optimization Plan')
        )
        
        optimization_results = await allocation_optimizer.optimize_utilization(allocation_plan)
        
        response = {
            "plan_id": allocation_plan.id,
            "optimization_results": optimization_results,
            "improvement_potential": {
                "efficiency_gain": optimization_results['optimized_efficiency'] - optimization_results['current_efficiency'],
                "cost_savings": optimization_results['potential_savings'],
                "implementation_complexity": "medium"
            },
            "next_steps": [
                "Review optimization recommendations",
                "Prioritize high-impact, low-risk actions",
                "Implement changes in phases",
                "Monitor results and adjust as needed"
            ],
            "optimization_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info("Resource utilization optimization completed")
        return response
        
    except Exception as e:
        logger.error(f"Error optimizing utilization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Utilization optimization failed: {str(e)}")


@router.get("/strategies", response_model=Dict[str, Any])
async def get_optimization_strategies():
    """
    Get available optimization strategies
    
    Returns:
        List of available optimization strategies and their descriptions
    """
    try:
        strategies = {
            "priority_based": {
                "name": "Priority Based",
                "description": "Prioritize high-priority requirements first",
                "best_for": "Crisis situations with clear priority hierarchy",
                "weights": {"priority": 0.5, "capability": 0.3, "availability": 0.2}
            },
            "cost_minimization": {
                "name": "Cost Minimization",
                "description": "Minimize total allocation cost",
                "best_for": "Budget-constrained scenarios",
                "weights": {"cost": 0.4, "priority": 0.3, "capability": 0.2, "availability": 0.1}
            },
            "time_minimization": {
                "name": "Time Minimization",
                "description": "Minimize time to resource deployment",
                "best_for": "Time-critical emergency situations",
                "weights": {"availability": 0.4, "priority": 0.3, "capability": 0.2, "efficiency": 0.1}
            },
            "balanced_optimization": {
                "name": "Balanced Optimization",
                "description": "Balance all factors for optimal allocation",
                "best_for": "General crisis response scenarios",
                "weights": {"priority": 0.25, "cost": 0.2, "capability": 0.2, "availability": 0.2, "efficiency": 0.15}
            },
            "capacity_maximization": {
                "name": "Capacity Maximization",
                "description": "Maximize resource capacity utilization",
                "best_for": "Long-term resource planning",
                "weights": {"efficiency": 0.4, "availability": 0.3, "capability": 0.2, "priority": 0.1}
            }
        }
        
        return {
            "available_strategies": strategies,
            "default_strategy": "balanced_optimization",
            "strategy_selection_guidance": {
                "emergency_situations": "time_minimization",
                "budget_constraints": "cost_minimization",
                "clear_priorities": "priority_based",
                "general_use": "balanced_optimization",
                "resource_planning": "capacity_maximization"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting optimization strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get strategies: {str(e)}")


@router.get("/constraints", response_model=Dict[str, Any])
async def get_allocation_constraints():
    """
    Get available allocation constraints
    
    Returns:
        List of available allocation constraints and their descriptions
    """
    try:
        constraints = {
            "budget_limit": {
                "name": "Budget Limit",
                "description": "Maximum budget for resource allocation",
                "type": "float",
                "example": 50000.0,
                "unit": "currency"
            },
            "time_limit": {
                "name": "Time Limit",
                "description": "Maximum time for resource deployment",
                "type": "duration",
                "example": "24 hours",
                "unit": "hours"
            },
            "location_constraint": {
                "name": "Location Constraint",
                "description": "Required location for resources",
                "type": "string",
                "example": "New York",
                "unit": "location"
            },
            "required_skills": {
                "name": "Required Skills",
                "description": "Mandatory skills for resource allocation",
                "type": "list",
                "example": ["incident_response", "security_analysis"],
                "unit": "skills"
            },
            "max_cost_per_hour": {
                "name": "Maximum Cost Per Hour",
                "description": "Maximum hourly cost for resources",
                "type": "float",
                "example": 200.0,
                "unit": "currency_per_hour"
            }
        }
        
        return {
            "available_constraints": constraints,
            "constraint_usage": {
                "budget_limit": "Set maximum total cost for allocation",
                "time_limit": "Limit deployment time for urgent situations",
                "location_constraint": "Ensure resources are in required location",
                "required_skills": "Ensure allocated resources have necessary capabilities",
                "max_cost_per_hour": "Control hourly resource costs"
            },
            "example_constraints": {
                "emergency_response": {
                    "budget_limit": 100000.0,
                    "time_limit": "4 hours",
                    "required_skills": ["emergency_response", "crisis_management"]
                },
                "cost_conscious": {
                    "budget_limit": 25000.0,
                    "max_cost_per_hour": 150.0
                },
                "location_specific": {
                    "location_constraint": "Data Center A",
                    "required_skills": ["on_site_support"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting allocation constraints: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get constraints: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint for resource allocation system
    
    Returns:
        System health status
    """
    try:
        # Basic health checks
        optimizer_status = "healthy" if allocation_optimizer else "unhealthy"
        assessor_status = "healthy" if resource_assessor else "unhealthy"
        
        return {
            "status": "healthy" if optimizer_status == "healthy" and assessor_status == "healthy" else "unhealthy",
            "optimizer_status": optimizer_status,
            "assessor_status": assessor_status,
            "active_allocations": len(allocation_optimizer.active_allocations),
            "allocation_history_count": len(allocation_optimizer.allocation_history),
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