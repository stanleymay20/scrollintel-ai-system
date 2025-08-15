"""
API Routes for External Resource Coordination

Provides REST API endpoints for coordinating with external partners,
managing resource requests, and activating partnership protocols
during crisis situations.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from ...engines.external_resource_coordinator import (
    ExternalResourceCoordinator, PartnerType, RequestStatus, ActivationLevel
)
from ...models.resource_mobilization_models import (
    ExternalPartner, ExternalResourceRequest, ResourceType, ResourcePriority
)
from ...models.crisis_models_simple import Crisis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/external-resources", tags=["external-resources"])

# Global coordinator instance
external_coordinator = ExternalResourceCoordinator()


@router.post("/coordinate/{crisis_id}", response_model=Dict[str, Any])
async def coordinate_external_resources(
    crisis_id: str,
    resource_requirements: List[Dict[str, Any]]
):
    """
    Coordinate with external partners for crisis response
    
    Args:
        crisis_id: ID of the crisis requiring external resources
        resource_requirements: List of resource requirements
        
    Returns:
        Comprehensive coordination results and partner responses
    """
    try:
        logger.info(f"Starting external resource coordination for crisis {crisis_id}")
        
        # Create crisis object
        crisis = Crisis(
            id=crisis_id,
            description=f"Crisis requiring external resource coordination: {crisis_id}"
        )
        
        # Convert requirements to ResourceRequirement objects
        from ...models.resource_mobilization_models import ResourceRequirement
        requirements = []
        
        for req_data in resource_requirements:
            try:
                requirement = ResourceRequirement(
                    crisis_id=crisis_id,
                    resource_type=ResourceType(req_data.get('resource_type', 'external_services')),
                    required_capabilities=req_data.get('required_capabilities', []),
                    quantity_needed=float(req_data.get('quantity_needed', 1.0)),
                    priority=ResourcePriority(req_data.get('priority', 2)),
                    duration_needed=timedelta(hours=req_data.get('duration_hours', 8)),
                    budget_limit=float(req_data.get('budget_limit', 0.0)),
                    justification=req_data.get('justification', ''),
                    requested_by=req_data.get('requested_by', 'api_user')
                )
                requirements.append(requirement)
            except (ValueError, TypeError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid requirement data: {str(e)}")
        
        # Perform coordination
        coordination_result = await external_coordinator.coordinate_with_partners(
            crisis, requirements
        )
        
        # Format response
        response = {
            "crisis_id": crisis_id,
            "coordination_successful": True,
            "coordination_summary": {
                "total_partners_contacted": coordination_result['total_partners_contacted'],
                "total_requests_submitted": coordination_result['total_requests_submitted'],
                "estimated_response_time_hours": coordination_result['estimated_response_time'].total_seconds() / 3600,
                "coordination_status": coordination_result['coordination_status']
            },
            "partner_activations": {
                "successful": len(coordination_result['activation_results']['successful_activations']),
                "failed": len(coordination_result['activation_results']['failed_activations']),
                "total_activated": coordination_result['activation_results']['total_activated'],
                "activation_details": [
                    {
                        "partner_id": act['partner_id'],
                        "partner_name": act['partner_name'],
                        "activation_level": act['activation_level'],
                        "activation_successful": act['activation_successful'],
                        "estimated_response_time_hours": act['estimated_response_time'].total_seconds() / 3600
                    }
                    for act in coordination_result['activation_results']['successful_activations']
                ]
            },
            "resource_requests": {
                "submitted": coordination_result['request_results']['total_submitted'],
                "failed_submissions": len(coordination_result['request_results']['failed_submissions']),
                "request_details": [
                    {
                        "request_id": req.id,
                        "partner_id": req.partner_id,
                        "resource_type": req.resource_type.value,
                        "quantity_requested": req.quantity_requested,
                        "urgency_level": req.urgency_level.value,
                        "duration_hours": req.duration_needed.total_seconds() / 3600,
                        "status": req.request_status
                    }
                    for req in coordination_result['request_results']['submitted_requests']
                ]
            },
            "monitoring_setup": coordination_result['response_monitoring'],
            "next_steps": [
                "Monitor partner responses and acknowledgments",
                "Follow up on pending requests within 2 hours",
                "Prepare for resource integration and deployment",
                "Maintain regular communication with activated partners"
            ],
            "coordination_timestamp": coordination_result['coordination_timestamp'].isoformat()
        }
        
        logger.info(f"External resource coordination completed for crisis {crisis_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in external resource coordination for crisis {crisis_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Coordination failed: {str(e)}")


@router.post("/requests/manage", response_model=Dict[str, Any])
async def manage_resource_requests(
    request_ids: List[str]
):
    """
    Manage external resource requests throughout their lifecycle
    
    Args:
        request_ids: List of request IDs to manage
        
    Returns:
        Request management results and status updates
    """
    try:
        logger.info(f"Managing {len(request_ids)} external resource requests")
        
        # Get requests from coordinator's active requests
        requests = []
        for request_id in request_ids:
            if request_id in external_coordinator.active_requests:
                requests.append(external_coordinator.active_requests[request_id])
            else:
                # Create mock request for demonstration
                mock_request = ExternalResourceRequest(
                    id=request_id,
                    crisis_id="unknown",
                    partner_id="unknown",
                    resource_type=ResourceType.EXTERNAL_SERVICES,
                    quantity_requested=1.0,
                    urgency_level=ResourcePriority.MEDIUM
                )
                requests.append(mock_request)
        
        # Manage requests
        management_results = await external_coordinator.manage_resource_requests(requests)
        
        # Format response
        response = {
            "management_summary": {
                "total_requests_managed": management_results['total_requests'],
                "successful_requests": len(management_results['successful_requests']),
                "failed_requests": len(management_results['failed_requests']),
                "pending_requests": len(management_results['pending_requests'])
            },
            "status_breakdown": management_results['request_status_summary'],
            "management_actions": [
                {
                    "request_id": action['request_id'],
                    "current_status": action['current_status'],
                    "action_taken": action['action_taken'],
                    "action_successful": action['action_successful'],
                    "next_check": action['next_check'].isoformat()
                }
                for action in management_results['management_actions']
            ],
            "request_categories": {
                "successful": management_results['successful_requests'],
                "failed": management_results['failed_requests'],
                "pending": management_results['pending_requests']
            },
            "next_steps": management_results['next_steps'],
            "management_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Resource request management completed")
        return response
        
    except Exception as e:
        logger.error(f"Error managing resource requests: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Request management failed: {str(e)}")


@router.post("/partners/{partner_id}/activate", response_model=Dict[str, Any])
async def activate_partnership(
    partner_id: str,
    activation_level: str = Query(..., description="Activation level (standby, alert, activated, full_deployment, emergency_response)"),
    crisis_context: Optional[Dict[str, Any]] = None
):
    """
    Activate partnership coordination protocols
    
    Args:
        partner_id: ID of partner to activate
        activation_level: Level of activation required
        crisis_context: Context information about the crisis
        
    Returns:
        Partnership activation results and status
    """
    try:
        logger.info(f"Activating partnership for partner {partner_id}")
        
        # Validate activation level
        try:
            level = ActivationLevel(activation_level)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid activation level: {activation_level}")
        
        # Set default crisis context if not provided
        if crisis_context is None:
            crisis_context = {
                "crisis_id": f"activation_{partner_id}_{datetime.now().timestamp()}",
                "activation_reason": "manual_activation",
                "urgency": "high"
            }
        
        # Activate partnership
        activation_result = await external_coordinator.activate_partnership_protocols(
            partner_id, level, crisis_context
        )
        
        # Format response
        response = {
            "partner_id": partner_id,
            "activation_summary": {
                "partner_name": activation_result['partner_name'],
                "activation_level": activation_result['activation_level'],
                "activation_successful": activation_result['activation_successful'],
                "estimated_response_time_hours": activation_result['estimated_response_time'].total_seconds() / 3600,
                "activation_timestamp": activation_result['activation_timestamp'].isoformat()
            },
            "activation_steps": [
                {
                    "step": step['step'],
                    "description": step['description'],
                    "status": step['status'],
                    "timestamp": step['timestamp'].isoformat()
                }
                for step in activation_result['activation_steps']
            ],
            "communication_setup": {
                "established_channels": len(activation_result['communication_setup']['established_channels']),
                "failed_channels": len(activation_result['communication_setup']['failed_channels']),
                "channel_details": activation_result['communication_setup']['established_channels']
            },
            "verification_result": activation_result['verification_result'],
            "next_actions": activation_result['next_actions'],
            "crisis_context": crisis_context
        }
        
        logger.info(f"Partnership activation completed for partner {partner_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating partnership for {partner_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Partnership activation failed: {str(e)}")


@router.get("/partners", response_model=Dict[str, Any])
async def get_external_partners(
    partner_type: Optional[str] = Query(None, description="Filter by partner type"),
    resource_type: Optional[str] = Query(None, description="Filter by available resource type"),
    min_reliability: Optional[float] = Query(None, description="Minimum reliability score")
):
    """
    Get list of external partners with optional filtering
    
    Args:
        partner_type: Filter by partner type
        resource_type: Filter by available resource type
        min_reliability: Minimum reliability score
        
    Returns:
        List of external partners matching criteria
    """
    try:
        logger.info("Getting external partners list")
        
        # Get all partners
        all_partners = await external_coordinator.partner_registry.get_all_partners()
        
        # Apply filters
        filtered_partners = all_partners
        
        if partner_type:
            filtered_partners = [p for p in filtered_partners if p.partner_type == partner_type]
        
        if resource_type:
            try:
                filter_resource_type = ResourceType(resource_type)
                filtered_partners = [
                    p for p in filtered_partners 
                    if filter_resource_type in p.available_resources
                ]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid resource type: {resource_type}")
        
        if min_reliability is not None:
            filtered_partners = [p for p in filtered_partners if p.reliability_score >= min_reliability]
        
        # Format response
        response = {
            "total_partners": len(filtered_partners),
            "filters_applied": {
                "partner_type": partner_type,
                "resource_type": resource_type,
                "min_reliability": min_reliability
            },
            "partners": [
                {
                    "partner_id": partner.id,
                    "name": partner.name,
                    "partner_type": partner.partner_type,
                    "available_resources": [rt.value for rt in partner.available_resources],
                    "service_capabilities": partner.service_capabilities,
                    "response_time_hours": partner.response_time_sla.total_seconds() / 3600,
                    "reliability_score": partner.reliability_score,
                    "cost_structure": partner.cost_structure,
                    "status": partner.status,
                    "last_engagement": partner.last_engagement.isoformat() if partner.last_engagement else None,
                    "contact_info": {
                        k: v for k, v in partner.contact_info.items() 
                        if k in ['primary_phone', 'primary_email', 'website']
                    }
                }
                for partner in filtered_partners
            ],
            "partner_summary": {
                "by_type": {},
                "by_resource": {},
                "average_reliability": sum(p.reliability_score for p in filtered_partners) / len(filtered_partners) if filtered_partners else 0,
                "average_response_time_hours": sum(p.response_time_sla.total_seconds() / 3600 for p in filtered_partners) / len(filtered_partners) if filtered_partners else 0
            },
            "query_timestamp": datetime.utcnow().isoformat()
        }
        
        # Calculate summary statistics
        for partner in filtered_partners:
            # By type
            partner_type_key = partner.partner_type
            response["partner_summary"]["by_type"][partner_type_key] = \
                response["partner_summary"]["by_type"].get(partner_type_key, 0) + 1
            
            # By resource
            for resource in partner.available_resources:
                resource_key = resource.value
                response["partner_summary"]["by_resource"][resource_key] = \
                    response["partner_summary"]["by_resource"].get(resource_key, 0) + 1
        
        logger.info(f"Retrieved {len(filtered_partners)} external partners")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting external partners: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get partners: {str(e)}")


@router.get("/partners/{partner_id}/performance", response_model=Dict[str, Any])
async def get_partner_performance(
    partner_id: str,
    time_window_days: int = Query(30, description="Time window for performance analysis in days")
):
    """
    Get performance metrics for specific partner
    
    Args:
        partner_id: ID of partner to analyze
        time_window_days: Time window for analysis in days
        
    Returns:
        Partner performance metrics and analysis
    """
    try:
        logger.info(f"Getting performance metrics for partner {partner_id}")
        
        time_window = timedelta(days=time_window_days)
        
        # Get performance data
        performance_data = await external_coordinator.monitor_partner_performance(
            [partner_id], time_window
        )
        
        if partner_id not in performance_data:
            raise HTTPException(status_code=404, detail=f"Partner {partner_id} not found or no performance data available")
        
        performance = performance_data[partner_id]
        
        # Format response
        response = {
            "partner_id": partner_id,
            "analysis_period": {
                "time_window_days": time_window_days,
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            "performance_summary": {
                "total_requests": performance.total_requests,
                "successful_requests": performance.successful_requests,
                "failed_requests": performance.failed_requests,
                "success_rate": (performance.successful_requests / performance.total_requests * 100) if performance.total_requests > 0 else 0,
                "reliability_score": performance.reliability_score,
                "quality_score": performance.quality_score,
                "cost_efficiency": performance.cost_efficiency
            },
            "response_metrics": {
                "average_response_time_hours": performance.average_response_time.total_seconds() / 3600,
                "average_fulfillment_time_hours": performance.average_fulfillment_time.total_seconds() / 3600,
                "last_engagement": performance.last_engagement.isoformat() if performance.last_engagement else None
            },
            "performance_trends": {
                "response_time_trend": performance.performance_trends.get("response_time", []),
                "quality_score_trend": performance.performance_trends.get("quality_score", []),
                "cost_efficiency_trend": performance.performance_trends.get("cost_efficiency", [])
            },
            "feedback_analysis": performance.feedback_summary,
            "performance_rating": self._calculate_performance_rating(performance),
            "recommendations": self._generate_performance_recommendations(performance)
        }
        
        logger.info(f"Performance analysis completed for partner {partner_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting partner performance for {partner_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")


@router.get("/requests/{request_id}/status", response_model=Dict[str, Any])
async def get_request_status(request_id: str):
    """
    Get status of external resource request
    
    Args:
        request_id: ID of request to check
        
    Returns:
        Current request status and details
    """
    try:
        logger.info(f"Getting status for request {request_id}")
        
        # Get request from active requests
        if request_id in external_coordinator.active_requests:
            request = external_coordinator.active_requests[request_id]
            
            # Get current status
            current_status = await external_coordinator.request_manager.get_request_status(request_id)
            
            response = {
                "request_id": request_id,
                "current_status": current_status,
                "request_details": {
                    "crisis_id": request.crisis_id,
                    "partner_id": request.partner_id,
                    "resource_type": request.resource_type.value,
                    "quantity_requested": request.quantity_requested,
                    "urgency_level": request.urgency_level.value,
                    "duration_hours": request.duration_needed.total_seconds() / 3600,
                    "budget_approved": request.budget_approved,
                    "requested_by": request.requested_by,
                    "request_time": request.request_time.isoformat()
                },
                "status_history": [
                    {
                        "status": "submitted",
                        "timestamp": request.request_time.isoformat(),
                        "notes": "Request submitted to partner"
                    },
                    {
                        "status": current_status,
                        "timestamp": datetime.utcnow().isoformat(),
                        "notes": f"Current status: {current_status}"
                    }
                ],
                "estimated_completion": self._estimate_completion_time(request, current_status),
                "next_actions": self._get_request_next_actions(current_status),
                "status_timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Request not found in active requests
            response = {
                "request_id": request_id,
                "current_status": "not_found",
                "error": "Request not found in active requests",
                "status_timestamp": datetime.utcnow().isoformat()
            }
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting request status for {request_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get request status: {str(e)}")


@router.get("/coordination/history", response_model=Dict[str, Any])
async def get_coordination_history(
    limit: int = Query(50, description="Maximum number of events to return"),
    event_type: Optional[str] = Query(None, description="Filter by event type")
):
    """
    Get coordination history and events
    
    Args:
        limit: Maximum number of events to return
        event_type: Filter by specific event type
        
    Returns:
        Coordination history and event details
    """
    try:
        logger.info("Getting coordination history")
        
        # Get coordination history
        history = external_coordinator.coordination_history
        
        # Apply filters
        if event_type:
            history = [event for event in history if event.event_type == event_type]
        
        # Sort by timestamp (most recent first) and limit
        history = sorted(history, key=lambda x: x.timestamp, reverse=True)[:limit]
        
        # Format response
        response = {
            "total_events": len(history),
            "filters_applied": {
                "limit": limit,
                "event_type": event_type
            },
            "coordination_events": [
                {
                    "event_id": event.id,
                    "partner_id": event.partner_id,
                    "event_type": event.event_type,
                    "event_description": event.event_description,
                    "timestamp": event.timestamp.isoformat(),
                    "initiated_by": event.initiated_by,
                    "status": event.status,
                    "metadata": event.metadata
                }
                for event in history
            ],
            "event_summary": {
                "by_type": {},
                "by_partner": {},
                "recent_activity": len([e for e in history if e.timestamp > datetime.utcnow() - timedelta(hours=24)])
            },
            "query_timestamp": datetime.utcnow().isoformat()
        }
        
        # Calculate summary statistics
        for event in history:
            # By type
            type_key = event.event_type
            response["event_summary"]["by_type"][type_key] = \
                response["event_summary"]["by_type"].get(type_key, 0) + 1
            
            # By partner
            partner_key = event.partner_id
            response["event_summary"]["by_partner"][partner_key] = \
                response["event_summary"]["by_partner"].get(partner_key, 0) + 1
        
        logger.info(f"Retrieved {len(history)} coordination events")
        return response
        
    except Exception as e:
        logger.error(f"Error getting coordination history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get coordination history: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint for external resource coordination system
    
    Returns:
        System health status
    """
    try:
        # Basic health checks
        coordinator_status = "healthy" if external_coordinator else "unhealthy"
        
        # Check partner registry
        try:
            partners = await external_coordinator.partner_registry.get_all_partners()
            partner_count = len(partners)
            registry_status = "healthy"
        except Exception:
            partner_count = 0
            registry_status = "unhealthy"
        
        return {
            "status": "healthy" if coordinator_status == "healthy" and registry_status == "healthy" else "unhealthy",
            "coordinator_status": coordinator_status,
            "registry_status": registry_status,
            "partner_count": partner_count,
            "active_requests": len(external_coordinator.active_requests),
            "coordination_history_count": len(external_coordinator.coordination_history),
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


# Helper methods
def _calculate_performance_rating(performance) -> str:
    """Calculate overall performance rating"""
    score = (performance.reliability_score + performance.quality_score + performance.cost_efficiency) / 3
    
    if score >= 0.9:
        return "excellent"
    elif score >= 0.8:
        return "good"
    elif score >= 0.7:
        return "satisfactory"
    elif score >= 0.6:
        return "needs_improvement"
    else:
        return "poor"


def _generate_performance_recommendations(performance) -> List[str]:
    """Generate performance improvement recommendations"""
    recommendations = []
    
    if performance.reliability_score < 0.8:
        recommendations.append("Improve reliability through better SLA adherence")
    
    if performance.quality_score < 0.8:
        recommendations.append("Focus on quality improvement initiatives")
    
    if performance.cost_efficiency < 0.8:
        recommendations.append("Review cost structure and negotiate better rates")
    
    if performance.average_response_time.total_seconds() > 7200:  # 2 hours
        recommendations.append("Work on reducing response times")
    
    if not recommendations:
        recommendations.append("Maintain current performance levels")
    
    return recommendations


def _estimate_completion_time(request, current_status: str) -> Optional[str]:
    """Estimate completion time based on request and status"""
    if current_status == RequestStatus.FULFILLED.value:
        return "completed"
    elif current_status == RequestStatus.IN_PROGRESS.value:
        estimated_time = datetime.utcnow() + timedelta(hours=4)
        return estimated_time.isoformat()
    elif current_status == RequestStatus.SUBMITTED.value:
        estimated_time = datetime.utcnow() + timedelta(hours=8)
        return estimated_time.isoformat()
    else:
        return None


def _get_request_next_actions(current_status: str) -> List[str]:
    """Get next actions based on request status"""
    actions = {
        RequestStatus.SUBMITTED.value: [
            "Wait for partner acknowledgment",
            "Follow up if no response within 2 hours"
        ],
        RequestStatus.IN_PROGRESS.value: [
            "Monitor progress with partner",
            "Prepare for resource integration"
        ],
        RequestStatus.FULFILLED.value: [
            "Coordinate resource deployment",
            "Confirm resource integration"
        ],
        RequestStatus.REJECTED.value: [
            "Contact alternative partners",
            "Review and adjust requirements"
        ],
        RequestStatus.FAILED.value: [
            "Escalate to management",
            "Initiate alternative sourcing"
        ]
    }
    
    return actions.get(current_status, ["Monitor request status", "Take appropriate action based on updates"])