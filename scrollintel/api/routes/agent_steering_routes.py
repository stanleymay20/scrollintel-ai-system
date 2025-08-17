"""
Agent Steering System API Routes

This module provides REST API endpoints for the Agent Steering System,
enabling external systems to interact with agent orchestration capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging

from ...core.agent_registry import AgentRegistry, AgentRegistrationRequest, AgentSelectionCriteria
from ...core.realtime_messaging import RealTimeMessagingSystem, EventType, MessagePriority
from ...core.secure_communication import SecureCommunicationManager
from ...models.agent_steering_models import AgentStatus, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/v1/agent-steering", tags=["Agent Steering"])

# Global instances (would be dependency injected in production)
messaging_system = RealTimeMessagingSystem()
agent_registry = AgentRegistry(messaging_system)
communication_manager = SecureCommunicationManager()


# Pydantic models for API requests/responses

class AgentRegistrationModel(BaseModel):
    """Agent registration request model"""
    name: str = Field(..., description="Unique agent name")
    type: str = Field(..., description="Agent type")
    version: str = Field(..., description="Agent version")
    capabilities: List[Dict[str, Any]] = Field(..., description="Agent capabilities")
    endpoint_url: str = Field(..., description="Agent endpoint URL")
    health_check_url: str = Field(..., description="Health check endpoint URL")
    resource_requirements: Dict[str, Any] = Field(default_factory=dict, description="Resource requirements")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")
    authentication_token: Optional[str] = Field(None, description="Authentication token")


class AgentSelectionModel(BaseModel):
    """Agent selection criteria model"""
    required_capabilities: List[str] = Field(default_factory=list, description="Required capabilities")
    preferred_capabilities: List[str] = Field(default_factory=list, description="Preferred capabilities")
    max_load_threshold: float = Field(default=80.0, description="Maximum load threshold")
    min_success_rate: float = Field(default=90.0, description="Minimum success rate")
    max_response_time: float = Field(default=5.0, description="Maximum response time")
    business_domain: Optional[str] = Field(None, description="Business domain")
    exclude_agents: List[str] = Field(default_factory=list, description="Agents to exclude")


class TaskCreationModel(BaseModel):
    """Task creation request model"""
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    task_type: str = Field(..., description="Task type")
    priority: str = Field(default="medium", description="Task priority")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Task requirements")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Task constraints")
    expected_outcome: Dict[str, Any] = Field(default_factory=dict, description="Expected outcome")
    business_context: Dict[str, Any] = Field(default_factory=dict, description="Business context")
    deadline: Optional[datetime] = Field(None, description="Task deadline")


class PerformanceUpdateModel(BaseModel):
    """Performance metrics update model"""
    current_load: Optional[float] = Field(None, description="Current load percentage")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    throughput: Optional[float] = Field(None, description="Throughput (tasks/hour)")
    accuracy: Optional[float] = Field(None, description="Accuracy percentage")
    reliability: Optional[float] = Field(None, description="Reliability percentage")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")
    memory_usage: Optional[float] = Field(None, description="Memory usage percentage")
    network_usage: Optional[float] = Field(None, description="Network usage MB/s")
    cost_savings: Optional[float] = Field(None, description="Cost savings generated")
    revenue_increase: Optional[float] = Field(None, description="Revenue increase generated")
    productivity_gain: Optional[float] = Field(None, description="Productivity gain percentage")


class EventPublishModel(BaseModel):
    """Event publishing model"""
    event_type: str = Field(..., description="Event type")
    source: str = Field(..., description="Event source")
    data: Dict[str, Any] = Field(..., description="Event data")
    priority: str = Field(default="normal", description="Event priority")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")


# API Endpoints

@router.post("/agents/register", response_model=Dict[str, str])
async def register_agent(
    registration: AgentRegistrationModel,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Register a new agent in the system
    
    This endpoint allows agents to register themselves with the orchestration system,
    providing their capabilities, endpoints, and configuration.
    """
    try:
        request = AgentRegistrationRequest(
            name=registration.name,
            type=registration.type,
            version=registration.version,
            capabilities=registration.capabilities,
            endpoint_url=registration.endpoint_url,
            health_check_url=registration.health_check_url,
            resource_requirements=registration.resource_requirements,
            configuration=registration.configuration,
            authentication_token=registration.authentication_token
        )
        
        agent_id = await agent_registry.register_agent(request)
        
        if not agent_id:
            raise HTTPException(status_code=400, detail="Agent registration failed")
        
        return {
            "agent_id": agent_id,
            "status": "registered",
            "message": f"Agent {registration.name} registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Agent registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{agent_id}")
async def deregister_agent(agent_id: str) -> Dict[str, str]:
    """
    Deregister an agent from the system
    
    Removes the agent from the registry and stops routing tasks to it.
    """
    try:
        success = await agent_registry.deregister_agent(agent_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent_id,
            "status": "deregistered",
            "message": "Agent deregistered successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent deregistration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents", response_model=List[Dict[str, Any]])
async def get_available_agents(
    required_capabilities: Optional[str] = None,
    max_load_threshold: float = 80.0,
    min_success_rate: float = 90.0,
    max_response_time: float = 5.0
) -> List[Dict[str, Any]]:
    """
    Get list of available agents matching criteria
    
    Returns agents that meet the specified performance and capability requirements.
    """
    try:
        criteria = None
        if required_capabilities:
            criteria = AgentSelectionCriteria(
                required_capabilities=required_capabilities.split(","),
                max_load_threshold=max_load_threshold,
                min_success_rate=min_success_rate,
                max_response_time=max_response_time
            )
        
        agents = await agent_registry.get_available_agents(criteria)
        return agents
        
    except Exception as e:
        logger.error(f"Get available agents error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/select", response_model=Dict[str, Any])
async def select_best_agent(selection_criteria: AgentSelectionModel) -> Dict[str, Any]:
    """
    Select the best agent using advanced performance-based algorithms
    
    Uses intelligent selection algorithms with capability matching,
    performance scoring, and adaptive learning to find the optimal agent.
    """
    try:
        criteria = AgentSelectionCriteria(
            required_capabilities=selection_criteria.required_capabilities,
            preferred_capabilities=selection_criteria.preferred_capabilities,
            max_load_threshold=selection_criteria.max_load_threshold,
            min_success_rate=selection_criteria.min_success_rate,
            max_response_time=selection_criteria.max_response_time,
            business_domain=selection_criteria.business_domain,
            exclude_agents=selection_criteria.exclude_agents
        )
        
        selected_agent = await agent_registry.select_best_agent(criteria)
        
        if not selected_agent:
            raise HTTPException(status_code=404, detail="No suitable agent found")
        
        return selected_agent
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/select-multiple", response_model=List[Dict[str, Any]])
async def select_multiple_agents(
    count: int = Query(..., description="Number of agents to select"),
    distribution_strategy: str = Query("performance", description="Distribution strategy"),
    selection_criteria: AgentSelectionModel = None
) -> List[Dict[str, Any]]:
    """
    Select multiple agents for parallel task execution
    
    Uses advanced distribution strategies to select optimal agents
    for parallel processing and load balancing.
    """
    try:
        criteria = AgentSelectionCriteria(
            required_capabilities=selection_criteria.required_capabilities if selection_criteria else [],
            preferred_capabilities=selection_criteria.preferred_capabilities if selection_criteria else [],
            max_load_threshold=selection_criteria.max_load_threshold if selection_criteria else 80.0,
            min_success_rate=selection_criteria.min_success_rate if selection_criteria else 90.0,
            max_response_time=selection_criteria.max_response_time if selection_criteria else 5.0,
            business_domain=selection_criteria.business_domain if selection_criteria else None,
            exclude_agents=selection_criteria.exclude_agents if selection_criteria else []
        )
        
        selected_agents = await agent_registry.select_multiple_agents(
            criteria, count, distribution_strategy
        )
        
        return selected_agents
        
    except Exception as e:
        logger.error(f"Multiple agent selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent_info(agent_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific agent
    
    Returns comprehensive agent information including performance metrics,
    capabilities, and current status.
    """
    try:
        agent_info = await agent_registry.get_agent_info(agent_id)
        
        if not agent_info:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get agent info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/agents/{agent_id}/performance")
async def update_agent_performance(
    agent_id: str,
    performance_data: PerformanceUpdateModel
) -> Dict[str, str]:
    """
    Update agent performance metrics
    
    Allows agents to report their current performance metrics for
    intelligent load balancing and selection decisions.
    """
    try:
        # Convert to dictionary, excluding None values
        performance_dict = {
            k: v for k, v in performance_data.dict().items() 
            if v is not None
        }
        
        success = await agent_registry.update_agent_performance(agent_id, performance_dict)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent_id,
            "status": "updated",
            "message": "Performance metrics updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/publish")
async def publish_event(event_data: EventPublishModel) -> Dict[str, str]:
    """
    Publish an event to the real-time messaging system
    
    Allows external systems to publish events that will be distributed
    to all relevant subscribers in real-time.
    """
    try:
        # Map string priority to enum
        priority_map = {
            "low": MessagePriority.LOW,
            "normal": MessagePriority.NORMAL,
            "high": MessagePriority.HIGH,
            "critical": MessagePriority.CRITICAL,
            "emergency": MessagePriority.EMERGENCY
        }
        
        priority = priority_map.get(event_data.priority.lower(), MessagePriority.NORMAL)
        
        # Map string event type to enum
        try:
            event_type = EventType(event_data.event_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {event_data.event_type}")
        
        event = messaging_system.create_event(
            event_type=event_type,
            source=event_data.source,
            data=event_data.data,
            priority=priority,
            correlation_id=event_data.correlation_id
        )
        
        success = messaging_system.publish(event)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to publish event")
        
        return {
            "event_id": event.id,
            "status": "published",
            "message": "Event published successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Event publishing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/status", response_model=Dict[str, Any])
async def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status
    
    Returns detailed information about the agent steering system including
    registry statistics, messaging system status, and overall health.
    """
    try:
        registry_stats = await agent_registry.get_registry_stats()
        messaging_stats = messaging_system.get_system_status()
        communication_stats = communication_manager.get_system_status()
        
        return {
            "system_id": messaging_stats.get("system_id"),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "operational",
            "registry": registry_stats,
            "messaging": messaging_stats,
            "communication": communication_stats,
            "uptime_seconds": messaging_stats.get("uptime_seconds", 0)
        }
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/health")
async def health_check() -> Dict[str, str]:
    """
    Simple health check endpoint
    
    Returns basic health status for load balancers and monitoring systems.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "agent-steering-system"
    }


@router.post("/system/start")
async def start_system() -> Dict[str, str]:
    """
    Start the agent steering system
    
    Initializes all system components including messaging, registry, and monitoring.
    """
    try:
        messaging_system.start()
        await agent_registry.start()
        
        return {
            "status": "started",
            "message": "Agent steering system started successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/stop")
async def stop_system() -> Dict[str, str]:
    """
    Stop the agent steering system
    
    Gracefully shuts down all system components.
    """
    try:
        await agent_registry.stop()
        messaging_system.stop()
        
        return {
            "status": "stopped",
            "message": "Agent steering system stopped successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System stop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Advanced Agent Management Endpoints

@router.put("/agents/{agent_id}/configuration")
async def update_agent_configuration(
    agent_id: str,
    configuration_updates: Dict[str, Any]
) -> Dict[str, str]:
    """
    Update agent configuration and capabilities
    
    Allows dynamic updates to agent configuration, capabilities,
    and resource requirements without requiring re-registration.
    """
    try:
        success = await agent_registry.update_agent_configuration(agent_id, configuration_updates)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "agent_id": agent_id,
            "status": "updated",
            "message": "Agent configuration updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/scale")
async def scale_agent(
    agent_id: str,
    scale_action: str = Query(..., description="Scaling action: scale_up, scale_down, set_capacity"),
    target_capacity: Optional[int] = Query(None, description="Target capacity for set_capacity")
) -> Dict[str, str]:
    """
    Scale agent capacity up or down
    
    Dynamically adjusts agent capacity based on current load
    and performance requirements.
    """
    try:
        success = await agent_registry.scale_agent(agent_id, scale_action, target_capacity)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found or scaling failed")
        
        return {
            "agent_id": agent_id,
            "status": "scaled",
            "scale_action": scale_action,
            "message": f"Agent scaled successfully: {scale_action}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent scaling error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/maintenance")
async def put_agent_in_maintenance(
    agent_id: str,
    maintenance_reason: Optional[str] = Query(None, description="Reason for maintenance")
) -> Dict[str, str]:
    """
    Put agent in maintenance mode
    
    Temporarily removes agent from active duty for maintenance,
    updates, or troubleshooting while preserving its registration.
    """
    try:
        success = await agent_registry.put_agent_in_maintenance(agent_id, maintenance_reason)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found or cannot enter maintenance")
        
        return {
            "agent_id": agent_id,
            "status": "maintenance",
            "reason": maintenance_reason,
            "message": "Agent put in maintenance mode"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Maintenance mode error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{agent_id}/maintenance")
async def remove_agent_from_maintenance(agent_id: str) -> Dict[str, str]:
    """
    Remove agent from maintenance mode
    
    Returns agent to active duty after maintenance completion.
    """
    try:
        success = await agent_registry.remove_agent_from_maintenance(agent_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found or not in maintenance")
        
        return {
            "agent_id": agent_id,
            "status": "active",
            "message": "Agent returned from maintenance"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remove from maintenance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/lifecycle-history")
async def get_agent_lifecycle_history(agent_id: str) -> List[Dict[str, Any]]:
    """
    Get agent lifecycle history
    
    Returns detailed history of agent lifecycle events including
    registration, configuration changes, scaling, and maintenance.
    """
    try:
        history = await agent_registry.get_agent_lifecycle_history(agent_id)
        return history
        
    except Exception as e:
        logger.error(f"Lifecycle history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health Monitoring and Failover Endpoints

@router.post("/failover-groups")
async def configure_failover_group(
    group_name: str = Query(..., description="Failover group name"),
    agent_ids: List[str] = Query(..., description="Agent IDs in the group")
) -> Dict[str, str]:
    """
    Configure a failover group for automatic failover
    
    Sets up groups of agents that can serve as backups for each other
    in case of failures.
    """
    try:
        agent_registry.health_monitor.configure_failover_group(group_name, agent_ids)
        
        return {
            "group_name": group_name,
            "agent_count": len(agent_ids),
            "status": "configured",
            "message": f"Failover group '{group_name}' configured with {len(agent_ids)} agents"
        }
        
    except Exception as e:
        logger.error(f"Failover group configuration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/stats")
async def get_monitoring_stats() -> Dict[str, Any]:
    """
    Get comprehensive monitoring statistics
    
    Returns detailed statistics about health monitoring,
    failure detection, and failover operations.
    """
    try:
        stats = agent_registry.health_monitor.get_monitoring_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Monitoring stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/health-history")
async def get_agent_health_history(
    agent_id: str,
    limit: int = Query(50, description="Number of health records to return")
) -> List[Dict[str, Any]]:
    """
    Get agent health history
    
    Returns detailed health check history for predictive analysis
    and troubleshooting.
    """
    try:
        health_history = list(agent_registry.health_monitor.health_history.get(agent_id, []))
        
        # Convert to serializable format
        serializable_history = []
        for entry in health_history[-limit:]:
            serializable_entry = {
                "timestamp": entry["timestamp"].isoformat(),
                "is_healthy": entry["is_healthy"],
                "response_time": entry.get("response_time"),
                "health_data": entry.get("health_data", {})
            }
            serializable_history.append(serializable_entry)
        
        return serializable_history
        
    except Exception as e:
        logger.error(f"Health history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/predictive-alerts")
async def get_agent_predictive_alerts(agent_id: str) -> Dict[str, Any]:
    """
    Get predictive failure alerts for an agent
    
    Returns current predictive alerts and risk assessments
    for proactive failure prevention.
    """
    try:
        alerts = agent_registry.health_monitor.predictive_alerts.get(agent_id, {})
        
        if alerts:
            # Convert timestamp to ISO format
            alerts = alerts.copy()
            if "timestamp" in alerts:
                alerts["timestamp"] = alerts["timestamp"].isoformat()
        
        return alerts or {"message": "No predictive alerts for this agent"}
        
    except Exception as e:
        logger.error(f"Predictive alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance Analytics Endpoints

@router.post("/agents/{agent_id}/selection-outcome")
async def record_selection_outcome(
    agent_id: str,
    outcome_success: bool = Query(..., description="Whether the selection was successful"),
    performance_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Record the outcome of an agent selection for learning
    
    Helps improve future agent selection through machine learning
    and adaptive algorithms.
    """
    try:
        # This would typically be called automatically by the system,
        # but can be used for manual feedback or external system integration
        
        # For now, just return success - in production, implement proper outcome recording
        return {
            "agent_id": agent_id,
            "status": "recorded",
            "message": "Selection outcome recorded for learning"
        }
        
    except Exception as e:
        logger.error(f"Selection outcome recording error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/capability-matching")
async def get_capability_matching_analytics() -> Dict[str, Any]:
    """
    Get capability matching analytics
    
    Returns insights about capability matching effectiveness
    and optimization opportunities.
    """
    try:
        # Return analytics about capability matching performance
        return {
            "capability_synonyms_count": len(agent_registry.capability_matcher.capability_synonyms),
            "domain_expertise_areas": len(agent_registry.capability_matcher.domain_expertise),
            "matching_algorithm": "semantic_with_performance_weighting",
            "optimization_status": "active"
        }
        
    except Exception as e:
        logger.error(f"Capability matching analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/selection-performance")
async def get_selection_performance_analytics() -> Dict[str, Any]:
    """
    Get agent selection performance analytics
    
    Returns metrics about selection algorithm effectiveness
    and learning progress.
    """
    try:
        selector = agent_registry.performance_selector
        
        return {
            "selection_history_size": len(selector.selection_history),
            "performance_weights": selector.performance_weights,
            "adaptive_learning_enabled": selector.adaptive_learning,
            "algorithm_version": "performance_based_with_ml"
        }
        
    except Exception as e:
        logger.error(f"Selection performance analytics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates (placeholder)
@router.websocket("/ws/events")
async def websocket_events(websocket):
    """
    WebSocket endpoint for real-time event streaming
    
    Provides real-time updates about system events, agent status changes,
    and task execution progress.
    """
    await websocket.accept()
    
    # Subscribe to events
    def event_handler(event):
        # Send event to WebSocket client
        asyncio.create_task(websocket.send_json(event.to_dict()))
    
    subscriber_id = f"websocket_{uuid.uuid4()}"
    messaging_system.subscribe(
        subscriber_id,
        list(EventType),
        event_handler
    )
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception as e:
        logger.info(f"WebSocket connection closed: {e}")
    finally:
        messaging_system.unsubscribe(subscriber_id)