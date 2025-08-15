"""
Agent Lifecycle Management System for ScrollIntel

This module provides comprehensive lifecycle management for agents including
registration, discovery, monitoring, scaling, and replacement capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
import threading
from collections import defaultdict, deque
import weakref

from .enhanced_specialized_agent import (
    EnhancedSpecializedAgent, AgentCapability, AgentStatus, 
    AgentConfiguration, AgentMetrics, HealthStatus
)
from .agent_monitoring import agent_monitor, HealthCheckStatus

logger = logging.getLogger(__name__)

class LifecycleState(Enum):
    """Agent lifecycle states"""
    CREATED = "created"
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    SCALING = "scaling"
    REPLACING = "replacing"
    DECOMMISSIONING = "decommissioning"

class ScalingDirection(Enum):
    """Scaling directions"""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"

class ReplacementReason(Enum):
    """Reasons for agent replacement"""
    HEALTH_FAILURE = "health_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    VERSION_UPDATE = "version_update"
    CONFIGURATION_CHANGE = "configuration_change"
    MANUAL_REQUEST = "manual_request"
    RESOURCE_OPTIMIZATION = "resource_optimization"

@dataclass
class AgentInstance:
    """Agent instance information"""
    instance_id: str
    agent_id: str
    agent_type: str
    capabilities: List[AgentCapability]
    configuration: AgentConfiguration
    lifecycle_state: LifecycleState = LifecycleState.CREATED
    health_status: Optional[HealthStatus] = None
    metrics: Optional[AgentMetrics] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    failure_count: int = 0
    restart_count: int = 0
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_ref: Optional[weakref.ReferenceType] = None

@dataclass
class ScalingPolicy:
    """Scaling policy configuration"""
    agent_type: str
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_request_rate: float = 100.0  # requests per minute
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScalingEvent:
    """Scaling event record"""
    id: str
    agent_type: str
    direction: ScalingDirection
    from_count: int
    to_count: int
    reason: str
    trigger_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    error: Optional[str] = None
    duration: float = 0.0

@dataclass
class ReplacementPlan:
    """Agent replacement plan"""
    id: str
    instance_id: str
    reason: ReplacementReason
    new_configuration: Optional[AgentConfiguration] = None
    new_version: Optional[str] = None
    graceful_shutdown: bool = True
    timeout: int = 300
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None

class AgentDiscovery:
    """
    Agent discovery and registration service.
    """
    
    def __init__(self):
        self._registered_agents: Dict[str, AgentInstance] = {}
        self._capability_index: Dict[str, Set[str]] = defaultdict(set)
        self._type_index: Dict[str, Set[str]] = defaultdict(set)
        self._discovery_handlers: List[Callable] = []
        self._lock = threading.RLock()
    
    def register_agent(self, agent: EnhancedSpecializedAgent, 
                      metadata: Optional[Dict[str, Any]] = None) -> AgentInstance:
        """Register an agent instance"""
        instance = AgentInstance(
            instance_id=str(uuid.uuid4()),
            agent_id=agent.agent_id,
            agent_type=type(agent).__name__,
            capabilities=agent.capabilities,
            configuration=agent.config,
            version=agent.version,
            metadata=metadata or {},
            agent_ref=weakref.ref(agent)
        )
        
        with self._lock:
            self._registered_agents[instance.instance_id] = instance
            
            # Update indexes
            for capability in agent.capabilities:
                self._capability_index[capability.value].add(instance.instance_id)
            
            self._type_index[instance.agent_type].add(instance.instance_id)
        
        # Notify discovery handlers
        self._notify_discovery_handlers("registered", instance)
        
        logger.info(f"Registered agent instance {instance.instance_id} ({instance.agent_type})")
        return instance
    
    def unregister_agent(self, instance_id: str) -> bool:
        """Unregister an agent instance"""
        with self._lock:
            if instance_id not in self._registered_agents:
                return False
            
            instance = self._registered_agents[instance_id]
            
            # Remove from indexes
            for capability in instance.capabilities:
                self._capability_index[capability.value].discard(instance_id)
                if not self._capability_index[capability.value]:
                    del self._capability_index[capability.value]
            
            self._type_index[instance.agent_type].discard(instance_id)
            if not self._type_index[instance.agent_type]:
                del self._type_index[instance.agent_type]
            
            del self._registered_agents[instance_id]
        
        # Notify discovery handlers
        self._notify_discovery_handlers("unregistered", instance)
        
        logger.info(f"Unregistered agent instance {instance_id}")
        return True
    
    def find_agents_by_capability(self, capability: str) -> List[AgentInstance]:
        """Find agents by capability"""
        with self._lock:
            instance_ids = self._capability_index.get(capability, set())
            return [self._registered_agents[iid] for iid in instance_ids 
                   if iid in self._registered_agents]
    
    def find_agents_by_type(self, agent_type: str) -> List[AgentInstance]:
        """Find agents by type"""
        with self._lock:
            instance_ids = self._type_index.get(agent_type, set())
            return [self._registered_agents[iid] for iid in instance_ids 
                   if iid in self._registered_agents]
    
    def get_agent_instance(self, instance_id: str) -> Optional[AgentInstance]:
        """Get agent instance by ID"""
        with self._lock:
            return self._registered_agents.get(instance_id)
    
    def list_all_agents(self) -> List[AgentInstance]:
        """List all registered agents"""
        with self._lock:
            return list(self._registered_agents.values())
    
    def get_healthy_agents(self, capability: Optional[str] = None,
                          agent_type: Optional[str] = None) -> List[AgentInstance]:
        """Get healthy agents matching criteria"""
        agents = []
        
        if capability:
            agents = self.find_agents_by_capability(capability)
        elif agent_type:
            agents = self.find_agents_by_type(agent_type)
        else:
            agents = self.list_all_agents()
        
        # Filter by health status
        healthy_agents = []
        for agent in agents:
            if (agent.lifecycle_state == LifecycleState.RUNNING and
                agent.health_status and 
                agent.health_status.status in ["healthy", "degraded"]):
                healthy_agents.append(agent)
        
        return healthy_agents
    
    def add_discovery_handler(self, handler: Callable):
        """Add discovery event handler"""
        self._discovery_handlers.append(handler)
    
    def _notify_discovery_handlers(self, event_type: str, instance: AgentInstance):
        """Notify discovery handlers of events"""
        for handler in self._discovery_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.create_task(handler(event_type, instance))
                else:
                    handler(event_type, instance)
            except Exception as e:
                logger.error(f"Discovery handler error: {e}")

class AgentScaler:
    """
    Automatic agent scaling based on metrics and policies.
    """
    
    def __init__(self, discovery: AgentDiscovery):
        self.discovery = discovery
        self._scaling_policies: Dict[str, ScalingPolicy] = {}
        self._scaling_history: deque = deque(maxlen=1000)
        self._last_scaling_times: Dict[str, datetime] = {}
        self._scaling_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._lock = threading.RLock()
    
    def set_scaling_policy(self, policy: ScalingPolicy):
        """Set scaling policy for an agent type"""
        with self._lock:
            self._scaling_policies[policy.agent_type] = policy
        logger.info(f"Set scaling policy for {policy.agent_type}")
    
    def get_scaling_policy(self, agent_type: str) -> Optional[ScalingPolicy]:
        """Get scaling policy for an agent type"""
        with self._lock:
            return self._scaling_policies.get(agent_type)
    
    async def start_scaling_monitor(self, check_interval: int = 60):
        """Start the scaling monitor"""
        self._scaling_task = asyncio.create_task(
            self._scaling_monitor_loop(check_interval)
        )
        logger.info("Started agent scaling monitor")
    
    async def stop_scaling_monitor(self):
        """Stop the scaling monitor"""
        self._shutdown_event.set()
        if self._scaling_task:
            self._scaling_task.cancel()
            try:
                await self._scaling_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped agent scaling monitor")
    
    async def _scaling_monitor_loop(self, interval: int):
        """Main scaling monitor loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._evaluate_scaling_needs()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scaling monitor error: {e}")
                await asyncio.sleep(interval)
    
    async def _evaluate_scaling_needs(self):
        """Evaluate scaling needs for all agent types"""
        with self._lock:
            policies = list(self._scaling_policies.values())
        
        for policy in policies:
            if not policy.enabled:
                continue
            
            try:
                await self._evaluate_agent_type_scaling(policy)
            except Exception as e:
                logger.error(f"Error evaluating scaling for {policy.agent_type}: {e}")
    
    async def _evaluate_agent_type_scaling(self, policy: ScalingPolicy):
        """Evaluate scaling needs for a specific agent type"""
        agents = self.discovery.find_agents_by_type(policy.agent_type)
        healthy_agents = [a for a in agents if a.lifecycle_state == LifecycleState.RUNNING]
        
        current_count = len(healthy_agents)
        
        # Check cooldown periods
        last_scaling = self._last_scaling_times.get(policy.agent_type)
        if last_scaling:
            time_since_scaling = (datetime.now() - last_scaling).total_seconds()
            if time_since_scaling < policy.scale_up_cooldown:
                return
        
        # Calculate average metrics
        if not healthy_agents:
            # No healthy agents, scale up to minimum
            if current_count < policy.min_instances:
                await self._scale_agents(policy, ScalingDirection.UP, 
                                       policy.min_instances - current_count,
                                       "No healthy agents available")
            return
        
        avg_cpu = sum(a.metrics.cpu_usage for a in healthy_agents if a.metrics) / len(healthy_agents)
        avg_memory = sum(a.metrics.memory_usage for a in healthy_agents if a.metrics) / len(healthy_agents)
        avg_load = sum(a.metrics.current_load for a in healthy_agents if a.metrics) / len(healthy_agents)
        
        # Determine scaling decision
        scaling_decision = self._make_scaling_decision(
            policy, current_count, avg_cpu, avg_memory, avg_load
        )
        
        if scaling_decision["action"] != ScalingDirection.MAINTAIN:
            await self._scale_agents(
                policy, 
                scaling_decision["action"],
                scaling_decision["count"],
                scaling_decision["reason"],
                scaling_decision["metrics"]
            )
    
    def _make_scaling_decision(self, policy: ScalingPolicy, current_count: int,
                             avg_cpu: float, avg_memory: float, avg_load: float) -> Dict[str, Any]:
        """Make scaling decision based on metrics"""
        
        # Check if we need to scale up
        scale_up_reasons = []
        if avg_cpu > policy.scale_up_threshold:
            scale_up_reasons.append(f"CPU usage {avg_cpu:.1f}% > {policy.scale_up_threshold}%")
        if avg_memory > policy.scale_up_threshold:
            scale_up_reasons.append(f"Memory usage {avg_memory:.1f}% > {policy.scale_up_threshold}%")
        if avg_load > policy.scale_up_threshold / 100:
            scale_up_reasons.append(f"Load {avg_load:.1f} > {policy.scale_up_threshold/100:.1f}")
        
        if scale_up_reasons and current_count < policy.max_instances:
            scale_count = min(2, policy.max_instances - current_count)  # Scale by 1-2 instances
            return {
                "action": ScalingDirection.UP,
                "count": scale_count,
                "reason": "; ".join(scale_up_reasons),
                "metrics": {"cpu": avg_cpu, "memory": avg_memory, "load": avg_load}
            }
        
        # Check if we need to scale down
        scale_down_reasons = []
        if avg_cpu < policy.scale_down_threshold:
            scale_down_reasons.append(f"CPU usage {avg_cpu:.1f}% < {policy.scale_down_threshold}%")
        if avg_memory < policy.scale_down_threshold:
            scale_down_reasons.append(f"Memory usage {avg_memory:.1f}% < {policy.scale_down_threshold}%")
        if avg_load < policy.scale_down_threshold / 100:
            scale_down_reasons.append(f"Load {avg_load:.1f} < {policy.scale_down_threshold/100:.1f}")
        
        if (len(scale_down_reasons) >= 2 and  # Multiple metrics indicate low usage
            current_count > policy.min_instances):
            scale_count = min(1, current_count - policy.min_instances)  # Scale down by 1
            return {
                "action": ScalingDirection.DOWN,
                "count": scale_count,
                "reason": "; ".join(scale_down_reasons),
                "metrics": {"cpu": avg_cpu, "memory": avg_memory, "load": avg_load}
            }
        
        return {
            "action": ScalingDirection.MAINTAIN,
            "count": 0,
            "reason": "Metrics within acceptable range",
            "metrics": {"cpu": avg_cpu, "memory": avg_memory, "load": avg_load}
        }
    
    async def _scale_agents(self, policy: ScalingPolicy, direction: ScalingDirection,
                          count: int, reason: str, metrics: Dict[str, float] = None):
        """Execute scaling action"""
        agents = self.discovery.find_agents_by_type(policy.agent_type)
        current_count = len([a for a in agents if a.lifecycle_state == LifecycleState.RUNNING])
        
        scaling_event = ScalingEvent(
            id=str(uuid.uuid4()),
            agent_type=policy.agent_type,
            direction=direction,
            from_count=current_count,
            to_count=current_count + (count if direction == ScalingDirection.UP else -count),
            reason=reason,
            trigger_metrics=metrics or {}
        )
        
        start_time = datetime.now()
        
        try:
            if direction == ScalingDirection.UP:
                success = await self._scale_up(policy, count)
            else:
                success = await self._scale_down(policy, count)
            
            scaling_event.success = success
            scaling_event.duration = (datetime.now() - start_time).total_seconds()
            
            if success:
                self._last_scaling_times[policy.agent_type] = datetime.now()
                logger.info(f"Successfully scaled {policy.agent_type} {direction.value} by {count}")
            else:
                logger.error(f"Failed to scale {policy.agent_type} {direction.value} by {count}")
        
        except Exception as e:
            scaling_event.success = False
            scaling_event.error = str(e)
            scaling_event.duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"Scaling error for {policy.agent_type}: {e}")
        
        finally:
            self._scaling_history.append(scaling_event)
    
    async def _scale_up(self, policy: ScalingPolicy, count: int) -> bool:
        """Scale up agents"""
        # This would create new agent instances
        # For now, just log the action
        logger.info(f"Would create {count} new instances of {policy.agent_type}")
        return True
    
    async def _scale_down(self, policy: ScalingPolicy, count: int) -> bool:
        """Scale down agents"""
        agents = self.discovery.find_agents_by_type(policy.agent_type)
        running_agents = [a for a in agents if a.lifecycle_state == LifecycleState.RUNNING]
        
        # Select agents to terminate (prefer least loaded)
        agents_to_terminate = sorted(running_agents, 
                                   key=lambda a: a.metrics.current_load if a.metrics else 0)[:count]
        
        for agent in agents_to_terminate:
            agent.lifecycle_state = LifecycleState.STOPPING
            # This would actually stop the agent
            logger.info(f"Would terminate agent instance {agent.instance_id}")
        
        return True
    
    def get_scaling_history(self, agent_type: Optional[str] = None) -> List[ScalingEvent]:
        """Get scaling history"""
        if agent_type:
            return [event for event in self._scaling_history if event.agent_type == agent_type]
        return list(self._scaling_history)

class AgentReplacer:
    """
    Manages agent replacement for health failures, updates, etc.
    """
    
    def __init__(self, discovery: AgentDiscovery):
        self.discovery = discovery
        self._replacement_plans: Dict[str, ReplacementPlan] = {}
        self._replacement_handlers: List[Callable] = []
        self._lock = threading.RLock()
    
    async def replace_agent(self, instance_id: str, reason: ReplacementReason,
                          new_configuration: Optional[AgentConfiguration] = None,
                          new_version: Optional[str] = None,
                          graceful: bool = True) -> ReplacementPlan:
        """Create and execute agent replacement plan"""
        
        instance = self.discovery.get_agent_instance(instance_id)
        if not instance:
            raise ValueError(f"Agent instance {instance_id} not found")
        
        plan = ReplacementPlan(
            id=str(uuid.uuid4()),
            instance_id=instance_id,
            reason=reason,
            new_configuration=new_configuration,
            new_version=new_version,
            graceful_shutdown=graceful
        )
        
        with self._lock:
            self._replacement_plans[plan.id] = plan
        
        # Execute replacement
        plan.executed_at = datetime.now()
        
        try:
            success = await self._execute_replacement(plan, instance)
            plan.success = success
            
            if success:
                logger.info(f"Successfully replaced agent {instance_id}")
            else:
                logger.error(f"Failed to replace agent {instance_id}")
        
        except Exception as e:
            plan.success = False
            plan.error = str(e)
            logger.error(f"Replacement error for {instance_id}: {e}")
        
        finally:
            plan.completed_at = datetime.now()
        
        # Notify handlers
        for handler in self._replacement_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(plan, instance)
                else:
                    handler(plan, instance)
            except Exception as e:
                logger.error(f"Replacement handler error: {e}")
        
        return plan
    
    async def _execute_replacement(self, plan: ReplacementPlan, 
                                 instance: AgentInstance) -> bool:
        """Execute agent replacement"""
        try:
            # Mark instance as being replaced
            instance.lifecycle_state = LifecycleState.REPLACING
            
            # Get the actual agent reference
            agent_ref = instance.agent_ref
            if not agent_ref:
                logger.error(f"No agent reference for instance {instance.instance_id}")
                return False
            
            agent = agent_ref()
            if not agent:
                logger.error(f"Agent reference is dead for instance {instance.instance_id}")
                return False
            
            # Graceful shutdown if requested
            if plan.graceful_shutdown:
                logger.info(f"Gracefully shutting down agent {instance.instance_id}")
                await agent.shutdown()
            
            # Create new agent with updated configuration/version
            new_config = plan.new_configuration or instance.configuration
            new_version = plan.new_version or instance.version
            
            # This would create a new agent instance
            # For now, just update the existing instance
            instance.configuration = new_config
            instance.version = new_version
            instance.lifecycle_state = LifecycleState.RUNNING
            instance.restart_count += 1
            
            logger.info(f"Replaced agent {instance.instance_id} with new configuration")
            return True
        
        except Exception as e:
            logger.error(f"Failed to execute replacement: {e}")
            instance.lifecycle_state = LifecycleState.FAILED
            return False
    
    def add_replacement_handler(self, handler: Callable):
        """Add replacement event handler"""
        self._replacement_handlers.append(handler)
    
    def get_replacement_plan(self, plan_id: str) -> Optional[ReplacementPlan]:
        """Get replacement plan by ID"""
        with self._lock:
            return self._replacement_plans.get(plan_id)
    
    def list_replacement_plans(self, instance_id: Optional[str] = None) -> List[ReplacementPlan]:
        """List replacement plans"""
        with self._lock:
            plans = list(self._replacement_plans.values())
            if instance_id:
                plans = [p for p in plans if p.instance_id == instance_id]
            return plans

class AgentLifecycleManager:
    """
    Main agent lifecycle management system.
    """
    
    def __init__(self):
        self.discovery = AgentDiscovery()
        self.scaler = AgentScaler(self.discovery)
        self.replacer = AgentReplacer(self.discovery)
        
        # Health monitoring integration
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Setup event handlers
        self.discovery.add_discovery_handler(self._on_agent_discovery_event)
    
    async def start(self):
        """Start the lifecycle management system"""
        await self.scaler.start_scaling_monitor()
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Agent lifecycle management system started")
    
    async def stop(self):
        """Stop the lifecycle management system"""
        self._shutdown_event.set()
        
        await self.scaler.stop_scaling_monitor()
        
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Agent lifecycle management system stopped")
    
    async def register_agent(self, agent: EnhancedSpecializedAgent,
                           metadata: Optional[Dict[str, Any]] = None) -> AgentInstance:
        """Register an agent with lifecycle management"""
        instance = self.discovery.register_agent(agent, metadata)
        
        # Start monitoring the agent
        await agent_monitor.add_agent(agent.agent_id, agent.name)
        
        # Update instance state
        instance.lifecycle_state = LifecycleState.RUNNING
        instance.started_at = datetime.now()
        instance.last_heartbeat = datetime.now()
        
        return instance
    
    async def unregister_agent(self, instance_id: str) -> bool:
        """Unregister an agent from lifecycle management"""
        instance = self.discovery.get_agent_instance(instance_id)
        if not instance:
            return False
        
        # Stop monitoring
        await agent_monitor.remove_agent(instance.agent_id)
        
        # Update state
        instance.lifecycle_state = LifecycleState.STOPPED
        instance.stopped_at = datetime.now()
        
        # Unregister from discovery
        return self.discovery.unregister_agent(instance_id)
    
    async def _health_monitor_loop(self):
        """Monitor agent health and trigger replacements if needed"""
        while not self._shutdown_event.is_set():
            try:
                await self._check_agent_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _check_agent_health(self):
        """Check health of all registered agents"""
        agents = self.discovery.list_all_agents()
        
        for instance in agents:
            if instance.lifecycle_state != LifecycleState.RUNNING:
                continue
            
            try:
                # Get health status from monitoring system
                health_status = agent_monitor.get_agent_status(instance.agent_id)
                instance.health_status = health_status
                instance.last_heartbeat = datetime.now()
                
                # Check if agent needs replacement
                if health_status.overall_status == HealthCheckStatus.FAIL:
                    instance.failure_count += 1
                    
                    # Replace after 3 consecutive failures
                    if instance.failure_count >= 3:
                        logger.warning(f"Agent {instance.instance_id} has failed health checks, replacing")
                        await self.replacer.replace_agent(
                            instance.instance_id,
                            ReplacementReason.HEALTH_FAILURE
                        )
                else:
                    # Reset failure count on successful health check
                    instance.failure_count = 0
                
                # Update metrics
                instance.metrics = health_status.metrics_summary
                
            except Exception as e:
                logger.error(f"Error checking health for agent {instance.instance_id}: {e}")
                instance.failure_count += 1
    
    async def _on_agent_discovery_event(self, event_type: str, instance: AgentInstance):
        """Handle agent discovery events"""
        logger.debug(f"Agent discovery event: {event_type} for {instance.instance_id}")
        
        if event_type == "registered":
            # Agent registered, start monitoring
            pass
        elif event_type == "unregistered":
            # Agent unregistered, cleanup
            pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        agents = self.discovery.list_all_agents()
        
        status = {
            "total_agents": len(agents),
            "running_agents": len([a for a in agents if a.lifecycle_state == LifecycleState.RUNNING]),
            "failed_agents": len([a for a in agents if a.lifecycle_state == LifecycleState.FAILED]),
            "scaling_policies": len(self.scaler._scaling_policies),
            "recent_scaling_events": len([e for e in self.scaler.get_scaling_history() 
                                        if (datetime.now() - e.timestamp).total_seconds() < 3600]),
            "recent_replacements": len([p for p in self.replacer.list_replacement_plans()
                                      if p.executed_at and (datetime.now() - p.executed_at).total_seconds() < 3600]),
            "agent_types": {}
        }
        
        # Group by agent type
        for agent in agents:
            agent_type = agent.agent_type
            if agent_type not in status["agent_types"]:
                status["agent_types"][agent_type] = {
                    "total": 0,
                    "running": 0,
                    "failed": 0,
                    "healthy": 0
                }
            
            status["agent_types"][agent_type]["total"] += 1
            
            if agent.lifecycle_state == LifecycleState.RUNNING:
                status["agent_types"][agent_type]["running"] += 1
                
                if (agent.health_status and 
                    agent.health_status.overall_status == HealthCheckStatus.PASS):
                    status["agent_types"][agent_type]["healthy"] += 1
            
            elif agent.lifecycle_state == LifecycleState.FAILED:
                status["agent_types"][agent_type]["failed"] += 1
        
        return status

# Global lifecycle manager instance
agent_lifecycle_manager = AgentLifecycleManager()

# Utility functions
async def start_lifecycle_management():
    """Start the global lifecycle management system"""
    await agent_lifecycle_manager.start()

async def stop_lifecycle_management():
    """Stop the global lifecycle management system"""
    await agent_lifecycle_manager.stop()

async def register_agent_with_lifecycle(agent: EnhancedSpecializedAgent,
                                       metadata: Optional[Dict[str, Any]] = None) -> AgentInstance:
    """Register an agent with lifecycle management"""
    return await agent_lifecycle_manager.register_agent(agent, metadata)

def set_scaling_policy(agent_type: str, **policy_kwargs) -> ScalingPolicy:
    """Set scaling policy for an agent type"""
    policy = ScalingPolicy(agent_type=agent_type, **policy_kwargs)
    agent_lifecycle_manager.scaler.set_scaling_policy(policy)
    return policy

async def replace_agent(instance_id: str, reason: ReplacementReason, **kwargs) -> ReplacementPlan:
    """Replace an agent instance"""
    return await agent_lifecycle_manager.replacer.replace_agent(instance_id, reason, **kwargs)

def get_lifecycle_status() -> Dict[str, Any]:
    """Get lifecycle management system status"""
    return agent_lifecycle_manager.get_system_status()

def find_healthy_agents(capability: Optional[str] = None, 
                       agent_type: Optional[str] = None) -> List[AgentInstance]:
    """Find healthy agents matching criteria"""
    return agent_lifecycle_manager.discovery.get_healthy_agents(capability, agent_type)