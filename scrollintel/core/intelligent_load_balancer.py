"""
Intelligent Load Balancer for ScrollIntel

This module provides advanced load balancing with intelligent routing algorithms,
real-time performance monitoring, and automatic failover capabilities.
"""

import asyncio
import logging
import time
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import uuid
import threading
from collections import defaultdict, deque
import statistics

from .enhanced_specialized_agent import (
    EnhancedSpecializedAgent, AgentRequest, AgentResponse, 
    AgentCapability, RequestPriority
)
from .agent_lifecycle import AgentInstance, agent_lifecycle_manager
from .agent_monitoring import agent_monitor, HealthCheckStatus

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Load balancing routing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"
    INTELLIGENT = "intelligent"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"

class HealthStatus(Enum):
    """Agent health status for routing"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class LoadBalancingDecision(Enum):
    """Load balancing decision types"""
    ROUTE = "route"
    QUEUE = "queue"
    REJECT = "reject"
    RETRY = "retry"
    ESCALATE = "escalate"

@dataclass
class RoutingMetrics:
    """Metrics for routing decisions"""
    response_time: float = 0.0
    success_rate: float = 1.0
    current_load: float = 0.0
    queue_length: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    availability: float = 1.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class AgentScore:
    """Scoring information for agent selection"""
    agent_id: str
    instance_id: str
    total_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    routing_metrics: Optional[RoutingMetrics] = None
    selection_reason: str = ""
    confidence: float = 0.0

@dataclass
class RoutingDecision:
    """Complete routing decision information"""
    request_id: str
    decision: LoadBalancingDecision
    selected_agent: Optional[AgentScore] = None
    alternative_agents: List[AgentScore] = field(default_factory=list)
    queue_position: Optional[int] = None
    estimated_wait_time: Optional[float] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT
    decision_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration"""
    default_strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT
    health_check_interval: int = 30
    metrics_collection_interval: int = 10
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60
    max_queue_size: int = 1000
    queue_timeout: int = 300
    enable_predictive_routing: bool = True
    enable_adaptive_weights: bool = True
    performance_window_size: int = 100
    weight_adjustment_factor: float = 0.1
    min_agent_weight: float = 0.1
    max_agent_weight: float = 2.0

class PerformancePredictor:
    """
    Predicts agent performance based on historical data.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._prediction_cache: Dict[str, Tuple[float, datetime]] = {}
        self._cache_ttl = 30  # seconds
    
    def record_performance(self, agent_id: str, response_time: float, 
                         success: bool, load: float):
        """Record agent performance data"""
        performance_data = {
            "timestamp": datetime.now(),
            "response_time": response_time,
            "success": success,
            "load": load,
            "score": self._calculate_performance_score(response_time, success, load)
        }
        
        self._performance_history[agent_id].append(performance_data)
        
        # Invalidate cache for this agent
        self._prediction_cache.pop(agent_id, None)
    
    def predict_performance(self, agent_id: str, current_load: float = 0.0) -> float:
        """Predict agent performance score"""
        # Check cache first
        if agent_id in self._prediction_cache:
            score, timestamp = self._prediction_cache[agent_id]
            if (datetime.now() - timestamp).total_seconds() < self._cache_ttl:
                return score
        
        history = self._performance_history.get(agent_id, deque())
        if not history:
            # No history, return neutral score
            return 0.5
        
        # Simple prediction based on recent performance trend
        recent_scores = [entry["score"] for entry in list(history)[-10:]]
        
        if len(recent_scores) < 3:
            predicted_score = statistics.mean(recent_scores)
        else:
            # Use weighted average with more weight on recent data
            weights = [i + 1 for i in range(len(recent_scores))]
            weighted_sum = sum(score * weight for score, weight in zip(recent_scores, weights))
            weight_sum = sum(weights)
            predicted_score = weighted_sum / weight_sum
        
        # Adjust for current load
        load_factor = max(0.1, 1.0 - current_load)
        predicted_score *= load_factor
        
        # Cache the prediction
        self._prediction_cache[agent_id] = (predicted_score, datetime.now())
        
        return predicted_score
    
    def _calculate_performance_score(self, response_time: float, success: bool, load: float) -> float:
        """Calculate performance score from metrics"""
        if not success:
            return 0.0
        
        # Normalize response time (assume 1 second is baseline)
        time_score = max(0.0, 1.0 - (response_time / 1.0))
        
        # Load penalty
        load_score = max(0.0, 1.0 - load)
        
        # Combined score
        return (time_score * 0.6 + load_score * 0.4)
    
    def get_agent_trend(self, agent_id: str) -> str:
        """Get performance trend for an agent"""
        history = self._performance_history.get(agent_id, deque())
        if len(history) < 5:
            return "insufficient_data"
        
        recent_scores = [entry["score"] for entry in list(history)[-10:]]
        older_scores = [entry["score"] for entry in list(history)[-20:-10]]
        
        if not older_scores:
            return "stable"
        
        recent_avg = statistics.mean(recent_scores)
        older_avg = statistics.mean(older_scores)
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "degrading"
        else:
            return "stable"

class CircuitBreaker:
    """
    Circuit breaker for agent fault tolerance.
    """
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self._agent_states: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "state": "closed",  # closed, open, half-open
            "failure_count": 0,
            "last_failure_time": None,
            "success_count": 0
        })
        self._lock = threading.RLock()
    
    def can_route_to_agent(self, agent_id: str) -> bool:
        """Check if requests can be routed to an agent"""
        with self._lock:
            state_info = self._agent_states[agent_id]
            
            if state_info["state"] == "closed":
                return True
            elif state_info["state"] == "open":
                # Check if timeout has passed
                if (state_info["last_failure_time"] and
                    (datetime.now() - state_info["last_failure_time"]).total_seconds() > self.timeout):
                    state_info["state"] = "half-open"
                    state_info["success_count"] = 0
                    return True
                return False
            elif state_info["state"] == "half-open":
                return True
            
            return False
    
    def record_success(self, agent_id: str):
        """Record successful request"""
        with self._lock:
            state_info = self._agent_states[agent_id]
            
            if state_info["state"] == "half-open":
                state_info["success_count"] += 1
                if state_info["success_count"] >= 3:  # Require 3 successes to close
                    state_info["state"] = "closed"
                    state_info["failure_count"] = 0
            elif state_info["state"] == "closed":
                state_info["failure_count"] = max(0, state_info["failure_count"] - 1)
    
    def record_failure(self, agent_id: str):
        """Record failed request"""
        with self._lock:
            state_info = self._agent_states[agent_id]
            state_info["failure_count"] += 1
            state_info["last_failure_time"] = datetime.now()
            
            if state_info["failure_count"] >= self.failure_threshold:
                state_info["state"] = "open"
                logger.warning(f"Circuit breaker opened for agent {agent_id}")
    
    def get_agent_state(self, agent_id: str) -> str:
        """Get circuit breaker state for an agent"""
        with self._lock:
            return self._agent_states[agent_id]["state"]
    
    def reset_agent(self, agent_id: str):
        """Reset circuit breaker for an agent"""
        with self._lock:
            self._agent_states[agent_id] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure_time": None,
                "success_count": 0
            }

class RequestQueue:
    """
    Intelligent request queue with priority handling.
    """
    
    def __init__(self, max_size: int = 1000, timeout: int = 300):
        self.max_size = max_size
        self.timeout = timeout
        self._queues: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        self._queue_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def enqueue(self, request: AgentRequest) -> bool:
        """Enqueue a request"""
        with self._lock:
            total_size = sum(len(queue) for queue in self._queues.values())
            
            if total_size >= self.max_size:
                return False
            
            self._queues[request.priority].append(request)
            self._queue_times[request.id] = datetime.now()
            return True
    
    def dequeue(self) -> Optional[AgentRequest]:
        """Dequeue highest priority request"""
        with self._lock:
            # Check queues in priority order
            for priority in sorted(RequestPriority, key=lambda p: p.value, reverse=True):
                queue = self._queues[priority]
                
                while queue:
                    request = queue.popleft()
                    queue_time = self._queue_times.pop(request.id, datetime.now())
                    
                    # Check if request has timed out
                    if (datetime.now() - queue_time).total_seconds() > self.timeout:
                        logger.warning(f"Request {request.id} timed out in queue")
                        continue
                    
                    return request
            
            return None
    
    def get_queue_length(self, priority: Optional[RequestPriority] = None) -> int:
        """Get queue length"""
        with self._lock:
            if priority:
                return len(self._queues[priority])
            return sum(len(queue) for queue in self._queues.values())
    
    def get_estimated_wait_time(self, priority: RequestPriority) -> float:
        """Estimate wait time for a priority level"""
        with self._lock:
            # Simple estimation based on queue length and average processing time
            higher_priority_count = sum(
                len(self._queues[p]) for p in RequestPriority 
                if p.value > priority.value
            )
            same_priority_count = len(self._queues[priority])
            
            # Assume 2 seconds average processing time
            return (higher_priority_count + same_priority_count * 0.5) * 2.0
    
    def cleanup_expired(self):
        """Remove expired requests from queues"""
        with self._lock:
            current_time = datetime.now()
            
            for priority, queue in self._queues.items():
                expired_requests = []
                
                for request in queue:
                    queue_time = self._queue_times.get(request.id, current_time)
                    if (current_time - queue_time).total_seconds() > self.timeout:
                        expired_requests.append(request)
                
                for request in expired_requests:
                    try:
                        queue.remove(request)
                        self._queue_times.pop(request.id, None)
                    except ValueError:
                        pass  # Request already removed

class IntelligentLoadBalancer:
    """
    Main intelligent load balancer with advanced routing capabilities.
    """
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.performance_predictor = PerformancePredictor(self.config.performance_window_size)
        self.circuit_breaker = CircuitBreaker(
            self.config.circuit_breaker_threshold,
            self.config.circuit_breaker_timeout
        )
        self.request_queue = RequestQueue(
            self.config.max_queue_size,
            self.config.queue_timeout
        )
        
        # Routing state
        self._agent_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._round_robin_counters: Dict[str, int] = defaultdict(int)
        self._routing_history: deque = deque(maxlen=1000)
        self._routing_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "last_used": None
        })
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._queue_cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._lock = threading.RLock()
    
    async def start(self):
        """Start the load balancer"""
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._queue_cleanup_task = asyncio.create_task(self._queue_cleanup_loop())
        logger.info("Intelligent load balancer started")
    
    async def stop(self):
        """Stop the load balancer"""
        self._shutdown_event.set()
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._queue_cleanup_task:
            self._queue_cleanup_task.cancel()
        
        # Wait for tasks to complete
        tasks = [t for t in [self._monitoring_task, self._queue_cleanup_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Intelligent load balancer stopped")
    
    async def route_request(self, request: AgentRequest) -> RoutingDecision:
        """Route a request to the best available agent"""
        start_time = time.time()
        
        try:
            # Find suitable agents
            suitable_agents = await self._find_suitable_agents(request)
            
            if not suitable_agents:
                # No suitable agents, queue the request
                if self.request_queue.enqueue(request):
                    return RoutingDecision(
                        request_id=request.id,
                        decision=LoadBalancingDecision.QUEUE,
                        queue_position=self.request_queue.get_queue_length(),
                        estimated_wait_time=self.request_queue.get_estimated_wait_time(request.priority),
                        routing_strategy=self.config.default_strategy,
                        decision_time=time.time() - start_time
                    )
                else:
                    return RoutingDecision(
                        request_id=request.id,
                        decision=LoadBalancingDecision.REJECT,
                        routing_strategy=self.config.default_strategy,
                        decision_time=time.time() - start_time,
                        metadata={"reason": "queue_full"}
                    )
            
            # Score and rank agents
            agent_scores = await self._score_agents(suitable_agents, request)
            
            if not agent_scores:
                return RoutingDecision(
                    request_id=request.id,
                    decision=LoadBalancingDecision.REJECT,
                    routing_strategy=self.config.default_strategy,
                    decision_time=time.time() - start_time,
                    metadata={"reason": "no_healthy_agents"}
                )
            
            # Select best agent
            selected_agent = agent_scores[0]
            alternative_agents = agent_scores[1:5]  # Top 5 alternatives
            
            # Record routing decision
            decision = RoutingDecision(
                request_id=request.id,
                decision=LoadBalancingDecision.ROUTE,
                selected_agent=selected_agent,
                alternative_agents=alternative_agents,
                routing_strategy=self.config.default_strategy,
                decision_time=time.time() - start_time
            )
            
            self._record_routing_decision(decision)
            
            return decision
            
        except Exception as e:
            logger.error(f"Error routing request {request.id}: {e}")
            return RoutingDecision(
                request_id=request.id,
                decision=LoadBalancingDecision.REJECT,
                routing_strategy=self.config.default_strategy,
                decision_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    async def _find_suitable_agents(self, request: AgentRequest) -> List[AgentInstance]:
        """Find agents suitable for handling the request"""
        suitable_agents = []
        
        # Get all healthy agents with required capabilities
        for capability in request.capabilities_required:
            agents = agent_lifecycle_manager.discovery.find_agents_by_capability(capability)
            
            for agent in agents:
                # Check if agent is healthy and available
                if (agent.lifecycle_state.value == "running" and
                    self.circuit_breaker.can_route_to_agent(agent.agent_id) and
                    agent not in suitable_agents):
                    
                    suitable_agents.append(agent)
        
        # If no specific capabilities required, get all healthy agents
        if not request.capabilities_required:
            all_agents = agent_lifecycle_manager.discovery.get_healthy_agents()
            suitable_agents.extend([a for a in all_agents if a not in suitable_agents])
        
        return suitable_agents
    
    async def _score_agents(self, agents: List[AgentInstance], 
                          request: AgentRequest) -> List[AgentScore]:
        """Score and rank agents for request routing"""
        agent_scores = []
        
        for agent in agents:
            try:
                score = await self._calculate_agent_score(agent, request)
                agent_scores.append(score)
            except Exception as e:
                logger.error(f"Error scoring agent {agent.agent_id}: {e}")
                continue
        
        # Sort by total score (descending)
        agent_scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return agent_scores
    
    async def _calculate_agent_score(self, agent: AgentInstance, 
                                   request: AgentRequest) -> AgentScore:
        """Calculate comprehensive score for an agent"""
        component_scores = {}
        
        # Get current metrics
        health_status = agent_monitor.get_agent_status(agent.agent_id)
        metrics = RoutingMetrics()
        
        if health_status and health_status.metrics_summary:
            summary = health_status.metrics_summary
            metrics.current_load = summary.get("current_load", {}).get("current", 0.0)
            metrics.cpu_usage = summary.get("cpu_usage", {}).get("current", 0.0)
            metrics.memory_usage = summary.get("memory_usage", {}).get("current", 0.0)
            metrics.response_time = summary.get("response_time", {}).get("stats", {}).get("mean", 0.0)
            metrics.error_rate = summary.get("error_rate", {}).get("current", 0.0)
            metrics.throughput = summary.get("throughput", {}).get("current", 0.0)
        
        # 1. Load score (lower load = higher score)
        load_score = max(0.0, 1.0 - metrics.current_load)
        component_scores["load"] = load_score
        
        # 2. Performance score (lower response time = higher score)
        if metrics.response_time > 0:
            perf_score = max(0.0, 1.0 - min(1.0, metrics.response_time / 5.0))  # 5s baseline
        else:
            perf_score = 1.0
        component_scores["performance"] = perf_score
        
        # 3. Health score
        if health_status:
            if health_status.overall_status == HealthCheckStatus.PASS:
                health_score = 1.0
            elif health_status.overall_status == HealthCheckStatus.WARN:
                health_score = 0.7
            else:
                health_score = 0.3
        else:
            health_score = 0.5
        component_scores["health"] = health_score
        
        # 4. Resource utilization score
        cpu_score = max(0.0, 1.0 - metrics.cpu_usage / 100.0)
        memory_score = max(0.0, 1.0 - metrics.memory_usage / 100.0)
        resource_score = (cpu_score + memory_score) / 2.0
        component_scores["resources"] = resource_score
        
        # 5. Error rate score (lower error rate = higher score)
        error_score = max(0.0, 1.0 - metrics.error_rate)
        component_scores["reliability"] = error_score
        
        # 6. Capability match score
        capability_score = self._calculate_capability_score(agent, request)
        component_scores["capability"] = capability_score
        
        # 7. Predictive score
        if self.config.enable_predictive_routing:
            predictive_score = self.performance_predictor.predict_performance(
                agent.agent_id, metrics.current_load
            )
            component_scores["predictive"] = predictive_score
        else:
            predictive_score = 0.5
        
        # 8. Priority bonus for high-priority requests
        priority_score = 1.0
        if request.priority == RequestPriority.CRITICAL:
            priority_score = 1.2
        elif request.priority == RequestPriority.HIGH:
            priority_score = 1.1
        component_scores["priority"] = priority_score
        
        # Calculate weighted total score
        weights = {
            "load": 0.20,
            "performance": 0.20,
            "health": 0.15,
            "resources": 0.15,
            "reliability": 0.10,
            "capability": 0.10,
            "predictive": 0.05,
            "priority": 0.05
        }
        
        total_score = sum(
            component_scores[component] * weight 
            for component, weight in weights.items()
        )
        
        # Apply agent weight if adaptive weights are enabled
        if self.config.enable_adaptive_weights:
            agent_weight = self._agent_weights[agent.agent_id]
            total_score *= agent_weight
        
        # Generate selection reason
        top_components = sorted(
            component_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        selection_reason = f"Top factors: {', '.join([f'{comp}({score:.2f})' for comp, score in top_components])}"
        
        return AgentScore(
            agent_id=agent.agent_id,
            instance_id=agent.instance_id,
            total_score=total_score,
            component_scores=component_scores,
            routing_metrics=metrics,
            selection_reason=selection_reason,
            confidence=min(1.0, total_score)
        )
    
    def _calculate_capability_score(self, agent: AgentInstance, 
                                  request: AgentRequest) -> float:
        """Calculate capability match score"""
        if not request.capabilities_required:
            return 1.0
        
        agent_capabilities = {cap.value for cap in agent.capabilities}
        required_capabilities = set(request.capabilities_required)
        
        # Perfect match
        if required_capabilities.issubset(agent_capabilities):
            # Bonus for exact match vs over-qualified
            if len(agent_capabilities) == len(required_capabilities):
                return 1.0
            else:
                return 0.9  # Slightly lower for over-qualified
        
        # Partial match
        matched = len(required_capabilities.intersection(agent_capabilities))
        total_required = len(required_capabilities)
        
        return matched / total_required if total_required > 0 else 0.0
    
    def _record_routing_decision(self, decision: RoutingDecision):
        """Record routing decision for analytics"""
        with self._lock:
            self._routing_history.append(decision)
            
            if decision.selected_agent:
                agent_id = decision.selected_agent.agent_id
                stats = self._routing_stats[agent_id]
                stats["total_requests"] += 1
                stats["last_used"] = datetime.now()
    
    async def record_request_result(self, request_id: str, agent_id: str, 
                                  response_time: float, success: bool):
        """Record the result of a routed request"""
        with self._lock:
            stats = self._routing_stats[agent_id]
            
            if success:
                stats["successful_requests"] += 1
                self.circuit_breaker.record_success(agent_id)
            else:
                stats["failed_requests"] += 1
                self.circuit_breaker.record_failure(agent_id)
            
            # Update average response time
            total_requests = stats["total_requests"]
            if total_requests > 0:
                current_avg = stats["average_response_time"]
                stats["average_response_time"] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )
        
        # Record for performance prediction
        agent_instance = agent_lifecycle_manager.discovery.get_agent_instance(agent_id)
        if agent_instance:
            current_load = 0.0
            if agent_instance.metrics:
                current_load = agent_instance.metrics.get("current_load", {}).get("current", 0.0)
            
            self.performance_predictor.record_performance(
                agent_id, response_time, success, current_load
            )
        
        # Update adaptive weights
        if self.config.enable_adaptive_weights:
            self._update_agent_weight(agent_id, success, response_time)
    
    def _update_agent_weight(self, agent_id: str, success: bool, response_time: float):
        """Update agent weight based on performance"""
        current_weight = self._agent_weights[agent_id]
        
        # Calculate performance factor
        if success and response_time < 2.0:  # Good performance
            performance_factor = 1.0 + self.config.weight_adjustment_factor
        elif success and response_time < 5.0:  # Average performance
            performance_factor = 1.0
        else:  # Poor performance
            performance_factor = 1.0 - self.config.weight_adjustment_factor
        
        # Update weight with bounds
        new_weight = current_weight * performance_factor
        new_weight = max(self.config.min_agent_weight, 
                        min(self.config.max_agent_weight, new_weight))
        
        self._agent_weights[agent_id] = new_weight
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                await self._update_agent_metrics()
                await asyncio.sleep(self.config.metrics_collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.metrics_collection_interval)
    
    async def _queue_cleanup_loop(self):
        """Background queue cleanup loop"""
        while not self._shutdown_event.is_set():
            try:
                self.request_queue.cleanup_expired()
                await asyncio.sleep(60)  # Cleanup every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _update_agent_metrics(self):
        """Update cached agent metrics"""
        # This would update routing metrics from the monitoring system
        # For now, it's handled in the scoring function
        pass
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        with self._lock:
            total_requests = sum(stats["total_requests"] for stats in self._routing_stats.values())
            successful_requests = sum(stats["successful_requests"] for stats in self._routing_stats.values())
            
            return {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0,
                "active_agents": len(self._routing_stats),
                "queue_length": self.request_queue.get_queue_length(),
                "circuit_breaker_open": len([
                    agent_id for agent_id in self._routing_stats.keys()
                    if self.circuit_breaker.get_agent_state(agent_id) == "open"
                ]),
                "routing_decisions": len(self._routing_history),
                "agent_weights": dict(self._agent_weights),
                "agent_stats": dict(self._routing_stats)
            }
    
    def get_agent_performance_trend(self, agent_id: str) -> str:
        """Get performance trend for an agent"""
        return self.performance_predictor.get_agent_trend(agent_id)
    
    async def process_queued_requests(self) -> int:
        """Process queued requests when agents become available"""
        processed = 0
        
        while True:
            request = self.request_queue.dequeue()
            if not request:
                break
            
            # Try to route the queued request
            decision = await self.route_request(request)
            
            if decision.decision == LoadBalancingDecision.ROUTE:
                processed += 1
                # In a real implementation, this would actually send the request
                logger.debug(f"Processed queued request {request.id}")
            else:
                # Re-queue if still can't route
                if decision.decision == LoadBalancingDecision.QUEUE:
                    self.request_queue.enqueue(request)
                break
        
        return processed

# Global load balancer instance
intelligent_load_balancer = IntelligentLoadBalancer()

# Utility functions
async def start_load_balancer(config: Optional[LoadBalancerConfig] = None):
    """Start the global load balancer"""
    if config:
        global intelligent_load_balancer
        intelligent_load_balancer = IntelligentLoadBalancer(config)
    
    await intelligent_load_balancer.start()

async def stop_load_balancer():
    """Stop the global load balancer"""
    await intelligent_load_balancer.stop()

async def route_request(request: AgentRequest) -> RoutingDecision:
    """Route a request using the global load balancer"""
    return await intelligent_load_balancer.route_request(request)

async def record_request_result(request_id: str, agent_id: str, 
                              response_time: float, success: bool):
    """Record request result for load balancer optimization"""
    await intelligent_load_balancer.record_request_result(
        request_id, agent_id, response_time, success
    )

def get_load_balancer_stats() -> Dict[str, Any]:
    """Get load balancer statistics"""
    return intelligent_load_balancer.get_load_balancer_stats()

async def process_queued_requests() -> int:
    """Process any queued requests"""
    return await intelligent_load_balancer.process_queued_requests()